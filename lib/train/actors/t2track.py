from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy, box_iou
import torch
from lib.train.admin import multigpu
from lib.utils.heapmap_utils import generate_heatmap

class T2Track_Actor(BaseActor):
    """ Actor for training the t2track"""
    def __init__(self, net, objective, loss_weight, settings, cfg):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg
        self.multi_modal_language = cfg.DATA.MULTI_MODAL_LANGUAGE
        self.warm_epoch = self.cfg.TRAIN.WARM_EPOCH

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'search_anno'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        self.epoch = data['epoch']
        n,b,_,_,_ = data['search_images'].shape   # n,b,c,h,w

        # b = data['search_images'].shape[1]   # n,b,c,h,w
        # search_list = data['search_images'].view(-1, *data['search_images'].shape[2:]).split(b,dim=0)  # (n*b, c, h, w)
        template_list = data['template_images'].view(-1, *data['template_images'].shape[2:]).split(b,dim=0)
        outputs_list = []
        new_outputs = {}

        self.net.memory_search.clear()
        for search_images in data['search_images']:
            search_images = search_images.unsqueeze(dim=0)
            search_list = search_images.view(-1, *search_images.shape[2:]).split(b, dim=0)  # (n*b, c, h, w)

            if self.epoch > self.warm_epoch and self.net.use_temporal:  # 使用预测框
                self.net.is_memory = True
                enc_opt = self.net(template_list=template_list,
                                   search_list=search_list,
                                   mode='encoder')
                outputs = self.net(feature=enc_opt, mode="decoder")
                memory_feature = self.net.append_new_memory(outputs['pred_boxes'])
                self.net.memory_search.append(memory_feature)
                outputs_list.append(outputs)
            else:
                self.net.is_memory = False
                enc_opt = self.net(template_list=template_list,
                                   search_list=search_list,
                                   mode='encoder')
                outputs = self.net(feature=enc_opt, mode="decoder")
                # break
                outputs_list.append(outputs)
                # break
        self.net.memory_search.clear()

        for key in outputs_list[0].keys():
            new_outputs[key] = torch.stack([o[key] for o in outputs_list], dim=0)

        return new_outputs

    def compute_losses(self, pred_dict, gt_dict, return_status=True):

        frame_num = gt_dict['search_anno'].shape[0]
        giou_loss_lists = []
        l1_loss_lists=[]
        location_loss_lists=[]
        iou_lists=[]

        weights = torch.ones(frame_num,dtype=torch.float32,device=gt_dict['search_anno'][0].device)
        weights = weights / weights.sum()

        # weights = torch.arange(1, frame_num, dtype=torch.float32, device=gt_dict['search_anno'][0].device) ** 2
        # weights = weights / (weights.sum() * 2)
        # weights = torch.cat([weights, torch.tensor([0.5], device=weights.device)], dim=0)

        for frame_idx in range(frame_num):
            gt_bbox = gt_dict['search_anno'][frame_idx]
            gt_bboxs = gt_dict['search_anno'][frame_idx:frame_idx+1]

            # gt gaussian map
            # gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
            gt_gaussian_maps = generate_heatmap(gt_bboxs, self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.ENCODER.STRIDE) # list of torch.Size([b, H, W])
            gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)  # torch.Size([b, 1, H, W])

            # Get boxes
            pred_boxes = pred_dict['pred_boxes'][frame_idx] # torch.Size([b, 1, 4])
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)

            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
            # compute giou and iou
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            # compute l1 loss
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            # compute location loss
            if 'score_map' in pred_dict:
                location_loss = self.objective['focal'](pred_dict['score_map'][frame_idx], gt_gaussian_maps)
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)

            giou_loss_lists.append(giou_loss)
            iou_lists.append(iou.mean())
            l1_loss_lists.append(l1_loss)
            location_loss_lists.append(location_loss)

        giou_loss = torch.stack(giou_loss_lists)  # [8]
        giou_loss = (giou_loss * weights).sum()
        iou = torch.stack(iou_lists)  # [8]
        iou = (iou * weights).sum()
        l1_loss = torch.stack(l1_loss_lists)  # [8]
        l1_loss = (l1_loss * weights).sum()
        location_loss = torch.stack(location_loss_lists)  # [8]
        location_loss = (location_loss * weights).sum()

        # weighted sum
        loss = (self.loss_weight['giou'] * giou_loss +
                self.loss_weight['l1'] * l1_loss +
                self.loss_weight['focal'] * location_loss )

        if return_status:
            # status for log
            mean_iou = iou.detach()#.mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss