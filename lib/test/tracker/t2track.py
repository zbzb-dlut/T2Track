from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.utils import sample_target, transform_image_to_crop
import cv2
from lib.utils.box_ops import box_xywh_to_xyxy, box_xyxy_to_cxcywh
from lib.test.utils.hann import hann2d
from lib.models.sutrack import build_sutrack
from lib.models.t2track import build_t2track
from lib.test.tracker.utils import Preprocessor
from lib.utils.box_ops import clip_box
import clip
import numpy as np
from lib.test.tracker.memoryupdate import MATRJudge
from lib.models.t2track.memorybankmanager import MemoryBankManager
from lib.models.t2track.Jump_freeze_reopen_manager import JumpFreezeReopenManager

class T2TRACK(BaseTracker):
    def __init__(self, params, dataset_name):
        super(T2TRACK, self).__init__(params)
        network = build_t2track(params.cfg)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.fx_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.ENCODER.STRIDE
        if self.cfg.TEST.WINDOW == True: # for window penalty
            self.output_window = hann2d(torch.tensor([self.fx_sz, self.fx_sz]).long(), centered=True).cuda()

        self.num_template = self.cfg.TEST.NUM_TEMPLATES

        self.debug = params.debug
        self.frame_id = 0
        self.prev_update_frame_id=0
        # self.current_search_factor=1.0
        # online update settings
        DATASET_NAME = dataset_name.upper()
        if hasattr(self.cfg.TEST.UPDATE_INTERVALS, DATASET_NAME):
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS[DATASET_NAME]
        else:
            self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS.DEFAULT
        print("Update interval is: ", self.update_intervals)

        if hasattr(self.cfg.TEST.UPDATE_THRESHOLD, DATASET_NAME):
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD[DATASET_NAME]
        else:
            self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD.DEFAULT
        print("Update threshold is: ", self.update_threshold)

        # mapping similar datasets
        if 'GOT10K' in DATASET_NAME:
            DATASET_NAME = 'GOT10K'
        if 'LASOT' in DATASET_NAME:
            DATASET_NAME = 'LASOT'
        if 'OTB' in DATASET_NAME:
            DATASET_NAME = 'TNL2K'

        # multi modal vision
        if hasattr(self.cfg.TEST.MULTI_MODAL_VISION, DATASET_NAME):
            self.multi_modal_vision = self.cfg.TEST.MULTI_MODAL_VISION[DATASET_NAME]
        else:
            self.multi_modal_vision = self.cfg.TEST.MULTI_MODAL_VISION.DEFAULT
        print("MULTI_MODAL_VISION is: ", self.multi_modal_vision)

        #multi modal language
        if hasattr(self.cfg.TEST.MULTI_MODAL_LANGUAGE, DATASET_NAME):
            self.multi_modal_language = self.cfg.TEST.MULTI_MODAL_LANGUAGE[DATASET_NAME]
        else:
            self.multi_modal_language = self.cfg.TEST.MULTI_MODAL_LANGUAGE.DEFAULT
        print("MULTI_MODAL_LANGUAGE is: ", self.multi_modal_language)

        #using nlp information
        if hasattr(self.cfg.TEST.USE_NLP, DATASET_NAME):
            self.use_nlp = self.cfg.TEST.USE_NLP[DATASET_NAME]
        else:
            self.use_nlp = self.cfg.TEST.USE_NLP.DEFAULT
        print("USE_NLP is: ", self.use_nlp)

        self.task_index_batch = None

        # self.mem_manager = MemoryBankManager(
        #     fx_sz=self.fx_sz,
        #     memory_encoder=self.network.memory_encoder,
        #     min_update_interval=30,
        #     confirm_frames=10,
        #     second_peak_min_ratio=0.30,
        #     peak_suppress_radius=1,
        #     device=self.feature.device if hasattr(self, "feature") else "cuda",
        # )



    def initialize(self, image, info: dict):

        self.network.is_memory=True
        self.network.pred_score_lists.clear()
        self.network.memory_search.clear()

        self.jump_mgr = JumpFreezeReopenManager()

        # get the initial templates
        z_patch_arr, resize_factor = sample_target(image, info['init_bbox'], self.params.template_factor,
                                       output_sz=self.params.template_size)
        z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        if self.multi_modal_vision and (template.size(1) == 3):
            template = torch.cat((template, template), axis=1)
        self.template_list = [template] * self.num_template

        self.state = info['init_bbox']
        prev_box_crop = transform_image_to_crop(torch.tensor(info['init_bbox']),
                                                torch.tensor(info['init_bbox']),
                                                resize_factor,
                                                torch.Tensor([self.params.template_size, self.params.template_size]),
                                                normalize=True)
        self.template_anno_list = [prev_box_crop.to(template.device).unsqueeze(0)]
        self.frame_id = 0

        # language information
        if self.multi_modal_language:
            if self.use_nlp:
                init_nlp = info.get("init_nlp")
            else:
                init_nlp = None
            text_data, _ = self.extract_token_from_nlp_clip(init_nlp)
            text_data = text_data.unsqueeze(0).to(template.device)
            with torch.no_grad():
                self.text_src = self.network.forward_textencoder(text_data=text_data)
        else:
            self.text_src = None


    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1 #self.current_search_factor
        x_patch_arr, resize_factor = sample_target(image, self.state, self.params.search_factor,
                                                   output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        if self.multi_modal_vision and (search.size(1) == 3):
            search = torch.cat((search, search), axis=1)
        search_list = [search]

        # run the encoder
        with torch.no_grad():
            enc_opt = self.network.forward_encoder(self.template_list,
                                                   search_list)

        # run the decoder
        with torch.no_grad():
            out_dict = self.network.forward_decoder(feature=enc_opt)

        # add hann windows
        pred_score_map = out_dict['score_map']
        if self.cfg.TEST.WINDOW == True: # for window penalty
            response = self.output_window * pred_score_map
        else:
            response = pred_score_map
        if 'size_map' in out_dict.keys():
            pred_boxes, conf_score = self.network.decoder.cal_bbox(response, out_dict['size_map'],
                                                                   out_dict['offset_map'], return_score=True)
        else:
            pred_boxes, conf_score = self.network.decoder.cal_bbox(response,
                                                               out_dict['offset_map'],
                                                                   return_score=True)

        # if len(self.network.pred_score_lists)<10:
        #     self.network.pred_score_lists.append(conf_score)
        #     self.network.append_new_memory(pred_boxes)
        # elif len(self.network.pred_score_lists)==10:
        #     self.avg_pred_score = sum(self.network.pred_score_lists) / len(self.network.pred_score_lists)
        #     self.network.pred_score_lists.append(conf_score)
        # else:
        #     if conf_score.item()>=self.avg_pred_score.item():
        #     self.network.append_new_memory(pred_boxes)

        # should_update, info = self.memory_gate.judge_update(
        #     frame_id=self.frame_id,
        #     score_map=pred_score_map,
        #     response_map=response,
        #     pred_box=pred_boxes,
        # )
        #
        # # print(info)
        # if should_update:
        #     # print(info)
        #     self.network.append_new_memory(pred_boxes)

        # if len(self.network.pred_score_lists)<3 and self.frame_id%10==0:
        #     self.network.pred_score_lists.append(conf_score)
        #     self.network.append_new_memory(pred_boxes)
        #     self.prev_update_frame_id = self.frame_id
        # elif len(self.network.pred_score_lists)==3:
        #     self.avg_pred_score = sum(self.network.pred_score_lists) / len(self.network.pred_score_lists)
        #     self.avg_pred_score = torch.clamp(self.avg_pred_score, min=0.4, max=0.8)
        #     self.network.pred_score_lists.append(conf_score)
        # else:
        #     if (self.frame_id-self.prev_update_frame_id)>=30:
        #         if conf_score.item()>=self.avg_pred_score.item():
        #             self.network.append_new_memory(pred_boxes)
        #             self.prev_update_frame_id = self.frame_id

        # # self.prev_box = pred_boxes
        #
        # should_write, info = self.mem_manager.step(
        #     frame_id=self.frame_id,
        #     feature_map=enc_opt,
        #     score_map=response,  # [1,1,14,14]
        #     pred_boxes=pred_boxes,  # [1,4], normalized cxcywh
        # )
        #
        # if should_write:
        #     # 如果你还想保留原来的 memory_search 列表，也可以同步写
        #     self.network.append_new_memory(pred_boxes)

        conf_val = conf_score.item()
        if len(self.network.pred_score_lists) < 6 and self.frame_id % 5 == 0:
            self.network.pred_score_lists.append(conf_score.detach())
            memory_feature=self.network.append_new_memory(pred_boxes)
            self.network.memory_search.append(memory_feature)
            self.prev_update_frame_id = self.frame_id
            _, _, _ = self.jump_mgr.update(
                pred_boxes=pred_boxes*self.fx_sz,
                conf_score=conf_val,
                avg_pred_score=None,
                curr_memory_feat=None,
                anchor_bank=None,
            )
        elif len(self.network.pred_score_lists) == 6:
            self.avg_pred_score = sum(self.network.pred_score_lists) / len(self.network.pred_score_lists)
            self.avg_pred_score = torch.clamp(self.avg_pred_score, min=0.5, max=0.8)
            self.network.pred_score_lists.append(conf_score.detach())
            _, _, _ = self.jump_mgr.update(
                pred_boxes=pred_boxes*self.fx_sz,
                conf_score=conf_val,
                avg_pred_score=None,
                curr_memory_feat=None,
                anchor_bank=None,
            )
        else:
            # 当前ROI memory特征（仅用于 reopen 时和 anchor 比）

            # if len(self.network.memory_search) >= 3:
            #     # 你自己的 append_new_memory 里已经能得到 memory_feature，
            #     # 这里如果没有现成的 curr_memory_feat，就先不传
            #     anchor_bank = self.network.memory_search[:3]

            if len(self.network.pred_score_lists)>=6:
                avg_pred_score = self.avg_pred_score.item()
            else:
                avg_pred_score = None

            if self.jump_mgr.state=="RECOVER_CANDIDATE":
                curr_memory_feat = self.network.append_new_memory(pred_boxes)
                anchor_bank = self.network.memory_search[:3]
                self.prev_update_frame_id = self.frame_id
            else:
                curr_memory_feat = None
                anchor_bank = None

            self.current_search_factor, self.memory_write_enabled, self.jump_info = self.jump_mgr.update(
                pred_boxes=pred_boxes*self.fx_sz,
                conf_score=conf_val,
                avg_pred_score=avg_pred_score,
                curr_memory_feat=curr_memory_feat,  # 如果你能拿到当前ROI特征就传进来
                anchor_bank=anchor_bank,
            )

            if (self.frame_id - self.prev_update_frame_id) >= 30:
                if conf_val >= self.avg_pred_score.item():
                    if self.memory_write_enabled:
                        memory_feature = self.network.append_new_memory(pred_boxes)
                        if len(self.network.memory_search) >= 6:
                            # log_memory_similarity(self.network.memory_search)
                            self.network.memory_search.pop(3)

                        self.network.memory_search.append(memory_feature)
                        self.prev_update_frame_id = self.frame_id



        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # update the template
        if self.num_template > 1:
            if (self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold):
                z_patch_arr, resize_factor = sample_target(image, self.state, self.params.template_factor,
                                                           output_sz=self.params.template_size)
                template = self.preprocessor.process(z_patch_arr)
                if self.multi_modal_vision and (template.size(1) == 3):
                    template = torch.cat((template, template), axis=1)
                self.template_list.append(template)
                if len(self.template_list) > self.num_template:
                    self.template_list.pop(1)

                prev_box_crop = transform_image_to_crop(torch.tensor(self.state),
                                                        torch.tensor(self.state),
                                                        resize_factor,
                                                        torch.Tensor(
                                                            [self.params.template_size, self.params.template_size]),
                                                        normalize=True)
                self.template_anno_list.append(prev_box_crop.to(template.device).unsqueeze(0))
                if len(self.template_anno_list) > self.num_template:
                    self.template_anno_list.pop(1)

        # for debug
        if image.shape[-1] == 6:
            image_show = image[:,:,:3]
        else:
            image_show = image
        if 1:#self.debug == 1:
            x1, y1, w, h = self.state
            image_BGR = cv2.cvtColor(image_show, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_BGR, (int(x1),int(y1)), (int(x1+w),int(y1+h)), color=(0,0,255), thickness=2)
            cv2.imshow('vis', image_BGR)
            cv2.waitKey(1)

        return {"target_bbox": self.state,
                "best_score": conf_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def extract_token_from_nlp_clip(self, nlp):
        if nlp is None:
            nlp_ids = torch.zeros(77, dtype=torch.long)
            nlp_masks = torch.zeros(77, dtype=torch.long)
        else:
            nlp_ids = clip.tokenize(nlp).squeeze(0)
            nlp_masks = (nlp_ids == 0).long()
        return nlp_ids, nlp_masks

def get_tracker_class():
    return T2TRACK
