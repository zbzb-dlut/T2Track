"""
SUTrack Model
"""
import torch
import math
from torch import nn
import torch.nn.functional as F
from .encoder import build_encoder
from .decoder import build_decoder
from .memory_encoder import build_memory_encoder

from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed

from collections import deque
from lib.utils.box_ops import box_cxcywh_to_xyxy
from lib.utils.jitter_bbox import jitter_bbox,sanitize_xyxy
from torchvision.ops import roi_align

import torch
import torch.nn.functional as F
import os

# def log_memory_similarity(memory_search, save_path="/home/b402/Desktop/B402/2026-1-31/UAVTracking2/T2Track-A1/memory_sim.txt"):
#     """
#     memory_search: list of [1, N, C] tensor
#     计算 memory_search[3] 和 memory_search[0:3] 的相似度，并写入一行txt
#     """
#
#     # 至少要有4帧
#     if len(memory_search) < 4:
#         return
#
#     # 当前要比较的帧（dynamic第一个）
#     feat_target = memory_search[3]  # [1,49,256]
#
#     # mean pooling
#     f_target = feat_target.mean(dim=1)  # [1,256]
#
#     sims = []
#     for i in range(3):
#         feat_ref = memory_search[i]
#         f_ref = feat_ref.mean(dim=1)
#
#         sim = F.cosine_similarity(f_target, f_ref, dim=-1)
#         sims.append(sim.item())
#
#     # 拼成一行字符串
#     line = " ".join([f"{s:.6f}" for s in sims])
#
#     # 写入文件（追加模式）
#     with open(save_path, "a") as f:
#         f.write(line + "\n")

class T2TRACK(nn.Module):
    """ This is the base class for SUTrack """
    def __init__(self,
                 encoder,
                 decoder,
                 memory_encoder=None,
                 num_frames=1,
                 num_template=1,
                 decoder_type="CENTER",
                 task_feature_type="average",
                 use_temporal=False,
                 history_len=0
                 ):
        """ Initializes the model.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder_type = decoder_type
        self.task_feature_type = task_feature_type
        self.num_patch_x = self.encoder.body.num_patches_x
        self.num_patch_z = self.encoder.body.num_patches_z
        self.fx_sz = int(math.sqrt(self.num_patch_x))
        self.fz_sz = int(math.sqrt(self.num_patch_z))

        self.decoder = decoder
        self.num_frames = num_frames
        self.num_template = num_template

        self.memory_encoder = memory_encoder
        self.history_len = history_len
        self.use_temporal = use_temporal
        self.memory_search = [] #deque(maxlen=self.history_len)
        self.is_memory = False

        self.pred_score_lists=[]
        self.box_history=deque(maxlen=30)
        self.avg_pred_score = 1.0


    def forward(self,
                template_list=None,
                search_list=None,
                feature=None,
                mode="encoder"):
        if mode == "encoder":
            return self.forward_encoder(template_list, search_list)
        elif mode == "decoder":
            return self.forward_decoder(feature)#, self.forward_task_decoder(feature)
        else:
            raise ValueError

    def forward_encoder(self, template_list, search_list):
        device = search_list[-1].device
        B = search_list[-1].shape[0]

        if self.use_temporal:
            if self.is_memory and len(self.memory_search)>0:
                history_memory = torch.stack(list(self.memory_search), dim=0)
            else:
                history_memory = None  # torch.zeros(B, self.history_len, self.encoder.num_channels, device=device)
        else:
            history_memory = None  # torch.zeros(B, self.history_len, self.encoder.num_channels, device=device)

        # Forward the encoder
        xz = self.encoder(template_list, search_list,history_memory)
        return xz

    def forward_decoder(self, feature, gt_score_map=None):

        # feature = feature[0]
        # if self.class_token:
        #     feature = feature[:,1:self.num_patch_x * self.num_frames+1]
        # else:
        #     feature = feature[:,0:self.num_patch_x * self.num_frames] # (B, HW, C)

        bs, HW, C = feature.size()
        if self.decoder_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.fx_sz, self.fx_sz)
            self.feature = feature

        if self.decoder_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.decoder(feature, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out
        elif self.decoder_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        elif self.decoder_type == "MLP":
            # run the mlp head
            score_map, bbox, offset_map = self.decoder(feature, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError
    def append_new_memory(self,pred_boxes):

        bs,C,_,_ = self.feature.shape
        pred_boxes_cxcywh = pred_boxes * self.fx_sz
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes_cxcywh)
        pred_boxes_xyxy = sanitize_xyxy(pred_boxes_xyxy, feat_sz=self.fx_sz)
        batch_idx = torch.arange(bs, device=pred_boxes.device, dtype=pred_boxes.dtype).unsqueeze(1)
        rois = torch.cat([batch_idx, pred_boxes_xyxy], dim=1)  # [B, 5]
        object_ROI_feat = roi_align(
            input=self.feature,
            boxes=rois,
            output_size=(self.fx_sz // 2, self.fx_sz // 2),
            spatial_scale=1.0,  # 你的 boxes 已经在 feature 坐标系
            sampling_ratio=2,
            aligned=True
        ).detach()

        memory_feature = (self.memory_encoder.forward_single_x(object_ROI_feat)
                          .reshape(bs, self.memory_encoder.out_dim, -1)
                          .permute(0, 2, 1))
        return memory_feature


def build_t2track(cfg):
    encoder = build_encoder(cfg)
    decoder = build_decoder(cfg, encoder)
    memory_encoder = build_memory_encoder(encoder)

    model = T2TRACK(
        encoder=encoder,
        decoder=decoder,
        memory_encoder=memory_encoder,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
        task_feature_type=cfg.MODEL.TASK_DECODER.FEATURE_TYPE,
        use_temporal=cfg.TRAIN.USE_TEMPORAL,
        history_len=cfg.DATA.SEARCH.HISTORY_LEN
    )

    return model
