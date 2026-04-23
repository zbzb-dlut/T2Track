"""
SUTrack Model
"""
import torch
import math
from torch import nn
import torch.nn.functional as F
from .encoder import build_encoder
# from lib.models.sutrack.clip import build_textencoder
from .decoder import build_decoder
# from lib.models.sutrack.task_decoder import build_task_decoder
from .neck import build_neck
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.utils.pos_embed import get_sinusoid_encoding_table, get_2d_sincos_pos_embed


class UAVTRACK(nn.Module):
    """ This is the base class for SUTrack """
    def __init__(self,
                 encoder,
                 decoder,
                 neck,
                 num_frames=1,
                 num_template=1,
                 text_encoder=None,
                 task_decoder=None,
                 decoder_type="CENTER",
                 task_feature_type="average"):
        """ Initializes the model.
        """
        super().__init__()
        self.encoder = encoder
        self.neck = neck
        self.decoder_type = decoder_type

        self.num_patch_x = self.encoder.body.num_patches_search
        self.num_patch_z = self.encoder.body.num_patches_template

        self.task_decoder = task_decoder
        self.decoder = decoder

        self.num_frames = num_frames
        self.num_template = num_template


    def forward(self,
                template_list=None,
                search_list=None,
                feature=None,
                template_feats=None,
                search_feats=None,
                mode="encoder"):
        if mode == "encoder":
            template_img = template_list[-1]
            search_img = search_list[-1]
            template_feats,search_feats = self.forward_encoder(z=template_img,x=search_img)
            return template_feats,search_feats
        elif mode == "decoder":
            return self.forward_decoder(feature)
        elif mode =='neck':
            return self.forward_neck(template_feats, search_feats)
        else:
            raise ValueError

    def forward_neck(self,template_feats,search_feats):
        # Forward the neck
        enc_opt = self.neck(template_feats,search_feats)
        return enc_opt

    def forward_encoder(self, z: torch.Tensor=None,x: torch.Tensor=None):
        z, x = self.encoder(z, x)
        return z,x

    def forward_decoder(self, feature, gt_score_map=None):

        bs, HW, C = feature.size()
        if self.decoder_type in ['CORNER', 'CENTER']:
            feature = feature.permute((0, 2, 1)).contiguous()
            feature = feature.view(bs, C, self.num_patch_x, self.num_patch_x)
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

def build_uavtrack(cfg):

    encoder = build_encoder(cfg)
    neck = build_neck(cfg,encoder)
    decoder = build_decoder(cfg, encoder)
    model = UAVTRACK(
        encoder=encoder,
        decoder=decoder,
        neck=neck,
        num_frames = cfg.DATA.SEARCH.NUMBER,
        num_template = cfg.DATA.TEMPLATE.NUMBER,
        decoder_type=cfg.MODEL.DECODER.TYPE,
        task_feature_type=cfg.MODEL.TASK_DECODER.FEATURE_TYPE
    )

    return model
