import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type
from .transformer import TwoWayTransformer,TransformerLayer
import math
from .common import LayerNorm2d, UpSampleLayer, OpSequential



class NECK(nn.Module):
    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        # feat_s=16,
        # feat_z=8,
        # posemb='learn',
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        # self.feat_s = feat_s
        # self.feat_z = feat_z
        self.dim = transformer_dim
        # self.patches_search_size = image_size // patch_size[0]
        # self.patches_template_size = template_size //patch_size[0]

        # self.num_patches_search = self.feat_s**2
        # self.num_patches_template = self.feat_z**2


        #
        # if posemb == 'learn':
        #     self.pos_embedding_x = torch.nn.Parameter(
        #         torch.empty(1,self.num_patches_search, self.dim, dtype=torch.float32))
        #     torch.nn.init.normal_(self.pos_embedding_x, std=1 / math.sqrt(self.dim))
        #
        #     self.pos_embedding_z = torch.nn.Parameter(
        #         torch.empty(1, self.num_patches_template, self.dim, dtype=torch.float32))
        #     torch.nn.init.normal_(self.pos_embedding_z, std=1 / math.sqrt(self.dim))



    def forward(self,template_feats: torch.Tensor, search_feats: torch.Tensor
        # image_embeddings: torch.Tensor,
        # image_pe: torch.Tensor,
        # sparse_prompt_embeddings: torch.Tensor,
        # dense_prompt_embeddings: torch.Tensor,
    ): #-> Tuple[torch.Tensor, torch.Tensor]
        """Predicts masks. See 'forward' for more details."""
        # Run the transformer
        hs, src = self.transformer(search_feats, template_feats)

        return src

def build_neck(cfg,encoder):
    dim = encoder.num_channels
    mlp_dim = encoder.mlp_dim
    neck=NECK(transformer=TransformerLayer(depth=2,
                                           embedding_dim=dim,
                                           mlp_dim=mlp_dim,
                                           num_heads=8,
                                           feat_s=encoder.body.num_patches_search,
                                           feat_z = encoder.body.num_patches_template,
                                           posemb='learn'
                                           ),
              transformer_dim=dim,
            )
    return neck