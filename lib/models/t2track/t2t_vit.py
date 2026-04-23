# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
T2T-ViT
"""
import torch
import torch.nn as nn

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
from .token_transformer import Token_transformer
from .token_performer import Token_performer
from .transformer_block import Block, get_sinusoid_encoding
import os
from .transformer import TwoWayTransformer

def load_pretrained(model, pretrained):
    model_state = model.state_dict()
    current_file_path = os.path.abspath(__file__)
    project_dir = os.path.abspath(os.path.join(current_file_path, '../../../../'))

    try:
        with open(os.path.join(project_dir,pretrained), "rb") as f:
            checkpoint = torch.load(f, map_location="cpu", weights_only=False)
    except Exception:
        # 第二次必须重新打开文件
        with open(os.path.join(project_dir,pretrained), "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")

    # with open(os.path.join(project_dir,pretrained), "rb") as f:
    #     checkpoint = torch.load(f, map_location="cpu")
        # checkpoint = torch.load(f, map_location="cpu",weights_only=False)
    if "module" in checkpoint.keys():
        pretrained_state = checkpoint["module"]
    elif "model" in checkpoint.keys():
        pretrained_state = checkpoint["model"]
    elif "net" in checkpoint.keys():
        pretrained_state = checkpoint["net"]
    elif 'state_dict' in checkpoint.keys():
        pretrained_state = checkpoint['state_dict']
    elif 'state_dict_ema' in checkpoint.keys():
        pretrained_state = checkpoint['state_dict_ema']
    else:
        pretrained_state = checkpoint

    matched_keys, mismatched_keys, unused_keys = [], [], []
    pretrained_dict = {}
    for k, v in pretrained_state.items():
        if k in model_state:
            # 检查尺寸是否匹配
            if v.shape == model_state[k].shape:
                pretrained_dict[k] = v
                matched_keys.append(k)
            else:
                mismatched_keys.append((k, pretrained_state[k].shape, model_state[k].shape))
                print(f"Skipping parameter '{k}' due to size mismatch: "
                      f"checkpoint shape {v.shape} != current model shape {model_state[k].shape}")
        else:
            unused_keys.append(k)
            print(f"Skipping parameter '{k}' as it's not in the current model")

    model_state.update(pretrained_dict)
    model.load_state_dict(model_state, strict=True)
    if len(mismatched_keys) == 0 and len(unused_keys) == 0:
        print("All keys matched successfully")
    return model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'search_size': (3, 224, 224),'template_size': (3, 112, 112), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'T2t_vit_7': _cfg(),
    'T2t_vit_10': _cfg(),
    'T2t_vit_12': _cfg(),
    'T2t_vit_14': _cfg(),
    'T2t_vit_19': _cfg(),
    'T2t_vit_24': _cfg(),
    'T2t_vit_t_14': _cfg(),
    'T2t_vit_t_19': _cfg(),
    'T2t_vit_t_24': _cfg(),
    'T2t_vit_14_resnext': _cfg(),
    'T2t_vit_14_wide': _cfg(),
}

class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """
    def __init__(self,
                 search_size,
                 template_size,
                 stride=16,
                 tokens_type='performer',
                 in_chans=3,
                 embed_dim=768,
                 token_dim=64):
        super().__init__()
        self.search_size = search_size
        self.template_size = template_size
        self.stride = stride
        self.num_patches_x = (search_size//stride)**2
        self.num_patches_z = (template_size//stride)**2

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            #nn.Unfold: 把每个局部 patch 当成一个 token，并把 patch 内所有像素拼到通道维里，只是对图像做重排，没有参数
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_transformer(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'performer':
            print('adopt performer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_performer(dim=in_chans*7*7, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim*3*3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'convolution':  # just for comparison with conolution, not our model
            # for this tokens type, you need change forward as three convolution operation
            print('adopt convolution layers for tokens-to-token')
            self.soft_split0 = nn.Conv2d(3, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))  # the 1st convolution
            self.soft_split1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 2nd convolution
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)) # the 3rd convolution

        # self.num_patches = (img_size // (4 * 2 * 2)) * (img_size // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x,z):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)
        z = self.soft_split0(z).transpose(1, 2)
        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        z = self.attention1(z)
        B, new_HW_x, C = x.shape
        _, new_HW_z, _ = z.shape
        x = x.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW_x)), int(np.sqrt(new_HW_x)))
        z = z.transpose(1,2).reshape(B, C, int(np.sqrt(new_HW_z)), int(np.sqrt(new_HW_z)))

        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)
        z = self.soft_split1(z).transpose(1, 2)

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        z = self.attention2(z)
        B, new_HW_x, C = x.shape
        _, new_HW_z, _ = z.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW_x)), int(np.sqrt(new_HW_x)))
        z = z.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW_z)), int(np.sqrt(new_HW_z)))

        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)
        z = self.soft_split2(z).transpose(1, 2)
        # final tokens
        x = self.project(x)
        z = self.project(z)

        return x,z

class T2T_ViT(nn.Module):
    def __init__(self,
                 search_size,
                 template_size,
                 stride=16,
                 # img_size=224,
                 tokens_type='performer',
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 token_dim=64,
                 use_temporal=False,
                 stage_num=3,
                 ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tokens_to_token = T2T_module(
            search_size=search_size,
            template_size=template_size,
            stride=stride,
            tokens_type=tokens_type,
            in_chans=in_chans,
            embed_dim=embed_dim,
            token_dim=token_dim)

        self.num_patches_x = self.tokens_to_token.num_patches_x
        self.num_patches_z = self.tokens_to_token.num_patches_z

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed_x = nn.Parameter(data=get_sinusoid_encoding(n_position=self.num_patches_x, d_hid=embed_dim), requires_grad=False)
        self.pos_embed_z = nn.Parameter(data=get_sinusoid_encoding(n_position=self.num_patches_z, d_hid=embed_dim), requires_grad=False)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer) for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.use_temporal=use_temporal
        if self.use_temporal:
            self.stage_num= stage_num
            self.stage_Block_num = depth//stage_num
            self.temporal_fuse_dim = self.embed_dim // 4
            self.Memory_Attention = nn.ModuleList([
                TwoWayTransformer(
                    depth=2,
                    embedding_dim=self.temporal_fuse_dim,
                    num_heads=num_heads,
                    mlp_dim=512
                ) for _ in range(stage_num)
            ])
            self.pos_embed_memory = nn.Parameter(data=get_sinusoid_encoding(n_position=self.num_patches_z, d_hid=self.temporal_fuse_dim), requires_grad=False)
            self.depth = depth
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, search,template,history_memory=None):
        x=search[-1]
        z=template[-1]
        B = x.shape[0]
        x,z = self.tokens_to_token(x,z)

        if self.use_temporal:
            if history_memory is not None:
                history_token = history_memory+self.pos_embed_memory.unsqueeze(0)
                history_token = history_token.permute(1,0,2,3).reshape(B,-1,self.temporal_fuse_dim)

        x = x + self.pos_embed_x
        z = z + self.pos_embed_z
        zx = torch.cat((z, x), dim=1)
        zx = self.pos_drop(zx)

        for idx,blk in enumerate(self.blocks):
            zx = blk(zx)
            if self.use_temporal and history_memory is not None and idx != (self.depth-1):
                if (idx+1)%self.stage_Block_num == 0:
                    x = zx[:, -self.num_patches_x:, :]
                    history_token,x = self.Memory_Attention[((idx+1)//self.stage_Block_num)-1](image_embedding=x,
                                                                                               memory_embedding=history_token)
                    zx[:, -self.num_patches_x:, :] = x

        zx = self.norm(zx)
        return zx[:, -self.num_patches_x:,:]

    def forward(self, search,template,history_memory=None):
        x = self.forward_features(search,template,history_memory)
        # x = self.head(x)
        return x

@register_model
def t2t_vit_7(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=7, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_7']
    if pretrained:
        model = load_pretrained(model, pretrained)
        # load_pretrained(
        #     model, num_classes=1000, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def t2t_vit_10(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=10, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_10']
    if pretrained:
        model = load_pretrained(model, pretrained)

        # load_pretrained(
        #     model, num_classes=1000, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def t2t_vit_12(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=12, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_12']
    if pretrained:
        model = load_pretrained(model, pretrained)
        # load_pretrained(
        #     model, num_classes=1000, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_14(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14']
    if pretrained:
        model = load_pretrained(model, pretrained)
        # load_pretrained(
        #     model, num_classes=1000, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def t2t_vit_19(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 448 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=448, depth=19, num_heads=7, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_19']
    if pretrained:
        model = load_pretrained(model, pretrained)
        # load_pretrained(
        #     model, num_classes=1000, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def t2t_vit_24(pretrained=False, **kwargs): # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_24']
    if pretrained:
        load_pretrained(
            model, num_classes=1000, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def t2t_vit_t_14(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='transformer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_14']
    if pretrained:
        load_pretrained(
            model, num_classes=1000, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def t2t_vit_t_19(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 448 ** -0.5)
    model = T2T_ViT(tokens_type='transformer', embed_dim=448, depth=19, num_heads=7, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_19']
    if pretrained:
        load_pretrained(
            model, num_classes=1000, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def t2t_vit_t_24(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='transformer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_24']
    if pretrained:
        load_pretrained(
            model, num_classes=1000, in_chans=kwargs.get('in_chans', 3))
    return model

# rexnext and wide structure
@register_model
def t2t_vit_14_resnext(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=384, depth=14, num_heads=32, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14_resnext']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def t2t_vit_14_wide(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=768, depth=4, num_heads=12, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14_wide']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model
