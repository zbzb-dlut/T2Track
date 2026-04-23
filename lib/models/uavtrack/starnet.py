"""
Implementation of Prof-of-Concept Network: StarNet.

We make StarNet as simple as possible [to show the key contribution of element-wise multiplication]:
    - like NO layer-scale in network design,
    - and NO EMA during training,
    - which would improve the performance further.

Created by: Xu Ma (Email: ma.xu1@northeastern.edu)
Modified Date: Mar/29/2024
"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
import math
import warnings
import os
from .common import UpSampleLayer, OpSequential
from .Freq_Fuse import TemplateGuidedFreqMod


model_urls = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}


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


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x



class StarNet(nn.Module):
    def __init__(self, base_dim=32,
                 depths=[3, 3, 12, 5],
                 mlp_ratio=4,
                 drop_path_rate=0.0,
                 patch_size=16,
                 search_size=128,
                 template_size=64,
                 num_classes=1000,
                 model_type: str='',
                 **kwargs):
        super().__init__()
        self.channels = []
        self.in_channel = base_dim
        self.num_patches_search = int(search_size/patch_size)
        self.num_patches_template = int(template_size/patch_size)
        # stem layer
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
        # build stages
        self.stages = nn.ModuleList()
        self.aux_fuse_stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            self.channels.append(embed_dim)
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))

            if i_layer in [0,1]:
                self.aux_fuse_stages.append(TemplateGuidedFreqMod(in_ch=self.in_channel,
                                                                  hidden_dim=self.in_channel*2))


        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.apply(self._init_weights)
        if model_type == 's4':
            self.fuse_stage2 = nn.Conv2d(self.channels[-2], self.channels[-1], kernel_size=1, bias=False)
            self.fuse_stage3 = OpSequential([
                nn.Conv2d(self.channels[-1], self.channels[-1], kernel_size=1, bias=False),
                UpSampleLayer(factor=2, mode='bicubic'),
            ])
            self.fuse = nn.Conv2d(self.channels[-1], self.channels[-1], kernel_size=1, bias=False)
            self.channel = self.channels[-1]
        elif model_type == 's3':
            self.fuse_stage2 = nn.Conv2d(self.channels[-2], self.channels[-2], kernel_size=1, bias=False)
            self.fuse_stage3 = OpSequential([
                nn.Conv2d(self.channels[-1], self.channels[-2], kernel_size=1, bias=False),
                UpSampleLayer(factor=2, mode='bicubic'),
            ])
            self.fuse = nn.Conv2d(self.channels[-2], self.channels[-2], kernel_size=1, bias=False)
            self.channel = self.channels[-2]

        self.z_mid_feat = []
        self.gate_map=[]


    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, z:torch.Tensor=None,x: torch.Tensor=None):
        if self.training:
            z_mid_feat = []
            x_mid_feat = []
            z,x = self.stem(z), self.stem(x) #output: 3->32,缩小2倍
            for idx, stage in enumerate(self.stages):
                z,x = stage(z),stage(x)
                if idx in [0, 1]:
                    _, _ = self.aux_fuse_stages[idx].forward_z(z=z,idx=idx)
                    x_out = self.aux_fuse_stages[idx].forward_x(x=x, idx=idx)
                    x = x + x_out
                z_mid_feat.append(z)
                x_mid_feat.append(x)
            x = self.fuse(self.fuse_stage2(x_mid_feat[-2]) + self.fuse_stage3(x_mid_feat[-1]))
            z = self.fuse(self.fuse_stage2(z_mid_feat[-2]) + self.fuse_stage3(z_mid_feat[-1]))
        elif z is None:
            x_mid_feat = []
            x = self.stem(x)
            for idx, stage in enumerate(self.stages):
                x = stage(x)
                if idx in [0,1]:
                    x_out = self.aux_fuse_stages[idx].forward_x(x=x, idx=idx)
                    x=x+x_out
                x_mid_feat.append(x)
            x = self.fuse(self.fuse_stage2(x_mid_feat[-2]) + self.fuse_stage3(x_mid_feat[-1]))
        elif x is None:
            self.z_mid_feat.clear()
            self.gate_map.clear()
            z = self.stem(z)
            for idx, stage in enumerate(self.stages):
                z = stage(z)
                if idx in [0, 1]:
                    _, _ = self.aux_fuse_stages[idx].forward_z(z=z, idx=idx)
                self.z_mid_feat.append(z)
            z = self.fuse(self.fuse_stage2(self.z_mid_feat[-2]) + self.fuse_stage3(self.z_mid_feat[-1]))

        return z, x

    # def forward(self, x):
    #     mid_feat = []
    #     x = self.stem(x) #output: 3->32,缩小2倍
    #     for stage in self.stages:
    #         x = stage(x)
    #         mid_feat.append(x)
    #     x = self.fuse_stage2(mid_feat[-2] + self.fuse_stage3(mid_feat[-1]))
    #     return x


@register_model
def starnet_s1(pretrained=False, **kwargs):
    model = StarNet(base_dim=24, depths=[2, 2, 8, 3], **kwargs)
    if pretrained:
        if pretrained:
            model = load_pretrained(model=model, pretrained=pretrained)
    return model


@register_model
def starnet_s2(pretrained=False, **kwargs):
    model = StarNet(base_dim=32, depths=[1, 2, 6, 2], **kwargs)
    if pretrained:
        if pretrained:
            model = load_pretrained(model=model, pretrained=pretrained)
    return model


@register_model
def starnet_s3(pretrained=False, **kwargs):
    model = StarNet(base_dim=32, depths=[2, 2, 8, 4],model_type='s3', **kwargs)
    if pretrained:
        if pretrained:
            model = load_pretrained(model=model, pretrained=pretrained)
    return model


@register_model
def starnet_s4(pretrained=False, **kwargs):
    model = StarNet(base_dim=32, depths=[3, 3, 12, 5], model_type='s4', **kwargs)
    if pretrained:
        model = load_pretrained(model=model,pretrained=pretrained)
    return model


# very small networks #
@register_model
def starnet_s050(pretrained=False, **kwargs):
    return StarNet(16, [1, 1, 3, 1], 3, **kwargs)


@register_model
def starnet_s100(pretrained=False, **kwargs):
    return StarNet(20, [1, 2, 4, 1], 4, **kwargs)


@register_model
def starnet_s150(pretrained=False, **kwargs):
    return StarNet(24, [1, 2, 4, 2], 3, **kwargs)
