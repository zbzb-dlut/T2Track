import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1) Patch Embedding
# =========================================================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=(256, 256), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W]
        return: [B, N, C]
        """
        x = self.proj(x)                  # [B, C, H/P, W/P]
        x = x.flatten(2).transpose(1, 2) # [B, N, C]
        return x


# =========================================================
# 2) TMoE
#    Paper Sec 3.3
# =========================================================
class TMoE(nn.Module):
    """
    TMoE for replacing a linear layer:
        y = E_s(x) + sum_i softmax(router(x))_i * E_r_i(E_c(x))
    where:
        - shared expert E_s : linear(d -> D), frozen
        - compression expert E_c : linear(d -> r)
        - routed experts E_r_i : linear(r -> D)
        - router : linear(d -> N_e)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_experts: int = 4,
        compress_dim: int = 64,
        bias: bool = True,
        freeze_shared: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_experts = num_experts
        self.compress_dim = compress_dim

        # shared expert: copy from pretrained linear in real implementation
        self.shared_expert = nn.Linear(in_dim, out_dim, bias=bias)

        # compression expert
        self.compress_expert = nn.Linear(in_dim, compress_dim, bias=bias)

        # routed experts
        self.routed_experts = nn.ModuleList([
            nn.Linear(compress_dim, out_dim, bias=bias)
            for _ in range(num_experts)
        ])

        # router
        self.router = nn.Linear(in_dim, num_experts, bias=True)

        if freeze_shared:
            for p in self.shared_expert.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., in_dim]
        return: [..., out_dim]
        """
        # router weights
        gate = self.router(x)               # [..., Ne]
        gate = F.softmax(gate, dim=-1)      # dense activation

        # shared expert
        y_shared = self.shared_expert(x)    # [..., D]

        # compression
        y_comp = self.compress_expert(x)    # [..., r]

        # routed experts
        routed_outputs = []
        for expert in self.routed_experts:
            routed_outputs.append(expert(y_comp))  # each: [..., D]

        # stack -> [..., Ne, D]
        y_routed_all = torch.stack(routed_outputs, dim=-2)

        # weighted sum
        gate = gate.unsqueeze(-1)           # [..., Ne, 1]
        y_routed = (gate * y_routed_all).sum(dim=-2)  # [..., D]

        # final output
        y = y_shared + y_routed
        return y


# =========================================================
# 3) M2SA: MoE-based Multi-head Self-Attention
#    Replaces Q, K, V, and output projection with TMoE
# =========================================================
class M2SA(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_experts: int = 4,
        compress_dim: int = 64,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # replace Q,K,V linear projections by TMoE
        self.q_proj = TMoE(dim, dim, num_experts=num_experts, compress_dim=compress_dim, bias=qkv_bias)
        self.k_proj = TMoE(dim, dim, num_experts=num_experts, compress_dim=compress_dim, bias=qkv_bias)
        self.v_proj = TMoE(dim, dim, num_experts=num_experts, compress_dim=compress_dim, bias=qkv_bias)

        # replace output projection by TMoE
        self.out_proj = TMoE(dim, dim, num_experts=num_experts, compress_dim=compress_dim, bias=proj_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C]
        """
        B, N, C = x.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, h, N, d]
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q * self.scale) @ k.transpose(-2, -1)   # [B, h, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                                   # [B, h, N, d]
        out = out.transpose(1, 2).reshape(B, N, C)       # [B, N, C]

        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out


# =========================================================
# 4) MFFN: MoE-based Feed Forward Network
#    Both linear layers replaced by TMoE
# =========================================================
class MFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        act_layer=nn.GELU,
        drop: float = 0.0,
        num_experts: int = 4,
        compress_dim: int = 64,
        bias: bool = True,
    ):
        super().__init__()
        self.fc1 = TMoE(dim, hidden_dim, num_experts=num_experts, compress_dim=compress_dim, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)

        self.fc2 = TMoE(hidden_dim, dim, num_experts=num_experts, compress_dim=compress_dim, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# =========================================================
# 5) TMoEBlock
#    Paper Eq. (3)
# =========================================================
class TMoEBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        num_experts: int = 4,
        compress_dim: int = 64,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.m2sa = M2SA(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
            num_experts=num_experts,
            compress_dim=compress_dim,
        )

        self.norm2 = norm_layer(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mffn = MFFN(
            dim=dim,
            hidden_dim=hidden_dim,
            drop=drop,
            num_experts=num_experts,
            compress_dim=compress_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.m2sa(self.norm1(x)) + x
        x = self.mffn(self.norm2(x)) + x
        return x


# =========================================================
# 6) Transformer Encoder
# =========================================================
class TMoETransformerEncoder(nn.Module):
    def __init__(
        self,
        depth: int,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        num_experts: int = 4,
        compress_dim: int = 64,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            TMoEBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                num_experts=num_experts,
                compress_dim=compress_dim,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


# =========================================================
# 7) Decoupled Double-MLP Head
#    Similar spirit to OSTrack-style head
# =========================================================
class DoubleMLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.cls_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.box_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, feat_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        feat_map: [B, Hs, Ws, C]
        returns:
            cls_logits: [B, Hs, Ws, 1]
            box_pred  : [B, Hs, Ws, 4]
        """
        cls_logits = self.cls_head(feat_map)
        box_pred = self.box_head(feat_map)
        return cls_logits, box_pred


# =========================================================
# 8) SPMTrack (paper-based reconstruction)
# =========================================================
class SPMTrack(nn.Module):
    def __init__(
        self,
        template_size=(128, 128),
        search_size=(256, 256),
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_experts=4,
        compress_dim=64,
        drop=0.0,
        attn_drop=0.0,
    ):
        super().__init__()

        self.template_size = template_size
        self.search_size = search_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # patch embed for template / search
        self.template_patch_embed = PatchEmbed(template_size, patch_size, in_chans, embed_dim)
        self.search_patch_embed = PatchEmbed(search_size, patch_size, in_chans, embed_dim)

        self.num_template_tokens = self.template_patch_embed.num_patches
        self.num_search_tokens = self.search_patch_embed.num_patches

        # positional embeddings
        self.template_pos_embed = nn.Parameter(torch.zeros(1, self.num_template_tokens, embed_dim))
        self.search_pos_embed = nn.Parameter(torch.zeros(1, self.num_search_tokens, embed_dim))

        # token-type embeddings
        self.type_embed_fg = nn.Parameter(torch.zeros(1, 1, embed_dim))   # foreground token in reference
        self.type_embed_bg = nn.Parameter(torch.zeros(1, 1, embed_dim))   # background token in reference
        self.type_embed_search = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # target state token
        self.state_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # encoder
        self.encoder = TMoETransformerEncoder(
            depth=depth,
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop,
            attn_drop=attn_drop,
            num_experts=num_experts,
            compress_dim=compress_dim,
        )

        # prediction head
        self.pred_head = DoubleMLPHead(embed_dim, hidden_dim=256)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.template_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.search_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.type_embed_fg, std=0.02)
        nn.init.trunc_normal_(self.type_embed_bg, std=0.02)
        nn.init.trunc_normal_(self.type_embed_search, std=0.02)
        nn.init.trunc_normal_(self.state_token, std=0.02)

    def build_reference_tokens(
        self,
        ref_imgs: List[torch.Tensor],
        ref_bbox_masks: Optional[List[torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """
        ref_imgs: list of N tensors, each [B, 3, Hz, Wz]
        ref_bbox_masks: list of N tensors, each [B, Nt]
            1 means foreground token, 0 means background token
        returns:
            list of N token tensors, each [B, Nt, C]
        """
        ref_tokens_all = []
        for i, img in enumerate(ref_imgs):
            tok = self.template_patch_embed(img)                    # [B, Nt, C]
            tok = tok + self.template_pos_embed

            if ref_bbox_masks is not None:
                mask = ref_bbox_masks[i].unsqueeze(-1).float()      # [B, Nt, 1]
                tok = tok + mask * self.type_embed_fg + (1.0 - mask) * self.type_embed_bg
            else:
                # if no bbox mask, treat all as foreground for simplicity
                tok = tok + self.type_embed_fg

            ref_tokens_all.append(tok)
        return ref_tokens_all

    def build_search_tokens(self, search_img: torch.Tensor) -> torch.Tensor:
        x = self.search_patch_embed(search_img)                     # [B, Nx, C]
        x = x + self.search_pos_embed + self.type_embed_search
        return x

    def forward(
        self,
        ref_imgs: List[torch.Tensor],
        search_img: torch.Tensor,
        prev_state_token: Optional[torch.Tensor] = None,
        ref_bbox_masks: Optional[List[torch.Tensor]] = None,
    ):
        """
        ref_imgs: N reference frames, each [B, 3, Hz, Wz]
        search_img: [B, 3, Hs, Ws]
        prev_state_token: [B, 1, C] or None
        ref_bbox_masks: list of N tensors [B, Nt]

        returns:
            cls_logits: [B, Hs/P, Ws/P, 1]
            box_pred  : [B, Hs/P, Ws/P, 4]
            new_state_token: [B, 1, C]
        """
        B = search_img.shape[0]

        # state token propagation
        if prev_state_token is None:
            H = self.state_token.expand(B, -1, -1)   # [B, 1, C]
        else:
            H = prev_state_token + self.state_token  # paper says add with input target state token

        # reference tokens
        ref_tokens_list = self.build_reference_tokens(ref_imgs, ref_bbox_masks)  # N * [B, Nt, C]

        # search tokens
        X = self.build_search_tokens(search_img)  # [B, Nx, C]

        # concat input: I = Concat(H, T1, ..., TN, X)
        I = torch.cat([H] + ref_tokens_list + [X], dim=1)  # [B, 1 + N*Nt + Nx, C]

        # encoder
        O = self.encoder(I)

        # split outputs
        idx = 0
        H_out = O[:, idx:idx+1, :]
        idx += 1

        for _ in range(len(ref_tokens_list)):
            idx += self.num_template_tokens

        X_out = O[:, idx:idx+self.num_search_tokens, :]  # [B, Nx, C]

        # target-state-token reweighting
        # U = X' @ H'^T  -> [B, Nx, 1]
        U = torch.matmul(X_out, H_out.transpose(-1, -2))  # [B, Nx, 1]

        # reweight search tokens
        X_weighted = X_out * U

        # reshape to feature map
        Hs = self.search_size[0] // self.patch_size
        Ws = self.search_size[1] // self.patch_size
        F_map = X_weighted.reshape(B, Hs, Ws, self.embed_dim)

        # prediction head
        cls_logits, box_pred = self.pred_head(F_map)

        return cls_logits, box_pred, H_out