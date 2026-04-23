import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TemplateGuidedFreqMod(nn.Module):
    """
    Template-Guided Frequency Modulation Module (TGFM)

    Inputs:
        z: template feature  (B, C, Ht, Wt)
        x: search feature    (B, C, Hs, Ws)

    Output:
        x_out: modulated search feature (B, C, Hs, Ws)
    """

    def __init__(
        self,
        in_ch: int,
        num_freq_basis: int = 8,
        hidden_dim: int = 128,
        reduction: int = 4,
        kernel_size: int = 3,
        use_residual: bool = True,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.num_freq_basis = num_freq_basis
        self.use_residual = use_residual

        # ---- Template → frequency descriptor → modulation coefficients ----
        self.freq_modulator = nn.Sequential(
            nn.Linear(2 * in_ch, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_freq_basis),
        )
        self.template_mlp = nn.Sequential(
            nn.Linear(in_ch, in_ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction, in_ch)
        )

        # 2. Search spatial response extractor
        self.spatial_conv = nn.Conv2d(
            in_ch, in_ch,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_ch
        )

        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // reduction, in_ch, 1)
        )
        self.channel_conv = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.act = nn.Sigmoid()

        # ---- Lightweight spatial refinement (optional but recommended) ----
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 1),
        )

        self._freq_basis_cache = {}  # cache freq bases for different resolutions

        self.template_prior = {'alpha':torch.Tensor,'p':torch.Tensor}

        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * in_ch, in_ch// reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch// reduction, in_ch)
        )

    def _build_freq_basis(self, H, W, device, dtype):
        """
        Build radial frequency bases φ_k(f), resolution-aware but template-free
        Returns: (K, H, W//2+1)
        """
        key = (H, W, device)
        if key in self._freq_basis_cache:
            return self._freq_basis_cache[key]

        Hf, Wf = H, W // 2 + 1
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, Hf, device=device),
            torch.linspace(0, 1, Wf, device=device),
            indexing="ij",
        )
        freq_radius = torch.sqrt(xx ** 2 + yy ** 2)  # [0, ~1.4]
        freq_radius = freq_radius / freq_radius.max()

        bases = []
        for k in range(self.num_freq_basis):
            center = (k + 0.5) / self.num_freq_basis
            width = 1.0 / self.num_freq_basis
            basis = torch.exp(-((freq_radius - center) ** 2) / (2 * width ** 2))
            bases.append(basis)

        bases = torch.stack(bases, dim=0).to(dtype)  # (K, H, Wf)
        self._freq_basis_cache[key] = bases
        return bases

    def forward_z(self, z: torch.Tensor,idx: int=0):
        B, C, _, _ = z.shape
        z_pool = F.adaptive_avg_pool2d(z, 1).view(B, C)
        p = self.template_mlp(z_pool).view(B, C, 1, 1)

        z_fft = torch.fft.rfft2(z.float(), norm="ortho")
        z_amp = torch.log(torch.abs(z_fft) + 1e-6)
        z_mean = z_amp.mean(dim=(2, 3))  # (B, C)
        z_var = z_amp.var(dim=(2, 3))  # (B, C)
        z_desc = torch.cat([z_mean, z_var], dim=1)  # (B, 2C)

        alpha = self.freq_modulator(z_desc)  # (B, K)
        alpha = torch.tanh(alpha)  # stabilize

        self.template_prior['alpha']=alpha #{: ,'p':p}
        self.template_prior['p']=p
        return alpha,p

    def forward_x(self, x: torch.Tensor,idx: int=0):
        """
        z: (B, C, Ht, Wt) template
        x: (B, C, Hs, Ws) search
        """

        alpha = self.template_prior['alpha']
        p = self.template_prior['p']

        B, C, Hs, Ws = x.shape
        device, dtype = x.device, x.dtype
        s_local = self.spatial_conv(x)
        s_local = self.channel_conv(s_local)

        s_global = self.global_branch(x)


        gate = self.act((s_local + s_global) * p)
        x_spatial = x * gate

        freq_bases = self._build_freq_basis(Hs, Ws, device, x_fft_dtype := torch.float32)
        theta = torch.einsum("bk,khw->bhw", alpha, freq_bases)
        theta = theta.unsqueeze(1)
        x_fft = torch.fft.rfft2(x.float(), norm="ortho")
        x_fft_mod = torch.complex(
            x_fft.real * (1.0 + 0.1*theta),
            x_fft.imag * (1.0 + 0.1*theta),
        )

        x_mod = torch.fft.irfft2(x_fft_mod, s=(Hs, Ws), norm="ortho")
        x_mod = x_mod.to(dtype)

        # ===== 5. Spatial refinement & output =====
        x_freq = self.refine(x_mod)

        u_pool = F.adaptive_avg_pool2d(torch.cat([x_spatial, x_freq], dim=1), 1).view(B, 2 * C)
        g = self.act(self.gate_mlp(u_pool)).view(B, C, 1, 1)

        x_out = g * x_spatial + (1.0 - g) * x_freq

        return x_out
