import torch
from collections import deque


class MATRJudge:
    """
    Reproduction of MATR-style current-frame feasibility judgment.

    Inputs:
        score_map: [H, W] or [1, H, W]
        response_map: [H, W] or [1, H, W]
            - In the paper, entropy E is computed from the Hann-windowed response map R
            - Peakiness P and confidence C are computed from score map S
        frame_id: current frame index, starting from 0 or 1
    """

    def __init__(
        self,
        tau_c=0.3,          # confidence threshold
        tau_p=1.6,          # peakiness threshold
        tau_e=2.0,          # entropy threshold
        theta_update=0.5,   # template write threshold (paper says quality threshold, exact default not explicitly fixed)
        min_update_interval=5,
        max_update_num=999999,
        entropy_bins=20,
        peak_topk=5,
        ema_gamma=0.1,
        sharpness_threshold=1.6,
    ):
        self.tau_c = tau_c
        self.tau_p = tau_p
        self.tau_e = tau_e

        self.theta_update = theta_update
        self.min_update_interval = min_update_interval
        self.max_update_num = max_update_num

        self.entropy_bins = entropy_bins
        self.peak_topk = peak_topk

        self.ema_gamma = ema_gamma
        self.sharpness_threshold = sharpness_threshold

        self.last_update_frame = -10**9
        self.update_count = 0
        self.occlusion_count = 0

        self.last_valid_state = None  # 可存 pred_box
        self.metric_history = deque(maxlen=100)

    def _to_2d(self, x):
        if isinstance(x, torch.Tensor):
            y = x.detach().float()
        else:
            y = torch.tensor(x).float()

        if y.dim() == 3:
            y = y.squeeze(0)
        elif y.dim() == 4:
            y = y.squeeze(0).squeeze(0)
        return y

    def compute_entropy(self, response_map):
        """
        Paper Eq. (18):
            E = - sum_i p_i log2 p_i
        where p_i is histogram probability over B bins.
        """
        R = self._to_2d(response_map)
        flat = R.reshape(-1)

        rmin = flat.min()
        rmax = flat.max()

        if float(rmax - rmin) < 1e-12:
            return 0.0

        hist = torch.histc(flat, bins=self.entropy_bins, min=float(rmin), max=float(rmax))
        prob = hist / (hist.sum() + 1e-12)
        prob = prob[prob > 0]
        entropy = -(prob * torch.log2(prob)).sum()
        return float(entropy.item())

    def compute_peakiness(self, score_map):
        """
        Paper Eq. (19):
            P = max(S) / mean(topk(S))
        Note:
            If topk includes max itself, P <= 1 always, which conflicts with the paper's later description.
            To make it meaningful, we use:
                P = max(S) / mean(next top-k neighbors excluding the max)
            This is the most reasonable operational reproduction.
        """
        S = self._to_2d(score_map)
        flat = S.reshape(-1)

        if flat.numel() < 2:
            return 1.0

        k = min(self.peak_topk + 1, flat.numel())
        vals = torch.topk(flat, k=k).values
        peak = vals[0]

        # exclude the global max
        ref = vals[1:] if vals.numel() > 1 else vals[:1]
        denom = ref.mean() + 1e-12
        peakiness = peak / denom
        return float(peakiness.item())

    def compute_confidence(self, score_map):
        """
        Paper Eq. (20):
            C = max(S)
        """
        S = self._to_2d(score_map)
        return float(S.max().item())

    def judge_occlusion(self, score_map, response_map):
        """
        Paper Eq. (21):
            O = I(C < tau_c AND P < tau_p AND E > tau_e)
        """
        E = self.compute_entropy(response_map)
        P = self.compute_peakiness(score_map)
        C = self.compute_confidence(score_map)

        occluded = (C < self.tau_c) and (P < self.tau_p) and (E > self.tau_e)

        info = {
            "entropy": E,
            "peakiness": P,
            "confidence": C,
            "occluded": bool(occluded),
        }
        return bool(occluded), info

    def judge_update(
        self,
        frame_id,
        score_map,
        response_map,
        pred_box=None,
    ):
        """
        Reproduce MATR update decision:
        Update allowed iff:
            1) not initial frame
            2) update_count below limit
            3) enough interval since last update
            4) confidence >= theta_update
            5) not occluded
        """
        occluded, info = self.judge_occlusion(score_map, response_map)
        C = info["confidence"]
        P = info["peakiness"]

        basic_ok = (
            frame_id > 0 and
            self.update_count < self.max_update_num and
            (frame_id - self.last_update_frame) >= self.min_update_interval
        )

        quality_ok = C >= self.theta_update
        non_occlusion_ok = not occluded

        should_update = basic_ok and quality_ok and non_occlusion_ok

        if occluded:
            self.occlusion_count += 1
        else:
            self.occlusion_count = 0
            if pred_box is not None:
                self.last_valid_state = pred_box

        # paper: reduce gamma when peakiness is low
        gamma = self.ema_gamma
        if P < self.sharpness_threshold:
            gamma = gamma * 0.5

        out = {
            **info,
            "basic_ok": bool(basic_ok),
            "quality_ok": bool(quality_ok),
            "non_occlusion_ok": bool(non_occlusion_ok),
            "should_update": bool(should_update),
            "ema_gamma": float(gamma),
            "occlusion_count": int(self.occlusion_count),
        }

        if should_update:
            self.last_update_frame = frame_id
            self.update_count += 1

        self.metric_history.append(out)
        return bool(should_update), out