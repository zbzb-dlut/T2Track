import torch
import torch.nn.functional as F
import numpy as np

class JumpFreezeReopenManager:
    def __init__(
        self,
        stat_window=30,
        memory_len=300,
        lambda_mad=10.0,
        recover_frames=30,
        warmup_len=30,
        expand_ratio=2.0,
    ):
        # self.normal_search_factor = normal_search_factor
        self.expand_ratio = expand_ratio
        # self.expanded_search_factor = normal_search_factor * expand_ratio

        self.stat_window = stat_window
        self.memory_len = memory_len
        self.lambda_mad = lambda_mad
        self.recover_frames = recover_frames
        self.warmup_len = warmup_len

        self.state = "NORMAL"   # NORMAL / FROZEN / RECOVER_CANDIDATE
        self.dyn_factor = 1.0
        self.memory_write_enabled = True

        self.center_history = []
        self.residual_history = []

        self.frozen_center_history = []
        self.frozen_residual_history = []

        self.prev_boxes = []
        self.recover_counter = 0

        self.trusted_bank = []  # list of [1, N, C]
        self.reid_recover_counter = 0

    def _to_box(self, pred_boxes):
        if isinstance(pred_boxes, torch.Tensor):
            return pred_boxes.detach().view(-1).cpu().numpy().astype(np.float32)
        return np.array(pred_boxes, dtype=np.float32).reshape(-1)

    def _get_center(self, pred_boxes):
        box = self._to_box(pred_boxes)
        return box[:2]   # cx, cy

    def _predict_center(self, hist):
        if len(hist) < 1:
            return None
        c1 = hist[-1]
        # c2 = hist[-2]
        return c1 #+ (c1 - c2)

    def _median_mad(self, arr):
        arr = np.array(arr, dtype=np.float32)
        med = np.median(arr)
        mad = np.median(np.abs(arr - med)) + 1e-6
        return med, mad

    def _compute_residual(self, center, center_hist):
        pred_center = self._predict_center(center_hist)
        if pred_center is None:
            return 0.0
        return float(np.linalg.norm(center - pred_center)) #float(np.sum((center - pred_center) ** 2))

    def _is_jump(self,
                 residual,
                 residual_hist=None,
                 ref_med=None,
                 ref_mad=None
                 ):
        if ref_med is not None and ref_mad is not None:
            jump = residual > (ref_med + self.lambda_mad * ref_mad)
            return jump, ref_med, ref_mad

        if residual_hist is None or len(residual_hist) < max(5, self.warmup_len):
            return False, 0.0, 0.0

        hist = residual_hist #[-self.stat_window:]
        med, mad = self._median_mad(hist)
        jump = residual > (med + self.lambda_mad * mad)
        return jump, med, mad

    def _anchor_similarity(self, curr_memory_feat, anchor_bank):
        """
        curr_memory_feat: [1, N, C]
        anchor_bank: list of [1, N, C]
        """
        if len(anchor_bank) == 0:
            return 1.0

        f_curr = curr_memory_feat.mean(dim=1)  # [1,C]
        sims = []
        for a in anchor_bank:
            f_a = a.mean(dim=1)
            sim = F.cosine_similarity(f_curr, f_a, dim=-1).item()
            sims.append(sim)
        return max(sims)

    # def _anchor_similarity(self, curr_memory_feat, anchor_bank):
    #     """
    #     返回:
    #         s_proto: 当前帧与anchor原型的相似度
    #         s_max:   当前帧与单个anchor的最大相似度
    #         anchor_ref: anchor内部相似度参考值
    #     """
    #     if len(anchor_bank) == 0:
    #         return 1.0, 1.0, 0.7
    #
    #     # 当前帧 pooled + normalize
    #     f_curr = curr_memory_feat.mean(dim=1)  # [1, C]
    #     f_curr = F.normalize(f_curr, dim=-1)
    #
    #     # anchor pooled + normalize
    #     anchor_feats = []
    #     for a in anchor_bank:
    #         fa = a.mean(dim=1)  # [1, C]
    #         fa = F.normalize(fa, dim=-1)
    #         anchor_feats.append(fa)
    #
    #     # prototype
    #     proto = torch.stack(anchor_feats, dim=0).mean(dim=0)  # [1, C]
    #     proto = F.normalize(proto, dim=-1)
    #     s_proto = F.cosine_similarity(f_curr, proto, dim=-1).item()
    #
    #     # max similarity to single anchor
    #     sims = [F.cosine_similarity(f_curr, fa, dim=-1).item() for fa in anchor_feats]
    #     s_max = max(sims)
    #
    #     # anchor self-consistency
    #     if len(anchor_feats) >= 2:
    #         pair_sims = []
    #         for i in range(len(anchor_feats)):
    #             for j in range(i + 1, len(anchor_feats)):
    #                 pair_sims.append(F.cosine_similarity(anchor_feats[i], anchor_feats[j], dim=-1).item())
    #         anchor_ref = float(np.median(pair_sims))  # 比 mean 更稳
    #     else:
    #         anchor_ref = 0.7
    #
    #     return s_proto, s_max, anchor_ref

    def update(
        self,
        pred_boxes,
        conf_score,
        avg_pred_score=None,
        curr_memory_feat=None,
        anchor_bank=None,
    ):
        """
        pred_boxes: [1,4], normalized cxcywh
        conf_score: float
        avg_pred_score: float
        curr_memory_feat: [1,N,C], 当前ROI特征（用于恢复时和anchor比）
        anchor_bank: list of anchor memory feats
        """
        center = self._get_center(pred_boxes)

        info = {
            "state": self.state,
            "jump": False,
            "residual": 0.0,
            "median": 0.0,
            "mad": 0.0,
            "dyn_factor": self.dyn_factor,
            "memory_write_enabled": self.memory_write_enabled,
            "recover_counter": self.recover_counter,
        }

        if self.state == "NORMAL":
            residual = self._compute_residual(center, self.center_history)
            jump, med, mad = self._is_jump(residual, self.residual_history)

            #jump, med, mad = self._is_jump(center, self.center_history)
            info["jump"] = jump
            info["residual"] = residual
            info["median"] = med
            info["mad"] = mad

            if jump:
                # 进入冻结
                self.state = "FROZEN"
                self.dyn_factor = self.expand_ratio
                self.memory_write_enabled = False
                self.recover_counter = 0
                # 清空冻结阶段缓存，开始新阶段统计
                self.frozen_center_history = [center]
                self.frozen_residual_history = []
                self.normal_med = med
                self.normal_mad = mad
            else:
                self.center_history.append(center)
                if len(self.center_history) > self.memory_len:
                    self.center_history.pop(0)

                self.residual_history.append(residual)
                if len(self.residual_history) > self.memory_len:
                    self.residual_history.pop(0)

        elif self.state == "FROZEN":
            residual = self._compute_residual(center, self.frozen_center_history)
            jump, med, mad = self._is_jump(residual,
                                           self.residual_history,
                                           self.normal_med,
                                           self.normal_mad
                                           )#self.frozen_residual_history)

            info["jump"] = jump
            info["residual"] = residual
            info["median"] = med
            info["mad"] = mad

            if jump:
                # 再次发生跳变，重新开始冻结阶段统计
                self.frozen_center_history = [center]
                self.frozen_residual_history = []
                self.recover_counter = 0
            else:
                self.frozen_center_history.append(center)
                if len(self.frozen_center_history) > self.memory_len:
                    self.frozen_center_history.pop(0)

                self.frozen_residual_history.append(residual)
                if len(self.frozen_residual_history) > self.memory_len:
                    self.frozen_residual_history.pop(0)

                # 连续稳定帧计数
                self.recover_counter += 1

                if self.recover_counter >= self.recover_frames:
                    self.state = "RECOVER_CANDIDATE"

        # ---------------- RECOVER_CANDIDATE ----------------
        elif self.state == "RECOVER_CANDIDATE":
            residual = self._compute_residual(center, self.frozen_center_history)
            jump, med, mad = self._is_jump(residual, self.frozen_residual_history)

            info["jump"] = jump
            info["residual"] = residual
            info["median"] = med
            info["mad"] = mad

            if jump:
                # 还不稳，退回冻结
                self.state = "FROZEN"
                self.recover_counter = 0
                self.frozen_center_history = [center]
                self.frozen_residual_history = []
            else:
                # 额外恢复门：conf恢复 + anchor相似度恢复
                if avg_pred_score is None:
                    conf_ok = True
                else:
                    conf_ok = conf_score >= avg_pred_score


                anchor_ok = True
                if curr_memory_feat is not None and anchor_bank is not None and len(anchor_bank) > 0:
                    s_anchor = self._anchor_similarity(curr_memory_feat, anchor_bank)
                    # 用 anchor 内部正常相似度的保守比例来判定 reopen
                    pair_sims = []
                    if len(anchor_bank) >= 2:
                        for i in range(len(anchor_bank)):
                            for j in range(i + 1, len(anchor_bank)):
                                f1 = anchor_bank[i].mean(dim=1)
                                f2 = anchor_bank[j].mean(dim=1)
                                pair_sims.append(F.cosine_similarity(f1, f2, dim=-1).item())
                        anchor_base = float(np.mean(pair_sims))
                    else:
                        anchor_base = 0.7
                    anchor_ok = s_anchor >= (1.0 * anchor_base)
                    info["s_anchor"] = s_anchor
                    info["anchor_base"] = anchor_base

                    # # 恢复门：prototype相似度达到anchor内部一致性水平
                    # anchor_ok = s_proto >= anchor_ref
                    #
                    # info["s_proto"] = s_proto
                    # info["s_max"] = s_max
                    # info["anchor_ref"] = anchor_ref

                if conf_ok and anchor_ok:
                    # 真正恢复
                    self.state = "NORMAL"
                    self.dyn_factor = 1.0
                    self.memory_write_enabled = True
                    self.recover_counter = 0

                    # 冻结阶段的稳定中心接管为新的正常统计
                    self.center_history = list(self.frozen_center_history[-self.memory_len:])
                    self.residual_history = list(self.frozen_residual_history[-self.memory_len:])
                else:
                    # 继续冻结
                    self.state = "FROZEN"
                    self.memory_write_enabled = False

        info["state"] = self.state
        info["dyn_factor"] = self.dyn_factor
        info["memory_write_enabled"] = self.memory_write_enabled
        info["recover_counter"] = self.recover_counter

        return self.dyn_factor, self.memory_write_enabled, info