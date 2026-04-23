import torch
import torch.nn.functional as F
from torchvision.ops import roi_align


def box_cxcywh_to_xyxy(box):
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def sanitize_xyxy(box_xyxy, feat_sz):
    box_xyxy[..., 0::2] = box_xyxy[..., 0::2].clamp(0, feat_sz - 1)
    box_xyxy[..., 1::2] = box_xyxy[..., 1::2].clamp(0, feat_sz - 1)
    # 保证 x2 > x1, y2 > y1
    box_xyxy[..., 2] = torch.maximum(box_xyxy[..., 2], box_xyxy[..., 0] + 1.0)
    box_xyxy[..., 3] = torch.maximum(box_xyxy[..., 3], box_xyxy[..., 1] + 1.0)
    return box_xyxy


class MemoryBankManager:
    """
    可直接嵌入模型的 memory bank 写入管理器

    依赖:
      - 当前 search feature: self.feature, shape [B,C,H,W]
      - memory encoder: self.memory_encoder
      - score_map: [1,1,Hs,Ws] 或 [1,Hs,Ws]
      - pred_boxes: [1,4], normalized cxcywh (相对于 feature size)
    """

    def __init__(
        self,
        fx_sz,
        memory_encoder,
        min_update_interval=30,
        confirm_frames=2,
        second_peak_min_ratio=0.30,
        peak_suppress_radius=1,
        device="cuda",
    ):
        self.fx_sz = fx_sz
        self.memory_encoder = memory_encoder
        self.min_update_interval = min_update_interval
        self.confirm_frames = confirm_frames
        self.second_peak_min_ratio = second_peak_min_ratio
        self.peak_suppress_radius = peak_suppress_radius
        self.device = device

        # memory banks
        self.anchor_bank = []   # fixed 3
        self.dynamic_bank = []  # rolling 3

        # state machine
        self.state = "TRACK"    # TRACK / FREEZE / RECOVER
        self.recover_count = 0

        # update tracking
        self.last_update_frame = -10**9

        # motion history
        self.prev_boxes = []    # store last few predicted boxes [4]
        self.motion_residual_hist = []

        # for debug
        self.debug_info = {}

    def _pool_feat(self, feat):
        # feat: [1, N, C] -> [1, C]
        return feat.mean(dim=1)

    def cosine_sim(self, feat1, feat2):
        f1 = self._pool_feat(feat1)
        f2 = self._pool_feat(feat2)
        return F.cosine_similarity(f1, f2, dim=-1).item()

    def max_sim_to_bank(self, feat, bank):
        if len(bank) == 0:
            return 0.0
        sims = [self.cosine_sim(feat, b) for b in bank]
        return max(sims)

    def avg_pairwise_sim(self, bank):
        if len(bank) < 2:
            return 0.7  # fallback
        sims = []
        for i in range(len(bank)):
            for j in range(i + 1, len(bank)):
                sims.append(self.cosine_sim(bank[i], bank[j]))
        return sum(sims) / len(sims)

    def motion_distance(self, curr_box):
        # curr_box: [4]
        if len(self.prev_boxes) < 2:
            return 0.0

        b1 = self.prev_boxes[-1]
        b2 = self.prev_boxes[-2]
        pred = b1 + (b1 - b2)
        d = torch.sum((curr_box - pred) ** 2).item()
        return d

    def motion_threshold(self):
        if len(self.motion_residual_hist) < 5:
            return 0.05
        hist = torch.tensor(self.motion_residual_hist[-20:], dtype=torch.float32)
        return (hist.mean() + hist.std(unbiased=False)).item()

    def get_top2_peaks(self, score_map):
        """
        返回:
          (y1, x1, v1), (y2, x2, v2 or None)
        如果第二峰无效，返回 None
        """
        if score_map.dim() == 4:
            sm = score_map[0, 0].detach()
        elif score_map.dim() == 3:
            sm = score_map[0].detach()
        else:
            sm = score_map.detach()

        H, W = sm.shape
        flat = sm.view(-1)
        idx1 = torch.argmax(flat)
        v1 = flat[idx1].item()
        y1 = int(idx1 // W)
        x1 = int(idx1 % W)

        # suppress neighborhood around peak1
        sm2 = sm.clone()
        r = self.peak_suppress_radius
        y0 = max(0, y1 - r)
        y1s = min(H, y1 + r + 1)
        x0 = max(0, x1 - r)
        x1s = min(W, x1 + r + 1)
        sm2[y0:y1s, x0:x1s] = -1e9

        idx2 = torch.argmax(sm2.view(-1))
        v2 = sm2.view(-1)[idx2].item()

        # 第二峰有效性：至少要达到第一峰一定比例
        if v2 <= 0 or v2 < self.second_peak_min_ratio * v1:
            return (y1, x1, v1), None

        y2 = int(idx2 // W)
        x2 = int(idx2 % W)
        return (y1, x1, v1), (y2, x2, v2)

    def candidate_box_from_peak(self, peak_y, peak_x, ref_box):
        """
        用当前预测框的 w,h，配合新 peak 位置，构造一个候选框
        ref_box: [1,4] normalized cxcywh
        """
        cand = ref_box.clone()
        # 把 peak 映射到 feature map坐标，再归一化
        cand[..., 0] = (peak_x + 0.5) / self.fx_sz
        cand[..., 1] = (peak_y + 0.5) / self.fx_sz
        return cand

    def extract_roi_memory_feature(self, feature_map, pred_boxes):
        """
        feature_map: [B,C,H,W]
        pred_boxes: [B,4], normalized cxcywh
        return: [B,N,C]
        """
        if feature_map.dim()==3:
            bs,L,C = feature_map.shape
            feature_map = feature_map.permute(0,2,1).reshape(bs,C,int(L**0.5),int(L**0.5))
        elif feature_map.dim()==4:
            bs, C, _, _ = feature_map.shape
        else:
            raise ValueError(f"Feature map dimension error: expected 3D or 4D tensor, got {feature_map.dim()}D instead.")

        pred_boxes_cxcywh = pred_boxes * self.fx_sz
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes_cxcywh)
        pred_boxes_xyxy = sanitize_xyxy(pred_boxes_xyxy, feat_sz=self.fx_sz)

        batch_idx = torch.arange(bs, device=pred_boxes.device, dtype=pred_boxes.dtype).unsqueeze(1)
        rois = torch.cat([batch_idx, pred_boxes_xyxy], dim=1)

        object_roi_feat = roi_align(
            input=feature_map,
            boxes=rois,
            output_size=(self.fx_sz // 2, self.fx_sz // 2),
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True
        ).detach()

        memory_feature = self.memory_encoder(object_roi_feat).reshape(bs, C, -1).permute(0, 2, 1)
        return memory_feature

    def update_state(self, s1_anchor, s2_anchor, motion_dist, has_second_peak):
        """
        无需手工设很多阈值：
        - anchor baseline = anchor内部平均相似度
        - motion baseline = 最近稳定历史均值+方差
        """
        anchor_base = self.avg_pairwise_sim(self.anchor_bank) if len(self.anchor_bank) >= 2 else 0.7
        motion_base = self.motion_threshold()

        delta_id = s1_anchor - s2_anchor if has_second_peak else s1_anchor - anchor_base

        # lost/freezing criteria
        low_identity = s1_anchor < anchor_base
        bad_motion = motion_dist > motion_base

        if self.state == "TRACK":
            if low_identity or bad_motion:
                self.state = "FREEZE"
                self.recover_count = 0

        elif self.state == "FREEZE":
            # 若又恢复了身份一致性，则进入 RECOVER
            if (not low_identity) and (not bad_motion):
                self.state = "RECOVER"
                self.recover_count = 1

        elif self.state == "RECOVER":
            if (not low_identity) and (not bad_motion):
                self.recover_count += 1
                if self.recover_count >= self.confirm_frames:
                    self.state = "TRACK"
                    self.recover_count = 0
            else:
                self.state = "FREEZE"
                self.recover_count = 0

        return {
            "anchor_base": anchor_base,
            "motion_base": motion_base,
            "delta_id": delta_id,
            "low_identity": low_identity,
            "bad_motion": bad_motion,
            "state": self.state,
        }

    def should_write_memory(self, frame_id, feature_map, score_map, pred_boxes):
        """
        pred_boxes: [1,4] normalized cxcywh, 当前主跟踪输出
        返回:
          should_write, selected_box, info
        """
        # 当前主候选
        feat1 = self.extract_roi_memory_feature(feature_map, pred_boxes)
        s1_anchor = self.max_sim_to_bank(feat1, self.anchor_bank) if len(self.anchor_bank) > 0 else 1.0
        s1_dynamic = self.max_sim_to_bank(feat1, self.dynamic_bank) if len(self.dynamic_bank) > 0 else 0.0

        # 第二候选（如果存在）
        peak1, peak2 = self.get_top2_peaks(score_map)
        has_second_peak = peak2 is not None

        if has_second_peak:
            y2, x2, _ = peak2
            cand2_box = self.candidate_box_from_peak(y2, x2, pred_boxes)
            feat2 = self.extract_roi_memory_feature(feature_map, cand2_box)
            s2_anchor = self.max_sim_to_bank(feat2, self.anchor_bank) if len(self.anchor_bank) > 0 else 0.0
        else:
            cand2_box = None
            s2_anchor = 0.0

        # motion
        curr_box = pred_boxes[0].detach().float()
        motion_dist = self.motion_distance(curr_box)

        # update state
        state_info = self.update_state(
            s1_anchor=s1_anchor,
            s2_anchor=s2_anchor,
            motion_dist=motion_dist,
            has_second_peak=has_second_peak,
        )

        # 最终写入条件
        # 只在 TRACK 状态且满足最小更新间隔时写入
        interval_ok = (frame_id - self.last_update_frame) >= self.min_update_interval
        anchor_base = state_info["anchor_base"]

        # 第一候选至少不要比 anchor 正常水平更差
        id_ok = s1_anchor >= anchor_base

        # 如果存在第二峰，要求第一峰身份更优；如果没有第二峰，则不额外惩罚
        if has_second_peak:
            rank_ok = s1_anchor > s2_anchor
        else:
            rank_ok = True

        motion_ok = motion_dist <= state_info["motion_base"]

        should_write = (
            self.state == "TRACK"
            and interval_ok
            and id_ok
            and rank_ok
            and motion_ok
        )

        selected_box = pred_boxes  # 默认写当前主输出

        info = {
            "state": self.state,
            "interval_ok": interval_ok,
            "id_ok": id_ok,
            "rank_ok": rank_ok,
            "motion_ok": motion_ok,
            "s1_anchor": s1_anchor,
            "s1_dynamic": s1_dynamic,
            "s2_anchor": s2_anchor,
            "motion_dist": motion_dist,
            **state_info
        }

        self.debug_info = info
        return should_write, selected_box, info

    def write_memory(self, frame_id, feature_map, pred_boxes):
        feat = self.extract_roi_memory_feature(feature_map, pred_boxes)

        # 先填 anchor
        if len(self.anchor_bank) < 3:
            self.anchor_bank.append(feat)
        else:
            # dynamic FIFO
            if len(self.dynamic_bank) >= 3:
                self.dynamic_bank.pop(0)
            self.dynamic_bank.append(feat)

        self.last_update_frame = frame_id

        # 更新运动残差统计
        curr_box = pred_boxes[0].detach().float()
        d = self.motion_distance(curr_box)
        if d > 0:
            self.motion_residual_hist.append(d)
            if len(self.motion_residual_hist) > 50:
                self.motion_residual_hist.pop(0)

        self.prev_boxes.append(curr_box)
        if len(self.prev_boxes) > 10:
            self.prev_boxes.pop(0)


    def step(self, frame_id, feature_map, score_map, pred_boxes):
        should_write, selected_box, info = self.should_write_memory(
            frame_id=frame_id,
            feature_map=feature_map,
            score_map=score_map,
            pred_boxes=pred_boxes,
        )

        # prev box 始终更新，用于状态估计
        curr_box = pred_boxes[0].detach().float()
        self.prev_boxes.append(curr_box)
        if len(self.prev_boxes) > 10:
            self.prev_boxes.pop(0)

        if should_write:
            self.write_memory(frame_id, feature_map, selected_box)

        return should_write, info