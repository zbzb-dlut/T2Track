
import numpy as np
import torch

def sanitize_xyxy(boxes, feat_sz=14.0, eps=1e-4):
    """
    boxes: [B,1,4] or [B,4], format = xyxy
    return: [B,4], 满足
        1) 0 <= x1,y1 < feat_sz
        2) 0 <= x2,y2 <= feat_sz
        3) x1 < x2, y1 < y2
    """

    if boxes.dim() == 3:
        boxes = boxes.squeeze(1)   # [B,1,4] -> [B,4]

    x1, y1, x2, y2 = boxes.unbind(-1)

    # 先裁到合法范围
    x1 = x1.clamp(0, feat_sz)
    y1 = y1.clamp(0, feat_sz)
    x2 = x2.clamp(0, feat_sz)
    y2 = y2.clamp(0, feat_sz)

    # 保证右下角严格大于左上角
    x2 = torch.maximum(x2, x1 + eps)
    y2 = torch.maximum(y2, y1 + eps)

    # 再裁一次，防止 x2/y2 因加 eps 超界
    x2 = x2.clamp(0, feat_sz)
    y2 = y2.clamp(0, feat_sz)

    # 如果 x2/y2 被裁回 feat_sz，反过来修 x1/y1
    x1 = torch.minimum(x1, x2 - eps).clamp(0, feat_sz)
    y1 = torch.minimum(y1, y2 - eps).clamp(0, feat_sz)

    return torch.stack([x1, y1, x2, y2], dim=-1)

def jitter_bbox(bbox, center_jitter=0.2, scale_jitter=0.2):
    """
    bbox: (..., 4) -> [x, y, w, h] normalized
    """
    x, y, w, h = bbox.unbind(-1)

    # center
    cx = x + 0.5 * w
    cy = y + 0.5 * h

    # center jitter
    delta_x = (torch.rand_like(cx) * 2 - 1) * center_jitter
    delta_y = (torch.rand_like(cy) * 2 - 1) * center_jitter

    cx = cx + delta_x * w
    cy = cy + delta_y * h

    # scale jitter（log-space更稳定）
    delta_w = (torch.rand_like(w) * 2 - 1) * scale_jitter
    delta_h = (torch.rand_like(h) * 2 - 1) * scale_jitter

    w = w * torch.exp(delta_w)
    h = h * torch.exp(delta_h)

    # back to xywh
    x = cx - 0.5 * w
    y = cy - 0.5 * h

    # clamp
    x = x.clamp(0.0, 1.0)
    y = y.clamp(0.0, 1.0)
    w = w.clamp(1e-6, 1.0)
    h = h.clamp(1e-6, 1.0)

    return torch.stack([x, y, w, h], dim=-1)