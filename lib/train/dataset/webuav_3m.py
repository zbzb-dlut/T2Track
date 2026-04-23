import os
import torch
import pandas as pd
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.dataset.depth_utils import get_x_frame  # 复用已有函数读取RGB+IR
from lib.train.admin import env_settings
import random
import numpy as np
from lib.train.data import jpeg4py_loader
import csv

class WebUAV_3M(BaseVideoDataset):
    """ VisDrone dataset loader for RGB+IR tracking """

    def __init__(self,
                 root=None,
                 image_loader=jpeg4py_loader,
                 split='train',
                 dtype='rgbrgb',
                 seq_ids=None,
                 data_fraction=None,
                 multi_modal_vision=False,
                 multi_modal_language=False,
                 use_nlp=False):
        """
        args:
            root - path to VisDrone root folder (Train/Val)
            split - train or val split
            dtype - 'rgbrgb', 'rgb', 'ir'
            seq_ids - optional list of sequence indices to use
            data_fraction - fraction of dataset to load
        """
        # 默认路径可以根据需要修改
        root = env_settings().webuav_dir if root is None else root
        # assert split in ['train', 'val', 'all'], f"Only support 'train', 'val', or 'all', got {split}"
        super().__init__('WebUAV_3M', root, image_loader)

        self.dtype = dtype
        # 获取序列列表
        self.sequence_list = self._get_sequence_list()

        # 选择指定序列
        if seq_ids is None:
            seq_ids = list(range(len(self.sequence_list)))
        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        # 数据抽样
        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))

        self.multi_modal_vision = multi_modal_vision
        self.multi_modal_language = multi_modal_language
        self.use_nlp = use_nlp

    def get_name(self):
        return 'WebUAV_3M'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True  # 通过标注判断可见性

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        absent_file = os.path.join(seq_path, "absent.txt")

        with open(absent_file, 'r', newline='') as f:
            absent = torch.ByteTensor([int(v.strip()) for v in f if v.strip()])
        target_visible = ~absent
        return target_visible

    def _get_sequence_list(self):
        """获取所有序列列表（支持 train_LT_xxx, train_ST_xxx 多个子文件夹）"""
        split_root = os.path.join(self.root, 'Train')  # Train 或 Val

        seq_list = []
        # 遍历 train_LT_001, train_ST_001 等子文件夹
        for subdir in sorted(os.listdir(split_root)):
            subdir_path = os.path.join(split_root, subdir)
            if not os.path.isdir(subdir_path):
                continue
            seq_list.append(os.path.join('Train',subdir))

        return seq_list

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def _read_bb_anno(self, seq_path):
        """读取bounding box标注，兼容逗号分隔和空格分隔"""
        def read_one(file_path):
            try:
                # 尝试按逗号分隔
                gt = pd.read_csv(file_path, delimiter=',', header=None, dtype=float).values
            except Exception:
                # 如果失败，改为空格/Tab 分隔
                # gt = pd.read_csv(file_path, delim_whitespace=True, header=None, dtype=float).values
                gt = pd.read_csv(file_path, sep='\s+', header=None, dtype=float).values

            return torch.tensor(gt, dtype=torch.float32)

        rgb_anno_file = os.path.join(seq_path,'groundtruth_rect.txt')
        rgb_gt = read_one(rgb_anno_file)

        return rgb_gt

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        rgb_bbox = self._read_bb_anno(seq_path)
        valid = (rgb_bbox[:, 2] > 0) & (rgb_bbox[:, 3] > 0)
        visible = self._read_target_visible(seq_path) & valid.byte()
        return {'bbox': rgb_bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):

        # rgb_frame_files = [f for f in os.listdir(os.path.join(seq_path, 'img')) if f.endswith('.jpg')]
        # rgb_frame_files.sort(key=lambda x: int(x.split('.')[0]))
        # # 拼接完整路径
        # rgb_frame_path = [os.path.join(os.path.join(seq_path, 'img'), f) for f in rgb_frame_files][frame_id]

        #rgb_frame_path = os.path.join(seq_path,'img/', f'{frame_id:06d}.jpg')
        rgb_frame_path = os.path.join(seq_path, 'img', '{:06}.jpg'.format(frame_id + 1))  # frames start from 1
        return rgb_frame_path

    def _get_frame(self, seq_path, frame_id):
        rgb_frame_path = self._get_frame_path(seq_path, frame_id)
        frame = self.image_loader(rgb_frame_path)
        if self.multi_modal_vision:
            frame = np.concatenate((frame, frame), axis=-1)
        return frame


    def get_frames(self, seq_id, frame_ids, anno=None):
        frame_ids = [1 if x == 0 else x for x in frame_ids]

        seq_path = self._get_sequence_path(seq_id)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            # 如果是二维bbox，按frame_ids选择
            if isinstance(value, torch.Tensor) and value.ndim >= 2:
                anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
            else:
                anno_frames[key] = value

        object_meta = OrderedDict({
            'object_class_name': None,
            'motion_class': None,
            'major_class': None,
            'root_class': None,
            'motion_adverb': None
        })

        # self.show(frame_list,anno_frames)
        return frame_list, anno_frames, object_meta

    # def get_annos(self, seq_id, frame_ids, anno=None):
    #     if anno is None:
    #         anno = self.get_sequence_info(seq_id)
    #
    #     anno_frames = {}
    #     for key, value in anno.items():
    #         anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
    #
    #     return anno_frames

    def show(self,frames,bbox):
        import numpy as np
        import cv2
        img = frames[0][:, :, :3]
        # ir = frames[0][:, :, 3:]

        if not isinstance(img, np.ndarray):
            img = img.cpu().numpy()
            # ir = ir.cpu().numpy()

        # 确保 dtype 正确
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
            # ir = (ir * 255).astype(np.uint8) if ir.max() <= 1 else ir.astype(np.uint8)
        # 确保内存连续
        img = np.ascontiguousarray(img)
        # ir = np.ascontiguousarray(ir)

        # alpha blending: alpha 控制 IR 的透明度
        alpha = 0.6  # IR 的透明度，可以调节 0~1
        # overlay = cv2.addWeighted(img, 1 - alpha, ir, alpha, 0)

        anno = bbox['bbox'][0]
        x1, y1, x2, y2 = int(anno[0]), int(anno[1]), int(anno[0] + anno[2]), int(
            anno[1] + anno[3])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # anno = bbox['bbox'][0]
        # x11, y11, x21, y21 = int(anno[0]), int(anno[1]), int(anno[0] + anno[2]), int(
        #     anno[1] + anno[3])
        # cv2.rectangle(overlay, (x11, y11), (x21, y21), (0, 0, 255), 2)

        # # === 定义鼠标回调函数 ===
        # def mouse_callback(event, x, y, flags, param):
        #     if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        #         print(f"Clicked at: ({x}, {y})")
        #         # 也可以把坐标画在图像上
        #         cv2.circle(overlay, (x, y), 5, (255, 0, 0), -1)
        #         cv2.putText(overlay, f"({x},{y})", (x + 10, y - 10),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        #         cv2.imshow("Image", overlay)

        # # 注册鼠标事件
        # cv2.namedWindow("Image")
        # cv2.setMouseCallback("Image", mouse_callback)

        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()