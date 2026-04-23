
import os
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text

class DTBDataset(BaseDataset):
    """
    Args:
        name: dataset name, should be 'OTB100', 'CVPR13', 'OTB50'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.dtb_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        class_name = sequence_name
        anno_path = '{}/{}/groundtruth_rect.txt'.format(self.base_path, sequence_name)
        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)
        frames_path = '{}/{}/img'.format(self.base_path, sequence_name)
        frames_list = ['{}/{:05d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]
        target_class = class_name
        return Sequence(sequence_name, frames_list, 'dtb', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):

        def get_file_names(folder_path):
            file_names = []
            for file in os.listdir(folder_path):
                file_names.append(file)
            return file_names
        folder_path = self.base_path
        sequence_list= get_file_names(folder_path)

        return sequence_list