import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class UAV123Dataset(BaseDataset):
    """ UAV123 dataset.
    Publication:
        A Benchmark and Simulator for UAV Tracking.
        Matthias Mueller, Neil Smith and Bernard Ghanem
        ECCV, 2016
        https://ivul.kaust.edu.sa/Documents/Publications/2016/A%20Benchmark%20and%20Simulator%20for%20UAV%20Tracking.pdf
    Download the dataset from https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx
    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.uav123_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        startFrame = sequence_info['startFrame']
        endFrame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                                                                           sequence_path=sequence_path, frame=frame_num,
                                                                           nz=nz, ext=ext) for frame_num in
                  range(startFrame + init_omit, endFrame + 1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'uav123_l', ground_truth_rect[init_omit:, :],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)


    # def _get_sequence_info_list(self):
    #     # {"name": "uav_bike1", "path": "data_seq/UAV123/bike1", "startFrame": 1, "endFrame": 3085, "nz": 6,"ext": "jpg",
    #     # "anno_path": "anno/UAV123/bike1.txt", "object_class": "vehicle"},
    #     sequence_info_list = [
    #                 {"name": "uav_person10", "path": "data_seq/UAV123/person10", "startFrame": 1, "endFrame": 1021, "nz": 6,
    #                  "ext": "jpg", "anno_path": "anno/UAV123/person10.txt", "object_class": "person"},
    #     ]
    #
    #     return sequence_info_list

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {'name': 'bike1', 'path': 'data_seq/UAV123/bike1', 'startFrame': 1, 'endFrame': 3085, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV20L/bike1.txt', 'object_class': 'vehicle'},
            {'name': 'bird1', 'path': 'data_seq/UAV123/bird1', 'startFrame': 1, 'endFrame': 2437, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV20L/bird1.txt', 'object_class': 'vehicle'},
            {'name': 'car1', 'path': 'data_seq/UAV123/car1', 'startFrame': 1, 'endFrame': 2629, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV20L/car1.txt', 'object_class': 'vehicle'},
            {'name': 'car3', 'path': 'data_seq/UAV123/car3', 'startFrame': 1, 'endFrame': 1717, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV20L/car3.txt', 'object_class': 'vehicle'},
            {'name': 'car6', 'path': 'data_seq/UAV123/car6', 'startFrame': 1, 'endFrame': 4861, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV20L/car6.txt', 'object_class': 'vehicle'},
            {'name': 'car8', 'path': 'data_seq/UAV123/car8', 'startFrame': 1, 'endFrame': 2575, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV20L/car8.txt', 'object_class': 'vehicle'},
            {'name': 'car9', 'path': 'data_seq/UAV123/car9', 'startFrame': 1, 'endFrame': 1879, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV20L/car9.txt', 'object_class': 'vehicle'},
            {'name': 'car16', 'path': 'data_seq/UAV123/car16', 'startFrame': 1, 'endFrame': 1993, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV20L/car16.txt', 'object_class': 'vehicle'},
            {'name': 'group1', 'path': 'data_seq/UAV123/group1', 'startFrame': 1, 'endFrame': 4873, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV20L/group1.txt', 'object_class': 'vehicle'},
            {'name': 'group2', 'path': 'data_seq/UAV123/group2', 'startFrame': 1, 'endFrame': 2683, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV20L/group2.txt', 'object_class': 'vehicle'},
            {'name': 'group3', 'path': 'data_seq/UAV123/group3', 'startFrame': 1, 'endFrame': 5527, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV20L/group3.txt', 'object_class': 'vehicle'},
            {'name': 'person2', 'path': 'data_seq/UAV123/person2', 'startFrame': 1, 'endFrame': 2623, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV20L/person2.txt', 'object_class': 'vehicle'},
            {'name': 'person4', 'path': 'data_seq/UAV123/person4', 'startFrame': 1, 'endFrame': 2743, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV20L/person4.txt', 'object_class': 'vehicle'},
            {'name': 'person5', 'path': 'data_seq/UAV123/person5', 'startFrame': 1, 'endFrame': 2101, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV20L/person5.txt', 'object_class': 'vehicle'},
            {'name': 'person7', 'path': 'data_seq/UAV123/person7', 'startFrame': 1, 'endFrame': 2065, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV20L/person7.txt', 'object_class': 'vehicle'},
            {'name': 'person14', 'path': 'data_seq/UAV123/person14', 'startFrame': 1, 'endFrame': 2923, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV20L/person14.txt', 'object_class': 'vehicle'},
            {'name': 'person17', 'path': 'data_seq/UAV123/person17', 'startFrame': 1, 'endFrame': 2347, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV20L/person17.txt', 'object_class': 'vehicle'},
            {'name': 'person19', 'path': 'data_seq/UAV123/person19', 'startFrame': 1, 'endFrame': 4357, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV20L/person19.txt', 'object_class': 'vehicle'},
            {'name': 'person20', 'path': 'data_seq/UAV123/person20', 'startFrame': 1, 'endFrame': 1783, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV20L/person20.txt', 'object_class': 'vehicle'},
            {'name': 'uav1', 'path': 'data_seq/UAV123/uav1', 'startFrame': 1, 'endFrame': 3469, 'nz': 6,
             'ext': 'jpg', 'anno_path': 'anno/UAV20L/uav1.txt', 'object_class': 'vehicle'}]

        return sequence_info_list
