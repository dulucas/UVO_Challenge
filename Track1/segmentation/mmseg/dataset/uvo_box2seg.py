# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from collections import OrderedDict

import mmcv
import numpy as np
from mmcv.utils import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

from mmseg.core import eval_metrics, intersect_and_union, pre_eval_to_metrics
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .pipelines import Compose

import cv2
import json
from pycocotools import mask as transform_mask

@DATASETS.register_module()
class UVODataset(Dataset):

    CLASSES = ('background', 'foreground')

    PALETTE = [[0, 0, 0], [255, 0, 0]]


    def __init__(self,
                 pipeline,
                 img_dir,
                 proposal_path,
                 gt_path,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = None
        self.proposal_path = proposal_path
        self.proposals = json.load(open(self.proposal_path, 'r'))
        self.proposals = sorted(self.proposals, key=lambda x:x['image_id'])

        id2annos = dict()
        for i in self.proposals:
            if i['image_id'] not in id2annos:
                id2annos[i['image_id']] = []
            id2annos[i['image_id']].append(i)
        proposals_part = []
        for key in id2annos.keys():
            ps = id2annos[key]
            ps = sorted(ps, key=lambda x:-x['score'])
            proposals_part.extend(ps[:100])
        self.proposals = proposals_part
        ########################
        #self.proposals = self.proposals[::1600]
        ########################
        self.gts = json.load(open(gt_path, 'r'))
        self.id2imgs = dict()
        for i in self.gts['images']:
            self.id2imgs[i['id']] = i['file_name']
        self.CLASSES, self.PALETTE = self.get_classes_and_palette(
            classes, palette)
        if test_mode:
            assert self.CLASSES is not None, \
                '`cls.CLASSES` or `classes` should be specified when testing'

        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)

        # load annotations
        self.img_infos = self.load_annotations(self.proposals)

    def __len__(self):
        """Total number of samples of data."""
        return len(self.img_infos)

    def load_annotations(self, proposals):

        img_infos = []
        for p in proposals:
            img_info = dict(filename=self.id2imgs[p['image_id']])
            img_info['pred_bbox'] = p['bbox']
            img_info['pred_score'] = p['score']
            img_info['image_id'] = p['image_id']
            img_info['pred_category_id'] = p['category_id']
            img_infos.append(img_info)
        img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos

    def format_results(self, result, info):
        out = dict()
        out['bbox'] = info['pred_bbox']
        out['score'] = info['pred_score']
        out['category_id'] = info['pred_category_id']
        out['image_id'] = info['image_id']

        crop_bbox = info['crop_bbox']
        img_ori_shape = info['img_shape_before_crop']
        img = np.zeros(img_ori_shape[:2]).astype(np.uint8)

        h,w = img[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]].shape
        mask = cv2.resize(result, (w,h), interpolation=cv2.INTER_NEAREST)
        img[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]] = mask

        out['segmentation'] = transform_mask.encode(np.array(img[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
        out['segmentation']['counts'] = out['segmentation']['counts'].decode("utf-8")

        return out

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir
        if self.custom_classes:
            results['label_map'] = self.label_map

    def __getitem__(self, idx):

        return self.prepare_test_img(idx)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by
                pipeline.
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def get_classes_and_palette(self, classes=None, palette=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.
            palette (Sequence[Sequence[int]]] | np.ndarray | None):
                The palette of segmentation map. If None is given, random
                palette will be generated. Default: None
        """
        if classes is None:
            self.custom_classes = False
            return self.CLASSES, self.PALETTE

        self.custom_classes = True
        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        if self.CLASSES:
            if not set(classes).issubset(self.CLASSES):
                raise ValueError('classes is not a subset of CLASSES.')

            # dictionary, its keys are the old label ids and its values
            # are the new label ids.
            # used for changing pixel labels in load_annotations.
            self.label_map = {}
            for i, c in enumerate(self.CLASSES):
                if c not in class_names:
                    self.label_map[i] = -1
                else:
                    self.label_map[i] = classes.index(c)

        palette = self.get_palette_for_custom_classes(class_names, palette)

        return class_names, palette

    def get_palette_for_custom_classes(self, class_names, palette=None):

        if self.label_map is not None:
            # return subset of palette
            palette = []
            for old_id, new_id in sorted(
                    self.label_map.items(), key=lambda x: x[1]):
                if new_id != -1:
                    palette.append(self.PALETTE[old_id])
            palette = type(self.PALETTE)(palette)

        elif palette is None:
            if self.PALETTE is None:
                palette = np.random.randint(0, 255, size=(len(class_names), 3))
            else:
                palette = self.PALETTE

        return palette


