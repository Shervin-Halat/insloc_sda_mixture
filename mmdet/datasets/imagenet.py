import csv
import io
import math
import os.path as op
import random

import cv2
import mmcv
import numpy as np
from mmcv.utils import build_from_cfg
from torch.utils.data import Dataset

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS, PIPELINES
from .pipelines import Compose

import os                                       ########
import pickle                                    ########



the_filename = '/mnt/e/Data/test/PackedGenerated(OneListPer)'               #######                             
# with open(the_filename, 'rb') as f:                                         #######
#     packed_generated_images = pickle.load(f)                                       #######

# packed_generated_images_filenames = [the_filename + '/' + i for i in os.listdir(the_filename)]

dirs = np.sort(np.array(os.listdir(the_filename),dtype = int)).astype(dtype = str)
packed_generated_images_filenames = [the_filename + '/' + i for i in dirs]



@DATASETS.register_module()
class ImageNetDataset(Dataset):
    CLASSES = None

    # def __init__(                         #####
    #     self,
    #     ann_file,
    #     pipeline,
    #     preprocess,                       #####
    #     data_root=None,
    #     img_prefix='',
    #     seg_prefix=None,
    # ):

    def __init__(
        self,
        ann_file,
        pipeline,
        data_root=None,
        img_prefix='',
        seg_prefix=None,
    ):

        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix

        self.data_infos = self.load_annotations(ann_file)
        self.flag = np.ones(len(self), dtype=np.uint8)

        # self.preprocess_pipeline = build_from_cfg(preprocess, PIPELINES)
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        txt_ptr = open(ann_file)
        txt = txt_ptr.readlines()
        all_imgs = [
            op.join(self.img_prefix,
                    each.strip().split()[0]) for each in txt
        ]
        txt_ptr.close()
        return all_imgs

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = None
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    # def __getitem__(self, idx):                   ######
    #     return self.prepare_train_img(idx)        ######

    def __getitem__(self,idx):                     ######
        return self.prepare_train_img(idx)          ######


    def prepare_train_img(self,idx):
        # Randomly choose a background image from the whole dataset
        # base_idx = int(np.random.randint(0, len(self.data_infos), 1))
        # base_info = self.data_infos[base_idx]

        # base_idx2 = int(np.random.randint(0, len(self.data_infos), 1))
        # base_info2 = self.data_infos[base_idx2]
        # base_info = [base_info, base_info2]

        # Load the current image as the foreground image
        # current_info = self.data_infos[idx]                                      #######
        # instance_info = [current_info]
        # instance_idx = [idx]

        # while True:                                                                             #######
        #     rand_instance_idx = random.randint(0,len(packed_generated_images)-1)
        #     if len(packed_generated_images[rand_instance_idx]) >= 1:
        #         break

        # rnd_filename = random.choice(packed_generated_images_filenames)       ###
        rnd_filename = packed_generated_images_filenames[idx]                   ###
        # print(idx)
        with open(rnd_filename, 'rb') as f:
            results0, results1 = pickle.load(f)
        
        # Compose data
        # results = dict(img_info=base_info, instance_info=instance_info)           ####
        # results['ins_idx'] = instance_idx                                         ####

        # qq3 = np.array(results1['img'].data.reshape(256,256,3))                   ######################################
        # np.save('/mnt/c/Users/sherw/OneDrive/Desktop/temp/test3.npy',qq3)         ######################################


        # import matplotlib.pyplot as plt                                           ########################################
        # qq = target_data['img'][1].detach().clone()
        # qq = np.einsum('ijk->jki', qq)                         
        # qq = qq.to('cpu')
        # plt.imshow(q)
        # plt.show()


        # Copy and paste foreground image onto two background images
        # results0, results1 = self.preprocess_pipeline(results)                    #######


        # print('\n','imagenet','\n')
        # print(idx)
        # print(results0['gt_labels'])
        # print(results0['gt_bboxes'],results0['img'].data.shape)
        # print(results1['gt_bboxes'],results1['img'].data.shape)
        

        # Augment two synthetic images
        results = self.pipeline(results0)
        results1 = self.pipeline(results1)
        results['target_data'] = results1
        
        
        # print('\n','imagenet','\n')
        # print(idx)
        # print(results['gt_labels'])
        # print(results['gt_bboxes'],results['img'].data.shape)
        # print(results1['gt_bboxes'],results1['img'].data.shape)

        return results