"""
created by xingxiangrui on 2019.5.23
this program is to :
    read badcase coco url and print some of them
"""
import torch.utils.data as data
import json
import os
import subprocess
from PIL import Image
import numpy as np
import torch
import pickle
from util import *
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import random


class coco_url_print():
    def __init__(self):
        # super(self).__init__()
        warnings.simplefilter("ignore")

        self.read_and_write_dir='/Users/baidu/Desktop/code/chun_ML_GCN/badcase_analyse/cls_gat_hist/'
        self.url_pkl_file_name='badcase_coco_url.pkl'
        self.url_pkl_file_path=self.read_and_write_dir+self.url_pkl_file_name
        self.output_category=24
        self.output_num=3


    # load and print coco url
    def run_coco_url_print(self):
        # loadint files
        with open(self.url_pkl_file_path,'rb') as f:
            print('loading ',self.url_pkl_file_path)
            coco_url_dict=pickle.load(f)
            random_idx=list(range(len(coco_url_dict[self.output_category])))
            random.shuffle(random_idx)

        print('category:', self.output_category,'output num:',self.output_num)
        for output_idx in range(self.output_num):
            print(coco_url_dict[self.output_category][random_idx[output_idx]])

        print('program end...')

if __name__ == '__main__':
    coco_url_print().run_coco_url_print()





