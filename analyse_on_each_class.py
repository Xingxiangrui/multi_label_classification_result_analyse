"""
created by xingxiangrui on 2019.5.15
this program is to read model validate results and labels
and analyse on each class

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


def scores_evaluation(scores_, targets_):
    print('evaluation start...')
    n, n_class = scores_.shape
    print('img_numbers=',n,'n_class=',n_class)
    Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
    cls_P,cls_R,cls_F1=np.zeros(n_class),np.zeros(n_class),np.zeros(n_class)
    for k in range(n_class):
        scores = scores_[:, k]         # all img scores on class_k
        targets = targets_[:, k]       # all img labels on class_k
        targets[targets == -1] = 0     # set img labels from -1 to 0
        Ng[k] = np.sum(targets == 1)   # ture:     all ture labels sum number
        Np[k] = np.sum(scores >= 0)    # positive: all predict positive sum number
        Nc[k] = np.sum(targets * (scores >= 0)) # true_positive: true_positive sum number
        cls_P[k]=Nc[k]/Np[k]
        cls_R[k]=Nc[k]/Ng[k]
        cls_F1[k]=(2 * cls_P[k] * cls_R[k]) / (cls_P[k] + cls_R[k])
    Np[Np == 0] = 1
    print('np.sum(Nc),true_positive=',np.sum(Nc))
    print('np.sum(Np),positive=',np.sum(Np))
    print('np.sum(Ng),ture=',np.sum(Ng))

    # for all labels num_imgs*n_classes
    OP = np.sum(Nc) / np.sum(Np)        # precision: true_positive/positive
    OR = np.sum(Nc) / np.sum(Ng)        # recall:    true_positive/true
    OF1 = (2 * OP * OR) / (OP + OR)     # F1_score: harmonic mean of precision and recall
    # average by class
    CP = np.sum(Nc / Np) / n_class      # precision: true_positive/positive
    CR = np.sum(Nc / Ng) / n_class      # recall:    true_positive/true
    CF1 = (2 * CP * CR) / (CP + CR)     # F1_score: harmonic mean of precision and recall

    return OP, OR, OF1, CP, CR, CF1,cls_P,cls_R,cls_F1

# load results on pkl files
# file in format { 0: [[ -4.905565    -7.9314375  ... -6.8639855   -8.622047    -8.28002]] numpy 1*80
#                  1: [[ -8.905565    -8.9314375  ... -5.8639855   -6.622047    -4.28002]] numpy 1*80
#                   ...
#                  1355: ..........    numpy 1*80 }
# label in format (0: [[1. 0. 0. 0. 0. 0. 0. 0. ..... 1 ]]  numpy 1*80
#                  1: [[0. 1. .....                  ...]]}
with open('checkpoint/coco/resnet101_on_coco/model_results.pkl', 'rb') as f:
    print("loading checkpoint/coco/resnet101_on_coco/model_results.pkl")
    model_results = pickle.load(f)
with open('checkpoint/coco/resnet101_on_coco/coco_labels_in_np.pkl','rb') as f:
    print("loading checkpoint/coco/resnet101_on_coco/model_results.pkl")
    labels=pickle.load(f)
# print('model_results[3]',model_results[3])
# print('labels[3]',labels[3])

# concat all numpy
total_results=model_results[0]
total_labels=labels[0]
for img_idx in range(len(model_results)-1):
    if img_idx%1000==0:
        print(img_idx,'/',len(model_results))
    total_results=np.append(total_results,model_results[img_idx+1],axis=0)
    total_labels=np.append(total_labels,labels[img_idx+1],axis=0)

print('np.shape(total_results)',np.shape(total_results))
print('np.shape(total_labels)',np.shape(total_labels))

with open('checkpoint/coco/resnet101_on_coco/model_results_numpy.pkl', 'wb') as f:
    print("writing checkpoint/coco/resnet101_on_coco/model_results_numpy.pkl")
    pickle.dump(total_results, f)
with open('checkpoint/coco/resnet101_on_coco/coco_labels_numpy.pkl','wb') as f:
    print("writing checkpoint/coco/resnet101_on_coco/oco_labels_numpy.pkl")
    pickle.dump(total_labels, f)

OP,OR,OF1,CP,CR,CF1,cls_P,cls_R,cls_F1=scores_evaluation(scores_=total_results,targets_=total_labels)

print('class_Precison',cls_P)
print('class_recall',cls_R)
print('class F1 score',cls_F1)
print('All evaluate results:')
print('OP: {OP:.4f}\t'
      'OR: {OR:.4f}\t'
      'OF1: {OF1:.4f}\t'
      'CP: {CP:.4f}\t'
      'CR: {CR:.4f}\t'
      'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))

all_evaluate_results={'OP':OP,'OR':OR,'OF1':OF1,'CP':CP,'CR':CR,'CF1':CF1,'cls_P':cls_P,'cls_R':cls_R,'cls_F1':cls_F1}
with open('checkpoint/coco/resnet101_on_coco/all_evaluate_results.pkl','wb') as f:
    print('writing checkpoint/coco/resnet101_on_coco/all_evaluate_results.pkl')
    pickle.dump(all_evaluate_results, f)















