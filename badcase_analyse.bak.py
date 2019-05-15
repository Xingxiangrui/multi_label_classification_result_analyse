"""
created by xingxiangrui on 2019.5.15
this program is to read model validate results and labels
and analyse precision, recall, F1 on each class
and print all bad examples

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




def evaluation_and_badcase(scores_, targets_):
    print('evaluation start...')
    n, n_class = scores_.shape
    print('img_numbers=',n,'n_class=',n_class)
    Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
    ture_Positive,true_negative,false_positive=np.zeros((n,n_class)),np.zeros((n,n_class)),np.zeros((n,n_class))
    cls_P,cls_R,cls_F1=np.zeros(n_class),np.zeros(n_class),np.zeros(n_class)
    for k in range(n_class):
        scores = scores_[:, k]         # all img scores on class_k
        targets = targets_[:, k]       # all img labels on class_k
        targets[targets == -1] = 0     # set img labels from -1 to 0
        Ng[k] = np.sum(targets == 1)   # ture:     all ture labels sum number
        Np[k] = np.sum(scores >= 0)    # positive: all predict positive sum number
        Nc[k] = np.sum(targets * (scores >= 0)) # true_positive: true_positive sum number
        ture_Positive[:,k]=np.where(targets * (scores >= 0)==1 ,1,0)
        true_negative[:,k]=np.where(targets * (scores < 0)==1 ,1,0)
        false_positive[:,k]=np.where((targets==0)*(scores >= 0)==1, 1,0)
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

    return OP, OR, OF1, CP, CR, CF1,cls_P,cls_R,cls_F1,ture_Positive,true_negative,false_positive


# loading results and labels
with open('checkpoint/coco/resnet101_on_coco/model_results_numpy.pkl', 'rb') as f:
    print("loading checkpoint/coco/resnet101_on_coco/model_results_numpy.pkl")
    total_results = pickle.load(f)
with open('checkpoint/coco/resnet101_on_coco/coco_labels_numpy.pkl','rb') as f:
    print("loading checkpoint/coco/resnet101_on_coco/coco_labels_numpy.pkl")
    total_labels = pickle.load(f)

print('np.shape(total_results)',np.shape(total_results))
print('np.shape(total_labels)',np.shape(total_labels))

# analyse on each class and store badcase
OP,OR,OF1,CP,CR,CF1,cls_P,cls_R,cls_F1,ture_Positive,true_negative,false_positive=evaluation_and_badcase(scores_=total_results,targets_=total_labels)

print('class_Precison',cls_P)
print('class_recall',cls_R)
print('class F1 score',cls_F1)
# print('ture_Positive',ture_Positive)
# print('true_negative',true_negative)
# print('false_positive',false_positive)

true_negative_num,true_positive_num,false_positive_num=0,0,0
ture_negative_dict,true_positive_dict,false_positive_dict={},{},{}


[img_num,class_num]=ture_Positive.shape
for class_idx in range(class_num):
    ture_negative_dict[class_idx]=[]
    true_positive_dict[class_idx]=[]
    false_positive_dict[class_idx]=[]
for img_idx in range(img_num):
    for class_idx in range(class_num):
        if true_negative[img_idx,class_idx]==1:
            true_negative_num=true_negative_num+1
            ture_negative_dict[class_idx].append(img_idx)
        if ture_Positive[img_idx,class_idx]==1:
            true_positive_num=true_positive_num+1
            true_positive_dict[class_idx].append(img_idx)
        if false_positive[img_idx,class_idx]==1:
            false_positive_num=false_positive_num+1
            false_positive_dict[class_idx].append(img_idx)
print('true_positive_num',true_positive_num)
print('true_negative_num=',true_negative_num)
print('false_positive_num',false_positive_num)

print('All evaluate results:')
print('OP: {OP:.4f}\t'
      'OR: {OR:.4f}\t'
      'OF1: {OF1:.4f}\t'
      'CP: {CP:.4f}\t'
      'CR: {CR:.4f}\t'
      'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))

# all_evaluate_results={'OP':OP,'OR':OR,'OF1':OF1,'CP':CP,'CR':CR,'CF1':CF1,'cls_P':cls_P,'cls_R':cls_R,'cls_F1':cls_F1}
# with open('checkpoint/coco/resnet101_on_coco/all_evaluate_results.pkl','wb') as f:
#     print('writing checkpoint/coco/resnet101_on_coco/all_evaluate_results.pkl')
#     pickle.dump(all_evaluate_results, f)

# write results into .txt file
fw = open('checkpoint/coco/resnet101_on_coco/result_analyse.txt','w')
fw.write('Total OP,OR,OF1,CP,CR,CF1:\n')
fw.write('OP: {OP:.4f}\t'
      'OR: {OR:.4f}\t'
      'OF1: {OF1:.4f}\t'
      'CP: {CP:.4f}\t'
      'CR: {CR:.4f}\t'
      'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))

coco_list_path = os.path.join('data/data/coco/data/val_anno.json')
coco_img_list = json.load(open(coco_list_path, 'r'))
# print(coco_img_list)
ture_negative_name_dict,true_positive_name_dict,false_positive_name_dict={},{},{}
for class_idx in range(class_num):
    ture_negative_name_dict[class_idx]=[]
    true_positive_name_dict[class_idx]=[]
    false_positive_name_dict[class_idx]=[]
for class_idx in range(len(false_positive_dict)):
    for object_idx in range(len(false_positive_dict[class_idx])):
        file_idx=false_positive_dict[class_idx][object_idx]
        item=coco_img_list[file_idx]
        false_positive_name_dict[class_idx].append(item['file_name'])
for class_idx in range(len(ture_negative_dict)):
    for object_idx in range(len(ture_negative_dict[class_idx])):
        file_idx=ture_negative_dict[class_idx][object_idx]
        item=coco_img_list[file_idx]
        ture_negative_name_dict[class_idx].append(item['file_name'])

with open('checkpoint/coco/resnet101_on_coco/ture_negative_name_dict.pkl','wb') as f:
    print('writing checkpoint/coco/resnet101_on_coco/ture_negative_name_dict.pkl')
    pickle.dump(ture_negative_name_dict, f)
with open('checkpoint/coco/resnet101_on_coco/false_positive_name_dict.pkl','wb') as f:
    print('writing checkpoint/coco/resnet101_on_coco/false_positive_name_dict.pkl')
    pickle.dump(false_positive_name_dict, f)


# for class_idx in range(0,1):
#     print('class_num',class_idx,'file_names',false_positive_name_dict[class_idx])




