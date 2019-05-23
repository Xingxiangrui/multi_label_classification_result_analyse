"""
created by xingxiangrui on 2019.5.15
this program is to :
    read model validate results and labels
    analyse precision, recall, F1 on each class
    print and store all bad examples file_name
    save bad example size histograms
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


class badcase_analyse():
    def __init__(self):
        # super(self).__init__()
        warnings.simplefilter("ignore")

        # self.load_model_validate_output_path='checkpoint/coco/resnet101_on_coco/model_results_numpy.pkl'
        # self.load_labels_path='checkpoint/coco/resnet101_on_coco/coco_labels_numpy.pkl'
        # self.load_validate_json_path='data/data/coco/data/val_anno.json'
        # self.write_ture_negative_path='checkpoint/coco/resnet101_on_coco/ture_negative_name_dict.pkl'
        # self.write_false_positive_path='checkpoint/coco/resnet101_on_coco/false_positive_name_dict.pkl'
        # self.html_path='checkpoint/coco/resnet101_on_coco/class_result.html'
        # self.write_class_result_into_html=True
        # self.write_badcase_name_list_into_dict = True
        # self.generate_badcase_size_histogram=True
        # self.area_annotation_document='data/data/coco/data/annotations/instances_val2014.json'
        # self.histo_img_dir='checkpoint/coco/resnet101_on_coco/badcase_size_histograms/'

        self.load_model_validate_output_path='checkpoint/coco/weight_decay_cls_gat_on_5_10/model_results_numpy.pkl'
        self.load_labels_path='checkpoint/coco/weight_decay_cls_gat_on_5_10/coco_labels_numpy.pkl'
        self.load_validate_json_path='data/data/coco/data/val_anno.json'
        self.write_ture_negative_path='checkpoint/coco/weight_decay_cls_gat_on_5_10/ture_negative_name_dict.pkl'
        self.write_false_positive_path='checkpoint/coco/weight_decay_cls_gat_on_5_10/false_positive_name_dict.pkl'
        self.html_path='checkpoint/coco/weight_decay_cls_gat_on_5_10/class_result.html'
        self.write_class_result_into_html=True
        self.write_badcase_name_list_into_dict = True
        self.generate_badcase_size_histogram=True
        self.area_annotation_document='data/data/coco/data/annotations/instances_val2014.json'
        self.histo_img_dir='checkpoint/coco/weight_decay_cls_gat_on_5_10/badcase_size_histograms/'
        self.badcase_coco_url_path='checkpoint/coco/weight_decay_cls_gat_on_5_10/badcase_coco_url.pkl'

    # evaluate model result on each category and generate badcase matrix
    def evaluation_and_badcase(self,scores_, targets_):

        print('evaluation start...')
        n, n_class = scores_.shape
        print('img_numbers=',n,'n_class=',n_class)

        # matrixs about prediction
        Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
        ture_Positive,true_negative,false_positive=np.zeros((n,n_class)),np.zeros((n,n_class)),np.zeros((n,n_class))
        cls_P,cls_R,cls_F1=np.zeros(n_class),np.zeros(n_class),np.zeros(n_class)
        cls_weight=np.zeros(n_class)

        # for each class calculate class P,R,F1,weight and label_predict 0,1 matrix
        for k in range(n_class):
            scores = scores_[:, k]         # all img scores on class_k
            targets = targets_[:, k]       # all img labels on class_k
            targets[targets == -1] = 0     # set img labels from -1 to 0
            Ng[k] = np.sum(targets == 1)   # ture:     all ture labels sum number on class_k
            Np[k] = np.sum(scores >= 0)    # positive: all predict positive sum number on class_k
            Nc[k] = np.sum(targets * (scores >= 0)) # true_positive: true_positive sum number on class_k
            cls_weight[k] = Ng[k] / n          # true samples weight in all samples of class_k
            ture_Positive[:,k]=np.where(targets * (scores >= 0)==1 ,1,0)     # label ture  predict positive
            true_negative[:,k]=np.where(targets * (scores < 0)==1 ,1,0)      # label ture  predict negative
            false_positive[:,k]=np.where((targets==0)*(scores >= 0)==1, 1,0) # label false predict positive
            cls_P[k]=Nc[k]/Np[k]
            cls_R[k]=Nc[k]/Ng[k]
            cls_F1[k]=(2 * cls_P[k] * cls_R[k]) / (cls_P[k] + cls_R[k])
        Np[Np == 0] = 1
        print('np.sum(Nc),true_positive=',np.sum(Nc))
        print('np.sum(Np),positive=',np.sum(Np))
        print('np.sum(Ng),ture=',np.sum(Ng))
        self.cls_weight=cls_weight

        # for all labels num_imgs*n_classes
        OP = np.sum(Nc) / np.sum(Np)        # precision: true_positive/positive
        OR = np.sum(Nc) / np.sum(Ng)        # recall:    true_positive/true
        OF1 = (2 * OP * OR) / (OP + OR)     # F1_score: harmonic mean of precision and recall
        # average by class
        CP = np.sum(Nc / Np) / n_class      # precision: true_positive/positive
        CR = np.sum(Nc / Ng) / n_class      # recall:    true_positive/true
        CF1 = (2 * CP * CR) / (CP + CR)     # F1_score: harmonic mean of precision and recall
        self.cls_P, self.cls_R, self.cls_F1 = cls_P, cls_R, cls_F1

        return OP, OR, OF1, CP, CR, CF1,cls_P,cls_R,cls_F1,ture_Positive,true_negative,false_positive

    # major function. run badcase analyse
    def run_badcase_analyse(self):

        # loading model output results and labels
        with open(self.load_model_validate_output_path, 'rb') as f:
            print('loading from',self.load_model_validate_output_path)
            total_results = pickle.load(f)
        with open(self.load_labels_path,'rb') as f:
            print('loading from',self.load_labels_path)
            total_labels = pickle.load(f)
        print('np.shape(total_results)',np.shape(total_results))
        print('np.shape(total_labels)',np.shape(total_labels))

        # analyse on each class and store badcase
        OP,OR,OF1,CP,CR,CF1,cls_P,cls_R,cls_F1,ture_Positive,true_negative,false_positive=self.evaluation_and_badcase(scores_=total_results,targets_=total_labels)
        # print('class_Precison',cls_P)
        # print('class_recall',cls_R)
        # print('class F1 score',cls_F1)
        # print('ture_Positive',ture_Positive)
        # print('laebl true predict negative',true_negative)
        # print('label false predict positive',false_positive)
        print('All evaluate results:')
        print('OP: {OP:.4f}\t'
              'OR: {OR:.4f}\t'
              'OF1: {OF1:.4f}\t'
              'CP: {CP:.4f}\t'
              'CR: {CR:.4f}\t'
              'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))

        # generate each category dict {category_idx: [img_idx1,img_idx2 ...  ]}
        true_negative_num,true_positive_num,false_positive_num=0,0,0
        ture_negative_dict,true_positive_dict,false_positive_dict={},{},{}
        [img_num,class_num]=ture_Positive.shape
        self.img_num, self.class_num=img_num,class_num
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
        self.ture_negative_dict,self.true_positive_dict,self.false_positive_dict= ture_negative_dict,true_positive_dict,false_positive_dict

        # write badcase of each class file_name into dict
        if self.write_badcase_name_list_into_dict==True:
            self.write_badcase_list_into_dict()

        # write each class results into html
        if self.write_class_result_into_html==True:
            self.write_clases_result_into_html()

        # generate badcase area size histogram
        if self.generate_badcase_size_histogram==True:
            self.badcase_area_histogram()

        # mojor program end

        # all_evaluate_results={'OP':OP,'OR':OR,'OF1':OF1,'CP':CP,'CR':CR,'CF1':CF1,'cls_P':cls_P,'cls_R':cls_R,'cls_F1':cls_F1}
        # with open('checkpoint/coco/resnet101_on_coco/all_evaluate_results.pkl','wb') as f:
        #     print('writing checkpoint/coco/resnet101_on_coco/all_evaluate_results.pkl')
        #     pickle.dump(all_evaluate_results, f)

    # write badcase file_names into dict {category_id:[file_name1,file_name2 ... ]}
    def write_badcase_list_into_dict(self):
        # load coco image file_name list
        coco_list_path = os.path.join(self.load_validate_json_path)
        coco_img_list = json.load(open(coco_list_path, 'r'))
        # result list {category_id:[file_name1,file_name2 ... ]}
        true_positive_dict,ture_negative_name_dict,true_positive_name_dict,false_positive_name_dict={},{},{},{}
        for class_idx in range(self.class_num):
            ture_negative_name_dict[class_idx]=[]
            true_positive_name_dict[class_idx]=[]
            false_positive_name_dict[class_idx]=[]
        false_positive_dict,ture_negative_dict,true_positive_dict=self.false_positive_dict,self.ture_negative_dict,self.true_positive_dict
        for class_idx in range(len(false_positive_dict)):
            for object_idx in range(len(false_positive_dict[class_idx])):
                file_idx=false_positive_dict[class_idx][object_idx]
                item=coco_img_list[file_idx]
                false_positive_name_dict[class_idx].append(item['file_name'])
        print('false_positive_name_dict generate done...')
        for class_idx in range(len(ture_negative_dict)):
            for object_idx in range(len(ture_negative_dict[class_idx])):
                file_idx=ture_negative_dict[class_idx][object_idx]
                item=coco_img_list[file_idx]
                ture_negative_name_dict[class_idx].append(item['file_name'])
        print('ture_negative_name_dict generate done...')
        self.ture_negative_name_dict=ture_negative_name_dict
        self.false_positive_name_dict=false_positive_name_dict
        # write results into pkl
        if not os.path.exists(self.write_ture_negative_path):
            with open(self.write_ture_negative_path,'wb') as f:
                print('writing to',self.write_ture_negative_path)
                pickle.dump(ture_negative_name_dict, f)
        if not os.path.exists(self.write_false_positive_path):
            with open(self.write_false_positive_path,'wb') as f:
                print('writing to',self.write_false_positive_path)
                pickle.dump(false_positive_name_dict, f)

    # write each class results into html
    def write_clases_result_into_html(self):
        print('writing result into html')
        index=[]
        for class_idx in range(self.class_num):
            index.append('class'+str(class_idx))
        with open('sk_spectral_cluster/coco_names.pkl', 'rb') as f:
            print("loading coco_names.pkl")
            names = pickle.load(f)
        df=pd.DataFrame(index=index)
        weight_list,precision_list,recall_list,F1_list=[],[],[],[]
        # df['weight'],df['precision'],df['recall'],df['F1']=[],[],[],[]
        self.cls_P, self.cls_R, self.cls_F1,self.cls_weight
        for class_idx in range(self.class_num):
            weight_list.append(self.cls_weight[class_idx])
            precision_list.append(self.cls_P[class_idx])
            recall_list.append(self.cls_R[class_idx])
            F1_list.append(self.cls_F1[class_idx])
        df['names']=names
        df['weight']=weight_list
        df['precision']=precision_list
        df['recall']=recall_list
        df['F1']=F1_list
        HEADER = '''
            <html>
                <head>
                    <meta charset="UTF-8">
                </head>
                <body>
            '''
        FOOTER = '''
                </body>
            </html>
            '''
        # write into html
        with open(self.html_path, 'w') as f:
            f.write(HEADER)
            f.write(df.to_html(classes='df'))
            f.write(FOOTER)

    # print badcase_area_histogram of each class
    def badcase_area_histogram(self):
        # make dir and load category names
        if not os.path.isdir(self.histo_img_dir):
            os.makedirs(self.histo_img_dir)
        with open('sk_spectral_cluster/coco_names.pkl', 'rb') as f:
            print("loading coco_names.pkl")
            names = pickle.load(f)
        self.names=names

        # loading json annotation
        with open(self.area_annotation_document) as f:
            print('loading:', self.area_annotation_document)
            instances_val = json.load(f)
        print('loading done.')
        annotations_list = instances_val['annotations']
        images_list = instances_val['images']
        coco_categories = instances_val['categories']

        # from predict idx to json category id list
        predict_idx_to_json_id = {}
        for idx in range(len(names)):
            predict_idx_to_json_id[idx] = coco_categories[idx]['id']

        # from image name find image id
        def from_image_name_find_id_and_URL(file_name, images_list):
            for image_idx in range(len(images_list)):
                # if (image_idx%10000==0):
                #     print('from image_name find id:',image_idx,'/',len(images_list))
                if file_name == images_list[image_idx]['file_name']:
                    img_id = images_list[image_idx]['id']
                    coco_url=images_list[image_idx]['coco_url']
                    break
            return img_id,coco_url

        # from image names
        def from_id_and_class_find_area(img_id, class_id, annotations_list):
            for anno_idx in range(len(annotations_list)):
                # if (anno_idx%100000==0):
                #     print('from annotation find area:',anno_idx,'/',len(annotations_list))
                if (annotations_list[anno_idx]['image_id'] == img_id):
                    area = annotations_list[anno_idx]['area']
                    if(annotations_list[anno_idx]['category_id'] == predict_idx_to_json_id[class_id]):
                        # print('match')
                        area = annotations_list[anno_idx]['area']
                        break
            return area

        # generate label true predict negative area list dict
        ltrue_pnegative_catagory_area_dict={}
        coco_badcase_img_url_dict={}
        for category_idx in range(self.class_num):
            ltrue_pnegative_catagory_area_dict[category_idx] = []
            coco_badcase_img_url_dict[category_idx]=[]
            for idx in range(len(self.ture_negative_name_dict[category_idx])):
                if idx%300==0:
                    print('category:',category_idx,'finding area:',idx,'/',len(self.ture_negative_name_dict[category_idx]))
                img_id,coco_url=from_image_name_find_id_and_URL(file_name=self.ture_negative_name_dict[category_idx][idx], images_list=images_list)
                ltrue_pnegative_catagory_area_dict[category_idx].append(from_id_and_class_find_area(img_id=img_id, class_id=category_idx, annotations_list=annotations_list))
                coco_badcase_img_url_dict[category_idx].append(coco_url)

        # save coco_url into dict and pkl
        if not os.path.exists(self.badcase_coco_url_path):
            with open(self.badcase_coco_url_path, 'wb') as f:
                print('writing to', self.badcase_coco_url_path)
                pickle.dump(coco_badcase_img_url_dict, f)


        # generate true positive area list dict
        true_positive_category_area_dict={}
        # for category_idx in range(self.class_num):
        #     true_positive_category_area_dict=[]
        #     for idx in range(len(self.))


        # from category area dict generate histogram and write into dir
        def write_dict_into_histogram(img_prefix,catagory_area_dict):
            # generate and save histograms of each category
            for category_idx in range(len(catagory_area_dict)):
                print(catagory_area_dict[category_idx])
                # print('writing into histogram,category',category_idx,self.names[category_idx])
                # generate histogram
                # plt.hist(catagory_area_dict[category_idx], bins=512, normed=0, facecolor='black', edgecolor='black',
                #          alpha=1, histtype='bar')
                plt.hist(catagory_area_dict[category_idx], bins=512,normed=0, facecolor='black', edgecolor='black',
                         alpha=1, histtype='bar')
                # histogram info
                plt.legend()
                plt.xlabel('badcase size')
                plt.ylabel('badcase numbers')
                # image name
                plt.title(img_prefix + ' category ' + str(category_idx) + ' badcase histogram')
                # save image
                img_filename = self.histo_img_dir + img_prefix + '_category_' + str(category_idx) +self.names[category_idx]+ '_histgoram.jpg'
                plt.savefig(img_filename)
                plt.close('all')

        # write true_positive histogram

        # write label true predict false histogram
        write_dict_into_histogram(img_prefix='label ture predict negative',catagory_area_dict=ltrue_pnegative_catagory_area_dict)



    # useless code, try histograms
    def hist_try(self):
        # make dir
        if not os.path.isdir(self.histo_img_dir):
            os.makedirs(self.histo_img_dir)

        # list histogram
        area_list=[235939.0035, 64153.92420000001, 797.3219000000005, 40377.498349999994, 825.0993000000005, 122283.17504999999, 110586.77519999997, 23235.967249999994, 7660.271999999997, 2845.3616000000015, 5078.0296, 39.01750000000018, 13050.5062, 1644.8198499999994, 9904.229450000003, 2675.0394999999994, 201191.88094999996, 41419.92445, 1120.6729999999989, 27064.562149999998, 6759.543450000003, 1151.2815999999998, 5645.972500000001, 70239.99469999998, 2468.0781500000003, 74.00769999999972, 274.57424999999995, 4085.4411499999997, 54892.3609, 20344.5, 4100.955550000003, 306.8878499999999, 598.1379500000002, 98200.96145, 1548.7608999999993, 48371.46090000002, 93.5650500000001, 996.3833999999999, 5999.726849999999, 65061.527849999984, 1343.6522999999995, 2444.3842, 20702.087949999997, 41612.6938, 8716.784449999997, 35314.11059999999, 1057.09515, 679.1063999999998, 32340.461000000007, 107332.58170000001, 564.0283000000005, 308.6485500000002, 6411.314050000004, 16892.101950000004, 2804.765700000001, 4544.119050000003, 848.3638999999996, 22555.577249999988, 704.9074499999999, 2714.924900000001, 400.17665000000056, 3047.8062999999997, 672.6902500000001, 1154.583900000002, 159.42344999999992, 2678.627699999999, 53877.39709999998, 6252.4588, 13623.237050000003, 1205.1451499999996, 3420.360850000001, 64.22899999999954, 1196.9678999999999, 5284.887600000001, 21816.5393, 40446.13375, 1009.3892000000008, 758.1965500000001, 13942.263000000003, 26605.372250000004, 1544.6851499999987, 11569.050200000003, 27346.60955, 7509.192800000006, 7443.993800000002, 187.2916000000001, 430.38455000000033, 1731.0626999999993, 22365.95915, 6107.7645999999995, 4146.846700000001, 5192.632750000001, 177.79995000000005, 2192.3364500000007, 30382.73005, 5084.077300000002, 24334.4352, 13273.328150000001, 19199.403049999997, 258.82960000000026, 171.6086499999996, 2579.294149999999, 10643.136300000002, 5276.648400000001, 130754.4461, 25967.8492, 98401.47165, 5083.535600000001, 620.5996500000009, 200.7367, 109093.38190000002, 14135.86125, 3862.85095, 1893.2924500000001, 1645.1986999999992, 3865.672500000002, 967.6888999999992, 2119.0998500000005, 1230.5245000000007, 1788.7488000000008, 693.7150500000005, 1072.0571000000002, 8818.327500000001, 1024.2146999999995, 25633.0359, 23320.623299999996, 673.1012000000009, 2222.3002000000015, 61014.46195, 1798.2479000000003, 84.1216999999999, 12116.474599999998, 4210.0095999999985, 314.3817499999999, 447.9805499999999, 228.29715000000022, 857.4145500000011, 519.8693000000002, 1650.575849999999, 5387.092499999999, 3240.4637500000003, 277831.7656500001, 6361.4247, 221.1024500000001, 15152.461450000003, 74614.90974999998, 4723.702600000005, 1759.2794500000005, 7691.003450000003, 449.2458000000002, 4290.22515, 49.50310000000002, 2441.497700000003, 95.18614999999988, 4379.470049999998, 2092.353499999998, 331.8804999999998, 3466.876549999998, 661.835500000002, 120996.0934, 629.6973000000003, 10204.53555, 3189.4067999999993, 1626.1462500000007, 311.83700000000005, 6595.661949999999, 766.3864500000003, 5832.719049999997, 622.8069000000003, 7538.644450000001, 90435.91854999997, 691.8538499999994, 19856.873300000003, 535.360100000001, 626.5313499999994, 160407.01739999998, 1773.1797999999994, 23848.098950000003, 412.3286000000004, 9071.85025, 77140.08305, 28072.59055000001, 3928.6560000000018, 928.5198500000002, 150.85995000000014, 11993.837499999998, 1933.3202499999993, 10484.912499999999, 104.0705, 2030.9136499999997, 995.7477499999995, 12849.491750000001, 2338.2236, 87.6514499999999, 24800.063850000002, 25153.20225, 1055.6573000000005, 694.9957499999987, 17828.574000000004, 76.1506500000003, 2596.9522000000006, 21308.426099999993, 3866.2829500000007, 3592.980500000001, 125.80120000000004, 296.64504999999923, 147.27640000000002, 16730.887099999993, 894.561599999999, 9031.985350000003, 2691.973500000001, 266.99935000000016, 2960.777250000001, 6210.0092, 433.22290000000044, 59792.20055, 4647.302199999999, 4789.46715, 3166.24635, 982.3338500000001, 5318.79285, 1205.97105, 30287.507150000005, 2542.6319000000003, 18771.75479999999, 988.8504000000003, 12833.495099999998, 6200.59555, 362.6848000000004, 1644.2095499999996, 4357.1161, 1328.4182000000008, 4486.79725, 40020.7544, 97421.5079, 12820.688849999999, 2030.1669000000004, 2788.8766499999983, 6761.3549, 1066.2487, 24087.15335, 466.37290000000036, 16821.8598, 149.65445000000034, 1246.0752999999997, 2882.825749999999, 26858.9004, 528.0869999999999, 19324.892450000003, 15629.004099999998, 98778.0215, 519.7051000000002, 43175.931500000006, 515.3512000000003, 1439.2336, 7132.1326, 1231.4716000000012, 486.8542000000008, 1815.2694999999983, 12663.202550000007, 59650.16855, 4806.4469500000005, 4913.544050000002, 8088.978949999999, 495.26579999999933, 11617.636250000003, 4081.1772499999997, 160451.48539999995, 16713.16865, 18227.059300000008, 26672.41565000001, 1043.1385999999998, 425.01955000000004, 130230.89709999993, 628.2395999999999, 22252.702399999995, 484.91110000000015, 210.3083500000005, 26.957550000000033, 5888.0771, 2141.0891, 220863.50235000002, 5737.141799999999, 3993.392899999999, 1562.9962500000004, 49597.06825, 128177.13039999997, 28657.39415000001, 1132.2413499999998, 26731.31054999999, 1059.2594, 4745.592499999999, 1920.1159499999983, 127.58095000000012, 12190.44565, 555.7345500000004, 9237.729949999997, 46.56134999999999, 236.1499999999998, 4298.163950000001, 216726.06719999996, 62.61789999999998, 2463.3468, 58960.8349, 5199.69665, 9304.32905, 9313.074099999998, 255.92869999999982, 69.30369999999995, 6764.494600000001, 6947.59405, 9963.741200000004, 2287.71385, 109.46640000000002, 2293.3109999999997, 16593.583899999998, 24518.62785, 355.55969999999905, 20953.521099999998, 2855.7088999999996, 3858.7823000000003, 8844.460499999997]
        plt.hist(area_list, bins=512, normed=0, facecolor='black', edgecolor='black', alpha=1, histtype = 'bar')
        # histogram info
        plt.xlabel('badcase size')
        plt.ylabel('badcase numbers')
        plt.title('class '+str(1)+'badcase histogram')
        # save histogram
        img_name=self.histo_img_dir+'class1.jpg'
        plt.savefig(img_name)

        # list histogram
        # plt.hist(area_list, bins=10, normed=0, facecolor='black', edgecolor='black', alpha=1, histtype = 'bar')
        plt.hist(area_list, normed=0, facecolor='black', edgecolor='black', alpha=1, histtype='bar')
        # histogram info
        plt.legend()
        plt.xlabel('badcase size')
        plt.ylabel('badcase numbers')
        plt.title('class '+str(2)+'badcase histogram')
        # save histogram
        img_name=self.histo_img_dir+'class2.jpg'
        plt.savefig(img_name)

    # useless code,try relation between coco id and names idx
    def coco_categories_names_id(self):

        # load json names id
        with open(self.area_annotation_document) as f:
            print('loading:',self.area_annotation_document)
            instances_val = json.load(f)
        print('loading done.')
        coco_categories=instances_val['categories']

        # load names id
        with open('sk_spectral_cluster/coco_names.pkl', 'rb') as f:
            print("loading coco_names.pkl")
            names = pickle.load(f)

        predict_idx_to_json_id={}
        for idx in range(len(names)):
            # print(idx,'names:', names[idx])
            # print(idx,'coco_json:','id',coco_categories[idx]['id'],'names',coco_categories[idx]['name'])
            predict_idx_to_json_id[idx]=coco_categories[idx]['id']
        # for idx in range(80,90,1):
        #     print(idx, 'coco_json:', 'id', coco_categories[idx]['id'], 'names', coco_categories[idx]['name'])


        # print('coco json categories id',coco_categories)



if __name__ == '__main__':
    badcase_analyse().run_badcase_analyse()
    # badcase_analyse().badcase_area_histogram()
    # badcase_analyse().hist_try()
    # badcase_analyse().coco_categories_names()




