"""
created by Xing xiangrui on 2019.5.14

"""


import os
import shutil
import time
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchnet as tnt
import torchvision.transforms as transforms
import torch.nn as nn
from util import *
import pickle
import numpy as np

tqdm.monitor_interval = 0


class Engine(object):
    def __init__(self, state={}):
        self.state = state
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('image_size') is None:
            self.state['image_size'] = 224

        if self._state('batch_size') is None:
            self.state['batch_size'] = 64

        if self._state('workers') is None:
            self.state['workers'] = 25

        if self._state('device_ids') is None:
            self.state['device_ids'] = None

        if self._state('evaluate') is None:
            self.state['evaluate'] = False

        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 90

        if self._state('epoch_step') is None:
            self.state['epoch_step'] = []

        # meters
        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        # time measure
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()
        # display parameters
        if self._state('use_pb') is None:
            self.state['use_pb'] = True
        if self._state('print_freq') is None:
            self.state['print_freq'] = 0

    def _state(self, name):
        if name in self.state:
            return self.state[name]

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        print('on start epoch reset meter_loss and batch_time and data_time')
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        print('On end epoch...')
        loss = self.state['meter_loss'].value()[0]
        print("self.state['meter_loss']:", self.state['meter_loss'])
        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}'.format(self.state['epoch'], loss=loss))
            else:
                print('Test: \t Loss {loss:.4f}'.format(loss=loss))
        return loss

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        pass

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        # record loss
        self.state['loss_batch'] = self.state['loss'].data[0]
        self.state['meter_loss'].add(self.state['loss_batch'])

        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))

    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        print('on_forward...')
        input_var = torch.autograd.Variable(self.state['input'])
        target_var = torch.autograd.Variable(self.state['target'])

        if not training:
            input_var.volatile = True
            target_var.volatile = True

        # compute output
        self.state['output'] = model(input_var)
        print('output is:',self.state['output'],'label is:',target_var)
        self.state['loss'] = criterion(self.state['output'], target_var)

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            optimizer.step()

    def init_learning(self, model, criterion):

        if self._state('train_transform') is None:
            #normalize = transforms.Normalize(mean=model.image_normalization_mean,
            #                                 std=model.image_normalization_std)
            #fixme resnet have no image_normalization_mean function so we use list to replace
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225] )
            self.state['train_transform'] = transforms.Compose([
                transforms.Resize((512, 512)),
                MultiScaleCrop(self.state['image_size'], scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
            print(self.state['train_transform'])

        if self._state('val_transform') is None:
            #normalize = transforms.Normalize(mean=model.image_normalization_mean,
            #                                 std=model.image_normalization_std)
            #fixme resnet have no image_normalization_mean function so we use list to replace
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.state['val_transform'] = transforms.Compose([
                Warp(self.state['image_size']),
                transforms.ToTensor(),
                normalize,
            ])

        self.state['best_score'] = 0

    def learning(self, model, criterion, train_dataset, val_dataset, optimizer=None):

        self.init_learning(model, criterion)

        # define train and val transform
        train_dataset.transform = self.state['train_transform']
        train_dataset.target_transform = self._state('train_target_transform')
        val_dataset.transform = self.state['val_transform']
        val_dataset.target_transform = self._state('val_target_transform')

        # data loading code
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.state['batch_size'], shuffle=True,
                                                   num_workers=self.state['workers'], drop_last=True)  # fixme

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=self.state['workers'])

        # val_loader = torch.utils.data.DataLoader(val_dataset,
        #                                          batch_size=16, shuffle=False,
        #                                          num_workers=self.state['workers'])

        # optionally resume from a checkpoint
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'])
                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['evaluate'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))

        if self.state['use_gpu']:
            train_loader.pin_memory = True
            val_loader.pin_memory = True
            cudnn.benchmark = True

            model = torch.nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()
        if self.state['loss_type'] != 'DeepMarLoss':  # fixme
            criterion = criterion.cuda()

        if self.state['evaluate']:
            self.validate(val_loader, model, criterion)
            return

        # TODO define optimizer

        for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
            self.state['epoch'] = epoch
            # lr = self.adjust_learning_rate(optimizer) #fixme
            # print('lr:{:.5f}'.format(lr)) # fixme
            self.adjust_learning_rate(optimizer)  # fixme: not return lr for printing

            # train for one epoch
            self.train(train_loader, model, criterion, optimizer, epoch)
            # evaluate on validation set
            prec1 = self.validate(val_loader, model, criterion)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > self.state['best_score']
            self.state['best_score'] = max(prec1, self.state['best_score'])
            self.save_checkpoint({
                'epoch': epoch + 1,
                'arch': self._state('arch'),
                'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
                'best_score': self.state['best_score'],
            }, is_best)

            print(' *** best={best:.3f}'.format(best=self.state['best_score']))
        return self.state['best_score']

    def train(self, data_loader, model, criterion, optimizer, epoch):

        # switch to train mode
        model.train()

        self.on_start_epoch(True, model, criterion, data_loader, optimizer)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Training')

        end = time.time()
        for i, (input, target) in enumerate(data_loader):
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(True, model, criterion, data_loader, optimizer)

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(async=True)

            self.on_forward(True, model, criterion, data_loader, optimizer)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            self.on_end_batch(True, model, criterion, data_loader, optimizer)

        self.on_end_epoch(True, model, criterion, data_loader, optimizer)

    def validate(self, data_loader, model, criterion):

        # switch to evaluate mode
        print('start model eval,fix BN and dropOut...')
        model.eval()

        self.on_start_epoch(False, model, criterion, data_loader)

        if self.state['use_pb']:
            data_loader = tqdm(data_loader, desc='Test')

        end = time.time()
        all_output_results={}
        all_labels={}
        for i, (input, target) in enumerate(data_loader):
            if(i%100==0):
                print('epoch ',i)
            # measure data loading time
            self.state['iteration'] = i
            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            self.on_start_batch(False, model, criterion, data_loader) #pass

            if self.state['use_gpu']:
                self.state['target'] = self.state['target'].cuda(async=True)

            output_data_np,labels_np=self.on_forward_analyse(False, model, criterion, data_loader)
            # output_and_labels={'output_data_np':output_data_np,'labels_np':labels_np}

            all_output_results[i]=output_data_np
            all_labels[i]=labels_np
            # print('all_output_results',all_output_results)

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()
            # measure accuracy
            # self.on_end_batch(False, model, criterion, data_loader)
        # all validate results and labels on coco
        # print('all_output_results',all_output_results)
        # print('all_labels',all_labels)

        # concat all numpy
        total_results = all_output_results[0]
        total_labels = all_labels[0]
        for img_idx in range(len(all_output_results) - 1):
            if img_idx % 1000 == 0:
                print(img_idx, '/', len(all_output_results))
            total_results= np.append(total_results, all_output_results[img_idx + 1], axis=0)
            total_labels = np.append(total_labels, all_labels[img_idx + 1], axis=0)
        with open('checkpoint/coco/weight_decay_cls_gat_on_5_10/model_results_numpy.pkl', 'wb') as f:
            print("writing checkpoint/coco/weight_decay_cls_gat_on_5_10/model_results_numpy.pkl")
            pickle.dump(total_results, f)
        with open('checkpoint/coco/weight_decay_cls_gat_on_5_10/coco_labels_numpy.pkl', 'wb') as f:
            print("writing checkpoint/coco/weight_decay_cls_gat_on_5_10/oco_labels_numpy.pkl")
            pickle.dump(total_labels, f)
        # score = self.on_end_epoch(False, model, criterion, data_loader)

        # return score

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if self._state('save_model_path') is not None:
            filename_ = filename
            filename = os.path.join(self.state['save_model_path'], filename_)
            if not os.path.exists(self.state['save_model_path']):
                os.makedirs(self.state['save_model_path'])
        print('save model {filename}'.format(filename=filename))
        torch.save(state, filename)
        if is_best:
            filename_best = 'model_best.pth.tar'
            if self._state('save_model_path') is not None:
                filename_best = os.path.join(self.state['save_model_path'], filename_best)
            shutil.copyfile(filename, filename_best)
            if self._state('save_model_path') is not None:
                if self._state('filename_previous_best') is not None:
                    os.remove(self._state('filename_previous_best'))
                filename_best = os.path.join(self.state['save_model_path'],
                                             'model_best_{score:.4f}.pth.tar'.format(score=state['best_score']))
                shutil.copyfile(filename, filename_best)
                self.state['filename_previous_best'] = filename_best

    def adjust_learning_rate(self, optimizer):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        # lr = args.lr * (0.1 ** (epoch // 30))
        # decay = 0.1 ** (sum(self.state['epoch'] >= np.array(self.state['epoch_step'])))
        # decay = 0.1 ** (self.state['epoch'] // self.state['epoch_step'])
        # lr = self.state['lr'] * decay # fixme
        # for i, param_group in enumerate(optimizer.param_groups):
        #     if i == 0:
        #         # print(param_group.)
        #         decay = 0.1 ** (self.state['epoch'] // self.state['epoch_step'])
        #         # print(self.state['lrp'])
        #         param_group['lr'] = decay * self.state['lrp']  # fixme
        #         # param_group['lr'] = lr
        #
        #         print('backbone learning rate', param_group['lr'])
        #     if i == 1:
        #         decay = 0.1 ** (self.state['epoch'] // self.state['epoch_step'])
        #         # print(self.state['lr'])
        #         param_group['lr'] = decay * self.state['lr']  # fixme
        #         # param_group['lr'] = lr
        #         print('head learning rate', param_group['lr'])

        for i, param_group in enumerate(optimizer.param_groups):
            if i == 0:
                # print(param_group.)
                if self.state['epoch']<30:
                    param_group['lrp']=0.01
                else:
                    up=self.state['epoch']-30
                    decay = 0.9 ** up
                    # print(self.state['lrp'])
                    param_group['lrp'] = decay * 0.01  # fixme
                    # param_group['lr'] = lr
                print('backbone learning rate', param_group['lrp'])

            if i == 1:
                if self.state['epoch']<30:
                    param_group['lr']=0.01
                else:
                    up=self.state['epoch']-30
                    decay = 0.9 ** up
                    # print(self.state['lrp'])
                    param_group['lr'] = decay * 0.01  # fixme
                    # param_group['lr'] = lr
                print('head learning rate', param_group['lr'])

        # return lr

    #fixme validate model-----------------------------------------------------------
    def validate_model(self, model, criterion, train_dataset, val_dataset, optimizer=None):

        print('validate model...')

        self.init_learning(model, criterion)

        print('init_learning done...')

        # define train and val transform
        train_dataset.transform = self.state['train_transform']
        train_dataset.target_transform = self._state('train_target_transform')
        print('train dataset transform done...')

        val_dataset.transform = self.state['val_transform']
        val_dataset.target_transform = self._state('val_target_transform')
        print('val dataset transfrom done...')


        # data loading code
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.state['batch_size'], shuffle=True,
                                                   num_workers=self.state['workers'], drop_last=True)  # fixme
        print('train loader done...')
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False,
                                                 num_workers=self.state['workers'])
        print('val loader done')
        # val_loader = torch.utils.data.DataLoader(val_dataset,
        #                                          batch_size=16, shuffle=False,
        #                                          num_workers=self.state['workers'])

        # optionally resume from a checkpoint
        if self._state('resume') is not None:
            if os.path.isfile(self.state['resume']):
                print("=> loading checkpoint '{}'".format(self.state['resume']))
                checkpoint = torch.load(self.state['resume'])
                self.state['start_epoch'] = checkpoint['epoch']
                self.state['best_score'] = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(self.state['evaluate'], checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(self.state['resume']))
        print('checkpoint resume done...')

        if self.state['use_gpu']:
            train_loader.pin_memory = True
            val_loader.pin_memory = True
            cudnn.benchmark = True
            model = torch.nn.DataParallel(model, device_ids=self.state['device_ids']).cuda()
            print('use gpu done...')
        if self.state['loss_type'] != 'DeepMarLoss':  # fixme
            criterion = criterion.cuda()
        print('loss tpye done...')

        # if self.state['evaluate']:
        #     self.validate(val_loader, model, criterion)
        #     return

        # TODO define optimizer

        # for epoch in range(self.state['start_epoch'], self.state['max_epochs']):
        #     self.state['epoch'] = epoch
        #     # lr = self.adjust_learning_rate(optimizer) #fixme
        #     # print('lr:{:.5f}'.format(lr)) # fixme
        #     self.adjust_learning_rate(optimizer)  # fixme: not return lr for printing
        #
        #     # train for one epoch
        #     self.train(train_loader, model, criterion, optimizer, epoch)
        #     # evaluate on validation set
        #     prec1 = self.validate(val_loader, model, criterion)
        #
        #     # remember best prec@1 and save checkpoint
        #     is_best = prec1 > self.state['best_score']
        #     self.state['best_score'] = max(prec1, self.state['best_score'])
        #     self.save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': self._state('arch'),
        #         'state_dict': model.module.state_dict() if self.state['use_gpu'] else model.state_dict(),
        #         'best_score': self.state['best_score'],
        #     }, is_best)
        print('start validate...')
        # prec1 = self.validate(val_loader, model, criterion)
        self.validate(val_loader, model, criterion)

        # print('validate reulst prec is :',prec1)
        # return prec1


class MultiLabelMAPEngine(Engine):
    def __init__(self, state):
        Engine.__init__(self, state)
        if self._state('difficult_examples') is None:
            self.state['difficult_examples'] = False
        self.state['ap_meter'] = AveragePrecisionMeter(self.state['difficult_examples'])

    def on_start_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        Engine.on_start_epoch(self, training, model, criterion, data_loader, optimizer)
        self.state['ap_meter'].reset()

    def on_end_epoch(self, training, model, criterion, data_loader, optimizer=None, display=True):
        map = 100 * self.state['ap_meter'].value().mean()
        loss = self.state['meter_loss'].value()[0]
        OP, OR, OF1, CP, CR, CF1 = self.state['ap_meter'].overall()
        OP_k, OR_k, OF1_k, CP_k, CR_k, CF1_k = self.state['ap_meter'].overall_topk(3)
        if display:
            if training:
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}\t'
                      'mAP {map:.3f}'.format(self.state['epoch'], loss=loss, map=map))
                print('OP: {OP:.4f}\t'
                      'OR: {OR:.4f}\t'
                      'OF1: {OF1:.4f}\t'
                      'CP: {CP:.4f}\t'
                      'CR: {CR:.4f}\t'
                      'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
            else:
                print('Test: \t Loss {loss:.4f}\t mAP {map:.3f}'.format(loss=loss, map=map))
                print('OP: {OP:.4f}\t'
                      'OR: {OR:.4f}\t'
                      'OF1: {OF1:.4f}\t'
                      'CP: {CP:.4f}\t'
                      'CR: {CR:.4f}\t'
                      'CF1: {CF1:.4f}'.format(OP=OP, OR=OR, OF1=OF1, CP=CP, CR=CR, CF1=CF1))
                print('OP_3: {OP:.4f}\t'
                      'OR_3: {OR:.4f}\t'
                      'OF1_3: {OF1:.4f}\t'
                      'CP_3: {CP:.4f}\t'
                      'CR_3: {CR:.4f}\t'
                      'CF1_3: {CF1:.4f}'.format(OP=OP_k, OR=OR_k, OF1=OF1_k, CP=CP_k, CR=CR_k, CF1=CF1_k))

        return map

    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        self.state['target_gt'] = self.state['target'].clone()
        self.state['target'][self.state['target'] == 0] = 1
        self.state['target'][self.state['target'] == -1] = 0

        input = self.state['input']
        self.state['input'] = input[0]
        self.state['name'] = input[1]

    def on_end_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        Engine.on_end_batch(self, training, model, criterion, data_loader, optimizer, display=False)

        # measure mAP
        self.state['ap_meter'].add(self.state['output'].data, self.state['target_gt'])
        if display and self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()[0]
            batch_time = self.state['batch_time'].value()[0]
            data_time = self.state['data_time'].value()[0]
            if training:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            else:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))


class GCNMultiLabelMAPEngine(MultiLabelMAPEngine):
    def on_forward(self, training, model, criterion, data_loader, optimizer=None, display=True):
        # print('on_forward in GCNMultiLabelMAPEngine...')
        feature_var = torch.autograd.Variable(self.state['feature']).float()
        target_var = torch.autograd.Variable(self.state['target']).float()
        inp_var = torch.autograd.Variable(self.state['input']).float().detach()  # one hot
        if not training:
            feature_var.volatile = True
            target_var.volatile = True
            inp_var.volatile = True

        # compute output
        self.state['output'] = model(feature_var, inp_var)
        # fixme==========================================
        if self.state['loss_type'] == 'DeepMarLoss':
            weights = self.state['DeepMarLoss'].weighted_label(target_var)
            self.state['loss'] = criterion(self.state['output'], target_var,
                                           weight=torch.autograd.Variable(weights.cuda()))
            print('DeepMarLoss,model output:', self.state['output'], 'target_var:', target_var)
        else:
            self.state['loss'] = criterion(self.state['output'], target_var)
            print('model output:', self.state['output'], 'label:', target_var)
        # fixme=========================================================
        # self.state['loss'] = criterion(self.state['output'], target_var)

        if training:
            optimizer.zero_grad()
            self.state['loss'].backward()
            nn.utils.clip_grad_norm(model.parameters(), max_norm=10.0)
            optimizer.step()

    def on_forward_analyse(self, training, model, criterion, data_loader, optimizer=None, display=True):
        # forward analyse and write results in output_data_np
        # print('on_forward_analyse in GCNMultiLabelMAPEngine...')
        feature_var = torch.autograd.Variable(self.state['feature']).float()
        target_var = torch.autograd.Variable(self.state['target']).float()
        inp_var = torch.autograd.Variable(self.state['input']).float().detach()  # one hot
        if not training:
            feature_var.volatile = True
            target_var.volatile = True
            inp_var.volatile = True

        # compute output
        self.state['output'] = model(feature_var, inp_var)
        # fixme==========================================
        # loss calculate
        # if self.state['loss_type'] == 'DeepMarLoss':
        #     weights = self.state['DeepMarLoss'].weighted_label(target_var)
        #     self.state['loss'] = criterion(self.state['output'], target_var,
        #                                    weight=torch.autograd.Variable(weights.cuda()))
        #     # print('DeepMarLoss,model output:', self.state['output'], 'target_var:', target_var)
        # else:
        #     self.state['loss'] = criterion(self.state['output'], target_var)
        #     print('model output:', self.state['output'], 'label:', target_var)
        # print('loss calculate done...')

        # cpu_labels=inp_var.type(torch.FloatTensor)
        # This is a [torch.FloatTensor of size 1x80x300]  ??unknown reason

        # .data-----.cpu()------.numpy
        output_data_np=self.state['output'].cpu().data.numpy()
        labels_np=target_var.cpu().data.numpy()

        return output_data_np,labels_np



    def on_start_batch(self, training, model, criterion, data_loader, optimizer=None, display=True):

        self.state['target_gt'] = self.state['target'].clone()
        self.state['target'][self.state['target'] == 0] = 1
        self.state['target'][self.state['target'] == -1] = 0

        input = self.state['input']
        self.state['feature'] = input[0]
        self.state['out'] = input[1]
        self.state['input'] = input[2]
