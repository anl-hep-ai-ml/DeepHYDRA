import os
import time
import warnings
import json
from collections import defaultdict
from functools import partial, partialmethod

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
#from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis
#from torchinfo import summary
#from bigtree import Node, tree_to_dataframe, tree_to_dot
#from bigtree.tree.search import find_child_by_name
from tqdm import tqdm #用来显示进度

from dataset_loaders.omni_anomaly_dataset import OmniAnomalyDataset
from dataset_loaders.hlt_datasets import HLTDataset
from dataset_loaders.eclipse_datasets import EclipseDataset
from exp.exp_basic import ExpBasic
from models.model import Informer
from models.sad_like_loss import *
from utils.tools import EarlyStopping, adjust_learning_rate
# from utils.torch_profiling import add_sub_mul_div_op_handler,\
#                                                 sum_op_handler,\
#                                                 mean_op_handler,\
#                                                 cumsum_op_handler
# from utils.fvcorewriter import FVCoreWriter
# from utils.torchinfowriter import TorchinfoWriter

#from torch_profiling_utils.torchinfowriter import TorchinfoWriter
#from torch_profiling_utils.fvcorewriter import FVCoreWriter


def log_gradients_in_model(model, summary_writer, step):
    for tag, value in model.named_parameters():
        if value.grad is not None:
            summary_writer.add_histogram(tag, value.cpu(), step)
            summary_writer.add_histogram(tag + "/grad", value.grad.cpu(), step)
#以上的log会写入叫runs的文件夹,在跑这个程序的文件夹下,生成的图横坐标是global step,纵坐标是loss
#终端输入: tensorboard --logdir=/lcrc/group/ATLAS/users/jj/DiHydra/analysis_scripts/runs 才能看以上histogram


class ExpInformer(ExpBasic):
    def __init__(self, args):
        super(ExpInformer, self).__init__(args)

    def _build_model(self):
        model = Informer(
            self.args.enc_in,
            self.args.dec_in, 
            self.args.c_out, 
            self.args.seq_len, 
            self.args.label_len,
            self.args.pred_len, 
            self.args.factor,
            self.args.d_model, 
            self.args.n_heads, 
            self.args.e_layers,
            self.args.d_layers, 
            self.args.d_ff,
            self.args.dropout, 
            self.args.attn,
            self.args.embed,
            self.args.freq,
            self.args.activation,
            self.args.output_attention,
            self.args.no_distil,
            self.args.no_mix,
            self.device).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model #而且它继承了torch.nn.Module的各种功能,比如model.train和model.eval

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'machine-1-1': OmniAnomalyDataset,
            'HLT_DCM_2018': HLTDataset,
            'HLT_PPD_2018': HLTDataset,
            'HLT_DCM_2022': HLTDataset,
            'HLT_PPD_2022': HLTDataset,
            'HLT_DCM_2023': HLTDataset,
            'HLT_PPD_2023': HLTDataset,
            'ECLIPSE_MEAN': EclipseDataset,
            'ECLIPSE_MEDIAN': EclipseDataset,}

        Data = data_dict[self.args.data]

        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size

        elif flag=='pred':
            shuffle_flag = False
            drop_last = False
            batch_size = 1

        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
        
        freq = args.freq

        dataset = None

        if Data == OmniAnomalyDataset:
            dataset = Data(dataset=self.args.data,
                            mode=flag,
                            size=[args.seq_len,
                                    args.label_len,
                                    args.pred_len],
                            features=args.features,
                            target=args.target,
                            inverse=args.inverse,
                            timeenc=timeenc,
                            freq=freq,
                            scaling_type='minmax',
                            scaling_source='train_set_fit')

        elif Data == HLTDataset:

            source = self.args.data.split('_')[1].lower()
            variant = int(self.args.data.split('_')[-1])

            dataset = Data(source=source,
                            variant=variant,
                            mode=flag,
                            size=[args.seq_len,
                                    args.label_len,
                                    args.pred_len],
                            features=args.features,
                            target=args.target,
                            inverse=args.inverse,
                            timeenc=timeenc,
                            freq=freq,
                            scaling_type='minmax',
                            scaling_source='train_set_fit',
                            applied_augmentations=\
                                    self.args.augmentations,
                            augmented_dataset_size_relative=\
                                    self.args.augmented_dataset_size_relative,
                            augmented_data_ratio=\
                                    self.args.augmented_data_ratio)

        elif Data == EclipseDataset:

            variant = self.args.data.split('_')[-1].lower()

            dataset = Data(variant=variant,
                            mode=flag,
                            size=[args.seq_len,
                                    args.label_len,
                                    args.pred_len],
                            features=args.features,
                            target=args.target,
                            inverse=args.inverse,
                            timeenc=timeenc,
                            freq=freq,
                            scaling_type='minmax',
                            scaling_source='train_set_fit',
                            applied_augmentations=\
                                    self.args.augmentations,
                            augmented_dataset_size_relative=\
                                    self.args.augmented_dataset_size_relative,
                            augmented_data_ratio=\
                                    self.args.augmented_data_ratio)

        data_loader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=shuffle_flag,
                                    num_workers=args.num_workers,
                                    drop_last=drop_last)#专门用来控制是否保留最后一个batch的样本,若最后一个batch的样本数量少于设置的batch_size,则会根据 drop_last 的值决定是否丢弃。

        return dataset, data_loader 
#为什么get_data函数分别返回的是dataset,data_loader,她们的作用分别是什么?
#dataset负责提供原始数据和定义数据的存取逻辑,data_loader是对dataset的封装,主要用于训练时按批次加载数据,处理效率更高

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim   


    def _select_criterion(self, loss='MSE'):

        if loss == 'MSE':
            return nn.MSELoss()

        elif loss == 'SMSE':
            return SADLikeLoss(eta=0.1)


    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for batch_x,batch_y,batch_x_mark,batch_y_mark in tqdm(vali_loader):
            if self.args.output_attention:
                pred, true, _ = self._process_one_batch(vali_data,
                                                            batch_x,
                                                            batch_y,
                                                            batch_x_mark,
                                                            batch_y_mark)
            else:
                pred, true = self._process_one_batch(vali_data,
                                                        batch_x,
                                                        batch_y,
                                                        batch_x_mark,
                                                        batch_y_mark)


            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss


    def train(self, setting):

        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_data.pickle_scaler(f'{path}/scaler.pkl')

        if self.args.loss == 'SMSE':
            labeled_train_data, labeled_train_loader =\
                            self._get_data(flag='labeled_train')

        train_steps_unlabeled = len(train_loader)

        train_steps_labeled = len(labeled_train_loader)\
                    if self.args.loss == 'SMSE' else 0

        delta = -1 if self.args.loss == 'SMSE' else 0

        early_stopping = EarlyStopping(patience=self.args.patience,
                                                        verbose=True,
                                                        delta=delta)
        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()#需要用GPU

        summary_writer = SummaryWriter()

        for epoch in range(self.args.train_epochs):

            train_loss = []
            preds_all = []
            y_actual_all = []
            
            self.model.train()
            
            epoch_time = time.time() #获取当前时间
            
            if self.args.loss == 'SMSE':
                for batch_index, (batch_x,\
                                    batch_y,\
                                    batch_x_mark,\
                                    batch_y_mark) in enumerate(tqdm(train_loader)):
                    
                    model_optim.zero_grad()

                    if self.args.output_attention:
                        pred, true, _ = self._process_one_batch(train_data,
                                                                    batch_x,
                                                                    batch_y,
                                                                    batch_x_mark,
                                                                    batch_y_mark)

                    else:
                        pred, true = self._process_one_batch(train_data,
                                                                batch_x,
                                                                batch_y,
                                                                batch_x_mark,
                                                                batch_y_mark)

                    preds_all.append(pred.detach().cpu().numpy())
                    y_actual_all.append(true.detach().cpu().numpy())

                    loss = criterion(pred, true)

                    train_loss.append(loss.item())
                     
                    summary_writer.add_scalar("Train loss",
                                                loss,
                                                batch_index +\
                                                    epoch*train_steps_unlabeled)

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

                    # Stop training in semi-supervised setting
                    # when an amount of data equal to the
                    # unlabelled training set length minus the
                    # labelled training set length has been reached
                    # to ensure that the models receive the same
                    # amount of data
                    #反正为了保证标签数据训练不要超过没有标签的数量,这里和下面的for循环是SMSE和MSE train的逻辑唯一不同的地方
                    if batch_index >= (len(train_loader) -\
                                        len(labeled_train_loader)):
                        break

                for batch_index, (batch_x,\
                                    batch_y,\
                                    batch_x_mark,\
                                    batch_y_mark,\
                                    label) in enumerate(tqdm(labeled_train_loader)):

                    label = label.to(self.device)
                    
                    model_optim.zero_grad()

                    if self.args.output_attention:
                        pred, true, _ = self._process_one_batch(train_data,
                                                                    batch_x,
                                                                    batch_y,
                                                                    batch_x_mark,
                                                                    batch_y_mark) 

                    else:
                        pred, true = self._process_one_batch(train_data,
                                                                batch_x,
                                                                batch_y,
                                                                batch_x_mark,
                                                                batch_y_mark)

                    preds_all.append(pred.detach().cpu().numpy())
                    y_actual_all.append(true.detach().cpu().numpy())

                    loss = criterion(pred, true, label)

                    train_loss.append(loss.item())

                    summary_writer.add_scalar("Train loss",
                                                loss, batch_index +\
                                                    train_steps_unlabeled -\
                                                    train_steps_labeled +\
                                                    epoch*train_steps_unlabeled)
                    
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

            else:#这里开始是MSE的train

                for batch_index, (batch_x,\
                                    batch_y,\
                                    batch_x_mark,\
                                    batch_y_mark) in enumerate(tqdm(train_loader)): #batch_index从enumerate来的
                    
                    model_optim.zero_grad()

                    if self.args.output_attention:

                        #可以在jupyter上打印一下内容,seq_len就是输入长度,label_len就是seq_len的其中一部分,用与提供预测的,pre_len就是预测
                        #在训练时,模型需要通过label_len部分帮助学习如何从历史输入seq_x过渡到未来的预测pred_len
                        #batch_x.shape = (batch_size, seq_len, feature_dim), 其中seq_len是时间步数,属于user定义的参数,(在informer.py定义的)
                        #batch_y.shape = (batch_size, label_len+pred_len, feature_dim), 其中label_len和pred_len属于传参,定义在在/lcrc/group/ATLAS/users/jj/DiHydra/transformer_based_detection/informers/dataset_loaders/hlt_datasets.py
                        #print("batch_y.shape is",batch_x.shape)  #batch_x.shape输出torch.Size([128, 16, 146])
                        #print("batch_y.shape is",batch_y.shape)  #batch_y.shape输出torch.Size([128, 9, 146])#146是输入的特征维度
                        #print("batch_x_mark.shape is",batch_x_mark.shape)#batch_x_mark.shape输出torch.Size([128, 16, 6])#6是时间编码的特征维度
                        #print("batch_y_mark.shape is",batch_y_mark.shape)#
                        pred, true, _ = self._process_one_batch(train_data,#在epoch的for循环里,逐批次batch加载和处理数据
                                                                    batch_x,
                                                                    batch_y,
                                                                    batch_x_mark,
                                                                    batch_y_mark)
                        #print("pred.shape is", pred.shape)
                    else:
                        pred, true = self._process_one_batch(train_data,
                                                                batch_x,
                                                                batch_y,
                                                                batch_x_mark,
                                                                batch_y_mark)

                        
                    preds_all.append(pred.detach().cpu().numpy())
                    y_actual_all.append(true.detach().cpu().numpy())

                    loss = criterion(pred, true)

                    train_loss.append(loss.item())
                    
                    summary_writer.add_scalar("Train loss",
                                                loss,
                                                batch_index + epoch*\
                                                    train_steps_unlabeled)
                    #global_step =batch_index + epoch * train_steps_unlabeled
                    #train_steps = total_data / batch 
                    #处理一个batch就等于完成一个 train step
                    #模型用了整个数据集训练了一遍,就完成一个 epoch

                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))    

            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_data, #在通过epoch循环里,结束了batch的循环后,就validate
                                    vali_loader,
                                    criterion)

            preds_all = early_stopping(vali_loss,#early_stop用的是vali_loss而不是train_loss
                                        self.model,
                                        preds_all,
                                        path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

            summary_writer.add_scalar("Validation loss", vali_loss, epoch)

            log_gradients_in_model(self.model,
                                    summary_writer,
                                    epoch)

        preds_all_np = np.array(preds_all)
        y_actual_all = np.array(y_actual_all)

        preds_all_np = preds_all_np.reshape(-1, preds_all_np.shape[-2], preds_all_np.shape[-1])
        y_actual_all = y_actual_all.reshape(-1, y_actual_all.shape[-2], y_actual_all.shape[-1])

        # Save results

        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'preds_all_train.npy', preds_all_np)
        np.save(folder_path + 'true_values_all_train.npy', y_actual_all)

        best_model_path = path +\
                '/checkpoint_informer.pth'

        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    def test(self, setting):

        # tqdm.__init__ = partialmethod(tqdm.__init__, disable=True) #这个决定要不要显示进度,注释掉就是要显示

        test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        best_model_path = path + '/checkpoint_informer.pth'
        self.model.load_state_dict(torch.load(best_model_path))#用torch加载已经训练好的模型
        #打印加载模型的所有参数
        #for param_tensor in model.state_dict():
            #print(f"Parameter name: {param_tensor}")
            #print(f"Parameter shape: {model.state_dict()[param_tensor].shape}")
            #print(f"Parameter values: {model.state_dict()[param_tensor]}\n")

        self.model.eval()
        
        preds_all = []
        y_actual_all = []
        
        with tqdm(total=len(test_loader)) as pbar:
            for count, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):

                if self.args.output_attention:
                
                    pred, true, attention =\
                            self._process_one_batch(test_data,
                                                        batch_x,
                                                        batch_y,
                                                        batch_x_mark,
                                                        batch_y_mark,
                                                        batch_x)

                else:
                    pred, true = self._process_one_batch(test_data,
                                                            batch_x,
                                                            batch_y,
                                                            batch_x_mark,
                                                            batch_y_mark)

                
                preds_all.append(pred.detach().cpu().numpy())
                y_actual_all.append(true.detach().cpu().numpy())

                pbar.update(1)#更新tqdm的进度条用的

        preds_all = np.array(preds_all)
        y_actual_all = np.array(y_actual_all)

        preds_all = preds_all.reshape(-1, preds_all.shape[-2], preds_all.shape[-1])
        #preds_all的shape[num_batches,batch_size,sequence_length,feature_dimension]变成
        #[num_batches*batch_size,sequence_length,feature_dimension]或者说第一个元素维度变成sample_num,由4维变成3维了
        #preds_all.shape[-2], preds_all.shape[-1]指的是保留最后一个和倒数第二个维度,其他都合并起来
        y_actual_all = y_actual_all.reshape(-1, y_actual_all.shape[-2], y_actual_all.shape[-1])
        
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'preds_all_test.npy', preds_all)
        np.save(folder_path + 'true_values_all_test.npy', y_actual_all)
        np.save(folder_path + 'labels_all_test.npy', test_data.get_labels())


    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, viz_data=None):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # Decoder input
        
        #为什么调用一次_process_one_batch,都要重置decoder,但是encoder不用重置呢？？
        if self.args.padding == 0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
           #解码器的初始化的tensor形状:[batch_size, pred_len, num_feature],算是占位符,用于后面训练后的填充

        elif self.args.padding == 1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()

        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        #将真实的历史数据和预测的数据拼接起来,tensor形状:[batch_size, label_len+pred_len, num_feature]

        # Encoder - decoder

        # FLOP and activation count retrieval

        # if not 'ECLIPSE' in self.args.data:
        #     output_dir = self.args.data.lower()
        # else:
        #     output_dir = 'eclipse'

        # output_filename = f'informer_{self.args.data.lower()}_'\
        #                         f'{self.args.loss.lower()}_'\
        #                         f'sl_{self.args.seq_len}_'\
        #                         f'll_{self.args.label_len}_'\
        #                         f'pl_{self.args.pred_len}_'\
        #                         f'ein_{self.args.enc_in}_'\
        #                         f'din_{self.args.dec_in}_'\
        #                         f'cout_{self.args.c_out}_'\
        #                         f'dm_{self.args.d_model}_'\
        #                         f'nh_{self.args.n_heads}_'\
        #                         f'el_{self.args.e_layers}_'\
        #                         f'dl_{self.args.d_layers}_'\
        #                         f'dff_{self.args.d_ff}_'\
        #                         f'f_{self.args.factor}_'\
        #                         f'attn_{self.args.attn}_'\
        #                         f'emb_{self.args.embed}_'\
        #                         f'act_{self.args.activation.lower()}'

        # fvcore_writer = FVCoreWriter(self.model, (batch_x,
        #                                             batch_x_mark,
        #                                             dec_inp,
        #                                             batch_y_mark))

        # # print(fvcore_writer.get_flop_dict('by_module'))
        # # print(fvcore_writer.get_flop_dict('by_operator'))
        # # print(fvcore_writer.get_activation_dict('by_module'))
        # # print(fvcore_writer.get_activation_dict('by_operator'))

        # fvcore_writer.write_flops_to_json('../../evaluation/computational_intensity_analysis/'
        #                                         f'data/{output_dir}/by_module/{output_filename}.json',
        #                                     'by_module')

        # fvcore_writer.write_flops_to_json('../../evaluation/computational_intensity_analysis/'
        #                                         f'data/{output_dir}/by_operator/{output_filename}.json',
        #                                     'by_operator')

        # fvcore_writer.write_activations_to_json('../../evaluation/activation_analysis/'
        #                                                 f'data/{output_dir}/by_module/{output_filename}.json',
        #                                             'by_module')

        # fvcore_writer.write_activations_to_json('../../evaluation/activation_analysis/'
        #                                                 f'data/{output_dir}/by_operator/{output_filename}.json',
        #                                             'by_operator')

        # torchinfo_writer = TorchinfoWriter(self.model,
        #                                     input_data=(batch_x,
        #                                                     batch_x_mark,
        #                                                     dec_inp,
        #                                                     batch_y_mark),
        #                                     verbose=0)

        # torchinfo_writer.construct_model_tree()

        # torchinfo_writer.show_model_tree(attr_list=['Parameters', 'MACs'])

        # torchinfo_writer.get_dataframe().to_pickle(
        #     f'../../evaluation/parameter_analysis/data/{output_dir}/{output_filename}.pkl')

        # exit()

        if self.args.use_amp:
            with torch.cuda.amp.autocast():

                if self.args.output_attention:
                    outputs, attention = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:

            if self.args.output_attention:
                outputs, attention = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, viz_data=viz_data)

            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)

        f_dim = -1 if self.args.features == 'MS' else 0
        #f_dim = 0:取第一个特征, f_dim = -1：取最后一个特征（如果只有一个特质,那么最后一个特证同时也是整个特征)
        #所以如果input是(32, 100, 1)（32个样本,每个100 时间步,3个特征),即使features == 'MS'或 ‘M',最后预测的都是(32, 100, 1)
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
        

        if self.args.output_attention:
            return outputs, batch_y, attention 
            #通过输出attention分数,了解模型在哪些时间步或特征上花费了更多“注意力”,
            #可以根据 Attention 可视化结果，可以调整模型的训练数据、超参数或结构
        else:
            return outputs, batch_y