
import os
import argparse
import random

import SingleTest
import data_process
# import train_process_debug as train_process
import train_process
import torch
import torch.nn.modules as nn
import dev_process
import test_process
import model
from transformers import BertTokenizer, RobertaTokenizer
import numpy as np
from util.write_file import WriteFile
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from model import ModelParam
from transformers import AutoTokenizer
from att_visualize import visualize_grid_attention_v2
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-run_type', type=int,
                       default=1, help='1: train, 2: debug train, 3: dev, 4: test')
    parse.add_argument('-save_model_path', type=str,
                       default='/hy-tmp/checkpoint', help='save the good model.pth path')
    parse.add_argument('-add_note', type=str, default='', help='Additional instructions when saving files')
    parse.add_argument('-gpu_num', type=str, default='0', help='gpu index')
    parse.add_argument('-gpu0_bsz', type=int, default=0,
                       help='the first GPU batch size')
    parse.add_argument('-epoch', type=int, default=40, help='train epoch num')
    parse.add_argument('-batch_size', type=int, default=16,
                       help='batch size number')
    parse.add_argument('-acc_grad', type=int, default=1, help='Number of steps to accumulate gradient on '
                                                              '(divide the batch_size and accumulate)')
    parse.add_argument('-lr', type=float, default=2e-5, help='learning rate')
    parse.add_argument('-fuse_lr', type=float, default=2e-5, help='learning rate')
    parse.add_argument('-min_lr', type=float,
                       default=1e-9, help='the minimum lr')
    parse.add_argument('-warmup_step_epoch', type=float,
                       default=2, help='warmup learning step')
    parse.add_argument('-num_workers', type=int, default=0,
                       help='loader dataset thread number')
    parse.add_argument('-l_dropout', type=float, default=0.1,
                       help='classify linear dropout')
    parse.add_argument('-train_log_file_name', type=str,
                       default='train_correct_log.txt', help='save some train log')
    parse.add_argument('-dis_ip', type=str, default='',
                       help='init_process_group ip and port')
    parse.add_argument('-optim_b1', type=float, default=0.9,
                       help='torch.optim.Adam betas_1')
    parse.add_argument('-optim_b2', type=float, default=0.999,
                       help='torch.optim.Adam betas_1')
    parse.add_argument('-data_path_name', type=str, default='10-flod-1',
                       help='train, dev and test data path name')
    parse.add_argument('-data_type', type=str, default='MVSA-single',
                       help='Train data type: MVSA-single and MVSA-multiple and HFM')
    parse.add_argument('-word_length', type=int,
                       default=200, help='the sentence\'s word length')
    parse.add_argument('-save_acc', type=float, default=-1, help='The default ACC threshold')
    parse.add_argument('-save_F1', type=float, default=-1, help='The default F1 threshold')
    parse.add_argument('-text_model', type=str, default='bert-base', help='language model')
    parse.add_argument('-loss_type', type=str, default='CE', help='Type of loss function')
    parse.add_argument('-optim', type=str, default='adam', help='Optimizer:adam, sgd, adamw')
    parse.add_argument('-activate_fun', type=str, default='gelu', help='Activation function')

    parse.add_argument('-image_model', type=str, default='resnet-50', help='Image model: resnet-18, resnet-34, resnet-50, resnet-101, resnet-152')
    parse.add_argument('-image_size', type=int, default=224, help='Image dim')
    parse.add_argument('-image_output_type', type=str, default='all',
                       help='"all" represents the overall features and regional features of the picture, and "CLS" represents the overall features of the picture')
    parse.add_argument('-text_length_dynamic', type=int, default=1, help='1: Dynamic length; 0: fixed length')
    parse.add_argument('-fuse_type', type=str, default='ave', help='att, ave, max')
    parse.add_argument('-tran_dim', type=int, default=768, help='Input dimension of text and picture encoded transformer')
    parse.add_argument('-tran_num_layers', type=int, default=4, help='4 The layer of transformer')
    parse.add_argument('-image_num_layers', type=int, default=2, help='2 The layer of image transformer')
    parse.add_argument('-train_fuse_model_epoch', type=int, default=0, help='20 The number of epoch of the model that only trains the fusion layer')
    parse.add_argument('-cl_loss_alpha', type=int, default=1, help='Weight of contrastive learning loss value')
    parse.add_argument('-cl_self_loss_alpha', type=int, default=1, help='Weight of contrastive learning loss value')
    parse.add_argument('-temperature', type=float, default=0.07, help='Temperature used to calculate contrastive learning loss')


    # 布尔类型的参数
    parse.add_argument('-cuda', action='store_true', default=True,
                       help='if True: use cuda. if False: use cpu')
    parse.add_argument('-fixed_image_model', action='store_true', default=False, help='是否固定图像模型的参数')

    opt = parse.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_num)
    opt.epoch = opt.train_fuse_model_epoch + opt.epoch

    dt = datetime.now()
    opt.save_model_path = opt.save_model_path + '/' + dt.strftime(
        '%Y-%m-%d-%H-%M-%S') + '-'
    if opt.add_note != '':
        opt.save_model_path += opt.add_note + '-'
    print('\n', opt.save_model_path, '\n')

    assert opt.batch_size % opt.acc_grad == 0
    opt.acc_batch_size = opt.batch_size // opt.acc_grad

    # CE（交叉熵），SCE（标签平滑的交叉熵），Focal（focal loss），Ghm（ghm loss）
    critertion = None
    if opt.loss_type == 'CE':
        critertion = nn.CrossEntropyLoss()

    cl_fuse_model = model.CLModel(opt)
    if opt.cuda is True:
        assert torch.cuda.is_available()
        if len(opt.gpu_num) > 1:
            if opt.gpu0_bsz > 0:
                cl_fuse_model = torch.nn.DataParallel(cl_fuse_model).cuda()
            else:
                print('multi-gpu')
                """
                单机多卡的运行方式，nproc_per_node表示使用的GPU的数量
                python -m torch.distributed.launch --nproc_per_node=2 main.py
                """
                print('当前GPU编号：', opt.local_rank)
                torch.cuda.set_device(opt.local_rank) # 在进行其他操作之前必须先设置这个
                torch.distributed.init_process_group(backend='nccl')
                cl_fuse_model = cl_fuse_model.cuda()
                cl_fuse_model = nn.parallel.DistributedDataParallel(cl_fuse_model, find_unused_parameters=True)
        else:
            cl_fuse_model = cl_fuse_model.cuda()
        critertion = critertion.cuda()

    print('Init Data Process:')
    tokenizer = None
    abl_path = ''
    if opt.text_model == 'bert-base':
        tokenizer = BertTokenizer.from_pretrained('/hy-tmp/pretrain/bert-base-uncased/vocab.txt')
        # tokenizer = AutoTokenizer.from_pretrained("/root/Sentiment Analysis/pretrain/bertweet-base-sentiment-analysis")
    if opt.data_type == 'HFM':
        data_path_root = abl_path + 'dataset/data/HFM/'
        train_data_path = data_path_root + 'train.json'
        dev_data_path = data_path_root + 'valid.json'
        test_data_path = data_path_root + 'test.json'
        # photo_path = data_path_root + '/dataset_image'
        photo_path = '/hy-tmp/dataset/data'
        image_coordinate = None
        data_translation_path = data_path_root + '/HFM.json'
    else:
        data_path_root = abl_path + 'dataset/data/' + opt.data_type + '/' + opt.data_path_name + '/'
        train_data_path = data_path_root + 'train.json'
        dev_data_path = data_path_root + 'dev.json'
        test_data_path = data_path_root + 'test.json'
        photo_path = abl_path + 'dataset/data/' + opt.data_type + '/dataset_image'
        image_coordinate = None
        data_translation_path = abl_path + 'dataset/data/' + opt.data_type + '/' + opt.data_type + '_translation.json'



    if opt.run_type == 1:

        train_loader, opt.train_data_len = data_process.data_process_train(opt, train_data_path, tokenizer, photo_path,
                                                                           data_type=1,
                                                                           data_translation_path=data_translation_path,
                                                                           image_coordinate=image_coordinate)

        opt.acc_batch_size = 1
        dev_loader, opt.dev_data_len = data_process.data_process(opt, dev_data_path, tokenizer, photo_path,
                                                                 data_type=2,
                                                                 data_translation_path=data_translation_path,
                                                                 image_coordinate=image_coordinate)
        test_loader, opt.test_data_len = data_process.data_process(opt, test_data_path, tokenizer, photo_path,
                                                                   data_type=3,
                                                                   data_translation_path=data_translation_path,
                                                                   image_coordinate=image_coordinate)
        opt.acc_batch_size = 16

        if opt.warmup_step_epoch > 0:
            opt.warmup_step = opt.warmup_step_epoch * len(train_loader)
            opt.warmup_num_lr = opt.lr / opt.warmup_step
        opt.scheduler_step_epoch = opt.epoch - opt.warmup_step_epoch

        if opt.scheduler_step_epoch > 0:
            opt.scheduler_step = opt.scheduler_step_epoch * len(train_loader)
            opt.scheduler_num_lr = opt.lr / opt.scheduler_step

        print(opt)
        opt.save_model_path = WriteFile(opt.save_model_path, 'train_correct_log.txt', str(opt) + '\n\n', 'a+',
                                        change_file_name=True)
        log_summary_writer = None
        log_summary_writer = SummaryWriter(log_dir=opt.save_model_path)
        log_summary_writer.add_text('Hyperparameter', str(opt), global_step=1)
        log_summary_writer.flush()

        print('\nTraining Begin')

        att = train_process.train_process(opt, train_loader, dev_loader, test_loader, cl_fuse_model, critertion, log_summary_writer)

        log_summary_writer.close()



    elif opt.run_type == 2:
        print('\nTest Begin')
        model_path = "/hy-tmp/checkpoint/2023-08-15-22-05-24-/08-16-00-35-09-Acc-0.69000.pth"
        cl_fuse_model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)

        # # 预训练权重
        # ckpt = torch.load(model_path)  # 加载预训练权重
        # model_dict = cl_fuse_model.state_dict()  # 得到我们模型的参数
        # # 判断预训练模型中网络的模块是否修改后的网络中也存在，并且shape相同，如果相同则取出
        # pretrained_dict = {k: v for k, v in ckpt.items() if k in model_dict and (v.shape == model_dict[k].shape)}
        # # 更新修改之后的 model_dict
        # model_dict.update(pretrained_dict)
        # # 加载我们真正需要的 state_dict
        # cl_fuse_model.load_state_dict(model_dict, strict=False)
    
        test_loader, opt.test_data_len = data_process.data_process(opt, test_data_path, tokenizer, photo_path,
                                                                   data_type=3,
                                                                   data_translation_path=data_translation_path,
                                                                   image_coordinate=image_coordinate)

        test_process.test_process(opt, critertion, cl_fuse_model, test_loader,0,0, epoch=1)

