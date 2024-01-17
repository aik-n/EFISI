"""
Name: test_process
Date: 2022/4/11 上午10:26
Version: 1.0
"""

from model import ModelParam
import torch
from util.write_file import WriteFile
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm
import numpy as np
from  torch.utils.tensorboard import SummaryWriter
# import tensorflow as tf
import math


def test_process(opt, critertion, cl_model, test_loader, last_F1=None, log_summary_writer: SummaryWriter=None, epoch=None):
    y_true = []
    y_pre = []
    total_labels = 0
    test_loss = 0

    predicts_all = None
    outputs_all = None

    orgin_param = ModelParam()

    with torch.no_grad():
        cl_model.eval()
        test_loader_tqdm = tqdm(test_loader, desc='Test Iteration')
        epoch_step_num = epoch * test_loader_tqdm.total
        step_num = 0
        for index, data in enumerate(test_loader_tqdm):
            texts_origin, img_text, img_text_bert_attention_mask, bert_attention_mask, image_origin, region_img_list, text_image_mask, labels, \
            _ = data
            # continue

            if opt.cuda is True:
                texts_origin = texts_origin.cuda()
                img_text = img_text.cuda()
                img_text_bert_attention_mask = img_text_bert_attention_mask.cuda()
                bert_attention_mask = bert_attention_mask.cuda()
                image_origin = image_origin.cuda()
                text_image_mask = text_image_mask.cuda()
                labels = labels.cuda()
                # region_img
                region_img_list = region_img_list
                region1 = []
                region2 = []
                region3 = []
                region_mask = []
                for img in region_img_list:
                    region1.append(img[0])
                    region2.append(img[1])
                    region3.append(img[2])
                    region_mask.append(img[3])
                # array2tensor
                region1 = (torch.stack(region1)).cuda()
                region2 = (torch.stack(region2)).cuda()
                region3 = (torch.stack(region3)).cuda()
                region_mask = region_mask

            orgin_param.set_data_param(texts=texts_origin, img_texts=img_text,
                                       img_text_bert_attention_mask=img_text_bert_attention_mask,
                                       bert_attention_mask=bert_attention_mask, images=image_origin,
                                       text_image_mask=text_image_mask, region1=region1, region2=region2,
                                       region3=region3, region_mask=region_mask)
            # augment_param.set_data_param(texts=texts_augment, bert_attention_mask=bert_attention_mask_augment,
            #                              images=image_augment, text_image_mask=text_image_mask_augment)
            origin_res, _, _, att = cl_model(orgin_param)

            print(origin_res)

            _, predicted = torch.max(origin_res, 1)
            total_labels += labels.size(0)
            y_true.extend(labels.cpu())
            y_pre.extend(predicted.cpu())

            labels_all = test_loader.dataset.label_list
            label = labels_all[0]
            predicts_all = predicted.cpu().numpy().tolist()
            predict = predicts_all[0]
            img_ids_all = test_loader.dataset.data_id_list
            img_id = img_ids_all[0]

            return img_id, predict, label, att



