
import torch
# from transformers import AdamW
from torch.optim import Adam, AdamW, SGD
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

import test_process
from util.write_file import WriteFile
import dev_process
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from model import ModelParam
import torch.nn.functional as F
from losses import SupConLoss



def train_process(opt, train_loader, dev_loader, test_loader, cl_model, critertion, log_summary_writer:SummaryWriter=None, tokenizer=None, image_id_list=None):

    supConloss = SupConLoss(temperature=0.07, contrast_mode='all', base_temperature=0.07)

    optimizer = None

    pre_train_model_param = [name for name, param in cl_model.named_parameters() if 'text_model' in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in cl_model.named_parameters() if n in pre_train_model_param],
            # "lr": opt.fuse_lr,
            "lr": 0,
        },
        {
            "params": [p for n, p in cl_model.named_parameters() if n not in pre_train_model_param],
            "lr": opt.fuse_lr,
        },
    ]

    if opt.optim == 'adam':
        optimizer = Adam(optimizer_grouped_parameters, betas=(opt.optim_b1, opt.optim_b2))
    elif opt.optim == 'adamw':
        optimizer = AdamW(optimizer_grouped_parameters, betas=(opt.optim_b1, opt.optim_b2))
    elif opt.optim == 'sgd':
        optimizer = SGD(optimizer_grouped_parameters, momentum=opt.momentum)

    orgin_param = ModelParam()
    augment_param = ModelParam()

    last_F1 = 0
    last_Accuracy = 0

    Test_last_F1 = 0
    Test_last_Accuracy = 0
    total_Acc = 0
    total_F1 = 0

    for epoch in trange(opt.epoch, desc='Epoch:'):
        y_true = []
        y_pre = []
        run_loss = 0
        total_labels = 0

        cl_model.train()
        cl_model.zero_grad()

        if epoch >= opt.train_fuse_model_epoch:
            optimizer.param_groups[0]['lr'] = opt.lr
            optimizer.param_groups[1]['lr'] = opt.lr

        train_loader_tqdm = tqdm(train_loader, desc='Train Iteration:')
        epoch_step_num = epoch * train_loader_tqdm.total
        step_num = 0
        for index, data in enumerate(train_loader_tqdm):
            texts_origin, img_text, img_text_bert_attention_mask, bert_attention_mask, image_origin, region_img_list, text_image_mask, labels,\
                target_labels = data

            if opt.cuda is True:
                texts_origin = texts_origin.cuda()
                img_text = img_text.cuda()
                img_text_bert_attention_mask = img_text_bert_attention_mask.cuda()
                bert_attention_mask = bert_attention_mask.cuda()
                image_origin = image_origin.cuda()

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
                region_mask = region_mask


                text_image_mask = text_image_mask.cuda()
                labels = labels.cuda()
                # texts_augment = texts_augment.cuda()
                # bert_attention_mask_augment = bert_attention_mask_augment.cuda()
                # image_augment = image_augment.cuda()
                # text_image_mask_augment = text_image_mask_augment.cuda()
                for i in range(len(target_labels)):
                    target_labels[i] = target_labels[i].cuda()

            orgin_param.set_data_param(texts=texts_origin, img_texts=img_text, img_text_bert_attention_mask = img_text_bert_attention_mask, bert_attention_mask=bert_attention_mask, images=image_origin, text_image_mask=text_image_mask, region1=region1, region_mask=region_mask)
            # augment_param.set_data_param(texts=texts_augment, bert_attention_mask=bert_attention_mask_augment, images=image_augment, text_image_mask=text_image_mask_augment)

            # origin_res, l_pos_neg, cl_lables, cl_self_loss, text_init = cl_model(orgin_param, augment_param, labels, target_labels)

            origin_res, text_init, cl_loss, att = cl_model(orgin_param)

            feature = text_init
            feature = F.normalize(feature, dim=1)
            scloss = supConloss(feature, labels)
            # print(celoss, scloss)

            classify_loss = critertion(origin_res, labels)

            # cl_loss = critertion(l_pos_neg, cl_lables)


            # loss = (classify_loss + cl_loss * opt.cl_loss_alpha + cl_self_loss * opt.cl_self_loss_alpha + scloss*0.1) / opt.acc_batch_size

            # loss = (classify_loss + scloss + cl_loss) / opt.acc_batch_size
            loss = classify_loss + 1*scloss + 0.01*cl_loss
            # optimizer.zero_grad
            loss.backward()
            # optimizer.step()
            train_loader_tqdm.set_description("Train:, l1: %.6f, l2: %.6f, l3: %.6f" %
                                              (classify_loss, scloss, cl_loss))

            if (index + 1) % opt.acc_grad == 0:
                if log_summary_writer:
                    log_summary_writer.add_scalar('train_info/loss', loss.item(), global_step=step_num + epoch_step_num)
                    log_summary_writer.add_scalar('train_info/classify_loss', classify_loss.item(), global_step=step_num + epoch_step_num)
                    # log_summary_writer.add_scalar('train_info/cl_loss', cl_loss.item(), global_step=step_num + epoch_step_num)
                    # log_summary_writer.add_scalar('train_info/cl_self_loss', cl_self_loss.item(), global_step=step_num + epoch_step_num)
                    log_summary_writer.add_scalar('train_info/lr', optimizer.param_groups[0]['lr'], global_step=step_num + epoch_step_num)
                    log_summary_writer.add_scalar('train_info/fuse_lr', optimizer.param_groups[1]['lr'], global_step=step_num + epoch_step_num)
                optimizer.step()
                optimizer.zero_grad()
            step_num += 1

            _, predicted = torch.max(origin_res, 1)
            y_true.extend(labels.cpu())
            y_pre.extend(predicted.cpu())
            run_loss += loss.item()
            total_labels += labels.size(0)

        run_loss /= total_labels
        y_true = np.array(y_true)
        y_pre = np.array(y_pre)
        train_accuracy = accuracy_score(y_true, y_pre)
        train_F1_weighted = f1_score(y_true, y_pre, average='weighted')
        train_R_weighted = recall_score(y_true, y_pre, average='weighted')
        train_precision_weighted = precision_score(y_true, y_pre, average='weighted')
        train_F1 = f1_score(y_true, y_pre, average='macro')
        train_R = recall_score(y_true, y_pre, average='macro')
        train_precision = precision_score(y_true, y_pre, average='macro')

        save_content = 'Epoch: %d:\nTrain: Accuracy: %.6f, F1(weighted): %.6f, Precision(weighted): %.6f, R(weighted): %.6f, F1(macro): %.6f, Precision: %.6f, R: %.6f, loss: %.6f' % \
                       (epoch, train_accuracy, train_F1_weighted, train_precision_weighted, train_R_weighted, train_F1, train_precision, train_R, run_loss)
        WriteFile(opt.save_model_path, 'train_correct_log.txt', save_content + '\n', 'a+')
        print(save_content, ' ' * 200)

        if log_summary_writer:
            log_summary_writer.add_scalar('train_info/loss_epoch', run_loss, global_step=epoch)
            log_summary_writer.add_scalar('train_info/acc', train_accuracy, global_step=epoch)
            log_summary_writer.add_scalar('train_info/f1_w', train_F1_weighted, global_step=epoch)
            log_summary_writer.add_scalar('train_info/r_w', train_R_weighted, global_step=epoch)
            log_summary_writer.add_scalar('train_info/p_w', train_precision_weighted, global_step=epoch)
            log_summary_writer.add_scalar('train_info/f1_ma', train_F1, global_step=epoch)
            log_summary_writer.flush()

        train_log = {
            "epoch": epoch,
            "train_accuracy": train_accuracy,
            "train_F1": train_F1,
            "train_R": train_R,
            "train_precision": train_precision,
            "train_F1_weighted": train_F1_weighted,
            "train_precision_weighted": train_precision_weighted,
            "train_R_weighted": train_R_weighted,
            "run_loss": run_loss
        }
        # debug：正常运行不要把下面的代码注释掉
        last_F1, last_Accuracy, now_acc, now_f1 = dev_process.dev_process(opt, critertion, cl_model, dev_loader, test_loader, last_F1, last_Accuracy, train_log, log_summary_writer)


        Test_last_F1, Test_last_Accuracy, total_Acc, total_F1 = test_process.test_process(opt, critertion, cl_model, test_loader, Test_last_F1, Test_last_Accuracy,epoch=1, now_Accuracy= now_acc,now_F1 = now_f1,last_total_acc = total_Acc,last_total_f1 = total_F1)
        
