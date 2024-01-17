
from model import ModelParam
import torch

from util.compare_to_save import compare_to_save
from util.write_file import WriteFile
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm
import numpy as np
from  torch.utils.tensorboard import SummaryWriter
# import tensorflow as tf
import math


def test_process(opt, critertion, cl_model, test_loader, last_F1=None, last_Accuracy=None, log_summary_writer: SummaryWriter=None, epoch=None,now_Accuracy=None, now_F1=None, last_total_acc=None,last_total_f1=None):
    y_true = []
    y_pre = []
    total_labels = 0
    test_loss = 0

    predicts_all = None
    outputs_all = None

    # last_Accuracy = 0

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

            orgin_param.set_data_param(texts=texts_origin, img_texts=img_text, img_text_bert_attention_mask=img_text_bert_attention_mask, bert_attention_mask=bert_attention_mask, images=image_origin,
                                       text_image_mask=text_image_mask, region1=region1, region2=region2, region3=region3,region_mask=region_mask)
            origin_res,_,_,att = cl_model(orgin_param)



            loss = critertion(origin_res, labels) / opt.acc_batch_size
            test_loss += loss.item()
            _, predicted = torch.max(origin_res, 1)
            total_labels += labels.size(0)
            y_true.extend(labels.cpu())
            y_pre.extend(predicted.cpu())



            test_loader_tqdm.set_description("Test Iteration, loss: %.6f" % loss)
            if log_summary_writer:
                log_summary_writer.add_scalar('test_info/loss', loss.item(), global_step=step_num + epoch_step_num)
            step_num += 1

            img_ids_all = test_loader.dataset.data_id_list
            labels_all = test_loader.dataset.label_list
            if predicts_all is None:
                predicts_all = predicted
                outputs_all = origin_res
            else:
                predicts_all = torch.cat((predicts_all, predicted), dim=0)
                outputs_all = torch.cat((outputs_all, origin_res), dim=0)

        with open('predict_new.txt', 'w', encoding='utf-8') as fout:

            predicts_all = predicts_all.cpu().numpy().tolist()
            outputs_all = outputs_all.cpu().numpy().tolist()
            assert len(img_ids_all) == len(predicts_all) == len(labels_all) == len(outputs_all)

            for i in range(len(img_ids_all)):
                img_id = img_ids_all[i]
                predict = predicts_all[i]
                label = labels_all[i]
                output = outputs_all[i]
                fout.write(f'{str(img_id)} {str(predict)} {str(label)} {str(output)} \n')

        test_loss /= total_labels
        y_true = np.array(y_true)
        y_pre = np.array(y_pre)
        test_accuracy = accuracy_score(y_true, y_pre)
        test_F1 = f1_score(y_true, y_pre, average='macro')
        test_R = recall_score(y_true, y_pre, average='macro')
        test_precision = precision_score(y_true, y_pre, average='macro')
        test_F1_weighted = f1_score(y_true, y_pre, average='weighted')
        test_R_weighted = recall_score(y_true, y_pre, average='weighted')
        test_precision_weighted = precision_score(y_true, y_pre, average='weighted')

        total_acc = (test_accuracy+now_Accuracy)/2
        total_f1 = (test_F1_weighted + now_F1) / 2
        save_content = 'Test : Accuracy: %.6f, F1(weighted): %.6f, Precision(weighted): %.6f, R(weighted): %.6f, F1(macro): %.6f, Precision: %.6f, R: %.6f, loss: %.6f,total_Accuracy: %.6f, total_F1: %.6f,' % \
            (test_accuracy, test_F1_weighted, test_precision_weighted, test_R_weighted, test_F1, test_precision, test_R, test_loss,total_acc,total_f1)

        print(save_content)



        if log_summary_writer:
            log_summary_writer.add_scalar('test_info/loss_epoch', test_loss, global_step=epoch)
            log_summary_writer.add_scalar('test_info/acc', test_accuracy, global_step=epoch)
            log_summary_writer.add_scalar('test_info/f1_w', test_F1_weighted, global_step=epoch)
            log_summary_writer.add_scalar('test_info/r_w', test_R_weighted, global_step=epoch)
            log_summary_writer.add_scalar('test_info/p_w', test_precision_weighted, global_step=epoch)
            log_summary_writer.add_scalar('test_info/f1_ma', test_F1, global_step=epoch)

            log_summary_writer.flush()

        if last_F1 is not None:
            WriteFile(
                opt.save_model_path, 'train_correct_log.txt', save_content + '\n', 'a+')

        # last_Accuracy, is_save_model, model_name = compare_to_save(last_Accuracy, test_accuracy, opt, cl_model,
        #                                                            None, None, 'TestAcc', opt.save_acc,
        #                                                            add_enter=False)
        if test_accuracy > last_Accuracy:
            torch.save(cl_model.state_dict(), '/hy-tmp/checkpoint/best_model.pth')
            last_Accuracy = test_accuracy

        if total_acc >last_total_acc:
            torch.save(cl_model.state_dict(), '/hy-tmp/checkpoint/total_best_model.pth')
            last_total_acc = total_acc
            last_total_f1 = total_f1

        return last_F1, last_Accuracy, last_total_acc, last_total_f1
