"""
Name: data_process
Date: 2022/4/11 上午10:25
Version: 1.0
"""
import os

from PIL import Image
from PIL import ImageFile
from PIL import TiffImagePlugin
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
import json
import torch.nn.utils.rnn as run_utils
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from util.image_augmentation.augmentations import RandAugment
import copy
import matplotlib.pyplot as plt


class SentenceDataset(Dataset):
    def __init__(self, opt, data_path, text_tokenizer, photo_path, image_transforms, data_type, data_translation_path=None, image_coordinate=None):
        self.data_type = data_type
        self.dataset_type = opt.data_type
        if opt.data_type == 'MVSA-multiple':
            self.photo_path = '/hy-tmp/dataset/multi/'
            self.region_photo_path = '/hy-tmp/dataset/multi_region/'
        if opt.data_type == 'MVSA-single':
            self.photo_path = '/hy-tmp/dataset/single/'
            self.region_photo_path = '/hy-tmp/dataset/single_region/'
        if opt.data_type == 'HFM':
            self.photo_path = '/hy-tmp/dataset/sarcasm/'
            self.region_photo_path = '/hy-tmp/dataset/sarcasm_region/'

        self.image_transforms = image_transforms

        file_read = open(data_path, 'r', encoding='utf-8')
        file_content = json.load(file_read)
        file_read.close()

        self.data_id_list = []
        self.text_list = []
        self.label_list = []
        self.img_text_list = []
        for data in file_content:
            self.data_id_list.append(data['id'])
            if opt.data_type == 'MVSA-single':
                f = open('/hy-tmp/dataset/single_ocr/' + data['id'] + '.txt', 'r', encoding='utf-8')
            if opt.data_type == 'MVSA-multiple':
                f = open('/hy-tmp/dataset/multi_ocr/' + data['id'] + '.txt', 'r', encoding='utf-8')
            if opt.data_type == 'HFM':
                f = open('/hy-tmp/dataset/sarcasm_ocr/' + data['id'] + '.txt', 'r', encoding='utf-8')
            text = f.read()
            out_text = text.replace('\n', ' ')
            self.img_text_list.append(out_text)

            self.text_list.append(data['text'])
            # self.text_list.append([data['text'], out_text])
            self.label_list.append(data['emotion_label'])

        if self.dataset_type != 'meme7k':
            self.image_id_list = [str(data_id) + '.jpg' for data_id in self.data_id_list]
        else:
            self.image_id_list = self.data_id_list

        file_read = open(data_translation_path, 'r', encoding='utf-8')
        file_content = json.load(file_read)
        file_read.close()

        if opt.text_model == 'bert-base':
            self.text_to_id = [text_tokenizer.encode(text) for text in
                                        tqdm(self.text_list, desc='convert text to token')]
            self.img_text_to_id = [text_tokenizer.encode(img_text) for img_text in
                               tqdm(self.img_text_list, desc='convert img_text to token')]

    def get_data_id_list(self):
        return self.data_id_list

    def __len__(self):
        return len(self.text_to_id)

    def __getitem__(self, index):
        image_path = self.photo_path + '/' + str(self.data_id_list[index]) + '.jpg'
        image_read = Image.open(image_path)
        image_read.load()

        image_origin = self.image_transforms(image_read)
        image_augment = image_origin

        data_id = str(self.data_id_list[index])

        # 区域目标图像处理部分
        region_image_path = self.region_photo_path + str(self.data_id_list[index])
        img_names = os.listdir(region_image_path)
        count = 0
        region_img_list = []
        for img_name in img_names:
            if img_name.lower().endswith(
                    ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                region_img_read = Image.open(os.path.join(region_image_path, img_name))

                region_img = self.image_transforms(region_img_read)

                region_img_list.append(image_origin)
                count += 1
                if count >= 3:
                    break
        # 处理没有region_img的情况
        if count == 0:
            region_img_list.append(image_origin)

            count += 1

        if count < 3:
            for i in range(3-count):
                x = torch.zeros((3, 224, 224))
                region_img_list.append(torch.FloatTensor(x))
        region_img_list.append(count)


        return self.text_to_id[index], image_origin, self.label_list[index], region_img_list, self.img_text_to_id[index]


class Collate():
    def __init__(self, opt):
        self.text_length_dynamic = opt.text_length_dynamic
        if self.text_length_dynamic == 1:
            # 使用动态的长度
            self.min_length = 1
        elif self.text_length_dynamic == 0:
            # 使用固定动的文本长度
            self.min_length = opt.word_length

        self.image_mask_num = 0
        if opt.image_output_type == 'cls':
            self.image_mask_num = 1
        elif opt.image_output_type == 'all':
            self.image_mask_num = 50

    def __call__(self, batch_data):
        return self._collate(batch_data)

    def _collate(self, batch_data):
        region_img_list = [b[3] for b in batch_data]

        text_to_id = [torch.LongTensor(b[0]) for b in batch_data]
        img_text_to_id = [torch.LongTensor(b[4]) for b in batch_data]
        image_origin = torch.FloatTensor([np.array(b[1]) for b in batch_data])
        label = torch.LongTensor([b[2] for b in batch_data])

        data_length = [text.size(0) for text in text_to_id]
        img_text_data_length = [text.size(0) for text in img_text_to_id]

        max_length = max(data_length)
        if max_length < self.min_length:
            # 这一步防止在后续的计算过程中，因为文本长度和mask长度不一致而出错
            text_to_id[0] = torch.cat((text_to_id[0], torch.LongTensor([0] * (self.min_length - text_to_id[0].size(0)))))
            max_length = self.min_length

        # img_text
        img_text_max_length = max(img_text_data_length)
        if img_text_max_length < self.min_length:
            # 这一步防止在后续的计算过程中，因为文本长度和mask长度不一致而出错
            img_text_to_id[0] = torch.cat(
                (img_text_to_id[0], torch.LongTensor([0] * (self.min_length - img_text_to_id[0].size(0)))))
            img_text_max_length = self.min_length
        if img_text_max_length < 200:
            text_to_id = text_to_id
        else:
            img_text_max_length = 200
            text_to_id_after = []
            sep = torch.LongTensor(torch.tensor([102]))
            for item in img_text_to_id:
                if item.__len__() < 200:
                    text_to_id_after.append(item)
                else:
                    item = item[:199]
                    item = torch.cat((item, sep))
                    text_to_id_after.append(item)
            img_text_to_id = text_to_id_after


        # 新预训练bert
        if max_length < 200:
            text_to_id = text_to_id
        else:
            max_length = 200
            text_to_id_after = []
            sep = torch.LongTensor(torch.tensor([102]))
            for item in text_to_id:
                if item.__len__() < 200:
                    text_to_id_after.append(item)
                else:
                    item = item[:199]
                    item = torch.cat((item, sep))
                    text_to_id_after.append(item)
            text_to_id = text_to_id_after



        text_to_id = run_utils.pad_sequence(text_to_id, batch_first=True, padding_value=0)
        img_text_to_id = run_utils.pad_sequence(img_text_to_id, batch_first=True, padding_value=0)
        # text_translation_to_id = run_utils.pad_sequence(text_translation_to_id, batch_first=True, padding_value=0)

        bert_attention_mask = []
        text_image_mask = []
        for length in data_length:
            if length >= 200:
                length = 200
            text_mask_cell = [1] * length
            text_mask_cell.extend([0] * (max_length - length))
            bert_attention_mask.append(text_mask_cell[:])

            text_mask_cell.extend([1] * self.image_mask_num)
            text_image_mask.append(text_mask_cell[:])

        # img_text
        img_text_bert_attention_mask = []
        for length in img_text_data_length:
            if length >= 200:
                length = 200
            text_mask_cell = [1] * length
            text_mask_cell.extend([0] * (img_text_max_length - length))
            img_text_bert_attention_mask.append(text_mask_cell[:])


        temp_labels = [label - 0, label - 1, label - 2]
        target_labels = []
        for i in range(3):
            temp_target_labels = []
            for j in range(temp_labels[0].size(0)):
                if temp_labels[i][j] == 0:
                    temp_target_labels.append(j)
            target_labels.append(torch.LongTensor(temp_target_labels[:]))

        return text_to_id, img_text_to_id,torch.LongTensor(img_text_bert_attention_mask), torch.LongTensor(bert_attention_mask), image_origin, region_img_list, torch.LongTensor(text_image_mask), label, \
               target_labels


def get_resize(image_size):
    for i in range(20):
        if 2**i >= image_size:
            return 2**i
    return image_size


def data_process(opt, data_path, text_tokenizer, photo_path, data_type, data_translation_path=None, image_coordinate=None):

    transform_base = transforms.Compose(
        [
            transforms.Resize(get_resize(opt.image_size)),
            transforms.CenterCrop(opt.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    # transform_train = copy.deepcopy(transform_base)
    transform_augment = copy.deepcopy(transform_base)
    transform_augment.transforms.insert(0, RandAugment(2, 14))
    transform_train = transform_augment
    # transform_train = [transform_train, transform_augment]

    transform_test_dev = transforms.Compose(
        [
            transforms.Resize(get_resize(opt.image_size)),
            transforms.CenterCrop(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    dataset = SentenceDataset(opt, data_path, text_tokenizer, photo_path, transform_train if data_type == 1 else transform_test_dev, data_type,
                              data_translation_path=data_translation_path, image_coordinate=image_coordinate)

    data_loader = DataLoader(dataset, batch_size=opt.acc_batch_size,
                             shuffle=True if data_type == 1 else False,
                             num_workers=opt.num_workers, collate_fn=Collate(opt), pin_memory=True if opt.cuda else False)
    return data_loader, dataset.__len__()

from torch.utils.data.sampler import WeightedRandomSampler

# 加权随机采样
def creater_sampler(train_set):
    label = []
    weight = []
    num_sample = 0
    positive = 0
    normal = 0
    negative = 0
    for line in train_set.label_list:
        label_ = line
        label.append(label_)
        if label_ == 0:
            positive += 1
            weight.append(0.5)
        elif label_ == 1:
            normal += 1
            weight.append(0.5)
        elif label_ == 2:
            negative += 1
            weight.append(0.4)

    num_sample = label.__len__()

    class_sample_count = [positive, normal, negative]
    # weights = 1 / torch.Tensor(class_sample_count)  # Multiple：0.3 0.3 0.4  Single：0.4 0.2 0.4

    sampler = WeightedRandomSampler(weight, num_sample, replacement=True)
    # print('样本权重:'+weight, '样本数量:'+num_sample)
    return sampler

def data_process_train(opt, data_path, text_tokenizer, photo_path, data_type, data_translation_path=None, image_coordinate=None):

    transform_base = transforms.Compose(
        [
            transforms.Resize(get_resize(opt.image_size)),
            transforms.CenterCrop(opt.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    # transform_train = copy.deepcopy(transform_base)
    transform_augment = copy.deepcopy(transform_base)
    transform_augment.transforms.insert(0, RandAugment(2, 14))
    transform_train = transform_augment
    # transform_train = [transform_train, transform_augment]

    transform_test_dev = transforms.Compose(
        [
            transforms.Resize(get_resize(opt.image_size)),
            transforms.CenterCrop(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    dataset = SentenceDataset(opt, data_path, text_tokenizer, photo_path, transform_train if data_type == 1 else transform_test_dev, data_type,
                              data_translation_path=data_translation_path, image_coordinate=image_coordinate)

    train_sampler = creater_sampler(dataset)
    data_loader = DataLoader(dataset, batch_size=opt.acc_batch_size,sampler=train_sampler,
                             num_workers=opt.num_workers, collate_fn=Collate(opt), pin_memory=True if opt.cuda else False)
    return data_loader, dataset.__len__()


class SingleSentenceDataset(Dataset):
    def __init__(self, opt, dataId, imgdir, ocr_text, data_path, text_tokenizer, photo_path, image_transforms, data_type, data_translation_path=None, image_coordinate=None):
        self.data_type = data_type
        self.dataset_type = opt.data_type
        if opt.data_type == 'MVSA-multiple':
            self.photo_path = '/hy-tmp/dataset/multi/'
            self.region_photo_path = '/hy-tmp/dataset/multi_region/'
        if opt.data_type == 'MVSA-single':
            self.photo_path = '/hy-tmp/dataset/single/'
            self.region_photo_path = '/hy-tmp/dataset/single_region/'
        if opt.data_type == 'HFM':
            self.photo_path = '/hy-tmp/dataset/sarcasm/'
            self.region_photo_path = '/hy-tmp/dataset/sarcasm_region/'

        self.image_transforms = image_transforms

        file_read = open(data_path, 'r', encoding='utf-8')
        file_content = json.load(file_read)
        file_read.close()

        self.imgdir = imgdir

        self.data_id_list = []
        self.text_list = []
        self.label_list = []
        self.img_text_list = []
        for data in file_content:
            if data['id'] == dataId:
                self.data_id_list.append(data['id'])
                if opt.data_type == 'MVSA-single':
                    f = open('/hy-tmp/dataset/single_ocr/' + data['id'] + '.txt', 'r', encoding='utf-8')
                elif opt.data_type == 'MVSA-multiple':
                    f = open('/hy-tmp/dataset/multi_ocr/' + data['id'] + '.txt', 'r', encoding='utf-8')
                elif opt.data_type == 'HFM':
                    f = open('/hy-tmp/dataset/sarcasm_ocr/' + data['id'] + '.txt', 'r', encoding='utf-8')
                text = f.read()
                out_text = text.replace('\n', ' ')
                if ocr_text != '':
                    self.img_text_list.append(' ')
                else:
                    self.img_text_list.append(out_text)

                self.text_list.append(data['text'])
                # self.text_list.append([data['text'], out_text])
                self.label_list.append(data['emotion_label'])
                break

        if self.dataset_type != 'meme7k':
            if data['id'] == dataId:
                self.image_id_list = [str(data_id) + '.jpg' for data_id in self.data_id_list]
        else:
            if data['id'] == dataId:
                self.image_id_list = self.data_id_list


        file_read = open(data_translation_path, 'r', encoding='utf-8')
        file_content = json.load(file_read)
        for line in file_content:
            if line['id'] == dataId:
                file_content1 = line
                self.data_translation_id_to_text_dict = {line['id']: line['text_translation']}
                break
        file_read.close()

        # new
        # if opt.text_model == 'bert-base':
        self.text_to_id = [text_tokenizer.encode(text) for text in
                                    tqdm(self.text_list, desc='convert text to token')]

        self.img_text_to_id = [text_tokenizer.encode(img_text) for img_text in
                           tqdm(self.img_text_list, desc='convert img_text to token')]

        self.text_translation_to_id = {index: text_tokenizer.encode(text) for
                                                  index, text in tqdm(self.data_translation_id_to_text_dict.items(), desc='convert translation_text to token')}

    def get_data_id_list(self):
        return self.data_id_list

    def __len__(self):
        return len(self.text_to_id)

    def __getitem__(self, index):
        image_path = self.photo_path + '/' + str(self.data_id_list[index]) + '.jpg'
        image_read = Image.open(image_path)
        image_read.load()

        image_origin = self.image_transforms(image_read)
        image_augment = image_origin

        data_id = str(self.data_id_list[index])

        # 区域目标图像处理部分
        region_image_path = self.region_photo_path + str(self.data_id_list[index])
        img_names = os.listdir(region_image_path)
        count = 0
        region_img_list = []
        # 测试指定单条数据
        if self.imgdir != '':
            for img_name in img_names:
                if img_name == '1.jpg':     # 指定图像
                    region_img = Image.open(os.path.join(region_image_path, img_name))
                    region_img = self.image_transforms(region_img)
                    region_img_list.append(torch.FloatTensor(region_img))
                    count += 1
                    break

        else:
            for img_name in img_names:
                if img_name.lower().endswith(
                        ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):

                    region_img_list.append(image_origin)
                    count += 1
                    if count >= 3:
                        break
        # 处理没有region_img的情况
        if count == 0:
            region_img_list.append(image_origin)
            count += 1

        if count < 3:
            for i in range(3-count):
                x = torch.zeros((3, 224, 224))
                region_img_list.append(torch.FloatTensor(x))

        region_img_list.append(count)


        if self.data_type == 1:
            image_augment = copy.deepcopy(image_read)
            image_augment = self.image_transforms(image_augment)
        return self.text_to_id[index], image_origin, self.label_list[index], region_img_list, self.img_text_to_id[index]
def Single_data_process(opt, dataId, imgdir, ocr_text, data_path, text_tokenizer, photo_path, data_type, data_translation_path=None, image_coordinate=None):

    transform_base = transforms.Compose(
        [
            transforms.Resize(get_resize(opt.image_size)),
            transforms.CenterCrop(opt.image_size),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    # transform_train = copy.deepcopy(transform_base)
    transform_augment = copy.deepcopy(transform_base)
    transform_augment.transforms.insert(0, RandAugment(2, 14))
    transform_train = transform_augment
    # transform_train = [transform_train, transform_augment]

    transform_test_dev = transforms.Compose(
        [
            transforms.Resize(get_resize(opt.image_size)),
            transforms.CenterCrop(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    dataset = SingleSentenceDataset(opt, dataId, imgdir, ocr_text, data_path, text_tokenizer, photo_path, transform_train if data_type == 1 else transform_test_dev, data_type,
                              data_translation_path=data_translation_path, image_coordinate=image_coordinate)

    data_loader = DataLoader(dataset, batch_size=opt.acc_batch_size,
                             shuffle=True if data_type == 1 else False,
                             num_workers=opt.num_workers, collate_fn=Collate(opt), pin_memory=True if opt.cuda else False)
    return data_loader, dataset.__len__()