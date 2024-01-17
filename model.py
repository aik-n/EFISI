
import torch.nn.modules as nn
import torchvision.models as cv_models
import torch
import os
from transformers import BertConfig, BertForPreTraining, RobertaForMaskedLM, RobertaModel, RobertaConfig, AlbertModel, \
    AlbertConfig, AutoModel, ViTModel
import math
import matplotlib.pyplot as plt

from contrastive_loss_with_temperature import ContrastiveLossWithTemperature
from pre_model import RobertaEncoder
import copy
import torch.nn.functional as F
from crossAttention import Cross_MultiAttention
import numpy as np
from DynamicLSTM import DynamicLSTM
from SimplifiedScaledDotProductAttention import SimplifiedScaledDotProductAttention


class ModelParam:
    def __init__(self, texts=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None, image_coordinate_position_token=None):
        self.texts = texts
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token

    def set_data_param(self, texts=None, img_texts=None, img_text_bert_attention_mask=None, images=None, bert_attention_mask=None, text_image_mask=None, segment_token=None, image_coordinate_position_token=None, region1=None, region2=None, region3=None, region_mask = None):
        self.texts = texts
        self.img_texts = img_texts
        self.img_text_bert_attention_mask = img_text_bert_attention_mask
        self.images = images
        self.bert_attention_mask = bert_attention_mask
        self.text_image_mask = text_image_mask
        self.segment_token = segment_token
        self.image_coordinate_position_token = image_coordinate_position_token
        self.region1 = region1
        self.region2 = region2
        self.region3 = region3
        self.region_mask = region_mask



def get_extended_attention_mask(attention_mask, input_shape):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


class ActivateFun(nn.Module):
    def __init__(self, opt):
        super(ActivateFun, self).__init__()
        self.activate_fun = opt.activate_fun

    def _gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        if self.activate_fun == 'relu':
            return torch.relu(x)
        elif self.activate_fun == 'gelu':
            return self._gelu(x)


class BertClassify(nn.Module):
    def __init__(self, opt, in_feature, dropout_rate=0.1):
        super(BertClassify, self).__init__()
        self.classify_linear = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_feature, 3),
            ActivateFun(opt)
        )

    def forward(self, inputs):
        return self.classify_linear(inputs)


class TextModel(nn.Module):
    def __init__(self, opt):
        super(TextModel, self).__init__()
        abl_path = '/hy-tmp/pretrain/'

        if opt.text_model == 'bert-base':
            # bert-base-uncased bertweet-base-sentiment-analysis
            self.config = BertConfig.from_pretrained(abl_path + 'bert-base-uncased/')
            self.model = BertForPreTraining.from_pretrained(abl_path + 'bert-base-uncased/', config=self.config)
            self.model = self.model.bert

        for param in self.model.parameters():
            param.requires_grad = True

        self.output_dim = self.model.encoder.layer[11].output.dense.out_features

    def get_output_dim(self):
        return self.output_dim

    def get_config(self):
        return self.config

    def get_encoder(self):
        model_encoder = copy.deepcopy(self.model.encoder)
        return model_encoder

    def forward(self, input, attention_mask):
        output = self.model(input, attention_mask=attention_mask)
        return output


class ImageModel(nn.Module):
    def __init__(self, opt):
        super(ImageModel, self).__init__()
        if opt.image_model == 'resnet-152':
            self.resnet = cv_models.resnet152(pretrained=True)
        elif opt.image_model == 'resnet-101':
            self.resnet = cv_models.resnet101(pretrained=True)
        elif opt.image_model == 'resnet-50':
            self.resnet = cv_models.resnet50(pretrained=True)
        elif opt.image_model == 'resnet-34':
            self.resnet = cv_models.resnet34(pretrained=True)
        elif opt.image_model == 'resnet-18':
            self.resnet = cv_models.resnet18(pretrained=True)
        self.resnet_encoder = nn.Sequential(*(list(self.resnet.children())[:-2]))
        self.resnet_avgpool = nn.Sequential(list(self.resnet.children())[-2])
        self.output_dim = self.resnet_encoder[7][2].conv3.out_channels

        for param in self.resnet.parameters():
            if opt.fixed_image_model:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def get_output_dim(self):
        return self.output_dim

    def forward(self, images):
        image_encoder = self.resnet_encoder(images)
        # image_encoder = self.conv_output(image_encoder)
        image_cls = self.resnet_avgpool(image_encoder)
        image_cls = torch.flatten(image_cls, 1)
        return image_encoder, image_cls


class FuseModel(nn.Module):
    def __init__(self, opt):
        super(FuseModel, self).__init__()
        self.cross_att2 = SimplifiedScaledDotProductAttention(d_model=768, h=2)
        self.clip_loss = ContrastiveLossWithTemperature()
        # from torch import nn
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.dropout = nn.Dropout(0.1)

        self.vit = ViTModel.from_pretrained('/hy-tmp/pretrain/vit-base-patch16-224')

        self.fuse_type = opt.fuse_type
        self.image_output_type = opt.image_output_type
        self.zoom_value = math.sqrt(opt.tran_dim)
        self.save_image_index = 0

        self.text_model = TextModel(opt)
        self.image_model = ImageModel(opt)

        self.text_config = copy.deepcopy(self.text_model.get_config())
        self.image_config = copy.deepcopy(self.text_model.get_config())

        self.text_config.num_attention_heads = opt.tran_dim // 64
        self.text_config.hidden_size = opt.tran_dim
        self.text_config.num_hidden_layers = opt.tran_num_layers

        self.image_config.num_attention_heads = opt.tran_dim // 64
        self.image_config.hidden_size = opt.tran_dim
        self.image_config.num_hidden_layers = opt.image_num_layers

        if self.text_config.is_decoder:
            self.use_cache = self.text_config.use_cache
        else:
            self.use_cache = False

        self.text_image_encoder = RobertaEncoder(self.text_config)
        self.image_encoder = RobertaEncoder(self.image_config)

        self.text_change = nn.Sequential(
            nn.Linear(self.text_model.get_output_dim(), opt.tran_dim),
            ActivateFun(opt)
        )
        self.image_change = nn.Sequential(
            nn.Linear(self.image_model.get_output_dim(), opt.tran_dim),
            ActivateFun(opt)
        )
        self.image_cls_change = nn.Sequential(
            nn.Linear(self.image_model.get_output_dim(), opt.tran_dim),
            ActivateFun(opt)
        )

        self.transformer_embedding_layernorm = nn.Sequential(
            nn.LayerNorm(opt.tran_dim),
            nn.Dropout(opt.l_dropout)
        )

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=opt.tran_dim, nhead=opt.tran_dim//64, dim_feedforward=opt.tran_dim * 4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=transformer_encoder_layer, num_layers=opt.tran_num_layers)

        if self.fuse_type == 'att':
            self.output_attention = nn.Sequential(
                nn.Linear(opt.tran_dim, opt.tran_dim // 2),
                ActivateFun(opt),
                nn.Linear(opt.tran_dim // 2, 1)
            )



    def forward(self, text_inputs, bert_attention_mask, image_inputs, text_image_mask, img_text=None, img_text_bert_attention_mask=None, region1=None, region2=None, region3=None, region_mask=None):
        text_encoder = self.text_model(text_inputs, attention_mask=bert_attention_mask)
        text_cls = text_encoder.pooler_output
        text_encoder = text_encoder.last_hidden_state
        text_init = self.text_change(text_encoder)



        # img_text处理部分
        if region1 is not None:
            img_text_encoder = self.text_model(img_text, attention_mask=img_text_bert_attention_mask)
            img_text_cls = img_text_encoder.pooler_output
            img_text_encoder = img_text_encoder.last_hidden_state
            img_text_init = self.text_change(img_text_encoder)



        image_encoder, image_cls = self.image_model(image_inputs)
        # region_img处理
        # if region1 is not None:
            # region1_encoder, region1_cls = self.image_model(region1)
            # image_encoder1 = region1_encoder.contiguous().view(region1_encoder.size(0), -1, region1_encoder.size(1))
            # image_encoder_init1 = self.image_change(image_encoder1)
            # image_cls_init1 = self.image_cls_change(region1_cls)
            # # bs*50*768
            # region1_init = torch.cat((image_cls_init1.unsqueeze(1), image_encoder_init1), dim=1)

            # region1_init = self.vit(region1, output_hidden_states=False)
            # # bs*197*768
            # region1_init = region1_init.last_hidden_state

        # if self.image_output_type == 'all':
        # image_encoder = image_encoder.contiguous().view(image_encoder.size(0), -1, image_encoder.size(1))
        # image_encoder_init = self.image_change(image_encoder)
        # image_cls_init = self.image_cls_change(image_cls)
        # image_init = torch.cat((image_cls_init.unsqueeze(1), image_encoder_init), dim=1)

        image_init = self.vit(image_inputs, output_hidden_states=False)
        image_init = image_init.last_hidden_state

        # else:
        #     image_cls_init = self.image_cls_change(image_cls)
        #     image_init = image_cls_init.unsqueeze(1)


        # image_mask = text_image_mask[:, -image_init.size(1):]
        # extended_attention_mask = get_extended_attention_mask(image_mask, image_init.size())
        #
        # image_init = self.image_encoder(image_init,
        #                                      attention_mask=None,
        #                                      head_mask=None,
        #                                      encoder_hidden_states=None,
        #                                      encoder_attention_mask=extended_attention_mask,
        #                                      past_key_values=None,
        #                                      use_cache=self.use_cache,
        #                                      output_attentions=self.text_config.output_attentions,
        #                                      output_hidden_states=(self.text_config.output_hidden_states),
        #                                      return_dict=self.text_config.use_return_dict
        #                                      )



        # region_img
        if region1 is not None:
            # region1_init = self.image_encoder(region1_init,
            #                                   attention_mask=None,
            #                                   head_mask=None,
            #                                   encoder_hidden_states=None,
            #                                   encoder_attention_mask=extended_attention_mask,
            #                                   past_key_values=None,
            #                                   use_cache=self.use_cache,
            #                                   output_attentions=self.text_config.output_attentions,
            #                                   output_hidden_states=(self.text_config.output_hidden_states),
            #                                   return_dict=self.text_config.use_return_dict
            #                                   )
            #
            # region_img1_init = region1_init.last_hidden_state

            # region_img2_init = region2_init.last_hidden_state
            # region_img3_init = region3_init.last_hidden_state

            # if img_text_init.shape[1] != 2:
            #     text_cls = text_init[:, 0, :]
            #     text_cls = torch.squeeze(text_cls, 1)
            #     img_text_cls = img_text_init[:, 0, :]
            #     img_text_cls = torch.squeeze(img_text_cls, 1)
            #     cl_loss = self.clip_loss(img_text_cls, text_cls, self.logit_scale)
            # else:
            #     cl_loss = 0

            # dropout数据增强，因为样本数量差距过大，即使设置采样权重也会导致少类别样本被大量的重复采样
            # text_init = self.dropout(text_init)
            # img_text_init = self.dropout(img_text_init)
            # region_img1_init = self.dropout(region_img1_init)


            # region图像和原始text间的对比损失
            text_cls = text_init[:, 0, :]
            text_cls = torch.squeeze(text_cls, 1)
            # img_cls = region1_init[:, 0, :]
            img_cls = image_init[:, 0, :]
            img_cls = torch.squeeze(img_cls, 1)
            cl_loss = self.clip_loss(img_cls, text_cls, self.logit_scale)

        if region1 is not None:

            # co-att
            region_img1_att, att = self.cross_att2(queries=text_init, keys=image_init,values=image_init)




        # text_init: bs*45*768  image_init: bs*50*768
        if region1 is not None:
            # text_image_cat = torch.cat((text_init, image_init, region_img1_init, region_img2_init, region_img3_init), dim=1)
            # text_init = self.dropout(text_init[:, 0, :])
            # region_img1_att = self.dropout(region_img1_att[:, 0, :])
            # img_text_init = self.dropout(img_text_init[:, 0, :])
            # text_init = self.LN(text_init)
            # region_img1_att = self.LN(region_img1_att)
            # img_text_init = self.LN(img_text_init)
            text_image_cat = torch.cat((text_init, region_img1_att, img_text_init), dim=1)

        else:
            text_image_cat = torch.cat((text_init, image_init), dim=1)

        if region1 is not None:
            extended_attention_mask: torch.Tensor = get_extended_attention_mask(bert_attention_mask, bert_attention_mask.size())
        else:
            extended_attention_mask: torch.Tensor = get_extended_attention_mask(text_image_mask, text_inputs.size())

        # region_img
#         if region1 is not None:
#             region_mask_count = region_mask
#             region_img_mask = []

#             for b in region_mask_count:
#                 region_img_mask1 = torch.ones(1, b * 40).cuda()
#                 region_img_mask2 = torch.zeros(1, (3 - b) * 40).cuda()
#                 region_img_mask.append(torch.cat((region_img_mask1, region_img_mask2), dim=1))

#             # region_img_mask1 = torch.ones(text_image_mask.shape[0], region_mask_count*50).cuda()
#             # region_img_mask2 = torch.zeros(text_image_mask.shape[0], (3-region_mask_count)*50).cuda()
#             # region_img_mask = torch.cat((region_img_mask1, region_img_mask2), dim=1)
#             # text_mask, _ = text_image_mask.split([text_image_mask.shape[1]-50, 50], dim=1)

#             region_img_mask = torch.squeeze(torch.stack(region_img_mask), dim=1)
#             # region_img_mask = torch.cat((text_mask, region_img_mask), dim=1)
#             region_img_mask1 = get_extended_attention_mask(region_img_mask, region_img_mask)

            # region_img_text_mask = torch.cat((text_mask, region_img_mask1), dim=3)
            # print(text_inputs.size())

        # 新的mask
        if region1 is not None:
            img_text_region_img_mask1 = torch.ones(img_text_init.shape[0], img_text_init.shape[1]).cuda()
            img_text_region_img_mask2 = get_extended_attention_mask(img_text_region_img_mask1, img_text_region_img_mask1.size())
            extended_attention_mask = torch.cat((extended_attention_mask, img_text_region_img_mask2), dim=3)

        # attn_output, attn_output_weights = self.multihead_attn(text_image_cat, text_image_cat, text_image_cat)

        # transformer
        text_image_transformer = self.text_image_encoder(text_image_cat,
                                                 attention_mask=None,
                                                 head_mask=None,
                                                 encoder_hidden_states=None,
                                                 encoder_attention_mask=None,
                                                 past_key_values=None,
                                                 use_cache=self.use_cache,
                                                 output_attentions=self.text_config.output_attentions,
                                                 output_hidden_states=(self.text_config.output_hidden_states),
                                                 return_dict=self.text_config.use_return_dict)
        # bs*95*768
        text_image_transformer = text_image_transformer.last_hidden_state
        text_image_transformer1 = text_image_transformer.permute(0, 2, 1).contiguous()

        if self.fuse_type == 'max':
            text_image_output = torch.max(text_image_transformer, dim=2)[0]
        elif self.fuse_type == 'att':
            text_image_output = text_image_transformer.permute(0, 2, 1).contiguous()
        elif self.fuse_type == 'ave':
            text_image_length = text_image_transformer.size(1)
            text_image_output = torch.sum(text_image_transformer, dim=1) / text_image_length

        else:
            raise Exception('fuse_type设定错误')
        if region1 is not None:
            return text_image_output, None, None, text_image_transformer1, cl_loss, att
        else:
            return text_image_output, None, None


class CLModel(nn.Module):
    def __init__(self, opt):
        super(CLModel, self).__init__()
        self.fuse_model = FuseModel(opt)
        self.temperature = opt.temperature
        self.set_cuda = opt.cuda



        self.output_classify = nn.Sequential(
            nn.Dropout(opt.l_dropout),
            nn.Linear(opt.tran_dim, opt.tran_dim // 2),
            ActivateFun(opt),
            nn.Linear(opt.tran_dim // 2, 2)
        )

    def forward(self, data_orgin: ModelParam, labels=None, target_labels=None):
        orgin_res, orgin_text_cls, orgin_image_cls, text_init, cl_loss, att = self.fuse_model(data_orgin.texts,data_orgin.bert_attention_mask,
                                                                                data_orgin.images, data_orgin.text_image_mask,
                                                                                data_orgin.img_texts,data_orgin.img_text_bert_attention_mask,
                                                                                data_orgin.region1, data_orgin.region2, data_orgin.region3, data_orgin.region_mask)

        output = self.output_classify(orgin_res)

        return output, text_init, cl_loss, att



class TensorBoardModel(nn.Module):
    def __init__(self, opt):
        super(TensorBoardModel, self).__init__()
        self.cl_model = CLModel(opt)

    def forward(self, texts, bert_attention_mask, images, text_image_mask,
                texts_augment, bert_attention_mask_augment, images_augment, text_image_mask_augment, label):
        orgin_param = ModelParam()
        augment_param = ModelParam()
        orgin_param.set_data_param(texts=texts, bert_attention_mask=bert_attention_mask, images=images, text_image_mask=text_image_mask)
        augment_param.set_data_param(texts=texts_augment, bert_attention_mask=bert_attention_mask_augment, images=images_augment, text_image_mask=text_image_mask_augment)
        return self.cl_model(orgin_param, augment_param, label, [torch.ones(1, dtype=torch.int64) for _ in range(3)])
