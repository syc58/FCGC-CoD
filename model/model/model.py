import json
import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import GaussianEncoder
from exp_generator import ExpGenerator
from lxrt.entry import LXRTEncoder
from loss import exp_generative_loss, structure_bce, kl_div_gaussian


class FCGC_CoD(nn.Module):
    def __init__(self, num_roi=36, nb_answer=2000, nb_vocab=2000, num_step=18, lang_dir=None, args=None, explainable=True):
        super(FCGC_CoD, self).__init__()
        self.nb_vocab = nb_vocab
        self.num_roi = num_roi
        self.nb_answer = nb_answer
        self.num_step = num_step
        self.img_size = 2048
        self.hidden_size = 768
        self.num_head = 4
        self.explainable = explainable
        self.args = args
        base_model = LXRTEncoder(max_seq_length=18)
        base_model.load("lxrt/model")

        self.bert_encoder = base_model.model
        self.bert_encoder.mode = 'lxr'
        self.IOF_ans_cls = nn.Sequential(nn.Linear(768, 768 * 2),
                                         nn.GELU(),
                                         nn.LayerNorm(768 * 2),
                                         nn.Linear(768 * 2, self.nb_answer))
        self.SOF_ans_cls = nn.Sequential(nn.Linear(768, 768 * 2),
                                         nn.GELU(),
                                         nn.LayerNorm(768 * 2),
                                         nn.Linear(768 * 2, self.nb_answer))
        # language generator
        if explainable:
            self.IOF_exp_generator = ExpGenerator(num_roi, nb_vocab, lang_dir, max_len=num_step)
            self.SOF_exp_generator = ExpGenerator(num_roi, nb_vocab, lang_dir, max_len=num_step)
            self.exp_var_feature = GaussianEncoder(768, 768)
            self.IOF_ans_cls = nn.Sequential(nn.Linear(768 * 2, 768 * 2),
                                             nn.GELU(),
                                             nn.LayerNorm(768 * 2),
                                             nn.Linear(768 * 2, self.nb_answer))
            self.SOF_ans_cls = nn.Sequential(nn.Linear(768 * 2, 768 * 2),
                                             nn.GELU(),
                                             nn.LayerNorm(768 * 2),
                                             nn.Linear(768 * 2, self.nb_answer))
            self.exp_norm = nn.LayerNorm(768)
            self.pre_exp_norm = nn.LayerNorm(768)

    def forward(self, img, box, cg, text_input, token_type, attention_mask, exp=None, valid_mask=None, ans=None, structure_gate=None):
        visual_IOF_mask = torch.add(cg, torch.eye(cg.shape[1]).to(cg.device)).to(img.device)
        visual_SOF_mask = (1.0 - cg).to(img.device)
        visual_object_mask = torch.ones(len(img), self.num_roi).to(img.device)

        IOF_feat_seq, IOF_pooled_output, IOF_hie_visn_feats = self.bert_encoder(input_ids=text_input, token_type_ids=token_type,
                                                                    attention_mask=attention_mask,
                                                                    visual_feats=(img, box),
                                                                    visual_attention_mask=visual_IOF_mask)

        SOF_feat_seq, SOF_pooled_output, SOF_hie_visn_feats = self.bert_encoder(input_ids=text_input, token_type_ids=token_type,
                                                                    attention_mask=attention_mask,
                                                                    visual_feats=(img, box),
                                                                    visual_attention_mask=visual_SOF_mask)
        IOF_que_feat = IOF_feat_seq[0]
        SOF_que_feat = SOF_feat_seq[0]
        IOF_visual_feat = IOF_feat_seq[1]
        SOF_visual_feat = SOF_feat_seq[1]
        IOF_cls_feat = IOF_pooled_output
        SOF_cls_feat = SOF_pooled_output

        if self.explainable:
            concat_mask = torch.cat(((1 - attention_mask).float().unsqueeze(1) * (-1e6),
                                     (1 - visual_object_mask).float().unsqueeze(1) * (-1e6)), dim=-1)
            IOF_mm_features = torch.cat((IOF_que_feat, IOF_visual_feat), dim=1)
            SOF_mm_features = torch.cat((SOF_que_feat, SOF_visual_feat), dim=1)
            IOF_pred_pro, IOF_pred_output, IOF_structure_gates, IOF_exp_feature, IOF_pred_exp_feature = self.IOF_exp_generator(exp, IOF_visual_feat, IOF_mm_features, concat_mask)
            SOF_pred_pro, SOF_pred_output, SOF_structure_gates, SOF_exp_feature, SOF_pred_exp_feature = self.SOF_exp_generator(exp, SOF_visual_feat, SOF_mm_features, concat_mask)

            if IOF_exp_feature is not None and SOF_exp_feature is not None:
                exp_feature = self.exp_norm(IOF_exp_feature + SOF_exp_feature)
                p_mean, p_var = self.exp_var_feature(exp_feature)
            else:
                p_mean, p_var = None, None
            pred_exp_feature = self.pre_exp_norm(IOF_pred_exp_feature + SOF_pred_exp_feature)
            q_mean, q_var, q_val = self.exp_var_feature(pred_exp_feature, sampling=4)
            IOF_ans_feat = torch.concat((IOF_cls_feat.unsqueeze(1).expand(-1, q_val.shape[1], -1), q_val), -1)
            SOF_ans_feat = torch.concat((SOF_cls_feat.unsqueeze(1).expand(-1, q_val.shape[1], -1), q_val), -1)
            IOF_output_ans_final = self.IOF_ans_cls(IOF_ans_feat)
            SOF_output_ans_final = self.SOF_ans_cls(SOF_ans_feat)
            pred_pro = torch.softmax(IOF_pred_pro + SOF_pred_pro, dim=-1)
            output_ans_final = torch.softmax(IOF_output_ans_final + SOF_output_ans_final, dim=-1)
        else:
            IOF_pred_pro, IOF_structure_gates, IOF_exp_feature, IOF_pred_exp_feature = None, None, None, None
            SOF_pred_pro, SOF_structure_gates, SOF_exp_feature, SOF_pred_exp_feature = None, None, None, None
            q_mean, q_var, p_mean, p_var = None, None, None, None
            IOF_output_ans_final = self.IOF_ans_cls(IOF_cls_feat)
            SOF_output_ans_final = self.SOF_ans_cls(SOF_cls_feat)
            output_ans_final = torch.softmax(IOF_output_ans_final + SOF_output_ans_final, dim=-1)

        if self.training:
            if output_ans_final.ndim > 2:
                ans = ans.unsqueeze(1)
            if self.explainable:
                IOF_ans_loss = F.cross_entropy(IOF_output_ans_final.view(-1, IOF_output_ans_final.size(-1)), ans.repeat(1, IOF_output_ans_final.size(1)).view(-1))
                IOF_exp_loss = exp_generative_loss(IOF_pred_pro, exp, valid_mask)
                IOF_structure_loss = structure_bce(IOF_structure_gates, structure_gate)
                SOF_ans_loss = F.cross_entropy(SOF_output_ans_final.view(-1, SOF_output_ans_final.size(-1)), ans.repeat(1, SOF_output_ans_final.size(1)).view(-1))
                SOF_exp_loss = exp_generative_loss(SOF_pred_pro, exp, valid_mask)
                SOF_structure_loss = structure_bce(SOF_structure_gates, structure_gate)
                ans_loss = F.cross_entropy(output_ans_final.view(-1, output_ans_final.size(-1)), ans.repeat(1, output_ans_final.size(1)).view(-1))
                exp_loss = exp_generative_loss(pred_pro, exp, valid_mask)
                kl_div = kl_div_gaussian(q_mean, q_var, p_mean, p_var)
                IOF_loss = IOF_ans_loss + IOF_exp_loss + IOF_structure_loss
                SOF_loss = SOF_ans_loss + SOF_exp_loss + SOF_structure_loss
                f_loss = ans_loss + exp_loss + kl_div
                loss = IOF_loss + SOF_loss + f_loss
            else:
                IOF_ans_loss = F.cross_entropy(IOF_output_ans_final, ans)
                SOF_ans_loss = F.cross_entropy(SOF_output_ans_final, ans)
                ans_loss = F.cross_entropy(output_ans_final, ans)
                loss = (ans_loss + IOF_ans_loss + SOF_ans_loss) / 3
            return loss.unsqueeze(0), output_ans_final.squeeze(1), pred_pro
        else:
            return output_ans_final.squeeze(1), pred_pro

# IOF
class FCGC_IOF(nn.Module):
    def __init__(self, num_roi=36, nb_answer=2000, nb_vocab=2000, num_step=18, lang_dir=None, args=None, explainable=True):
        super(FCGC_IOF, self).__init__()
        self.nb_vocab = nb_vocab
        self.num_roi = num_roi
        self.nb_answer = nb_answer
        self.num_step = num_step
        self.img_size = 2048
        self.hidden_size = 768
        self.num_head = 4
        self.explainable = explainable
        self.args = args
        base_model = LXRTEncoder(max_seq_length=18)
        base_model.load("lxrt/model")

        self.bert_encoder = base_model.model
        self.bert_encoder.mode = 'lxr'
        self.IOF_ans_cls = nn.Sequential(nn.Linear(768, 768 * 2),
                                         nn.GELU(),
                                         nn.LayerNorm(768 * 2),
                                         nn.Linear(768 * 2, self.nb_answer))
        # language generator
        if explainable:
            self.IOF_exp_generator = ExpGenerator(num_roi, nb_vocab, lang_dir, max_len=num_step)
            self.exp_var_feature = GaussianEncoder(768, 768)
            self.IOF_ans_cls = nn.Sequential(nn.Linear(768 * 2, 768 * 2),
                                             nn.GELU(),
                                             nn.LayerNorm(768 * 2),
                                             nn.Linear(768 * 2, self.nb_answer))

    def forward(self, img, box, cg, text_input, token_type, attention_mask, exp=None, valid_mask=None, ans=None, structure_gate=None):
        visual_IOF_mask = torch.add(cg, torch.eye(cg.shape[1]).to(cg.device)).to(img.device)
        visual_object_mask = torch.ones(len(img), self.num_roi).to(img.device)
        IOF_feat_seq, IOF_pooled_output, IOF_hie_visn_feats = self.bert_encoder(input_ids=text_input, token_type_ids=token_type,
                                                                    attention_mask=attention_mask,
                                                                    visual_feats=(img, box),
                                                                    visual_attention_mask=visual_IOF_mask)

        IOF_que_feat = IOF_feat_seq[0]
        IOF_visual_feat = IOF_feat_seq[1]
        IOF_cls_feat = IOF_pooled_output

        if self.explainable:
            concat_mask = torch.cat(((1 - attention_mask).float().unsqueeze(1) * (-1e6),
                                     (1 - visual_object_mask).float().unsqueeze(1) * (-1e6)), dim=-1)
            IOF_mm_features = torch.cat((IOF_que_feat, IOF_visual_feat), dim=1)
            IOF_pred_pro, IOF_pred_output, IOF_structure_gates, IOF_exp_feature, IOF_pred_exp_feature = self.IOF_exp_generator(exp, IOF_visual_feat, IOF_mm_features, concat_mask)

            if IOF_exp_feature is not None:
                p_mean, p_var = self.exp_var_feature(IOF_exp_feature)
            else:
                p_mean, p_var = None, None
            q_mean, q_var, q_val = self.exp_var_feature(IOF_pred_exp_feature, sampling=4)
            IOF_ans_feat = torch.concat((IOF_cls_feat.unsqueeze(1).expand(-1, q_val.shape[1], -1), q_val), -1)
            IOF_output_ans_final = self.IOF_ans_cls(IOF_ans_feat)
        else:
            IOF_pred_pro, IOF_structure_gates, IOF_exp_feature, IOF_pred_exp_feature = None, None, None, None
            q_mean, q_var, p_mean, p_var = None, None, None, None
            IOF_output_ans_final = self.IOF_ans_cls(IOF_cls_feat)

        if self.training:
            if IOF_output_ans_final.ndim > 2:
                ans = ans.unsqueeze(1)
            if self.explainable:
                IOF_ans_loss = F.cross_entropy(IOF_output_ans_final.view(-1, IOF_output_ans_final.size(-1)), ans.repeat(1, IOF_output_ans_final.size(1)).view(-1))
                IOF_exp_loss = self.args.alpha * exp_generative_loss(IOF_pred_pro, exp, valid_mask)
                IOF_structure_loss = self.args.beta * structure_bce(IOF_structure_gates, structure_gate)
                kl_div = kl_div_gaussian(q_mean, q_var, p_mean, p_var)
                IOF_loss = IOF_ans_loss + IOF_exp_loss + IOF_structure_loss
                loss = IOF_loss + kl_div
            else:
                IOF_ans_loss = F.cross_entropy(IOF_output_ans_final, ans)
                loss = IOF_ans_loss
            return loss.unsqueeze(0), IOF_output_ans_final.squeeze(1), IOF_pred_pro
        else:
            return IOF_output_ans_final.squeeze(1), IOF_pred_pro

# SOF
class FCGC_SOF(nn.Module):
    def __init__(self, num_roi=36, nb_answer=2000, nb_vocab=2000, num_step=18, lang_dir=None, args=None, explainable=True):
        super(FCGC_SOF, self).__init__()
        self.nb_vocab = nb_vocab
        self.num_roi = num_roi
        self.nb_answer = nb_answer
        self.num_step = num_step
        self.img_size = 2048
        self.hidden_size = 768
        self.num_head = 4
        self.explainable = explainable
        self.args = args
        base_model = LXRTEncoder(max_seq_length=18)
        base_model.load("lxrt/model")

        self.bert_encoder = base_model.model
        self.bert_encoder.mode = 'lxr'
        self.SOF_ans_cls = nn.Sequential(nn.Linear(768, 768 * 2),
                                         nn.GELU(),
                                         nn.LayerNorm(768 * 2),
                                         nn.Linear(768 * 2, self.nb_answer))
        # language generator
        if explainable:
            self.SOF_exp_generator = ExpGenerator(num_roi, nb_vocab, lang_dir, max_len=num_step)
            self.exp_var_feature = GaussianEncoder(768, 768)
            self.SOF_ans_cls = nn.Sequential(nn.Linear(768 * 2, 768 * 2),
                                             nn.GELU(),
                                             nn.LayerNorm(768 * 2),
                                             nn.Linear(768 * 2, self.nb_answer))

    def forward(self, img, box, cg, text_input, token_type, attention_mask, exp=None, valid_mask=None, ans=None, structure_gate=None):
        visual_SOF_mask = (1.0 - cg).to(img.device)
        visual_object_mask = torch.ones(len(img), self.num_roi).to(img.device)

        SOF_feat_seq, SOF_pooled_output, SOF_hie_visn_feats = self.bert_encoder(input_ids=text_input, token_type_ids=token_type,
                                                                    attention_mask=attention_mask,
                                                                    visual_feats=(img, box),
                                                                    visual_attention_mask=visual_SOF_mask)

        SOF_que_feat = SOF_feat_seq[0]
        SOF_visual_feat = SOF_feat_seq[1]
        SOF_cls_feat = SOF_pooled_output
        if self.explainable:
            concat_mask = torch.cat(((1 - attention_mask).float().unsqueeze(1) * (-1e6),
                                     (1 - visual_object_mask).float().unsqueeze(1) * (-1e6)), dim=-1)
            SOF_mm_features = torch.cat((SOF_que_feat, SOF_visual_feat), dim=1)
            SOF_pred_pro, SOF_pred_output, SOF_structure_gates, SOF_exp_feature, SOF_pred_exp_feature = self.SOF_exp_generator(exp, SOF_visual_feat, SOF_mm_features, concat_mask)
            if SOF_exp_feature is not None:
                p_mean, p_var = self.exp_var_feature(SOF_exp_feature)
            else:
                p_mean, p_var = None, None
            q_mean, q_var, q_val = self.exp_var_feature(SOF_pred_exp_feature, sampling=4)

            SOF_ans_feat = torch.concat((SOF_cls_feat.unsqueeze(1).expand(-1, q_val.shape[1], -1), q_val), -1)
            SOF_output_ans_final = self.SOF_ans_cls(SOF_ans_feat)
        else:
            SOF_pred_pro, SOF_structure_gates, SOF_exp_feature, SOF_pred_exp_feature = None, None, None, None
            q_mean, q_var, p_mean, p_var = None, None, None, None
            SOF_output_ans_final = self.SOF_ans_cls(SOF_cls_feat)

        if self.training:
            if SOF_output_ans_final.ndim > 2:
                ans = ans.unsqueeze(1)
            if self.explainable:
                SOF_ans_loss = F.cross_entropy(SOF_output_ans_final.view(-1, SOF_output_ans_final.size(-1)), ans.repeat(1, SOF_output_ans_final.size(1)).view(-1))
                SOF_exp_loss = self.args.alpha * exp_generative_loss(SOF_pred_pro, exp, valid_mask)
                SOF_structure_loss = self.args.beta * structure_bce(SOF_structure_gates, structure_gate)
                kl_div = kl_div_gaussian(q_mean, q_var, p_mean, p_var)
                SOF_loss = SOF_ans_loss + SOF_exp_loss + SOF_structure_loss
                loss = SOF_loss + kl_div
            else:
                SOF_ans_loss = F.cross_entropy(SOF_output_ans_final, ans)
                loss = SOF_ans_loss
            return loss.unsqueeze(0), SOF_output_ans_final.squeeze(1), SOF_pred_pro
        else:
            return SOF_output_ans_final.squeeze(1), SOF_pred_pro