
import nltk
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
import lib.SMART_globvars as gv

class Schema_head_predictior(nn.Module):
    def __init__(self, args):
        super(Schema_head_predictior, self).__init__()
        
        self.args = args
        weights = ResNet50_Weights.DEFAULT
        im_backbone = resnet50(weights=weights)
        self.preprocess = weights.transforms()
        self.out_dim, self.max_val = 64, 101
        self.q_dim, self.h_sz = 768, 128

        im_feat_size = im_backbone.fc.weight.shape[1]
        modules = list(im_backbone.children())[:-1]
        self.im_cnn = nn.Sequential(*modules)
        
        self.qv_fusion = nn.Sequential(
            nn.Linear(self.out_dim * 2, self.out_dim),  # for flava its *2.
            nn.ReLU(),
            nn.Linear(self.out_dim, self.out_dim),
            nn.ReLU(),
        )
        self.q_MLP = nn.Sequential(
            nn.Linear(self.q_dim, self.h_sz), nn.ReLU(), nn.Linear(self.h_sz, self.out_dim))
        self.i_MLP = nn.Sequential(
            nn.Linear(im_feat_size, self.h_sz), nn.ReLU(), nn.Linear(self.h_sz, self.out_dim))
        self.qvo_fusion = nn.Sequential(nn.Linear(self.out_dim, self.max_val))
        
    def encode_image(self, im):
        if self.args.mode == 'schema_head_train':
            x = self.im_cnn(im).squeeze()
        else:
            with torch.no_grad():
                x = self.im_cnn(im).squeeze()
                                
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        return x
            
    def encode_text(self, sentence):
        word_feats = gv.word_embed(sentence)
        return word_feats

    def forward(self, im, question):
        im_feat = self.encode_image(im)
        im_feat = self.i_MLP(im_feat)
        
        q_feat = self.encode_text(question)
        q_feat = self.q_MLP(q_feat.mean(1))
        
        qv_feat = self.qv_fusion(torch.cat([im_feat, q_feat], dim=1))
        qv_feat = qv_feat.unsqueeze(1)
        
        qvo_feat = self.qvo_fusion(qv_feat).squeeze()
        return qvo_feat