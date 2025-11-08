#-------------------Model.py-------------------------
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Resnet50FPN(nn.Module):
    def __init__(self):
        super(Resnet50FPN, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        children = list(self.resnet.children())
        self.conv1 = nn.Sequential(*children[:4])
        self.conv2 = children[4]
        self.conv3 = children[5]
        self.conv4 = children[6]
    def forward(self, im_data):
        feat = OrderedDict()
        feat_map = self.conv1(im_data)
        feat_map = self.conv2(feat_map)
        feat_map3 = self.conv3(feat_map)
        feat_map4 = self.conv4(feat_map3)
        feat['map3'] = feat_map3
        feat['map4'] = feat_map4
        return feat

class SiameseSimilarity(nn.Module):
    
    """Compute similarity maps between image features and a set of exemplar features.
    Inputs:
      - img_feat: [B, C, H, W]
      - exemplar_feats: [E, C, h, w]
    Output:
      - sim_maps: [B, E, H, W]  (one similarity map per exemplar)"""
      
    def __init__(self, in_channels, embed_dim=128):
        super(SiameseSimilarity, self).__init__()
        
        # small encoder to project to embed_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim), nn.ReLU(inplace=True),)

    def forward(self, img_feat, exemplar_feats):
        """
        img_feat: [B, C, H, W]
        exemplar_feats: [E, C, h, w]
        returns sim_maps: [B, E, H, W]
        """
        B = img_feat.shape[0]
        device = img_feat.device
        # image embedding: [B, D, H, W]
        img_emb = self.encoder(img_feat)                      # [B, D, H, W]
        print("img_emb shape: ", img_emb.shape)
        
        # exemplar embedding: [E, D, h, w]
        ex_emb = self.encoder(exemplar_feats)                 # [E, D, h, w]
        print("ex_emb shape: ", ex_emb.shape)
        # global pool exemplar embedding -> [E, D, 1, 1]
        ex_vec = F.adaptive_avg_pool2d(ex_emb, (1,1))         # [E, D, 1, 1]
        print("ex_vec shape: ", ex_vec.shape)

        # add batch dim for broadcasting
        ex_vec = ex_vec.unsqueeze(0)                          # [1, E, D, 1, 1]
        print("ex_vec shape: ", ex_vec.shape)
        img_emb_exp = img_emb.unsqueeze(1)                    # [B, 1, D, H, W]
        print("img_emb_exp shape: ", img_emb_exp.shape)

        # compute dot-product similarity across D -> [B, E, H, W]
        sim = (img_emb_exp * ex_vec).sum(dim=2)
        print("sim shape: ", sim.shape)

        # normalize to get cosine-like similarity
        img_norm = torch.norm(img_emb_exp, p=2, dim=2, keepdim=True).clamp(min=1e-6)
        print("img_norm shape: ", img_norm.shape)
        ex_norm = torch.norm(ex_vec, p=2, dim=2, keepdim=True).clamp(min=1e-6)
        print("ex_norm shape: ", ex_norm.shape)
        denom = (img_norm * ex_norm).squeeze(2)
        print("denom shape: ", denom.shape)
        sim = sim / denom
        print("sim shape: ", sim.shape)

        return sim   # [B, E, H, W]



class CountRegressor(nn.Module):
    def __init__(self, input_channels,pool='mean'):
        super(CountRegressor, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

    def forward(self, im):
        num_sample =  im.shape[0]
        if num_sample == 1:
            output = self.regressor(im)
            if self.pool == 'mean':
                output = torch.mean(output, dim=(0),keepdim=True)  
                return output
            elif self.pool == 'max':
                output, _ = torch.max(output, 0,keepdim=True)
                return output
        else:
            for i in range(0,num_sample):
                output = self.regressor(im[i])
                if self.pool == 'mean':
                    output = torch.mean(output, dim=(0),keepdim=True)
                elif self.pool == 'max':
                    output, _ = torch.max(output, 0,keepdim=True)
                if i == 0:
                    Output = output
                else:
                    Output = torch.cat((Output,output),dim=0)
            return Output


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):                
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def weights_xavier_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            
            
