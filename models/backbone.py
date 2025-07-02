import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.resnet import resnet50

class ResEncoder(nn.Module):
    def __init__(self, modal='video_encoder', pretrained=False):
        super(ResEncoder, self).__init__()
        resnet = resnet50()
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3 # [:3]
        self.layer4 = resnet.layer4

        self.conv_256 = nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.bn_256 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        if pretrained:
            self.load_model(modal)

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.shape
        x = x.view(-1, c, h, w)

        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)

        e1 = self.layer1(x_)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        
        feat = F.relu(self.bn_256(self.conv_256(e3)))
        # x = x.view(batch_size, num_frames, x.shape[1],  x.shape[2],  x.shape[3])
        return feat
    
    def load_model(self):
        ckpt_path = 'pretrained/resnet.pt'
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if 'module.flow_net' in key:
                key = key.split('module.flow_net.')[-1]
                if key in self.state_dict():
                    state_dict[key] = value
        self.load_state_dict(state_dict, strict=True)
    
class Prediction(nn.Module):
    def __init__(self, input_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Prediction, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, n_hidden_1),
            nn.LayerNorm(n_hidden_1),
            nn.Dropout(p=0.25, inplace=False),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            nn.Dropout(p=0.25, inplace=False),
            nn.Linear(n_hidden_2, out_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        return x