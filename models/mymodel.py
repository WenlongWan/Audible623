import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.blocks import Similarity_matrix, TransEncoder
from models.backbone import ResEncoder, Prediction

class MyModel(nn.Module):
    def __init__(self, num_frames, flo_dim = 3):
        super(MyModel, self).__init__()
        self.num_frames = num_frames

        self.video_net = ResEncoder()
        self.flow_net = ResEncoder()
        self.d2f_net = ResEncoder()

        self.sim_decoder = Similarity_Decoder()
        self.disent = Disentangler(768)
        self.from_scratch_layers = ['sim_decoder']

        self.v2f = Self_Attn(256, 512)
        self.v2df = Self_Attn(256, 512)

        self.load_model()
        # self.set_require_grad()
        
    def forward(self, video, diff, d2f):
        b, n, c, h, w = video.shape
        input = video.view(-1, c, h, w)
        x = self.video_net(input)
        f1 = self.flow_net(diff[:, :, :3])      # [b,f,256,7,7]
        f2 = self.flow_net(diff[:, :, 3:])      # [b,f,256,7,7
        df1 = self.d2f_net(d2f[:, :, :3])
        df2 = self.d2f_net(d2f[:, :, 3:])
                           
        f = torch.cat((f1, f2), dim = 1)
        df = torch.cat((df1, df2), dim = 1)
        v2f_attn, _ = self.v2f(x, f)
        v2df_attn, _ = self.v2df(x, df)
        x_fuse = torch.cat((x, v2f_attn, v2df_attn), dim = 1)    # [b,f,256]

        out = self.sim_decoder(x_fuse.view(b, n, -1, 7, 7))     # [b,f,2]
        # sideout = self.video_decoder(e0, e1, e2, x_fuse.view(b * n, -1, 7, 7))

        m1, m2, m = self.disent(x_fuse)
        out_mlt = self.sim_decoder(m.view(b, n, -1, 7, 7).repeat(1, 1, 768, 1, 1)) 
        return out, out_mlt, m1.view(b, n, -1), m2.view(b, n, -1), m # [b*n, 1, 112, 112]  
    
    def set_require_grad(self):
        for param in self.video_net.parameters():
            param.requires_grad = True
        for param in self.flow_net.parameters():
            param.requires_grad = True
        for param in self.d2f_net.parameters():
            param.requires_grad = True
        for param in self.sim_decoder.parameters():
            param.requires_grad = True
        for param in self.v2f.parameters():
            param.requires_grad = True
        for param in self.v2df.parameters():
            param.requires_grad = False
        for param in self.disent.parameters():
            param.requires_grad = False
        for param in self.sim_decoder.parameters():
            param.requires_grad = False

    def load_model(self):
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if 'module' in key:
                key = key.split('module.')[-1]
                if key in self.state_dict():
                    state_dict[key] = value
        self.load_state_dict(state_dict, strict=False)

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for name, value in self.named_parameters():
            module_name = name.split('.')[0]
            if 'weight' in name and value.requires_grad:
                if module_name in self.from_scratch_layers:
                    groups[2].append(value)
                else:
                    groups[0].append(value)
            if 'bias' in name and value.requires_grad:
                if module_name in self.from_scratch_layers:
                    groups[3].append(value)
                else:
                    groups[1].append(value)
        return groups
    
class Similarity_Decoder(nn.Module):
    def __init__(self):
        super(Similarity_Decoder, self).__init__()
        self.num_frames = 64

        self.conv3D = nn.Conv3d(in_channels=768,
                                out_channels=512,
                                kernel_size=3,
                                padding=(3, 1, 1),
                                dilation=(3, 1, 1))
        self.bn1 = nn.BatchNorm3d(512)
        self.SpatialPooling = nn.MaxPool3d(kernel_size=(1, 7, 7))

        self.sims = Similarity_matrix()
        self.conv3x3 = nn.Conv2d(in_channels=4,         # [512, 768, 3, 3, 3]
                                 out_channels=32,
                                 kernel_size=3,
                                 padding=1)

        self.bn2 = nn.BatchNorm2d(32)

        self.dropout1 = nn.Dropout(0.25)
        self.input_projection = nn.Linear(self.num_frames * 32, 512)  # 线性投射层
        self.ln1 = nn.LayerNorm(512)

        self.transEncoder = TransEncoder(d_model=512, n_head=4, dropout=0.2, dim_ff=512, num_layers=1,
                                         num_frames=self.num_frames)
        self.cls = Prediction(512, 512, 256, 2)  
        self.load_model()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv3D(x)))        # -> [b,512,f,h,w]

        x = self.SpatialPooling(x)                  # -> [b,512,f,1,1]
        x = x.squeeze(3).squeeze(3)                 # -> [b,512,f]
        x = x.transpose(1, 2)                       # -> [b,f,512]
        # -------- similarity matrix ---------
        x_sims = F.relu(self.sims(x, x, x))         # -> [b,4,f,f]
        x = F.relu(self.bn2(self.conv3x3(x_sims)))  # [b,32,f,f]
        x = self.dropout1(x)
        
        x = x.permute(0, 2, 3, 1)                   # [b,f,f,32]
        # --------- transformer encoder ------
        x = x.flatten(start_dim=2)                  # [b,f,32*f]
        x = F.relu(self.input_projection(x))        # [b,f,512]
        x = self.ln1(x)
        x = x.transpose(0, 1)                       # [f,b,512]
        x = self.transEncoder(x)  
        x = x.transpose(0, 1)                       # [b,f,512]
        x = self.cls(x)
        return x
    
class Disentangler(nn.Module):
    def __init__(self, cin = 512, imsize = 112):
        super(Disentangler, self).__init__()

        self.activation_head = nn.Conv2d(cin, 1, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)
        self.upsample = False
        self.imsize = 112

    def forward(self, x):
        N, C, H, W = x.size()
        m = torch.sigmoid(self.bn_head(self.activation_head(x)))

        m_ = m.reshape(N, 1, H * W)                          # [N, 1, H*W]
        x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()   # [N, H*W, C]
        m1 = torch.matmul(m_, x) / (H * W)                # [N, 1, C]
        m2 = torch.matmul(1 - m_, x) / (H * W)            # [N, 1, C]
        return m1.reshape(x.size(0), -1), m2.reshape(x.size(0), -1), m

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim, key_in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = key_in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)

        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x, v):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        # query = video, key + value = flow
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(v).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) / math.sqrt(self.chanel_in//8) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        return out,attention

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.num_frames = 64
        # [b 1 7 7]
        self.conv3D = nn.Conv3d(in_channels=768,
                                out_channels=512,
                                kernel_size=3,
                                padding=(3, 1, 1),
                                dilation=(3, 1, 1))
        self.bn1 = nn.BatchNorm3d(512)
        self.SpatialPooling = nn.MaxPool3d(kernel_size=(1, 7, 7))

        self.sims = Similarity_matrix(input_size=512)
        self.conv3x3 = nn.Conv2d(in_channels=4,         # [512, 768, 3, 3, 3]
                                 out_channels=32,
                                 kernel_size=3,
                                 padding=1)
        
        self.cls = Prediction(512, 512, 256, 2)  

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv3D(x)))        # -> [b,512,f,h,w]

        x = self.SpatialPooling(x)                  # -> [b,512,f,1,1]
        x = x.squeeze(3).squeeze(3)                 # -> [b,512,f]
        x = x.transpose(1, 2)                       # -> [b,f,512]
        # -------- similarity matrix ---------
        x_sims = F.relu(self.sims(x, x, x))         # -> [b,4,f,f]
        x = F.relu(self.bn2(self.conv3x3(x_sims)))  # [b,32,f,f]
        x = self.dropout1(x)
        
        x = x.permute(0, 2, 3, 1)                   # [b,f,f,32]
        # --------- transformer encoder ------
        x = x.flatten(start_dim=2)                  # [b,f,32*f]
        x = F.relu(self.input_projection(x))        # [b,f,512]
        x = self.ln1(x)
        x = self.cls(x)
        return x
    

if __name__ == '__main__':
    model = MyModel(64)
    video = torch.Tensor(1, 64, 3, 112, 112)
    flow = torch.Tensor(1, 64, 6, 112, 112)
    model(video, flow)