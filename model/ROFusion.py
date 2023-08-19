import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

NbTxAntenna = 12
NbRxAntenna = 16
NbVirtualAntenna = NbTxAntenna * NbRxAntenna

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


class Bottleneck(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None,expansion=4):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(expansion*planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = F.relu(residual + out)
        return out

class MIMO_PreEncoder(nn.Module):
    def __init__(self, in_layer,out_layer,kernel_size=(1,12),dilation=(1,16),use_bn = False):
        super(MIMO_PreEncoder, self).__init__()
        self.use_bn = use_bn

        self.conv = nn.Conv2d(in_layer, out_layer, kernel_size, 
                              stride=(1, 1), padding=0,dilation=dilation, bias= (not use_bn) )

        self.bn = nn.BatchNorm2d(out_layer)
        self.padding = int(NbVirtualAntenna/2)

    def forward(self,x):
        width = x.shape[-1]
        x = torch.cat([x[...,-self.padding:],x,x[...,:self.padding]],axis=3)
        x = self.conv(x)
        x = x[...,int(x.shape[-1]/2-width/2):int(x.shape[-1]/2+width/2)]

        if self.use_bn:
            x = self.bn(x)
        return x

class FPN_BackBone(nn.Module):

    def __init__(self, num_block,channels,block_expansion,mimo_layer,use_bn=True):
        super(FPN_BackBone, self).__init__()

        self.block_expansion = block_expansion
        self.use_bn = use_bn

        # pre processing block to reorganize MIMO channels
        self.pre_enc = MIMO_PreEncoder(32,mimo_layer,
                                        kernel_size=(1,NbTxAntenna),
                                        dilation=(1,NbRxAntenna),
                                        use_bn = True)

        self.in_planes = mimo_layer

        self.conv = conv3x3(self.in_planes, self.in_planes)
        self.bn = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        # Residuall blocks
        self.block1 = self._make_layer(Bottleneck, planes=channels[0], num_blocks=num_block[0])
        self.block2 = self._make_layer(Bottleneck, planes=channels[1], num_blocks=num_block[1])
        self.block3 = self._make_layer(Bottleneck, planes=channels[2], num_blocks=num_block[2])
        self.block4 = self._make_layer(Bottleneck, planes=channels[3], num_blocks=num_block[3])

    def forward(self, x):

        x = self.pre_enc(x)
        # 4x192x512x256
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        # Backbone
        features = {}
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        
        features['x0'] = x
        features['x1'] = x1
        features['x2'] = x2
        features['x3'] = x3
        features['x4'] = x4

        return features


    def _make_layer(self, block, planes, num_blocks):
        if self.use_bn:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * self.block_expansion,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(planes * self.block_expansion)
            )
        else:
            downsample = nn.Conv2d(self.in_planes, planes * self.block_expansion,
                                   kernel_size=1, stride=2, bias=True)

        layers = []
        layers.append(block(self.in_planes, planes, stride=2, downsample=downsample,expansion=self.block_expansion))
        self.in_planes = planes * self.block_expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, stride=1,expansion=self.block_expansion))
            self.in_planes = planes * self.block_expansion
        return nn.Sequential(*layers)

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            out = self.downsample(out)

        return out

class RangeAngle_Decoder(nn.Module):
    def __init__(self, ):
        super(RangeAngle_Decoder, self).__init__()

        # Top-down layers
        self.deconv4 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))
        
        self.conv_block4 = BasicBlock(48,128)
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=(2,1), padding=1, output_padding=(1,0))
        self.conv_block3 = BasicBlock(192,256)

        self.L3  = nn.Conv2d(192, 224, kernel_size=1, stride=1,padding=0)
        self.L2  = nn.Conv2d(160, 224, kernel_size=1, stride=1,padding=0)

    def forward(self,features):

        T4 = features['x4'].transpose(1, 3) 
        T3 = self.L3(features['x3']).transpose(1, 3)
        T2 = self.L2(features['x2']).transpose(1, 3)

        S4 = torch.cat((self.deconv4(T4),T3),axis=1)
        S4 = self.conv_block4(S4)
        
        S43 = torch.cat((self.deconv3(S4),T2),axis=1)
        out = self.conv_block3(S43)

        return out

class Image_feature_extractor(nn.Module):
    '''Extract Image feature'''

    def __init__(self):
        super(Image_feature_extractor,self).__init__()
        
        self.model = models.resnet18(pretrained=True)
        # self.model.load_state_dict(torch.load('./model/resnet18-5c106cde.pth'))
        self.model.eval()

        self.up = nn.Upsample(size = [540,960], mode='bilinear', align_corners=True)


    def forward(self,x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x) 

        x = self.up(x)
        
        return x

class Fusion(nn.Module):
    def __init__(self,num_points):
        super(Fusion, self).__init__()
        self.conv1 = torch.nn.Conv1d(256, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)

        self.e_conv1 = torch.nn.Conv1d(256, 64, 1)
        self.e_conv2 = torch.nn.Conv1d(64, 128, 1)

        self.conv5 = torch.nn.Conv1d(256, 512, 1)
        self.conv6 = torch.nn.Conv1d(512, 1024, 1)

        self.num_points = num_points
        self.ap1 = torch.nn.MaxPool1d(self.num_points)

    def forward(self, x, emb):
        # x: torch.Size([bs, 256, num]) RA
        # emb: torch.Size([bs, 32, num]) Img
        # bs = 1
        x = F.relu(self.conv1(x))  
        emb = F.relu(self.e_conv1(emb)) 
        pointfeat_1 = torch.cat((x, emb), dim=1) 

        x = F.relu(self.conv2(x)) 
        emb = F.relu(self.e_conv2(emb)) 
        pointfeat_2 = torch.cat((x, emb), dim=1) 

        x = F.relu(self.conv5(pointfeat_2)) 
        x = F.relu(self.conv6(x)) 

        ########### global feature
        ap_x = self.ap1(x).repeat(1,1,self.num_points)

        return torch.cat([pointfeat_1, pointfeat_2, ap_x], 1) 

class Detection_Header_center(nn.Module):
    def __init__(self,num_points):
        super(Detection_Header_center, self).__init__()
        self.num_points = num_points

        self.conv1 = nn.Conv1d(1408, 144, 1)
        self.bn1 = nn.BatchNorm1d(144)
        self.conv2 = nn.Conv1d(144, 96, 1)
        self.bn2 = nn.BatchNorm1d(96)

        self.center_rcls = nn.Conv1d(96, 11, 1)
        self.center_acls = nn.Conv1d(96, 5, 1)
        self.reg_r = nn.Conv1d(96, 1, 1)
        self.reg_a = nn.Conv1d(96, 1, 1)

        self.linear1 = nn.Linear(self.num_points,self.num_points)
        self.linear2 = nn.Linear(self.num_points,self.num_points)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)

        ## point-wise classification
        center_rcls_bin = self.center_rcls(x) # center:torch.size([1,7,num of points])
        center_acls_bin = self.center_acls(x) # center:torch.size([1,5,num of points])

        ## point-wise regression
        reg_r = self.reg_r(x) 
        reg_r = self.linear1(reg_r)
        reg_a = self.reg_a(x) 
        reg_a = self.linear2(reg_a)

        return torch.cat([center_rcls_bin, center_acls_bin, reg_r, reg_a], dim=1).permute(0,2,1).reshape(1,-1,18)

class ROFusion(nn.Module):
    def __init__(self,mimo_layer,channels,blocks,regression_layer = 2):
        super(ROFusion, self).__init__()
        self.FPN = FPN_BackBone(num_block=blocks,channels=channels,block_expansion=4, mimo_layer = mimo_layer,use_bn = True)
        self.cnn = Image_feature_extractor()
        for params in self.cnn.parameters():
            params.requires_grad = False
        self.RA_decoder = RangeAngle_Decoder()

        self.fusion = Fusion(num_points = 50)

        self.detection_header = Detection_Header_center(num_points = 50)

    def forward(self,x):
        radar_RD = x[0] # radar_RD: torch.Size([bs, 32, 512, 256])
        image = x[1] # img:torch.Size([bs, 3, 540, 960])
        img_idx = x[2]
        ra_idx=x[3]

        ####################################### image feature
        image_feature = self.cnn(image) # image_feature: torch.Size([bs, di, 540, 960])
        bs, di, _, _ = image_feature.size()
        # emb = image_feature.reshape(di,bs,540,960).contiguous().view(di, -1) # emb: torch.Size([di, bs*540*960])
        emb = image_feature.transpose(0,1).contiguous().view(di, -1) # emb: torch.Size([di, bs*540*960])


        ####################################### radar feature
        features= self.FPN(radar_RD)
        RA = self.RA_decoder(features) # RA: torch.size([bs, 256, 128, 224])
        ra_bs, ra_di, _, _ = RA.size()
        # emb_RA = RA.reshape(ra_di, ra_bs, 128, 224).contiguous().view(ra_di, -1) # emb_RA: torch.Size([ra_di, ra_bs*128*224])
        emb_RA = RA.transpose(0,1).contiguous().view(ra_di, -1) # emb_RA: torch.Size([ra_di, ra_bs*128*224])

        ####################################### hybrid point feature
        img_idx = img_idx.repeat(di, 1) # img_idx: torch.Size([di, num])
        ra_idx = ra_idx.repeat(ra_di, 1) # ra_idx: torch.Size([ra_di, num]

        emb = torch.stack(torch.split(torch.gather(emb,1,img_idx),50,dim=1)) 
        emb_RA = torch.stack(torch.split(torch.gather(emb_RA,1,ra_idx),50,dim=1))

        ####################################### fusion
        RA_fusion = self.fusion(emb_RA, emb) # RA_fusion: torch.Size([1, 1408, num of target*50])

        ####################################### detection header
        out = self.detection_header(RA_fusion)  # out: torch.size([bs, 3, 128, 224]) ----> torch.size([1, num of target, 4])

        return out