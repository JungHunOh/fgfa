import torch.nn as nn
import pretrained_resnet
import random
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np
from util import conv, predict_flow, deconv, smooth_l1_loss, crop_like, correlate, warp
import torchvision

class EmbedNet(nn.Module):
    def __init__(self):
        super(EmbedNet, self).__init__()
        self.em_conv1 = nn.Conv2d(in_channels=1024 ,out_channels=512, kernel_size=1, stride =1) # in ch  TODO
        self.em_conv2 = nn.Conv2d(in_channels=512 ,out_channels=512, kernel_size=3, stride =1, padding=1) # in ch 
        self.em_conv3 = nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=1, stride =1) # in ch 
        
        for l in [self.em_conv1, self.em_conv2, self.em_conv3, ]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.zeros_(l.bias)

    def forward(self,x):
        x = F.relu(self.em_conv1(x))
        x = F.relu(self.em_conv2(x))
        x = self.em_conv3(x)
        return x

def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, 1:target.size(2)+1, 1:target.size(3)+1]

class FlowNet(nn.Module):
    def __init__(self):
        super(FlowNet, self).__init__()

        self.flow_conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        self.conv6_1 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)

        self.Convolution1 = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)
        self.Convolution2 = nn.Conv2d(1026, 2, kernel_size=3, stride=1, padding=1)
        self.Convolution3 = nn.Conv2d(770, 2, kernel_size=3, stride=1, padding=1)
        self.Convolution4 = nn.Conv2d(386, 2, kernel_size=3, stride=1, padding=1)
        self.Convolution5 = nn.Conv2d(194, 2, kernel_size=3, stride=1, padding=1)

        self.deconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2)
        self.deconv4 = nn.ConvTranspose2d(1026, 256, kernel_size=4, stride=2)
        self.deconv3 = nn.ConvTranspose2d(770, 128, kernel_size=4, stride=2)
        self.deconv2 = nn.ConvTranspose2d(386, 64, kernel_size=4, stride=2)

        self.upsample_flow6to5 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2)
        self.upsample_flow5to4 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2)
        self.upsample_flow4to3 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2)
        self.upsample_flow3to2 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2)

        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.avgpool = nn.AvgPool2d(2, stride=2, ceil_mode=True)

    def forward(self, x):
        x = self.avgpool(x)
        conv1 = self.flow_conv1(x)
        relu1 = self.relu(conv1)
        conv2 = self.conv2(relu1)
        relu2 = self.relu(conv2)
        conv3 = self.conv3(relu2)
        relu3 = self.relu(conv3)
        conv3_1 = self.conv3_1(relu3)
        relu4 = self.relu(conv3_1)
        conv4 = self.conv4(relu4)
        relu5 = self.relu(conv4)
        conv4_1 = self.conv4_1(relu5)
        relu6 = self.relu(conv4_1)
        conv5 = self.conv5(relu6)
        relu7 = self.relu(conv5)
        conv5_1 = self.conv5_1(relu7)
        relu8 = self.relu(conv5_1)
        conv6 = self.conv6(relu8) 
        relu9 = self.relu(conv6)
        conv6_1 = self.conv6_1(relu9)
        relu10 = self.relu(conv6_1)

        Convolution1 = self.Convolution1(relu10)
        upsample_flow6to5 = self.upsample_flow6to5(Convolution1)
        deconv5 = self.deconv5(relu10)
        crop_upsampled_flow6to5 = crop_like(upsample_flow6to5, relu8)
        crop_deconv5 = crop_like(deconv5, relu8)
        relu11 = self.relu(crop_deconv5)

        concat2 = torch.cat((relu8, relu11, crop_upsampled_flow6to5), dim=1)
        Convolution2 = self.Convolution2(concat2)
        upsample_flow5to4 = self.upsample_flow5to4(Convolution2)
        deconv4 = self.deconv4(concat2)
        crop_upsampled_flow5to4 = crop_like(upsample_flow5to4, relu6)
        crop_deconv4 = crop_like(deconv4, relu6)
        relu12 = self.relu(crop_deconv4)

        concat3 = torch.cat((relu6, relu12, crop_upsampled_flow5to4), dim=1)
        Convolution3 = self.Convolution3(concat3)
        upsample_flow4to3 = self.upsample_flow4to3(Convolution3)
        deconv3 = self.deconv3(concat3)
        crop_upsampled_flow4to3 = crop_like(upsample_flow4to3, relu4)
        crop_deconv3 = crop_like(deconv3, relu4)
        relu13 = self.relu(crop_deconv3)

        concat4 = torch.cat((relu4, relu13, crop_upsampled_flow4to3), dim=1)
        Convolution4 = self.Convolution4(concat4)
        upsample_flow3to2 = self.upsample_flow3to2(Convolution4)
        deconv2 = self.deconv2(concat4)
        crop_upsampled_flow3to2 = crop_like(upsample_flow3to2, relu2)
        crop_deconv2 = crop_like(deconv2, relu2)
        relu14 = self.relu(crop_deconv2)

        concat5 = torch.cat((relu2, relu14, crop_upsampled_flow3to2), dim=1)
        concat5 = self.avgpool(concat5)
        Convolution5 = self.Convolution5(concat5)


        return Convolution5 * 2.5


class FGFA_Backbone(nn.Module):
    
    def __init__(self):
        
        super(FGFA_Backbone, self).__init__()

        self.backbone = pretrained_resnet.FGFA_Resnet()
        self.embednet = EmbedNet()
        self.checkpoint = torch.load('flownet.ckpt')
        self.flownet = FlowNet()
        self.flownet.load_state_dict(self.checkpoint['state_dict'])

    def compute_weight(self,embed_flow, embed_conv_feat):

        def l2normalization(tensor):
            norm = torch.norm(tensor, dim=1, keepdim=True) + 1e-10
            
            return tensor / norm

        embed_flow_norm = l2normalization(embed_flow)
        embed_conv_norm = l2normalization(embed_conv_feat)
        weight = torch.sum(embed_flow_norm * embed_conv_norm, dim=1, keepdim=True)
        return weight

    def forward(self, ref_images, guide_images):
        """
        ref_images: tensor([1, c, h, w])
        guided_images: list[tensor([c, h, w])]

        return: flow-guided aggregation feature map of which shape is tensor([b, c, h, w])
        """
        num_refs = len(guide_images)

        concat_imgs = torch.cat([ref_images, [guide_image.unsqeeze(0) for guide_image in guide_images]], dim=0)
        concat_feats = self.backbone(concat_imgs)
        
        img_cur, imgs_ref = torch.split(concat_imgs, (1, num_refs), dim=0)
        img_cur_copies = img_cur.repeat(num_refs, 1, 1, 1)
        concat_imgs_pair = torch.cat([img_cur_copies, imgs_ref], dim=1)

        flow = self.flownet(concat_imgs_pair)

        feats_cur, feats_refs = torch.split(concat_feats, (1, num_refs), dim=0)
        warped_feats_refs = warp(feats_refs, flow)

        concat_feats = torch.cat([feats_cur, warped_feats_refs], dim=0)
        concat_embed_feats = self.embednet(concat_feats)
        embed_cur, embed_refs = torch.split(concat_embed_feats, (1, num_refs), dim=0)

        unnormalized_weights = self.compute_weight(embed_refs, embed_cur)
        weights = F.softmax(unnormalized_weights, dim=0)

        feats = torch.sum(weights * warped_feats_refs, dim=0, keepdim=True)
        
        print(feats.shape)
        input()
        return feats

        ref_ids = []
        base_feats = [] # list[tensor[C_b, H_b, W_b]]
        
        batch_size = ref_images.shape[0]

        for guide_image in guide_images:
            
            base_feats.append(self.base(guide_image['image'].unsqueeze(0)))

            for bef in guide_image['bef']:
                if len(ref_ids) == batch_size:
                    break

                if bef not in ref_ids:
                    ref_ids.append(bef)

            for aft in guide_image['aft']:
                if len(ref_ids) == batch_size:
                    break

                if aft not in ref_ids:
                    ref_ids.append(aft)

        ref_ids = sorted(list(set(ref_ids)))

        base_feats_dict = [] 
        for j, ref_id  in enumerate(ref_ids):
            base_feat = dict()
            base_feat['bef'] = [] # list[tensor[1,C_b, H_b, W_b]]
            base_feat['aft'] = []
            base_feat['im_bef'] = [] # list[tensor[C, H, W]]
            base_feat['im_aft'] = []
            base_feat['ref'] = None
            for i, guide_image in enumerate(guide_images):

                for bef in guide_image['bef']:
                    if bef == ref_id:
                        base_feat['bef'].append(base_feats[i])
                        base_feat['im_bef'].append(guide_images[i]['image'])
                
                for aft in guide_image['aft']: 
                    if aft == ref_id:
                        base_feat['aft'].append(base_feats[i])
                        base_feat['im_aft'].append(guide_images[i]['image'])
                
                if ref_id == guide_image['ref']:
                    base_feat['ref'] = base_feats[i]
            
            if base_feat['ref'] is None:
                base_feat['ref'] = self.base(ref_images[j].unsqueeze(0))

            base_feats_dict.append(base_feat) 

        fgfa_features = [] 

        for i, base_feat in enumerate(base_feats_dict):
            
            len_bef = len(base_feat['bef'])
            len_aft = len(base_feat['aft'])

            # tensor[n, c_b, h_b, w_b]
            if len_bef > 0:
                base_bef_feats = torch.cat(base_feat['bef'], dim=0)
            else:
                base_bef_feats = []

            if len_aft > 0:
                base_aft_feats = torch.cat(base_feat['aft'], dim=0)
            else:
                base_aft_feats = []
        
            # list[tensor[1, 2c, h, w]]
            concat_bef = [torch.cat([ref_images[i].unsqueeze(0), im_bef.unsqueeze(0)], dim=1) for im_bef in base_feat['im_bef']]
            concat_aft = [torch.cat([ref_images[i].unsqueeze(0), im_aft.unsqueeze(0)], dim=1) for im_aft in base_feat['im_aft']]           
            concat_data = torch.cat(concat_bef + concat_aft, dim=0) # tensor[n_bef+n_aft, 2c, h, w]
        
            # tensor[n_bef + n_aft, 2, h_f, w_f]
            flows = self.flownet(concat_data)
            # tensor[n, c_b, h_b, w_b]
            if len_bef > 0:
                warp_bef = warp(base_bef_feats, flows[:len_bef])
            else:
                warp_bef = []

            if len_aft > 0:
                warp_aft = warp(base_aft_feats, flows[0-len_aft:])
            else:
                warp_aft = []
            
            if len_bef > 0 and len_aft > 0:
                concat_embed_data = torch.cat([base_feat['ref'], warp_bef, warp_aft], dim=0) # [1+n_bef+n_aft, c_b, h_b, w_b)

            elif len_bef == 0:
                concat_embed_data = torch.cat([base_feat['ref'], warp_aft], dim=0) 

            elif len_aft == 0:
                concat_embed_data = torch.cat([base_feat['ref'], warp_bef], dim=0) 

            embed_outputs = self.embednet(concat_embed_data)
        
            unnormalize_weights = [self.compute_weight(embed_output.unsqueeze(0), embed_outputs[0].unsqueeze(0)) for embed_output in embed_outputs[1:]]
            unnormalize_weights = torch.cat(unnormalize_weights, dim=0)

            weights = self.softmax(unnormalize_weights)

            # tile the channel dim of weights
            weights = [weight.repeat(1, 1024, 1, 1) for weight in weights]# Tensor? TODO

            #fgfa_feature = base_feat['ref']
            fgfa_feature = 0

            if len_bef > 0 and len_aft > 0:
                warp_feats = torch.cat([warp_bef, warp_aft], dim=0) # [1+n_bef+n_aft, c_b, h_b, w_b)

            elif len_bef == 0:
                warp_feats = warp_aft

            elif len_aft == 0:
                warp_feats = warp_bef
            
            for weight, warp_feat in zip(weights, warp_feats):
                fgfa_feature += weight * warp_feat.unsqueeze(0)
            
            fgfa_features.append(fgfa_feature)
        
        fgfa_feature = torch.cat(fgfa_features, dim=0) 

        #print('fgfa_feature: ',fgfa_feature, fgfa_feature.shape)
        #input()
        
        return fgfa_feature


