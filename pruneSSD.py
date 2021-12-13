import torch
from torchvision import datasets
from torchvision.transforms import transforms
from torch import nn
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
from torch.utils.tensorboard import SummaryWriter
import sys
from tqdm import tqdm
from torchsummary import summary
from vision.nn.mobilenet_v2 import MobileNetV2, InvertedResidual

import numpy as np
from mythop import clever_format, profile

class pruneSSD():
    def __init__(self,model,savePath,pruningRate):
        self.model = model
        self.savePath = savePath
        self.pruningRate = pruningRate
    def getCfg(self,wantPrune=False):
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = self.model
        block_count = 0
        # model = MobileNetV2(width_mult=1.0, use_batch_norm=True)
        layer_id = 1
        cfg = []
        cfg_mask = []
        skip = False
        nextBreak = False
        for m in model.modules():
            if isinstance(m, nn.ModuleList) and nextBreak:
                skip = True
                break
            if isinstance(m, InvertedResidual):
                block_count+=1
                
            if isinstance(m, InvertedResidual) and block_count == 14:
                skip = True
                continue
            
            if isinstance(m, InvertedResidual)and block_count <= 21:
                skip = False
                if block_count == 21:
                    nextBreak = True
                    
                continue
            if isinstance(m, InvertedResidual) and block_count != 14:
                skip = False
                continue
            
            if isinstance(m, nn.Conv2d) and not skip:
                
                out_channels = m.weight.data.shape[0]

                
                if layer_id > 3 and m.groups!=1:

                    prune_prob_stage = self.pruningRate
                    weight_copy = m.weight.data.abs().clone().cpu().numpy()
                    L1_norm = np.sum(weight_copy, axis=(1,2,3))
                    num_keep = int(out_channels * (1 - prune_prob_stage))
                    arg_max = np.argsort(L1_norm)
                    arg_max_rev = arg_max[::-1][:num_keep]
                    mask = torch.zeros(out_channels)
                    mask[arg_max_rev.tolist()] = 1
                    cfg_mask.append(mask)
                    cfg.append(num_keep)
                    layer_id += 1
                    continue
                layer_id += 1
        
        assert len(cfg) == 19
        if not wantPrune:
            return cfg
        else:
            return cfg_mask
    def prune(self,newmodel):
        cfg_mask = self.getCfg(wantPrune=True)
        newmodel = newmodel
        # newmodel = MobileNetV2(width_mult=1.0, use_batch_norm=True,cfg=cfg)
        # newmodel = MobileNetV2(num_classes=10)
        newmodel.cuda()
        model = self.model
        start_mask = torch.ones(3)
        layer_id_in_cfg = 0
        conv_count = 1
        prune=False
        finishPruningFilter=False
        blockCheck = False
        block_count = 0
        baseNetCheck = True
        nextBreak = False
        for [m0, m1] in zip(model.modules(), newmodel.modules()):
            
            if isinstance(m0, InvertedResidual):
                if (block_count == 0):
                    block_count+=1
                    continue
                blockCheck = True
                block_count+=1
                
                # continue

            if isinstance(m0, InvertedResidual) and block_count == 14 and not nextBreak:
                baseNetCheck = False
                continue
            if isinstance(m0, InvertedResidual) and block_count != 14 and not nextBreak:
                baseNetCheck = True
            if isinstance(m0, InvertedResidual) and block_count <= 21 and not nextBreak:
                baseNetCheck = True
                if block_count == 21:
                    nextBreak = True
            if isinstance(m0, nn.ModuleList) and nextBreak:
                baseNetCheck = False


            if isinstance(m0, nn.Conv2d):

                if conv_count != 1 and m0.kernel_size==(1, 1) and blockCheck and baseNetCheck:
                    mask = cfg_mask[layer_id_in_cfg]
                    inputMask = torch.arange(m0.in_channels)
                    temp = m0.weight.data[:, inputMask.tolist(), :, :]
                    idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                    if idx.size == 1:
                        idx = np.resize(idx, (1,))
                    w = temp[idx.tolist(), :, :, :].clone()
                    # w = m0.weight.data[idx.tolist(), :, :, :].clone()
                    m1.weight.data = w.clone()
                    conv_count += 1
                    prune = True
                    blockCheck = False
                    continue


                if m0.groups!=1 and block_count > 1 and baseNetCheck:
                    mask = cfg_mask[layer_id_in_cfg]
                    idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                    if idx.size == 1:
                        idx = np.resize(idx, (1,))
                    w = m0.weight.data[idx.tolist(), :, :, :].clone()
                    m1.weight.data = w.clone()
                    m1.groups = w.clone().shape[0]
                    conv_count += 1
                    finishPruningFilter = True
                    prune = True
                    continue

                # if conv_count == 1:
                #     m1.weight.data = m0.weight.data.clone()
                #     conv_count += 1
                #     continue

                if conv_count != 1 and m0.kernel_size==(1, 1) and finishPruningFilter and baseNetCheck:
                    mask = cfg_mask[layer_id_in_cfg]
                    idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                    if idx.size == 1:
                        idx = np.resize(idx, (1,))
                    w = m0.weight.data[:, idx.tolist(), :, :].clone()
                    m1.weight.data = w.clone()

                    conv_count += 1
                    layer_id_in_cfg += 1
                    finishPruningFilter=False
                    continue
                conv_count+=1
                m1.weight.data = m0.weight.data.clone()
            elif isinstance(m0, nn.BatchNorm2d):
                assert isinstance(m1, nn.BatchNorm2d), "There should not be bn layer here."
                if prune and baseNetCheck:
                    mask = cfg_mask[layer_id_in_cfg]
                    idx = np.squeeze(np.argwhere(np.asarray(mask.cpu().numpy())))
                    if idx.size == 1:
                        idx = np.resize(idx, (1,))
                    m1.weight.data = m0.weight.data[idx.tolist()].clone()
                    m1.bias.data = m0.bias.data[idx.tolist()].clone()
                    m1.running_mean = m0.running_mean[idx.tolist()].clone()
                    m1.running_var = m0.running_var[idx.tolist()].clone()
                    # m1.running_mean.copy_(m0.running_mean[idx.tolist()].clone())
                    # m1.running_var.copy_(m0.running_var[idx.tolist()].clone())
                    prune = False
                    continue
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
            elif isinstance(m0, nn.Linear) and baseNetCheck:
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()

        model.cuda()
        newmodel.cuda()
        # print(model)
        print(newmodel)
        # input = torch.randn(32,3,32,32).cuda()
        # print(newmodel(input))
        input = torch.randn(32, 3, 3, 3).cuda()

        flops, params = profile(model, inputs=(input,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        flopsnew, paramsnew = profile(newmodel, inputs=(input,), verbose=False)
        flopsnew, paramsnew = clever_format([flopsnew, paramsnew], "%.3f")
        print("flops:{}->{}, params: {}->{}".format(flops, flopsnew, params, paramsnew))
        torch.save({'state_dict': newmodel.state_dict()}, self.savePath)
        # torch.save({'state_dict': newmodel.state_dict()}, "pruned/L1PrunedMobilenet-SSD.pth")

        # savePath = "./pruned/retrained.pth"
        # r = retrain(newmodel,savePath)
        # r.train()
        print("Done!")
        return newmodel