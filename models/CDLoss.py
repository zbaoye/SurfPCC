import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
chamfer_dist = chamfer_3DDist()



class CDLoss(nn.Module):
    def __init__(self, args):
        super(CDLoss, self).__init__()
        self.args = args
        
    def loss_on_cd(self, deformation_p, p1):
        thisbatchsize = deformation_p.size()[0]
        dist1, dist2, _, _ = chamfer_dist(deformation_p, p1)
        output = (torch.sum(dist1) + torch.sum(dist2))*0.5
        cd = output/thisbatchsize
        return cd
    
    def forward(self, gen_points_batch, train_points_dense_batch):
         
        loss_stages={}
        
        loss_cd = self.loss_on_cd(gen_points_batch, train_points_dense_batch)
        loss = loss_cd * self.args.weight_cd
        loss_stages['cd_loss'] = loss_cd.item()   
            
        return loss, loss_stages
