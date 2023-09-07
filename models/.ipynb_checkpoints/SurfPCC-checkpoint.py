import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.init as init
import copy
import math
import numpy as np
import time

from models.pointops.functions import pointops
from models.utils import index_points, indexing_by_id
from pointnet2 import pointnet2_utils as pn2_utils
from compressai.entropy_models import EntropyBottleneck

from CDLoss import CDLoss

## DGCNN
class DGCNN_multi_knn_c5(nn.Module):
    def __init__(self, emb_dims=512, args=None):
        super(DGCNN_multi_knn_c5, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv1.weight, gain=1.0)
        self.conv2 = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv2.weight, gain=1.0)
        self.conv3 = nn.Conv2d(64*2, 128, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv3.weight, gain=1.0)
        self.conv4 = nn.Conv2d(128*2, 256, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv4.weight, gain=1.0)
        self.conv5 = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        init.xavier_normal_(self.conv5.weight, gain=1.0)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(emb_dims)
    def forward(self, x, if_relu_atlast = False):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x) # This sub model get the graph-based features for the following 2D convs
        # The x is similar with 2D image
        if self.args.if_bn == True: x = F.relu(self.bn1(self.conv1(x)))
        else: x = F.relu(self.conv1(x))
        x1 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x1)
        if self.args.if_bn == True: x = F.relu(self.bn2(self.conv2(x))) 
        else: x = F.relu(self.conv2(x))
        x2 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x2)
        if self.args.if_bn == True: x = F.relu(self.bn3(self.conv3(x))) 
        else: x = F.relu(self.conv3(x))
        x3 = x.max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x3)
        if self.args.if_bn == True: x = F.relu(self.bn4(self.conv4(x))) 
        else: x = F.relu(self.conv4(x))
        x4 = x.max(dim=-1, keepdim=False)[0]
        x = torch.cat((x1, x2, x3, x4), dim=1).unsqueeze(3)
        if if_relu_atlast == False:
            return torch.tanh(self.conv5(x)).view(batch_size, -1, num_points)
        x = F.relu(self.conv5(x)).view(batch_size, -1, num_points)
        return x
#### The knn function used in graph_feature ####
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx
#### The edge_feature used in DGCNN ####
def get_graph_feature(x, k=4):
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size()
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2,1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    return feature

def convert_str_2_list(str_):
    words = str_.split(' ')
    trt = [int(x) for x in words]
    return trt

#MLP#
class MLPNet_relu(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0, if_bn = False):
        super().__init__()
        list_layers = []
        last = nch_input
        for i, outp in enumerate(nch_layers):
            weights = torch.nn.Conv1d(last, outp, 1)
            init.xavier_normal_(weights.weight, gain=1.0)
            list_layers.append(weights)
            if if_bn==True:
                list_layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
            list_layers.append(torch.nn.ReLU())
            last = outp
            
        self.layers = torch.nn.Sequential(*list_layers)
    def forward(self, inp):
        out = self.layers(inp)
        return out

    
#### Encoder ####
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        
        self.args = args # the args
        self.emb_dims = args.emb_dims # the dim of the embedded feture
        self.emb_nn_sparse = DGCNN_multi_knn_c5(emb_dims=args.emb_dims, args=self.args) 
        self.emb_nn_sparse_local = DGCNN_multi_knn_c5(emb_dims=args.emb_dims, args=self.args) 
        self.mlp_fitting_str = self.args.mlp_fitting_str
        self.mlp_fitting = convert_str_2_list(self.mlp_fitting_str) # the channels of the layers in the MLP
        self.fitting_mlp = MLPNet_relu(2*self.emb_dims, self.mlp_fitting, b_shared=True, if_bn =False).layers
        self.reconstruct_out_p = torch.nn.Conv1d(self.mlp_fitting[-1], self.args.dim, 1)
        self.feature_merge = torch.nn.Sequential(self.fitting_mlp, self.reconstruct_out_p)
    
    def forward(self, xyz):
        xyz = xyz.permute(0,2,1)  
        B,N,C = xyz.shape
        # Get the anchor point
        downsample_num = self.args.training_up_ratio
        xyz_fps_id = pn2_utils.furthest_point_sample(xyz.contiguous(), round(N/downsample_num)).long()
        points_sparse = xyz.index_select(1, xyz_fps_id.squeeze(0))
        _, num_point,_ = points_sparse.shape  # [B,N/rate,C]
        points_sparse = points_sparse.contiguous()
        
        # Get local patch and extract F^l
        local_neighbour_indexes = pointops.knnquery_heap(downsample_num+1,xyz.contiguous(),points_sparse.contiguous()).long() 
        local_neighbour_indexes = local_neighbour_indexes[:,:,1:]
        points_sparse_local_patch_form = index_points(xyz.permute(0,2,1), local_neighbour_indexes) 
        points_sparse_local_patch_form = points_sparse_local_patch_form.permute(0,2,3,1)
        points_sparse_local_patch_form = points_sparse_local_patch_form - points_sparse.view(B, num_point,1,3)
        points_sparse_local_patch_form = points_sparse_local_patch_form.view(B*num_point, self.args.training_up_ratio, 3)
        points_sparse_local_embedding = self.emb_nn_sparse_local(points_sparse_local_patch_form.transpose(1,2))
        points_sparse_local_embedding= torch.max(points_sparse_local_embedding,dim=-1,keepdim=False)[0].view(B, num_point,-1).permute(0,2,1) 

        # Get anchor patch and extract F^a
        local_neighbour_indexes = pointops.knnquery_heap(self.args.neighbor_k+1,points_sparse,points_sparse).long() # bs, point_num, 18
        local_neighbour_indexes = local_neighbour_indexes[:,:,1:self.args.neighbor_k+1]
        points_sparse_anchor_patch_form = index_points(points_sparse.permute(0,2,1), local_neighbour_indexes) # bs,3,points_sparse_num,17
        points_sparse_anchor_patch_form = points_sparse_anchor_patch_form.permute(0,2,3,1)
        points_in_local_patch_form = points_sparse_anchor_patch_form - points_sparse.view(B, num_point,1,3)
        points_in_local_patch_form = points_in_local_patch_form.view(B*num_point, self.args.neighbor_k, 3)
        sparse_embedding = self.emb_nn_sparse(points_in_local_patch_form.transpose(1,2))  # B*num_point, self.emb_dims, self.neighbor_k
        sparse_embedding = torch.max(sparse_embedding,dim=-1,keepdim=False)[0].view(B, num_point,-1).permute(0,2,1) 

        # Feature compression
        sparse_embedding = torch.cat((sparse_embedding, points_sparse_local_embedding),dim=1)
        feat = self.feature_merge(sparse_embedding)
        feat = feat
        points_sparse = points_sparse.permute(0,2,1)
        return points_sparse, feat


#### Decoder ####
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        
        self.args = args 
        self.emb_dims = args.emb_dims # the dim of the embedded feture
        self.up_ratio = -1 # the upsampling factor
        self.over_sampling_up_ratio = -1 # the scale of over-sampling
        self.mlp_fitting_str = self.args.mlp_fitting_str
        self.mlp_fitting = convert_str_2_list(self.mlp_fitting_str) 
        self.emb_nn_sparse = DGCNN_multi_knn_c5(emb_dims=self.emb_dims, args=self.args)
        self.fitting_mlp = MLPNet_relu(self.emb_dims+(self.args.pe_out_L*4+2)+self.args.dim, self.mlp_fitting).layers
        self.reconstruct_out_p = torch.nn.Conv1d(self.mlp_fitting[-1], 3, 1)
        init.xavier_normal_(self.reconstruct_out_p.weight, gain=1.0)
        self.convert_feature_to_point_2to3 = torch.nn.Sequential(self.fitting_mlp, self.reconstruct_out_p)   # the Neural Field Fuction (MLP) 

    def forward(self, points_sparse, latent_feats):
        points_sparse = points_sparse.permute(0,2,1)
        points_sparse =points_sparse.contiguous()
        
        thisbatchsize, num_point, C = points_sparse.shape
        self.up_ratio = self.args.training_up_ratio
        self.over_sampling_up_ratio = int(self.up_ratio * self.args.over_sampling_scale)

        # sample from 2d plane
        uv_sampling_coors_1 = uniform_random_sample(thisbatchsize, num_point, self.over_sampling_up_ratio-4)
        uv_sampling_coors_2 = fix_sample(thisbatchsize, num_point, 4)
        uv_sampling_coors_ = torch.cat((uv_sampling_coors_1, uv_sampling_coors_2), dim=2) 
        uv_sampling_coors = copy.deepcopy(uv_sampling_coors_.detach())
        uv_sampling_coors = uv_sampling_coors.detach().contiguous()   # thisbatchsize, num_point, self.over_sampling_up_ratio, 2
        uv_sampling_coors.requires_grad=True

        # Get anchor patch and extract F^a
        local_neighbour_indexes = pointops.knnquery_heap(self.args.neighbor_k+1,points_sparse.contiguous(),points_sparse.contiguous()).long() 
        local_neighbour_indexes = local_neighbour_indexes[:,:,1:]
        points_sparse_local_patch_form = index_points(points_sparse.permute(0,2,1), local_neighbour_indexes) # 
        points_sparse_local_patch_form = points_sparse_local_patch_form.permute(0,2,3,1)
        points_in_local_patch_form = points_sparse_local_patch_form - points_sparse.view(thisbatchsize, num_point,1,3)
        points_in_local_patch_form = points_in_local_patch_form.view(thisbatchsize*num_point, self.args.neighbor_k, 3)
        sparse_embedding = self.emb_nn_sparse(points_in_local_patch_form.transpose(1,2)) 
        sparse_embedding = torch.max(sparse_embedding,dim=-1,keepdim=False)[0].view(thisbatchsize, num_point,-1).permute(0,2,1) 
        
        # feature decompress
        sparse_embedding = sparse_embedding.permute(0,2,1)  # thisbatchsize, self.args.num_point, self.emb_dims*2
        sparse_embedding = torch.cat((sparse_embedding, latent_feats.permute(0,2,1)),dim=2)

        uv_sampling_coors_id_in_sparse = torch.arange(num_point).view(1,-1,1).long()
        uv_sampling_coors_id_in_sparse = uv_sampling_coors_id_in_sparse.expand(thisbatchsize,-1,self.over_sampling_up_ratio).reshape(thisbatchsize,-1,1)
        
        # mapping 2d plane to 3d point cloud
        upsampled_p = self.convert_uv_to_xyz(uv_sampling_coors.reshape(thisbatchsize,-1,2), uv_sampling_coors_id_in_sparse, sparse_embedding, points_sparse) 

        # sampling from 3d point
        upsampled_p_fps_id = pn2_utils.furthest_point_sample(upsampled_p.contiguous(), self.up_ratio * num_point)
        querying_points_3d = pn2_utils.gather_operation(upsampled_p.permute(0, 2, 1).contiguous(), upsampled_p_fps_id)
        querying_points_3d = querying_points_3d.permute(0,2,1).contiguous()

        return upsampled_p, querying_points_3d
    

    def convert_uv_to_xyz(self, uv_coor, uv_coor_idx_in_sparse, sparse_embedding, points_sparse):
        # uv_coor                | should be in size : thisbatchsize, All2dQueryPointNum, 2
        # uv_coor_idx_in_sparse  | should be in size : thisbatchsize, All2dQueryPointNum, 1
        # sparse_embedding       | should be in size : thisbatchsize, sparse_point_num, embedding_dim
        # points_sparse          | should be in size : thisbatchsize, sparse_point_num, 3
        thisbatchsize = uv_coor.size()[0]
        All2dQueryPointNum = uv_coor.size()[1]
        coding_dim = 4*self.args.pe_out_L + 2
        uv_encoded = position_encoding(uv_coor.reshape(-1,2).contiguous(), self.args.pe_out_L).view(thisbatchsize, All2dQueryPointNum, coding_dim).permute(0,2,1) # bs, coding_dim, All2dQueryPointNum
        indexed_sparse_feature = indexing_by_id(sparse_embedding, uv_coor_idx_in_sparse)  
        # bs, All2dQueryPointNum, 1, embedding_num 
        indexed_sparse_feature = indexed_sparse_feature.view(thisbatchsize, All2dQueryPointNum, -1).transpose(2,1)  
        # bs, embedding_num, All2dQueryPointNum
        coding_with_feature = torch.cat((indexed_sparse_feature, uv_encoded), dim=1)
        out_p = self.convert_feature_to_point_2to3(coding_with_feature).view(thisbatchsize, -1, All2dQueryPointNum).permute(0,2,1)
        indexed_center_points = indexing_by_id(points_sparse, uv_coor_idx_in_sparse).view(thisbatchsize, All2dQueryPointNum, 3)
        out_p = out_p + indexed_center_points
        return out_p
    
    
#### Convert a string to num_list ####      
def convert_str_2_list(str_):
    words = str_.split(' ')
    trt = [int(x) for x in words]
    return trt
#### Compute the position code for uv or xyz. ####
def position_encoding(input_uv, pe_out_L):
    ## The input_uv should be with shape (-1, X)
    ## The returned tensor should be with shape (-1, X+2*X*L)
    ## X = 2/3 if the input is uv/xyz.
    trt = input_uv
    for i in range(pe_out_L):
        trt = torch.cat((trt, torch.sin(input_uv*(2**i)*(3.14159265))) , dim=-1 )
        trt = torch.cat((trt, torch.cos(input_uv*(2**i)*(3.14159265))) , dim=-1 )
    return trt
#### Sample uv by a fixed manner. #### 
def fix_sample(thisbatchsize, num_point, up_ratio, if_random=False):
    if up_ratio == 4:
        one_point_fixed = [ [ [0,0] for i in range(2)] for j in range(2) ]
        for i in range(2):
            for j in range(2):
                one_point_fixed[i][j][0] = (i/1) *2 -1
                one_point_fixed[i][j][1] = (j/1) *2 -1
        one_point_fixed = np.array(one_point_fixed).reshape(4,2)
        one_batch_uv2d_random_fixed = np.expand_dims(one_point_fixed,axis=0)
        one_batch_uv2d_random_fixed = np.expand_dims(one_batch_uv2d_random_fixed,axis=0)
        one_batch_uv2d_random_fixed = np.tile(one_batch_uv2d_random_fixed,[thisbatchsize, num_point, 1,1])
        one_batch_uv2d_random_fixed_tensor = torch.from_numpy(one_batch_uv2d_random_fixed).cuda().float()
        return one_batch_uv2d_random_fixed_tensor
    else:
        print('This up_ratio ('+str(up_ratio)+') is not supported now. You can try the random mode!')
        exit()
#### Sample uv uniformly in (-1,1). #### 
def uniform_random_sample(thisbatchsize, num_point, up_ratio):
    # return : randomly and uniformly sampled uv_coors   |   Its shape should be : thisbatchsize, num_point, up_ratio, 2
    res_ = torch.rand(thisbatchsize*num_point, 4*up_ratio, 3)*2-1
    res_ = res_.cuda()
    res_[:,:,2:]*=0
    furthest_point_index = pn2_utils.furthest_point_sample(res_,up_ratio)
    uniform_res_ = pn2_utils.gather_operation(res_.permute(0, 2, 1).contiguous(), furthest_point_index)
    uniform_res_ = uniform_res_.permute(0,2,1).contiguous()
    uniform_res_ = uniform_res_[:,:,:2].view(thisbatchsize, num_point, up_ratio, 2)
    return uniform_res_


######## SurfPCC: MAIN NETWORK ########
#### The main network ####
class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        
        self.args = args # the args
        self.encoder = Encoder(self.args)
        self.feats_eblock = EntropyBottleneck(args.dim)
        self.decoder = Decoder(self.args)
        self.loss = CDLoss(self.args)
        
        self.latent_xyzs_analysis = nn.Sequential(
            nn.Conv1d(3, args.hidden_dim, 1),
            nn.GroupNorm(args.ngroups, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(args.hidden_dim, args.dim, 1))
        
        self.xyzs_eblock = EntropyBottleneck(args.dim)
        
        self.latent_xyzs_synthesis = nn.Sequential(
            nn.Conv1d(args.dim, args.hidden_dim, 1),
            nn.GroupNorm(args.ngroups, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(args.hidden_dim, 3, 1))

        self.mse_loss = nn.MSELoss()
    
    def forward(self, xyz):
        # xyz [B,C,N]
        xyz = xyz[:,0:3,:]
        points_num = xyz.shape[0] * xyz.shape[2]
        
        points_sparse, points_sparse_feats = self.encoder(xyz) # points_sparse [B,C,N/rate]
        
        # feats bpp calculation
        latent_feats_hat, latent_feats_likelihoods = self.feats_eblock(points_sparse_feats) # [B,C,N/rate]
        feats_size = (torch.log(latent_feats_likelihoods).sum()) / (-math.log(2))
        feats_bpp = feats_size / points_num

        # anchor bpp calculation
        gt_latent_xyzs = points_sparse  #(B,C,N)
        analyzed_latent_xyzs = self.latent_xyzs_analysis(points_sparse)
        analyzed_latent_xyzs_hat, analyzed_latent_xyzs_likelihoods = self.xyzs_eblock(analyzed_latent_xyzs)
        pred_latent_xyzs = self.latent_xyzs_synthesis(analyzed_latent_xyzs_hat)
        xyzs_size = (torch.log(analyzed_latent_xyzs_likelihoods).sum()) / (-math.log(2))
        xyzs_bpp = xyzs_size / points_num
        
        upsampled_p, querying_points_3d = self.decoder(pred_latent_xyzs, latent_feats_hat)
        

        # CD loss
        total_loss, loss_stages  = self.loss(upsampled_p, xyz.permute(0,2,1))
        
        # MSE loss
        latent_xyzs_loss = self.mse_loss(gt_latent_xyzs, pred_latent_xyzs)
        total_loss = total_loss + latent_xyzs_loss * self.args.latent_xyzs_coe
        loss_stages['latent_xyzs_loss'] = latent_xyzs_loss.item()
        
        # bpp loss
        bpp =  feats_bpp + xyzs_bpp
        bpp_loss = bpp * self.args.bpp_lambda
        total_loss = total_loss + bpp_loss
        loss_stages['bpp_loss'] = bpp.item()
        
        return upsampled_p, querying_points_3d, total_loss, loss_stages, bpp, points_sparse, pred_latent_xyzs
         