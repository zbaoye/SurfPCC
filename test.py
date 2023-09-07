import torch
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import importlib
import time
import math
import argparse

from models.utils import save_pcd, AverageMeter, str2bool
from dataset.dataset import CompressDataset
from metrics.density import get_density_metric
from metrics.F1Score import get_f1_score
from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
chamfer_dist = chamfer_3DDist()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from scipy.io import savemat


def make_dirs(save_dir):
    gt_patch_dir = os.path.join(save_dir, 'patch/gt')
    if not os.path.exists(gt_patch_dir):
        os.makedirs(gt_patch_dir)
    pred_patch_dir = os.path.join(save_dir, 'patch/pred')
    if not os.path.exists(pred_patch_dir):
        os.makedirs(pred_patch_dir)
    gt_merge_dir = os.path.join(save_dir, 'merge/gt')
    if not os.path.exists(gt_merge_dir):
        os.makedirs(gt_merge_dir)
    pred_merge_dir = os.path.join(save_dir, 'merge/pred')
    if not os.path.exists(pred_merge_dir):
        os.makedirs(pred_merge_dir)

    return gt_patch_dir, pred_patch_dir, gt_merge_dir, pred_merge_dir




def load_model(args, model_path):
    # load model
    
    # create the model
    MODEL = importlib.import_module(args.model)
    model = MODEL.get_model(args).cuda()
    
    model.load_state_dict(torch.load(model_path))
    # update entropy bottleneck
    model.feats_eblock.update(force=True)
    model.xyzs_eblock.update(force=True)
    model.eval()

    return model




def compress(args, model, xyzs):
    # input: (b, c, n)

    encode_start = time.time()
    
    normal = xyzs[:,3:6,:]
    xyz = xyzs[:,0:3,:]
    
    # encoder forward

    latent_xyzs, latent_feats = model.encoder(xyz, normal) # points_sparse [B,N/rate,C]
    feats_size = latent_feats.size()[2:]

    # compress latent feats
    latent_feats_str = model.feats_eblock.compress(latent_feats)
    
    analyzed_latent_xyzs = model.latent_xyzs_analysis(latent_xyzs)

    # decompress size
    xyzs_size = analyzed_latent_xyzs.size()[2:]
    latent_xyzs_str = model.xyzs_eblock.compress(analyzed_latent_xyzs)

    
    encode_time = time.time() - encode_start
    points_num = xyzs.shape[0] * xyzs.shape[2]
    feats_bpp = (sum(len(s) for s in latent_feats_str) * 8.0) / points_num
    xyzs_bpp = (sum(len(s) for s in latent_xyzs_str) * 8.0) / points_num
    
    actual_bpp = feats_bpp + xyzs_bpp
    
    return latent_xyzs_str, xyzs_size, latent_feats_str, feats_size, encode_time, actual_bpp, xyzs_bpp




def decompress(args, model, latent_xyzs_str, xyzs_size, latent_feats_str, feats_size):
    decode_start = time.time()
    # decompress latent xyzs
    
    analyzed_latent_xyzs_hat = model.xyzs_eblock.decompress(latent_xyzs_str, xyzs_size)

    latent_xyzs_hat = model.latent_xyzs_synthesis(analyzed_latent_xyzs_hat)

    latent_feats_hat = model.feats_eblock.decompress(latent_feats_str, feats_size)


    _, querying_points_3d = model.decoder(latent_xyzs_hat, latent_feats_hat)

    decode_time = time.time() - decode_start

    return querying_points_3d, decode_time, glued_points, latent_xyzs_hat



def test_xyzs(args):
    # load data

    test_dataset = CompressDataset(data_path=args.test_data_path, cube_size=args.test_cube_size)

    # test_dataset = CompressDataset(data_path=args.test_data_path, cube_size=args.test_cube_size)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size)
    # indicate the last patch number of each full point cloud
    pcd_last_patch_num = test_dataset.pcd_last_patch_num

    # set up folders for saving point clouds
    model_path = args.model_path
    experiment_id = model_path.split('/')[-3]
    save_dir = os.path.join(args.output_path, experiment_id, 'pcd')
    gt_patch_dir, pred_patch_dir, gt_merge_dir, pred_merge_dir = make_dirs(save_dir)

    # load model
    print('model_path',model_path)
    model = load_model(args, model_path)

    # metrics
    patch_bpp = AverageMeter()
    patch_chamfer_loss = AverageMeter()
    patch_psnr = AverageMeter()
    patch_psnr_glued = AverageMeter()
    patch_density_metric = AverageMeter()
    patch_encode_time = AverageMeter()
    patch_decode_time = AverageMeter()
    pcd_num = 0
    pcd_bpp = AverageMeter()
    pcd_xyzs_bpp = AverageMeter()
    pcd_chamfer_loss = AverageMeter()
    pcd_glued_chamfer_loss = AverageMeter()
    pcd_psnr = AverageMeter()
    pcd_psnr_glued = AverageMeter()
    pcd_density_metric = AverageMeter()

    # merge xyzs
    pcd_gt_patches = []
    pcd_pred_patches = []
    pcd_pred_patches_glued = []
    pcd_gt_normals = []

    # test

    for i, input_dict in enumerate(test_loader):
        # input: (b, n, c)
        input = input_dict['xyzs'].cuda()
        # normals : (b, n, c)
        gt_normals = input_dict['normals'].cuda()
        # (b, c, n)
        input = torch.cat((input, gt_normals), dim=2)
        input = input.permute(0, 2, 1).contiguous()
        gt_patches = input[:, :3, :].contiguous()

        # compress
        time1 = time.time()
        latent_xyzs_str, xyzs_size, latent_feats_str, feats_size, encode_time, \
        actual_bpp, xyzs_bpp = compress(args, model, input)
        time1 = time.time() - time1


        # update metrics
        patch_encode_time.update(encode_time)
        patch_bpp.update(actual_bpp)
        pcd_bpp.update(actual_bpp)
        pcd_xyzs_bpp.update(xyzs_bpp)

        # decompress
        time1 = time.time()
        pred_patches, decode_time, glued_points, latent_xyzs \
                = decompress(args, model, latent_xyzs_str, xyzs_size, latent_feats_str, feats_size)
        patch_decode_time.update(decode_time)
        time1 = time.time() - time1


        # calculate metrics
        # (b, 3, n) -> (b, n, 3)
        # print('gt_patches',gt_patches.shape)
        # print('pred_patches',pred_patches.shape)
        gt_patches = gt_patches.permute(0, 2, 1).contiguous()
        sparse_patch = latent_xyzs.permute(0, 2, 1).contiguous()
        # chamfer distance

        # scale patches to original size: (n, 3)
        original_gt_patches = test_dataset.scale_to_origin(gt_patches.detach().cpu(), i).squeeze(0).numpy()
        original_pred_patches = test_dataset.scale_to_origin(pred_patches.detach().cpu(), i).squeeze(0).numpy()
        glued_points_patches = test_dataset.scale_to_origin(glued_points.detach().cpu(), i).squeeze(0).numpy()
        sparse_patches = test_dataset.scale_to_origin(sparse_patch.detach().cpu(), i).squeeze(0).numpy()
        # save patches
        gt_normals = gt_normals.squeeze(0).detach().cpu().numpy()
        save_pcd(gt_patch_dir, '%06d_gt.ply' % i, original_gt_patches, gt_normals)
        save_pcd(pred_patch_dir, '%06d_rec.ply' % i, original_pred_patches)
        # merge patches
        pcd_gt_patches.append(original_gt_patches)
        pcd_pred_patches.append(original_pred_patches)
        pcd_pred_patches_glued.append(glued_points_patches)
        pcd_gt_normals.append(gt_normals)
        # generate the full point cloud
        if i == pcd_last_patch_num[pcd_num] - 1:
            gt_pcd = np.concatenate((pcd_gt_patches), axis=0)
            pred_pcd = np.concatenate((pcd_pred_patches), axis=0)
            pred_pcd_glued = np.concatenate((pcd_pred_patches_glued), axis=0)

            # save the full point cloud
            print("pcd:", pcd_num, "pcd bpp:", pcd_bpp.get_avg(), "pcd chamfer loss:", pcd_chamfer_loss.get_avg())
            save_pcd(gt_merge_dir, '%04d.ply' % pcd_num, gt_pcd, np.concatenate((pcd_gt_normals), axis=0))
            save_pcd(pred_merge_dir, '%04d_rec.ply' % pcd_num, pred_pcd)


            # reset
            pcd_num += 1
            pcd_bpp.reset()
            pcd_gt_patches.clear()
            pcd_pred_patches.clear()
            pcd_pred_patches_glued.clear()
            pcd_gt_normals.clear()
    print("avg patch bpp:", patch_bpp.get_avg())
    print("avg encode time:", patch_encode_time.get_avg())
    print("avg decode time:", patch_decode_time.get_avg())



def reset_model_args(test_args, model_args):
    for arg in vars(test_args):
        setattr(model_args, arg, getattr(test_args, arg))




def parse_test_args():
    parser = argparse.ArgumentParser(description='Test Arguments')

    # dataset
    parser.add_argument('--opt', default='', type=str, help='operation name')
    parser.add_argument('--model', default='', type=str, help='model name')
    parser.add_argument('--dataset', default='shapenet', type=str, help='shapenet or semantickitti')
    parser.add_argument('--model_path', default='path to ckpt', type=str, help='path to ckpt')
    parser.add_argument('--batch_size', default=1, type=int, help='the test batch_size must be 1')
    parser.add_argument('--bpp_lambda', default=1, type=float, help='bpp loss coefficient')

    parser.add_argument('--test_data_path', default='./data/shapenet/shapenet_test_cube_size_22.pkl', type=str, help='path to val dataset')
    parser.add_argument('--test_cube_size', default=22, type=int, help='cube size of val dataset')
    parser.add_argument('--output_path', default='./output', type=str, help='output path')
    
    parser.add_argument('--peak', default=None, type=float, help='peak value for PSNR calculation')

    parser.add_argument('--training_up_ratio', type=int, default=21,help='The Upsampling Ratio during training') 
    parser.add_argument('--testing_up_ratio', type=int, default=21, help='The Upsampling Ratio during testing')  
    parser.add_argument('--over_sampling_scale', type=float, default=4, help='The scale for over-sampling')

    parser.add_argument('--emb_dims', type=int, default=50, metavar='N',help='Dimension of embeddings')

    parser.add_argument('--pe_out_L', type=int, default=5, metavar='N',help='The parameter L in the position code')

    # for phase train
    parser.add_argument('--learning_rate', type=float, default=0.005)

    parser.add_argument('--if_bn', type=int, default=0, help='If using batch normalization')
    parser.add_argument('--neighbor_k', type=int, default=9, help='The number of neighbour points used in DGCNN')
    parser.add_argument('--mlp_fitting_str', type=str, default='256 128 64', metavar='None',help='mlp layers of the part surface fitting (default: None)')

    #loss terms weights
    parser.add_argument('--latent_xyzs_coe', default=1, type=float, help='latent xyzs loss coefficient')
    parser.add_argument('--weight_cd', type=float, default = 0.01)

    parser.add_argument('--if_fix_sample', type=int, default=0, help='whether to use fix sampling')

    
    parser.add_argument('--dim', default=8, type=int, help='feature dimension')
    parser.add_argument('--hidden_dim', default=64, type=int, help='hiddem dimension')
    parser.add_argument('--ngroups', default=1, type=int, help='groups for groupnorm')

    args = parser.parse_args()
    return args




if __name__ == "__main__":
    test_args = parse_test_args()
    assert test_args.dataset in ['shapenet']
    # the test batch_size must be 1
    assert test_args.batch_size == 1

    # model_args.model_path  = "output/%s_%s_%s_%s/ckpt/ckpt-best.pth" % (model_args.opt, model_args.model, str(model_args.bpp_lambda), str(model_args.latent_xyzs_coe))
    test_args.model_path  = "output/%s_%s_%s/ckpt/ckpt-best.pth" % (test_args.opt, test_args.model, str(test_args.bpp_lambda))


    test_xyzs(test_args)

