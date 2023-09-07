import torch
import os
import sys
sys.path.append(os.getcwd())
import importlib
import time
import argparse
from datetime import datetime
import torch.optim as optim
from models.utils import  save_pcd, AverageMeter, str2bool
from dataset.dataset import CompressDataset
from torch.optim.lr_scheduler import StepLR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
# torch.set_num_threads(8)


def train(args):
    start = time.time()

    # load data
    train_dataset = CompressDataset(data_path=args.train_data_path, cube_size=args.train_cube_size, batch_size=args.batch_size)
    val_dataset = CompressDataset(data_path=args.val_data_path, cube_size=args.val_cube_size, batch_size=args.batch_size)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, pin_memory=True, num_workers=8)

    # set up folders for checkpoints
    str_time = datetime.now().isoformat()
    print('Experiment Time:', str_time)
    checkpoint_name = "%s_%s_%s" % (args.opt, args.model, str(args.bpp_lambda))
    print('Checkpoint Name',checkpoint_name)
    checkpoint_dir = os.path.join(args.output_path, checkpoint_name, 'ckpt')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    sample_dir = os.path.join(args.output_path, checkpoint_name, 'sample')
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    # create the model
    MODEL = importlib.import_module(args.model)
    model = MODEL.get_model(args).cuda()
    print('Training Arguments:', args)
    print('Model Architecture:', model)
    
    if len(args.pretrained)>=1: 
        model_dict =  model.state_dict()
        model_dict_pretrained = torch.load(args.pretrained)
        model_dict_new = { 'decoder.'+k :v  for k,v in model_dict_pretrained.items()}
        model_dict.update(model_dict_new)
        model.load_state_dict(model_dict,True)

    # optimizer for autoencoder
    parameters = set(p for n, p in model.named_parameters() if not n.endswith(".quantiles"))
    optimizer = optim.Adam(parameters, lr=args.lr)
    # lr scheduler
    scheduler_steplr = StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)
    # optimizer for entropy bottleneck
    aux_parameters = set(p for n, p in model.named_parameters() if n.endswith(".quantiles"))
    aux_optimizer = optim.Adam(aux_parameters, lr=args.aux_lr)
    
    # best validation metric
    best_val_chamfer_loss = float('inf')

    # train
    step = 0
    for epoch in range(args.epochs):
        epoch_loss = AverageMeter()
        epoch_cd_loss = AverageMeter()
        epoch_latent_xyzs_loss = AverageMeter()
        epoch_bpp_loss = AverageMeter()
        epoch_aux_loss = AverageMeter()
        
        model.train()
        for i, input_dict in enumerate(train_loader):
            # input: (b, n, c)
            input = input_dict['xyzs'].cuda()
            input = input.permute(0, 2, 1).contiguous()

            # model forward
            upsampled_p, querying_points, loss, loss_items, bpp, points_sparse, rec_latent_xyzs = model(input)
            epoch_loss.update(loss.item())
            epoch_cd_loss.update(loss_items['cd_loss'])
            epoch_latent_xyzs_loss.update(loss_items['latent_xyzs_loss'])
            epoch_bpp_loss.update(loss_items['bpp_loss'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update the parameters of entropy bottleneck
            aux_loss = model.feats_eblock.loss() + model.xyzs_eblock.loss()
            epoch_aux_loss.update(aux_loss.item())
            aux_optimizer.zero_grad()
            aux_loss.backward()
            aux_optimizer.step()

            # save samples
            if (i+1) % args.print_freq == 0:
                print("train epoch: %d/%d, iters: %d/%d, loss: %f, avg chamfer loss: %f, "
                      "avg latent xyzs loss: %f, avg bpp loss: %f, avg aux loss: %f" %
                      (epoch+1, args.epochs, i+1, len(train_loader), epoch_loss.get_avg(), epoch_cd_loss.get_avg(),
                       epoch_latent_xyzs_loss.get_avg(), epoch_bpp_loss.get_avg(), epoch_aux_loss.get_avg()))
                save_pcd(sample_dir,  "%08d_rec.ply" % step, querying_points.detach().cpu().squeeze(0).numpy())
                save_pcd(sample_dir,  "%08d_gt.ply" % step, input.permute(0, 2, 1).detach().cpu().squeeze(0).numpy()[:,:3])
                save_pcd(sample_dir,  "%08d_sparse.ply" % step, points_sparse.permute(0, 2, 1).detach().cpu().squeeze(0).numpy())
                save_pcd(sample_dir,  "%08d_sparse_rec.ply" % step, rec_latent_xyzs.permute(0, 2, 1).detach().cpu().squeeze(0).numpy())
            step = step + 1
        scheduler_steplr.step()
        # print loss
        interval = time.time() - start
        print("train epoch: %d/%d, time: %d mins %.1f secs, loss: %f, avg chamfer loss: %f, "
                      "avg latent xyzs loss: %f, avg bpp loss: %f, avg aux loss: %f" %
              (epoch+1, args.epochs, interval/60, interval%60, epoch_loss.get_avg(), epoch_cd_loss.get_avg(),
                       epoch_latent_xyzs_loss.get_avg(), epoch_bpp_loss.get_avg(), epoch_aux_loss.get_avg()))


        # validation
        model.eval()
        val_chamfer_loss = AverageMeter()
        val_bpp = AverageMeter()

        for input_dict in val_loader:
            # xyzs: (b, n, c)
            input = input_dict['xyzs'].cuda()
            input = input.permute(0, 2, 1).contiguous()
            # model forward
            upsampled_p, querying_points, loss, loss_items, bpp, points_sparse, points_sparse_gt = model(input)

            val_chamfer_loss.update(loss.item())
            val_bpp.update(bpp.item())

        # print loss
        print("val epoch: %d/%d, val bpp: %f, val chamfer loss: %f" %
              (epoch+1, args.epochs, val_bpp.get_avg(), val_chamfer_loss.get_avg()))

        # save checkpoint
        cur_val_chamfer_loss = val_chamfer_loss.get_avg()
        if  cur_val_chamfer_loss < best_val_chamfer_loss or (epoch+1) % args.save_freq == 0:
            model_name = 'ckpt-best.pth' if cur_val_chamfer_loss < best_val_chamfer_loss else 'ckpt-epoch-%02d.pth' % (epoch+1)
            model_path = os.path.join(checkpoint_dir, model_name)
            torch.save(model.state_dict(), model_path)
            if cur_val_chamfer_loss < best_val_chamfer_loss:
                best_val_chamfer_loss = cur_val_chamfer_loss




def parse_train_args():
    parser = argparse.ArgumentParser(description='Training Arguments')
    
    parser.add_argument('--opt', default='test', type=str, help='operation name')
    parser.add_argument('--model', default='SurfPCC', type=str, help='model name')
    parser.add_argument('--pretrained', default='', type=str, help='')
    parser.add_argument('--dataset', default='shapenet', type=str, help='shapenet or semantickitti')
    parser.add_argument('--batch_size', default=1, type=int, help='the performance will degrade if batch_size is larger than 1!')
    parser.add_argument('--bpp_lambda', default=1.0, type=float, help='bpp loss coefficient')
    parser.add_argument('--epochs', default=50, type=int, help='training epochs')
    
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for backbone')
    parser.add_argument('--aux_lr', default=1e-3, type=float, help='learning rate for entropy model')
    parser.add_argument('--lr_decay_step', default=15, type=int, help='learning rate decay step size')
    parser.add_argument('--gamma', default=0.5, type=float, help='gamma for scheduler_steplr')
    parser.add_argument('--train_data_path', default='./data/shapenet/shapenet_train_cube_size_22.pkl', type=str, help='path to train dataset')
    parser.add_argument('--train_cube_size', default=22, type=int, help='cube size of train dataset')
    parser.add_argument('--val_data_path', default='./data/shapenet/shapenet_val_cube_size_22.pkl', type=str, help='path to val dataset')
    parser.add_argument('--val_cube_size', default=22, type=int, help='cube size of val dataset')
    # normal compression
    parser.add_argument('--print_freq', default=5000, type=int, help='loss print frequency')
    parser.add_argument('--save_freq', default=1, type=int, help='save frequency')


    parser.add_argument('--output_path', default='./output', type=str, help='output path')
    
    parser.add_argument('--training_up_ratio', type=int, default=21,help='The Upsampling Ratio during training') 
    parser.add_argument('--testing_up_ratio', type=int, default=21, help='The Upsampling Ratio during testing')  
    parser.add_argument('--over_sampling_scale', type=float, default=4, help='The scale for over-sampling')
    parser.add_argument('--emb_dims', type=int, default=50, metavar='N',help='Dimension of embeddings')
    parser.add_argument('--pe_out_L', type=int, default=5, metavar='N',help='The parameter L in the position code')
    parser.add_argument('--feature_unfolding_nei_num', type=int, default=4, metavar='N',help='The number of neighbour points used while feature unfolding')
    parser.add_argument('--repulsion_nei_num', type=int, default=5, metavar='N',help='The number of neighbour points used in repulsion loss')
    
    # for phase train
    parser.add_argument('--if_bn', type=int, default=0, help='If using batch normalization')
    parser.add_argument('--neighbor_k', type=int, default=9, help='The number of neighbour points used in DGCNN')
    parser.add_argument('--mlp_fitting_str', type=str, default='256 128 64', metavar='None',help='mlp layers of the part surface fitting (default: None)')
    
    #loss terms weights
    parser.add_argument('--latent_xyzs_coe', default=1, type=float, help='latent xyzs loss coefficient')
    parser.add_argument('--weight_cd', type=float, default = 1)
    
    parser.add_argument('--if_fix_sample', type=int, default=0, help='whether to use fix sampling')
    
    parser.add_argument('--dim', default=8, type=int, help='feature dimension')
    parser.add_argument('--hidden_dim', default=64, type=int, help='hiddem dimension')
    parser.add_argument('--ngroups', default=1, type=int, help='groups for groupnorm')
    
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    train_args = parse_train_args()
        
    train(train_args)
