import torch
import numpy as np
import os
from torch.utils.data import Dataset
import pickle as pkl
import random
import json
import h5py
from glob import glob
import open3d as o3d

class Dataset_2048(Dataset):
    def __init__(self, root, dataset_name='shapenetcorev2', num_points=2048, split='train', load_name=True, load_file=True,):

        self.root = os.path.join(root, dataset_name + '_' + 'hdf5_2048')
        self.dataset_name = dataset_name
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.load_file = load_file

        self.path_h5py_all = []
        self.path_name_all = []
        self.path_file_all = []

        if self.split in ['train','trainval','all']:
            self.get_path('train')
        if self.split in ['val','trainval','all']:
            self.get_path('val')
        if self.split in ['test', 'all']:   
            self.get_path('test')

        self.path_h5py_all.sort()
        data, label= self.load_h5py(self.path_h5py_all)

        if self.load_name:
            self.path_name_all.sort()
            self.name = self.load_json(self.path_name_all)    # load label name

        if self.load_file:
            self.path_file_all.sort()
            self.file = self.load_json(self.path_file_all)    # load file name
        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0) 


    def get_path(self, type):
        path_h5py = os.path.join(self.root, '%s*.h5'%type)
        print(path_h5py)
        self.path_h5py_all += glob(path_h5py)
        if self.load_name:
            path_json = os.path.join(self.root, '%s*_id2name.json'%type)
            self.path_name_all += glob(path_json)
        if self.load_file:
            path_json = os.path.join(self.root, '%s*_id2file.json'%type)
            self.path_file_all += glob(path_json)
        return 

    def load_h5py(self, path):
        all_data = []
        all_label = []
        all_seg = []
        for h5_name in path:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)

        return all_data, all_label

    def load_json(self, path):
        all_data = []
        for json_name in path:
            j =  open(json_name, 'r+')
            data = json.load(j)
            all_data += data
        return all_data

    def __getitem__(self, item):
        point_set = self.data[item][:self.num_points]

        # convert numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        label = self.label[item]
        return point_set, label

    def __len__(self):
        return self.data.shape[0]

class CompressDataset(Dataset):
    def __init__(self, data_path, map_size=100, cube_size=5, batch_size=1, points_num=1024):
        self.data_path = data_path
        with open(self.data_path, 'rb')as f:
            self.data = pkl.load(f)
        self.init_data()

        self.map_size = map_size
        self.cube_size = cube_size
        self.dim_cube_num = np.ceil(map_size / cube_size).astype(int)
        self.batch_size = batch_size
        # the points_num should be larger than or equal to the min_num of each cube
        self.points_num = points_num


    def init_data(self):
        # data :{pcd_idx: {'points': {cube_idx: ...}, 'meta_data':{'shift':..., 'min_points':..., 'max_points':...}}}
        self.patch_num = []
        for pcd_idx in self.data.keys():
            cur_patch_num = len(self.data[pcd_idx]['points'].keys())
            self.patch_num.append(cur_patch_num)
        # the last patch num of each full point cloud
        self.pcd_last_patch_num = np.cumsum(self.patch_num)


    def get_pcd_and_patch(self, idx):
        diff = idx + 1 - self.pcd_last_patch_num
        pcd_idx = np.where(diff <= 0)[0][0]
        if pcd_idx > 0:
            patch_idx = idx - self.pcd_last_patch_num[pcd_idx-1]
        else:
            # the first pcd
            patch_idx = idx

        return pcd_idx, patch_idx


    def __getitem__(self, idx):
        pcd_idx, patch_idx = self.get_pcd_and_patch(idx)
        cubes = list(self.data[pcd_idx]['points'].keys())
        # indicate which cube
        cube_x = cubes[patch_idx] // self.dim_cube_num ** 2
        cube_y  = (cubes[patch_idx] - cube_x * self.dim_cube_num ** 2) // self.dim_cube_num
        cube_z = cubes[patch_idx] % self.dim_cube_num
        # the coordinate of center point
        center = [(cube_x + 0.5) * self.cube_size, (cube_y + 0.5) * self.cube_size, (cube_z + 0.5) * self.cube_size]
        xyzs = self.data[pcd_idx]['points'][cubes[patch_idx]][:, :3]
        normals = self.data[pcd_idx]['points'][cubes[patch_idx]][:, 3:]
        # normalize to [-1, 1]
        xyzs = 2 * (xyzs - center) / self.cube_size
        xyzs = torch.tensor(xyzs).float()
        normals = torch.tensor(normals).float()
        input_dict = {}
        if self.batch_size == 1:
            input_dict['xyzs'] = xyzs
            input_dict['normals'] = normals
        else:
            sample_idx = random.sample(range(xyzs.shape[0]), self.points_num)
            sample_idx = torch.tensor(sample_idx).long()
            input_dict['xyzs'] = xyzs[sample_idx, :]
            input_dict['normals'] = normals[sample_idx, :]

        return input_dict


    def __len__(self):
        return sum(self.patch_num)


    # scale to original size
    def scale_to_origin(self, xyzs, idx):
        pcd_idx, patch_idx = self.get_pcd_and_patch(idx)
        cubes = list(self.data[pcd_idx]['points'].keys())
        # indicate which cube
        cube_x = cubes[patch_idx] // self.dim_cube_num ** 2
        cube_y  = (cubes[patch_idx] - cube_x * self.dim_cube_num ** 2) // self.dim_cube_num
        cube_z = cubes[patch_idx] % self.dim_cube_num
        # the coordinate of center point
        center = [(cube_x + 0.5) * self.cube_size, (cube_y + 0.5) * self.cube_size, (cube_z + 0.5) * self.cube_size]
        center = torch.tensor(center).float().to(xyzs.device)
        # scale to the cube coordinate
        xyzs = xyzs * self.cube_size / 2 + center
        # scale to the original coordinate
        xyzs = xyzs / 100
        meta_data = self.data[pcd_idx]['meta_data']
        shift, max_coord, min_coord = meta_data['shift'], meta_data['max_coord'], meta_data['min_coord']
        xyzs = xyzs * (max_coord - min_coord) + min_coord + shift

        return xyzs

class CompressDataset_20W(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.filenames = glob(self.data_path+"*")
        self.filelen = len(self.filenames)
        print(self.filelen)
    
    def __getitem__(self, idx):
        pcd = o3d.io.read_point_cloud(self.data_path+str(idx)+".ply")
        xyzs = np.array(pcd.points).astype(np.float32)
        return xyzs
        
    def __len__(self):
        return self.filelen
        
class CompressDataset_3W(Dataset):
    def __init__(self, data_path, split='train'):
        self.data_path = data_path
        if (split=='test'):
            self.filenames = glob(self.data_path+"*/"+split+"/*.ply")
        elif (split=='train'):
            self.filenames = glob(self.data_path+"*/"+split+"/*.ply")
            file_len = len(self.filenames)
            self.filenames = self.filenames[::5]
        elif (split=='val'):
            split = 'train'
            self.filenames = glob(self.data_path+"*/"+split+"/*.ply")
            file_len = len(self.filenames)
            self.filenames = self.filenames[1::200]
    
    def __getitem__(self, idx):
        pcd = o3d.io.read_point_cloud(self.filenames[idx])
        xyzs = np.array(pcd.points).astype(np.float32)
        return xyzs
              
    def __len__(self):
        return len(self.filenames)
        

if __name__ == "__main__":
    dataset = CompressDataset_3W(data_path = '/home/zby/datasets/ShapeNet_SP_32768/', split='val')
    test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
    for i, input_dict in enumerate(test_loader):
        print(input_dict.shape)