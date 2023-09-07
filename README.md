# Environment
conda create -n surfpcc
conda activate surfpcc
conda install python==3.7
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install compressai
conda install -c conda-forge einops tensorboard
conda install -c open3d-admin open3d
conda install -c anaconda scikit-learn
conda install pyyaml pandas tqdm
pip install addict plyfile

cd models/Chamfer3D
python setup.py install

# How to use
## 生成训练数据
python prepare_shapenet.py

## 训练
python train.py --model SurPCC --bpp_lambda 0.001 --training_up_ratio 21

## 测试
python test.py --model SurPCC --bpp_lambda 0.001 --training_up_ratio 21

# 第三方库
## Manifold
https://github.com/hjwdzh/Manifold
## NeuralPoints
https://github.com/WanquanF/NeuralPoints
## DPCC
https://github.com/yunhe20/D-PCC