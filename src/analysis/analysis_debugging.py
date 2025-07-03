import rootutils
rootutils.setup_root("/home/phisch/multimodal", indicator=".project-root", pythonpath=True)

import os
from pathlib import Path
import torch
import numpy as np
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from torchvision import transforms

from src.analysis import rsa, cka
from src.data.data_module import MyDataModule
from src.model.model_module import LitMML
from src.model.utils import get_model_and_processor

config_str = """
model:
  image_encoder_name : 'google/vit-base-patch16-224'
  text_encoder_name : 'google-bert/bert-base-uncased'
  tokenizer :
    use_fast: False

datasets:
    - 'CIFAR10'

root: '/home/data'

use_subset_probe:
  value: False
  subset_fraction: 0.1

dataset:
  train:
    - 'coco'
    - 'vg'
    - 'cc3m'
  val:
    - 'coco_val'
    - 'cifar10'
    - 'caltech101'
  transforms: 
    enabled: True #False #True
    RandAugment:
      num_ops: 3
      magnitude: 8
  max_seq_length: 72
  coco:
    root: '/pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/COCO'
    split_train : '/pfss/mlde/workspaces/mlde_wsp_PI_Roig/bhavin/students/phillipscholl/multimodal/my_datasets/coco_karpathy_train.json'
    split_val : '/pfss/mlde/workspaces/mlde_wsp_PI_Roig/bhavin/students/phillipscholl/multimodal/my_datasets/coco_karpathy_val.json'
    split_test : '/pfss/mlde/workspaces/mlde_wsp_PI_Roig/bhavin/students/phillipscholl/multimodal/my_datasets/coco_karpathy_test.json'
  vg:
    data_dir: '/pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/VG_Bhavin/VG'
  cc3m:
    data_dir: '/pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/CC3m/h5'
  cifar10:
    root: '/pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/cifar10'
    download: false
  caltech101 :
    root: '/pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/caltech101'
    download: false
  use_subset:
    value: true
    subset_fraction: 1.0 
  use_subset_probe:
    value: false
    subset_fraction: 1.0
    

dataloader:
  train:
    shuffle : True
    batch_size : 512 #512 #1024 #512 #1792 #1920 #2048 #256 #512 #1536 #2048  #896 #960 #1024 #256 #180
    #batch_size : 192
    num_workers: 8
    persistent_workers: True
    pin_memory: True
  test:
    batch_size: 128
    shuffle: False
    num_workers: 8
    #persistent_workers: True
    pin_memory: True
  coco_val:
    shuffle : False
    #batch_size : 
    batch_size : 512 #512 #1024 #512 #1792 #1024 #1920 #2048 #256 #512 #1536 #896 #960 #1024 #256 #180
    num_workers : 8
    persistent_workers: True
    pin_memory: True
  
  cifar10_val:
    batch_size: 512 #512
    shuffle: False
    num_workers: 8
    persistent_workers: True
    pin_memory: True

  caltech101_val:
    batch_size: 512
    shuffle: False
    num_workers: 8
    persistent_workers: True
    pin_memory: True


loss:
  losses:
    # - 'contrastive'
    # - 'image_text_matching'
    - 'SimCLR'
  contrastive:
  #  temperature : 1.
  image_text_matching:
    arg1: ''

optimizer:
  name : "AdamW"
  lr: 2e-04
  kwargs:
    weight_decay : 0.1
    betas : [0.9, 0.95]

scheduler:
  enabled: True #True
  name: CosineWarmup
  monitor: 'loss-val'
  interval: 'step'
  kwargs:
    initial_lr: 1e-08
    num_warmup_steps: 'epoch'
    num_training_steps: 'all'


"""
config = OmegaConf.create(config_str)

SAVE_FIGURES_FIG = Path(
    "src/analysis/figures"
)

model, processor = get_model_and_processor(config)
net = LitMML(
    model,
    processor,
    loss_cfg=config.loss,
    optimizer_cfg=config.optimizer,
    scheduler_cfg=config.scheduler,
    # augmentation=augmentation,
)
data_module = MyDataModule(
    config,
    processor,
    local_dev=False,
    augmentation=None,
    num_views=2,
)
callback_dataloaders = data_module.get_test_dataloaders()
cifar10_test = callback_dataloaders["CIFAR10"]["test"]
device = torch.device("cuda:0")

def get_cifar10_reps(cifar_dataloader, forward_func):
    """ Get representations for CIFAR10 dataset """
    # rep_dict: list of class specific image features for every class index
    rep_dict = {i: [] for i in range(10)}
    for i, (img, lbl) in enumerate(cifar_dataloader):
        feats = forward_func(img.to(device)).detach() # get image features
        # print (feats.shape)
        for idx, element in enumerate(lbl.flatten()):
            assert element < 10
            rep_dict[element.item()].append(feats[idx, ...]) # 
    return rep_dict

CKPT_DIR = "/data/bhavin/ckpts_old"
INTERESTING_CHECKPOINTS = OmegaConf.load("src/analysis/model_info.yaml")
# INTERESTING_CHECKPOINTS = {
#     0.2 : 'n4pejzi5', # 'sunny-elevator-89',
#     0.4 : 'vnfcvlch', #'fragrant-flower-90',
#     0.6 : '0fsftphe', # 'exalted-night-92',
# }

REPS = {}
for key in INTERESTING_CHECKPOINTS:
    files = os.listdir(os.path.join(CKPT_DIR, key))
    files.remove("last.ckpt")
    fname = os.path.join(CKPT_DIR, key, files[0])
    checkpoint = torch.load(fname, map_location="cpu")
    state_dict = {
        k[6:]: v for k, v in checkpoint["state_dict"].items() if "model." in k
    } # extract model state dict (remove "model." from key)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    rep_dict = get_cifar10_reps(
        cifar10_test, lambda x: model.get_image_features(pixel_values=x)
    )

    mean_dict = [None] * len(rep_dict)
    for cifar_cateory in sorted(rep_dict):
        mean_vec = torch.stack(rep_dict[cifar_cateory]).mean(0).flatten().to(device)
        assert mean_vec.shape[0] == 512
        mean_dict[cifar_cateory] = mean_vec.cpu().numpy()

    REPS[key] = mean_dict
