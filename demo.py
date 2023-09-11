import glob
import os
import sys
from os.path import expanduser

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch.utils.data import DataLoader

from datasets import data_transform, get_dataset
from load_model_from_ckpt import get_sampler, init_samples, load_model
from runners.ncsn_runner import conditioning_fn

home = expanduser("~")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

EXP_PATH = os.path.join(home, "scratch/MCVD_SMMNIST_pred")
DATA_PATH = os.path.join(home, "scratch/Datasets/MNIST")


ckpt_path = glob.glob(os.path.join(EXP_PATH, "logs", "checkpoint_*.pt"))[0]
scorenet, config = load_model(ckpt_path, device)
sampler = get_sampler(config)


dataset, test_dataset = get_dataset(
    DATA_PATH, config, video_frames_pred=config.data.num_frames
)

torch.manual_seed(0)

test_loader = DataLoader(
    test_dataset,
    batch_size=config.training.batch_size,
    shuffle=True,
    num_workers=config.data.num_workers,
    drop_last=True,
)
test_iter = iter(test_loader)
test_x, test_y = next(test_iter)


test_x = data_transform(config, test_x)
real, cond, cond_mask = conditioning_fn(
    config,
    test_x,
    num_frames_pred=config.data.num_frames,
    prob_mask_cond=getattr(config.data, "prob_mask_cond", 0.0),
    prob_mask_future=getattr(config.data, "prob_mask_future", 0.0),
)


init = init_samples(len(real), config)
pred = sampler(
    init, scorenet, cond=cond, cond_mask=cond_mask, subsample=100, verbose=True
)

print(pred.shape)
# frames = rearrange(pred, "1 d h w -> h (d w)")
# img = Image.fromarray((frames.cpu().numpy() * 255).astype(np.uint8), mode="L")
# img.save("pred.png")
