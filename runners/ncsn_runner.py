import logging
import os
import time
from collections import OrderedDict
from functools import partial

import numpy as np
import psutil
import torch
import yaml
from torch.distributions.gamma import Gamma
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

import wandb
from datasets import data_transform, get_dataset
from debug_utils import p
from losses import get_optimizer, warmup_lr
from losses.dsm import anneal_dsm_score_estimation
from main import dict2namespace, namespace2dict
from models import (
    FPNDM_sampler,
    anneal_Langevin_dynamics,
    anneal_Langevin_dynamics_consistent,
    ddim_sampler,
    ddpm_sampler,
)
from models.ema import EMAHelper

__all__ = ["NCSNRunner"]


def count_training_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_proc_mem():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


def get_GPU_mem():
    try:
        num = torch.cuda.device_count()
        mem = 0
        for i in range(num):
            mem_free, mem_total = torch.cuda.mem_get_info(i)
            mem += (mem_total - mem_free) / 1024**3
        return mem
    except:
        return 0


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def convert_time_stamp_to_hrs(time_day_hr):
    time_day_hr = time_day_hr.split(",")
    if len(time_day_hr) > 1:
        days = time_day_hr[0].split(" ")[0]
        time_hr = time_day_hr[1]
    else:
        days = 0
        time_hr = time_day_hr[0]
    # Hr
    hrs = time_hr.split(":")
    return float(days) * 24 + float(hrs[0]) + float(hrs[1]) / 60 + float(hrs[2]) / 3600


def stretch_image(X, ch, imsize):
    return (
        X.reshape(len(X), -1, ch, imsize, imsize)
        .permute(0, 2, 1, 4, 3)
        .reshape(len(X), ch, -1, imsize)
        .permute(0, 1, 3, 2)
    )


# Make and load model
def load_model(ckpt_path, device):
    # Parse config file
    with open(os.path.join(os.path.dirname(ckpt_path), "config.yml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # Load config file
    config = dict2namespace(config)
    config.device = device
    # Load model
    scorenet = get_model(config)
    assert scorenet is not None
    if config.device != torch.device("cpu"):
        scorenet = torch.nn.DataParallel(scorenet)
        states = torch.load(ckpt_path, map_location=config.device)
    else:
        states = torch.load(ckpt_path, map_location="cpu")
        states[0] = OrderedDict([(k.replace("module.", ""), v) for k, v in states[0].items()])
    scorenet.load_state_dict(states[0], strict=False)
    if config.model.ema:
        ema_helper = EMAHelper(mu=config.model.ema_rate)
        ema_helper.register(scorenet)
        ema_helper.load_state_dict(states[-1])
        ema_helper.ema(scorenet)
    scorenet.eval()
    return scorenet, config


def get_model(config):
    version = getattr(config.model, "version", "SMLD").upper()
    arch = getattr(config.model, "arch", "ncsn")
    depth = getattr(config.model, "depth", "deep")

    if arch == "unetmore":
        from models.better.ncsnpp_more import UNetMore_DDPM  # This lets the code run on CPU when 'unetmore' is not used

        return UNetMore_DDPM(config).to(config.device)  # .to(memory_format=torch.channels_last).to(config.device)
    elif arch in ["unetmore3d", "unetmorepseudo3d"]:
        from models.better.ncsnpp_more import UNetMore_DDPM  # This lets the code run on CPU when 'unetmore' is not used

        # return UNetMore_DDPM(config).to(memory_format=torch.channels_last_3d).to(config.device) # channels_last_3d doesn't work!
        return UNetMore_DDPM(config).to(config.device)  # .to(memory_format=torch.channels_last).to(config.device)

    else:
        Exception("arch is not valid [ncsn, unet, unetmore, unetmore3d]")


class NCSNRunner:
    def __init__(self, args, config, config_uncond):
        self.args = args
        self.config = config
        self.config_uncond = config_uncond
        self.version = getattr(self.config.model, "version", "SMLD")
        args.log_sample_path = os.path.join(args.log_path, "samples")
        os.makedirs(args.log_sample_path, exist_ok=True)

    def train(self):
        def save_state(checkpoint_id=None):
            state = {
                "net": scorenet.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "total_train_time": self.total_train_time,
                "time_elapsed": self.time_elapsed,
            }
            if self.config.model.ema:
                state["ema"] = ema_helper.state_dict()

            if checkpoint_id is not None:
                filename = f"checkpoint_{checkpoint_id}.pt"
            else:
                filename = "checkpoint.pt"

            logging.info(f"Saving {filename} in {self.args.log_path}")
            torch.save(state, os.path.join(self.args.log_path, filename))

        wandb.init(
            project="mcvd",
            config=namespace2dict(self.config),
            resume=self.args.resume_training,
        )

        dataset, test_dataset = get_dataset(
            self.args.data_path,
            self.config,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
        )
        dataloader_test = DataLoader(
            test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            drop_last=True,
        )
        dataloader_iter = cycle(dataloader)
        dataloader_test_iter = cycle(dataloader_test)

        scorenet = get_model(self.config)
        scorenet = torch.nn.DataParallel(scorenet)

        logging.info(f"Number of parameters: {count_parameters(scorenet)}")
        logging.info(f"Number of trainable parameters: {count_training_parameters(scorenet)}")

        optimizer = get_optimizer(self.config, scorenet.parameters())

        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            logging.info(f"Number of GPUs : {num_devices}")
            for i in range(num_devices):
                logging.info(torch.cuda.get_device_properties(i))
        else:
            logging.info(f"Running on CPU!")

        start_step = 0
        self.total_train_time = 0
        self.time_elapsed = 0

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(scorenet)

        if self.args.resume_training:
            state = torch.load(os.path.join(self.args.log_path, "checkpoint.pt"))
            scorenet.load_state_dict(state["net"])
            ### Make sure we can resume with different eps
            state["optimizer"]["param_groups"][0]["eps"] = self.config.optim.eps
            optimizer.load_state_dict(state["optimizer"])
            start_step = state["step"]
            self.total_train_time = state["total_train_time"]
            self.time_elapsed = state["time_elapsed"]
            if self.config.model.ema:
                ema_helper.load_state_dict(state["ema"])
            logging.info(f"Resuming training from checkpoint.pt in {self.args.log_path} at step {start_step}.")

        net = scorenet.module if hasattr(scorenet, "module") else scorenet

        # Initial samples
        n_init_samples = min(30, self.config.training.batch_size)
        init_samples_shape = (
            n_init_samples,
            self.config.data.channels,
            self.config.data.image_size,
            self.config.data.image_size,
        )
        if self.version == "SMLD":
            init_samples = torch.rand(init_samples_shape, device=self.config.device)
            init_samples = data_transform(self.config, init_samples)
        elif self.version == "DDPM" or self.version == "DDIM" or self.version == "FPNDM":
            if getattr(self.config.model, "gamma", False):
                used_k, used_theta = net.k_cum[0], net.theta_t[0]
                z = (
                    Gamma(
                        torch.full(init_samples_shape, used_k),
                        torch.full(init_samples_shape, 1 / used_theta),
                    )
                    .sample()
                    .to(self.config.device)
                )
                init_samples = z - used_k * used_theta  # we don't scale here
            else:
                init_samples = torch.randn(init_samples_shape, device=self.config.device)

        # Sampler
        sampler = self.get_sampler()

        self.start_time = time.time()

        for step in range(start_step, self.config.training.n_steps):
            X, cond = next(dataloader_iter)

            optimizer.zero_grad()
            lr = warmup_lr(
                optimizer,
                step,
                getattr(self.config.optim, "warmup", 0),
                self.config.optim.lr,
            )
            scorenet.train()
            step += 1

            # Data
            X = X.to(self.config.device)
            cond = cond.to(self.config.device)

            # Loss
            itr_start = time.time()
            loss = anneal_dsm_score_estimation(
                scorenet,
                X,
                cond=cond,
                loss_type=getattr(self.config.training, "loss_type", "a"),
                gamma=getattr(self.config.model, "gamma", False),
                L1=getattr(self.config.training, "L1", False),
                all_frames=getattr(self.config.model, "output_all_frames", False),
            )

            # Optimize
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                scorenet.parameters(),
                getattr(self.config.optim, "grad_clip", np.inf),
            )
            optimizer.step()

            # Training time
            self.total_train_time += time.time() - itr_start

            wandb.log(
                {
                    "loss_train": loss.item(),
                    "lr": lr,
                    "grad_norm": grad_norm.item(),
                    "time_train_h": self.total_train_time / 3600,
                },
                step=step,
            )

            if self.config.model.ema:
                ema_helper.update(scorenet)

            # save state for resuming
            if step % self.config.training.checkpoint_freq == 0 and step != 0:
                save_state()

            # save sample weights for debugging from time to time
            if step % self.config.training.snapshot_freq == 0:
                save_state(step)

            test_scorenet = None
            # Get test_scorenet
            if (
                step == 1
                or step % self.config.training.val_freq == 0
                or (step % self.config.training.sample_freq == 0 and step != 0)
            ):
                if self.config.model.ema:
                    test_scorenet = ema_helper.ema_copy(scorenet)
                else:
                    test_scorenet = scorenet

                test_scorenet.eval()

            # Validation
            if step == 1 or step % self.config.training.val_freq == 0:
                test_X, test_cond = next(dataloader_test_iter)

                test_X = test_X.to(self.config.device)
                test_cond = test_cond.to(self.config.device)

                with torch.no_grad():
                    test_dsm_loss = anneal_dsm_score_estimation(
                        test_scorenet,
                        test_X,
                        cond=test_cond,
                        loss_type=getattr(self.config.training, "loss_type", "a"),
                        gamma=getattr(self.config.model, "gamma", False),
                        L1=getattr(self.config.training, "L1", False),
                        all_frames=getattr(self.config.model, "output_all_frames", False),
                    )
                wandb.log({"loss_test": test_dsm_loss.item()}, step=step)

            # Sample from model
            if step % self.config.training.sample_freq == 0 and step != 0:
                logging.info(f"Saving images in {self.args.log_sample_path}")

                test_X, test_cond = next(iter(dataloader_test))

                test_X = test_X.to(self.config.device)
                test_cond = test_cond.to(self.config.device)

                all_samples = sampler(
                    init_samples,
                    test_scorenet,
                    cond=test_cond,
                    just_beta=False,
                    final_only=True,
                    denoise=self.config.sampling.denoise,
                    subsample_steps=getattr(self.config.sampling, "subsample", None),
                    clip_before=getattr(self.config.sampling, "clip_before", True),
                    verbose=False,
                    log=False,
                    gamma=getattr(self.config.model, "gamma", False),
                ).to("cpu")[-1]

                gt_samples = test_X.cpu()

                # compute some metrics here
                # TODO: add metrics here

                result_image = torch.cat([gt_samples, all_samples], dim=3)
                result_image = make_grid(result_image, nrow=6, normalize=True, range=(-1, 1), pad_value=1, padding=6)

                save_image(
                    result_image,
                    os.path.join(self.args.log_sample_path, f"image_grid_{step}.png"),
                )

                if wandb.run is not None:
                    image = wandb.Image(result_image.permute(1, 2, 0).numpy(), caption=f"step: {step}")
                    wandb.log({"samples_train": image}, step=step)

                del all_samples

            del test_scorenet

            self.time_elapsed += time.time() - self.start_time
            self.start_time = time.time()

            wandb.log({"time_elapsed_h": self.time_elapsed / 3600}, step=step)

        save_state("final")
        logging.info(f"training done")

    def get_sampler(self):
        # Sampler
        if self.version == "SMLD":
            consistent = getattr(self.config.sampling, "consistent", False)
            sampler = anneal_Langevin_dynamics_consistent if consistent else anneal_Langevin_dynamics
        elif self.version == "DDPM":
            sampler = partial(ddpm_sampler)
        elif self.version == "DDIM":
            sampler = partial(ddim_sampler, config=self.config)
        elif self.version == "FPNDM":
            sampler = partial(FPNDM_sampler, config=self.config)

        return sampler
