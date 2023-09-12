import torch
from torch.utils.data import Dataset


class EventDataset(Dataset):
    def __init__(self, n_frames=3):
        super().__init__()

        self.n_frames = n_frames

    def __len__(self):
        return 100

    def __getitem__(self, index):
        gt = torch.zeros((3, 128, 128))
        gt[:, ::2, ::2] = 1
        gt[:, 1::2, 1::2] = -1

        cond = torch.zeros((18, 128, 128))

        return gt, cond
