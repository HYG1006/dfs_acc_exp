"""Disk-backed image datasets used by generation and metric evaluation."""

import bisect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def as_chw_uint8(image) -> torch.Tensor:
    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected HWC uint8 RGB image, got {array.shape} {array.dtype}")
    # Copy avoids PyTorch's warning about read-only memory-mapped NumPy arrays.
    return torch.from_numpy(np.array(array, copy=True)).permute(2, 0, 1)


class NpyImageDataset(Dataset):
    def __init__(self, path):
        self.path = Path(path)
        self.array = np.load(self.path, mmap_mode="r")
        if self.array.dtype != np.uint8 or self.array.ndim != 4 or self.array.shape[-1] != 3:
            raise ValueError(f"invalid image NPY: {self.path}, {self.array.shape}, {self.array.dtype}")

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return as_chw_uint8(self.array[index])


class ShardedNpyImageDataset(Dataset):
    def __init__(self, directory):
        self.directory = Path(directory)
        metadata_path = self.directory / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"missing {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text())
        self.limit = int(self.metadata["num_samples"])
        self.paths = sorted(self.directory.glob("rank-*.npy"))
        if not self.paths:
            raise FileNotFoundError(f"no rank-*.npy files in {self.directory}")

        self.arrays = [np.load(path, mmap_mode="r") for path in self.paths]
        self.ends = []
        total = 0
        for path, array in zip(self.paths, self.arrays):
            if array.dtype != np.uint8 or array.ndim != 4 or tuple(array.shape[1:]) != (256, 256, 3):
                raise ValueError(f"invalid sample shard: {path}, {array.shape}, {array.dtype}")
            total += len(array)
            self.ends.append(total)
        if total < self.limit:
            raise ValueError(f"shards contain {total} images but metadata requests {self.limit}")

    def __len__(self):
        return self.limit

    def __getitem__(self, index):
        if index < 0:
            index += self.limit
        if not 0 <= index < self.limit:
            raise IndexError(index)
        shard = bisect.bisect_right(self.ends, index)
        start = 0 if shard == 0 else self.ends[shard - 1]
        return as_chw_uint8(self.arrays[shard][index - start])

