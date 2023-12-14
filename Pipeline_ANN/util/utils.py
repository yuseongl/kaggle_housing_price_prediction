import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
  def __init__(self, *args:list[np.array]):
    assert all(args[0].shape[0] == arg.shape[0] for arg in args), "Size mismatch."
    self.data = args
  def __getitem__(self, index):
    return tuple(x[index] for x in self.data)
  def __len__(self):
    return self.data[0].shape[0]