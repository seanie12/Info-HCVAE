import linecache
import subprocess
import h5py
import torch
from torch.utils.data import Dataset, TensorDataset


class HarvestingQADataset(Dataset):
    def __init__(self, filename, ratio):
        self.filename = filename
        self.total_size = int(int(subprocess.check_output("wc -l " + filename, shell=True).split()[0]) * ratio)

    def __getitem__(self, idx):
        line = linecache.getline(self.filename, idx + 1)
        str_loaded = line.split("\t")

        input_ids = str_loaded[0].split()
        input_mask = str_loaded[1].split()
        segment_ids = str_loaded[2].split()
        start_position = str_loaded[3]
        end_position = str_loaded[4]

        input_ids = torch.tensor([int(idx) for idx in input_ids], dtype=torch.long)
        input_mask = torch.tensor([int(idx) for idx in input_mask], dtype=torch.long)
        segment_ids = torch.tensor([int(idx) for idx in segment_ids], dtype=torch.long)
        start_position = torch.tensor([int(start_position)], dtype=torch.long)
        end_position = torch.tensor([int(end_position)], dtype=torch.long)

        return input_ids, input_mask, segment_ids, start_position, end_position

    def __len__(self):
        return self.total_size


class HarvestingQADatasetH5(Dataset):
    def __init__(self, filename, mode="r"):
        self.filename = filename
        self.mode = mode
        self.fdata = h5py.File(filename, mode)
        self.input_ids = self.fdata["qas/input_ids"]
        self.input_masks = self.fdata["qas/input_masks"]
        self.segment_ids = self.fdata["qas/segment_ids"]
        self.start_positions = self.fdata["qas/start_positions"]
        self.end_positions = self.fdata["qas/end_positions"]
        self.total_size = self.start_positions.len()

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.input_ids[idx, :], dtype=torch.long)
        input_mask = torch.tensor(self.input_masks[idx, :], dtype=torch.long)
        segment_ids = torch.tensor(self.segment_ids[idx, :], dtype=torch.long)
        start_position = torch.tensor([self.start_positions[idx]], dtype=torch.long)
        end_position = torch.tensor([self.end_positions[idx]], dtype=torch.long)

        return input_ids, input_mask, segment_ids, start_position, end_position

    def __len__(self):
        return self.total_size

    def rewrite_start_end_positions(self, rewrite_idx, new_start_position, new_end_position):
        """ Only use for QA filtering """
        assert self.mode == "a"
        assert rewrite_idx < self.total_size
        self.start_positions[rewrite_idx] = new_start_position
        self.end_positions[rewrite_idx] = new_end_position

    def close_file(self):
        self.fdata.close()