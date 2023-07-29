import os
import torch
import numpy as np
import json
from os.path import exists
from torch.utils.data import Dataset
from ctypes import *

class go_string(Structure):
    _fields_ = [
        ("p", c_char_p),
        ("n", c_int)]

class EigenVecDataset(Dataset):
    """Eigen Vector dataset."""

    def __init__(self, config_filename, simulator_dir, repetition, ev_idx, transform=None):
        """
        Arguments:
            config_filename (string): Name of the json config file for xg-simulator.
            simulator_dir (string): Directory of xg-simulator.
            repetition (int): Number of calls to xg-simulator to export channel eigen
                vectors.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ev_idx = ev_idx
        self.config_file = os.path.join(simulator_dir, "configs")
        self.config_file = os.path.join(self.config_file, config_filename)
        with open(self.config_file, 'r') as config_file:
            config_data = json.load(config_file)
            self.numUEs = config_data['layout']['macroSites'] * config_data['layout']['sectorPerSite'] * config_data['layout']['uePerSector']
            self.antPerPanel = config_data['bsAntennaParams']['m'] * config_data['bsAntennaParams']['n'] * config_data['bsAntennaParams']['p']
        self.transform = transform
        config_filesplit = os.path.splitext(config_filename)
        cache_name = "{}_{}.pt".format(config_filesplit[0], repetition, config_filesplit[1])
        if exists(cache_name):
            self.eigen_buffer = torch.load(cache_name)
            return
        lib = cdll.LoadLibrary(os.path.join(simulator_dir, 'build/_output/chanexp.so'))
        f = self.config_file.encode('utf-8')
        b = go_string(c_char_p(f), len(f))
        lib.chanexp.restype = c_char_p
        EigenFloats = c_float * (2 * self.antPerPanel) * 13 * 2 * self.numUEs
        EigenBuffer = EigenFloats()
        self.eigen_buffer = np.zeros((self.numUEs*repetition, 2, 13, self.antPerPanel*2), dtype=np.float32)
        for i in range(repetition):
            lib.chanexp(b, EigenBuffer)
            eigen_buffer = np.ctypeslib.as_array(EigenBuffer, (self.numUEs, 2, 13, self.antPerPanel*2))
            self.eigen_buffer[i*self.numUEs:(i+1)*self.numUEs,:] = eigen_buffer
        self.eigen_buffer = torch.from_numpy(self.eigen_buffer)
        if not exists(cache_name):
            torch.save((self.eigen_buffer), cache_name)

    def __len__(self):
        return len(self.eigen_buffer)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ev = self.eigen_buffer[idx, self.ev_idx, :]
        sample = {'ev': ev}

        if self.transform:
            sample = self.transform(sample)

        return sample
