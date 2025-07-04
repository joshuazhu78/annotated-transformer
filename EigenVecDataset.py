import os
import torch
import numpy as np
import json
from os.path import exists
from torch.utils.data import Dataset
from ctypes import *
from utils import readConfig

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
        self.config_file, config_data = readConfig(simulator_dir, config_filename)
        self.numUEs = config_data['layout']['numOfUEs']
        self.stepSize = int(config_data['simulationTime']['stepSize'] * 1000)
        self.numSteps = int(config_data['simulationTime']['simulationLen'] / config_data['simulationTime']['stepSize'])
        self.antPerPanel = config_data['bsAntennaParams']['m'] * config_data['bsAntennaParams']['n'] * config_data['bsAntennaParams']['p']
        self.ueAntPerPanel = config_data['utAntennaParams']['m']* config_data['utAntennaParams']['n'] * config_data['utAntennaParams']['p']
        self.maxRank = min(self.antPerPanel, self.ueAntPerPanel)
        self.ueSpeed = config_data['ueSpeed']
        self.transform = transform
        config_filesplit = os.path.splitext(config_filename)
        cache_name = "{}_{}kmH_{}UEs_{}s_{}ms_{}tx_{}rx.pt".format(config_filesplit[0], self.ueSpeed, repetition*self.numUEs, self.numSteps, self.stepSize, self.antPerPanel, self.ueAntPerPanel)
        if exists(cache_name):
            self.eigen_buffer = torch.load(cache_name)
            return
        lib = cdll.LoadLibrary(os.path.join(simulator_dir, 'build/_output/chanexp.so'))
        f = self.config_file.encode('utf-8')
        b = go_string(c_char_p(f), len(f))
        lib.getNumOfPrecodingSBs.restype = c_int
        numOfPrecodingSBs = lib.getNumOfPrecodingSBs(b)
        lib.chanexp.restype = c_char_p
        EigenFloats = c_float * (2 * self.antPerPanel) * (numOfPrecodingSBs*self.numSteps) * self.maxRank * self.numUEs
        EigenBuffer = EigenFloats()
        self.eigen_buffer = np.zeros((self.numUEs*repetition, self.maxRank, numOfPrecodingSBs*self.numSteps, 2 * self.antPerPanel), dtype=np.float32)
        for i in range(repetition):
            lib.chanexp(b, EigenBuffer)
            eigen_buffer = np.ctypeslib.as_array(EigenBuffer, (self.numUEs, self.maxRank, numOfPrecodingSBs*self.numSteps, 2 * self.antPerPanel))
            self.eigen_buffer[i*self.numUEs:(i+1)*self.numUEs,:] = eigen_buffer
        self.eigen_buffer = torch.from_numpy(self.eigen_buffer)
        if not exists(cache_name):
            torch.save((self.eigen_buffer), cache_name)

    def __len__(self):
        return len(self.eigen_buffer)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ev = self.eigen_buffer[idx, self.ev_idx, :, :]
        sample = {'ev': ev}

        if self.transform:
            sample = self.transform(sample)

        return sample
