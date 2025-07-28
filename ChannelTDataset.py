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

class ChannelTDataset(Dataset):
    """Eigen Vector dataset."""

    def __init__(self, config_filename, simulator_dir, timeDomainChannel, numUEs, transform=None):
        """
        Arguments:
            config_filename (string): Name of the json config file for xg-simulator.
            simulator_dir (string): Directory of xg-simulator.
            timeDomainChannel (bool): whether the channel is for time domain or freq domain
            numUEs (int): number of UEs to generate for data set, set to -1 to use SLS configuration
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.config_file, config_data = readConfig(simulator_dir, config_filename)
        self.numUEs = numUEs
        self.stepSize = int(config_data['simulationTime']['stepSize'] * 1000)
        self.numSteps = int(config_data['simulationTime']['simulationLen'] / config_data['simulationTime']['stepSize'])
        self.bsAntPerPanel = config_data['bsAntennaParams']['m'] * config_data['bsAntennaParams']['n'] * config_data['bsAntennaParams']['p']
        self.ueAntPerPanel = config_data['utAntennaParams']['m']* config_data['utAntennaParams']['n'] * config_data['utAntennaParams']['p']
        self.transform = transform
        self.ueSpeed = config_data['ueSpeed']
        numUEsPerDrop = config_data["layout"]['numOfUEs']
        numOfDrops = int(self.numUEs/numUEsPerDrop)
        config_filesplit = os.path.splitext(config_filename)
        cache_name = "{}_{}_{}kmH_{}ue_{}s_{}ms_{}rx_{}tx.pt".format(config_filesplit[0], timeDomainChannel, self.ueSpeed, self.numUEs, self.numSteps, self.stepSize, self.ueAntPerPanel, self.bsAntPerPanel)
        if exists(cache_name):
            self.h_buffer, self.t_buffer = torch.load(cache_name)
            return
        lib = cdll.LoadLibrary(os.path.join(simulator_dir, 'build/_output/chanexp.so'))
        f = self.config_file.encode('utf-8')
        b = go_string(c_char_p(f), len(f))
        lib.getComChanLength.restype = c_int
        if timeDomainChannel:
            self.chanLen = lib.getComChanLength(b)
        else:
            self.chanLen = lib.getNumOfRBs(b)
        lib.chanfexp.restype = c_char_p
        self.h_buffer = np.zeros((self.numUEs, self.numSteps, self.ueAntPerPanel, self.bsAntPerPanel, self.chanLen * 2), dtype=np.float32)
        self.t_buffer = np.zeros((self.numUEs, self.chanLen), dtype=np.float32)
        for i in range(numOfDrops):
            ChanFFloats = c_float * (self.chanLen * 2) * self.bsAntPerPanel * self.ueAntPerPanel *  self.numSteps * numUEsPerDrop
            ChanFBuffer = ChanFFloats()
            ChanLFloats = c_float * self.chanLen * numUEsPerDrop
            ChanLBuffer = ChanLFloats()
            lib.chanfexp(b, timeDomainChannel, -1, ChanFBuffer, ChanLBuffer)
            h_buffer = np.ctypeslib.as_array(ChanFBuffer, (numUEsPerDrop, self.numSteps, self.ueAntPerPanel, self.bsAntPerPanel, self.chanLen * 2))
            t_buffer = np.ctypeslib.as_array(ChanLBuffer, (numUEsPerDrop, self.chanLen))
            self.h_buffer[i*numUEsPerDrop:(i+1)*numUEsPerDrop,:] = h_buffer
            self.t_buffer[i*numUEsPerDrop:(i+1)*numUEsPerDrop,:] = t_buffer
        self.h_buffer = torch.from_numpy(self.h_buffer)
        self.t_buffer = torch.from_numpy(self.t_buffer.astype(int))
        if not exists(cache_name):
            torch.save((self.h_buffer, self.t_buffer), cache_name)

    def __len__(self):
        return len(self.h_buffer)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        win = int(list(self.h_buffer.shape)[1] / 2)
        h = self.h_buffer[idx, :win, :, :, :]
        h_pred = self.h_buffer[idx, win:, :, :, :]
        t = self.t_buffer[idx, :]
        sample = {'h': h, 'h_pred': h_pred, 't': t}

        if self.transform:
            sample = self.transform(sample)

        return sample
