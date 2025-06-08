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

class ComChanDataset(Dataset):
    """Composite channel dataset."""

    def __init__(self, config_filename, simulator_dir, train_drops, test_drops, train, test_from_train, transform=None):
        """
        Arguments:
            config_filename (string): Name of the json config file for xg-simulator.
            simulator_dir (string): Directory of xg-simulator.
            train_drops (int): Number of channels to export from xg-simulator for training.
            test_drops (int): Number of channels to export from xg-simulator for validating.
            train (bool), whether this is for train or validate
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.config_file = os.path.join(simulator_dir, "configs")
        self.config_file = os.path.join(self.config_file, config_filename)
        with open(self.config_file, 'r') as config_file:
            config_data = json.load(config_file)
            #self.scenario = config_data['layout']['scenario']
        self.transform = transform
        numOfChans = train_drops + test_drops
        config_filesplit = os.path.splitext(config_filename)
        cache_name = "{}_hcom_{}.pt".format(config_filesplit[0], numOfChans)
        if exists(cache_name):
            h_buffer, n_buffer = torch.load(cache_name)
            if train:
                self.h_buffer = h_buffer[0:train_drops,:]
                self.n_buffer = n_buffer[0:train_drops]
            else:
                if test_from_train:
                    self.h_buffer = h_buffer[0:test_drops,:]
                    self.n_buffer = n_buffer[0:test_drops:]
                else:
                    self.h_buffer = h_buffer[train_drops:,:]
                    self.n_buffer = n_buffer[train_drops:]
            return
        self.numOfChans = numOfChans
        lib = cdll.LoadLibrary(os.path.join(simulator_dir, 'build/_output/chanexp.so'))
        f = self.config_file.encode('utf-8')
        b = go_string(c_char_p(f), len(f))
        lib.getComChanLength.restype = c_int
        comChanLength = lib.getComChanLength(b)
        lib.chantexp.restype = c_char_p
        hFloats = c_float * (comChanLength) * numOfChans
        hBuffer = hFloats()
        nFloats = c_float * (numOfChans)
        nBuffer = nFloats()
        lib.chantexp(b, self.numOfChans, hBuffer, nBuffer)

        h_buffer = np.ctypeslib.as_array(hBuffer, (self.numOfChans, comChanLength))
        h_buffer = torch.from_numpy(h_buffer)

        n_buffer = np.ctypeslib.as_array(nBuffer, (self.numOfChans))
        n_buffer = torch.from_numpy(n_buffer.astype(int))

        if train:
            self.h_buffer = h_buffer[0:train_drops,:]
            self.n_buffer = n_buffer[0:train_drops]
        else:
            if test_from_train:
                self.h_buffer = h_buffer[0:test_drops,:]
                self.n_buffer = n_buffer[0:test_drops:]
            else:
                self.h_buffer = h_buffer[train_drops:,:]
                self.n_buffer = n_buffer[train_drops:]

        if not exists(cache_name):
            torch.save((h_buffer, n_buffer), cache_name)

    def __len__(self):
        return len(self.h_buffer)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        h = self.h_buffer[idx, :]
        n = self.n_buffer[idx]
        sample = {'h': h, 'n': n}

        if self.transform:
            sample = self.transform(sample)

        return sample
