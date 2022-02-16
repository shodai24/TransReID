from typing import Any, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import os
from torch import Tensor
import parse
import re

class TrainStat():
    def __init__(self) -> None:
        self.epoch = []
        self.iter = []
        self.loss = []
        self.train_acc = []
        self.lr = []
        self.map = []
        self.cmc_r1 = []
        self.cmc_r5 = []
        self.cmc_r10 = []
    
    def add(self, epoch, n_iter, loss, train_acc, lr):
        self.epoch.append(epoch)
        if len(self.iter) == 0:
            self.iter.append(float(n_iter))
        else:    
            self.iter.append(float(50 + self.iter[-1]))
        self.loss.append(loss)
        if type(train_acc) is Tensor:
            train_acc = Tensor.cpu(train_acc).item()
        self.train_acc.append(train_acc)
        self.lr.append(lr)
    
    def add_valid(self, mAP, cmc_r1, cmc_r5, cmc_r10):        
        padding = [np.NaN for i in range(self.pad_len - 1)]
        
        self.map = self.map + padding
        self.map.append(mAP)
        
        self.cmc_r1 = self.cmc_r1 + padding
        self.cmc_r1.append(cmc_r1)
        
        self.cmc_r5 = self.cmc_r5 + padding
        self.cmc_r5.append(cmc_r5)
        
        self.cmc_r10 = self.cmc_r10 + padding
        self.cmc_r10.append(cmc_r10)

    def pad(self, list_in: List[Any], n: int):
        padding = [np.NaN for i in range(self.pad_len)]
        list_in = list_in + padding
    
    def plot(self):
        self.df = pd.DataFrame({'epoch': self.epoch,
                                'n_iter': self.iter,
                                'loss' : self.loss,
                                'training accuracy' : self.train_acc,
                                'learning rate' : self.lr,
                                'mean average precision': self.map,
                                'CMC Rank-1': self.cmc_r1,
                                'CMC Rank-5': self.cmc_r5,
                                'CMC Rank-10': self.cmc_r10})
        cols = self.df.columns.tolist()
        cols.remove('n_iter')
        self.fig = go.Figure()
        for col in cols:
                self.fig.add_trace(go.Line(x=self.df['n_iter'], y=self.df[col], name=col, mode='lines+markers'))
    
    def show(self):
        self.fig.show() 
    
    def save(self, dir):
        dateObj = datetime.now()
        timestamp = dateObj.strftime("%d-%b-%Y-%H-%M-%S")
        filename = "train-stat_" + timestamp
        self.fig.write_html(os.path.join(dir, filename))  

    def set_pad_length(self, epoch_length, log_period=50):
        self.pad_len = (epoch_length // log_period)
    
    def load_train_log(self, filepath):
        # open train log
        with open(filepath, 'r') as file:
            no_pad_length_set = True
            for line in file:
                if 'LOG_PERIOD' in line:
                    p = re.compile(r'\d+')
                    self.log_period = int(p.findall(line).pop())
                if 'INFO: Epoch[' in line:
                    index = line.find('Epoch')
                    line = line[index:]
                    form = "Epoch[{0}] Iteration[{1}/{2}] Loss: {3}, Acc: {4}, Base Lr: {5}"
                    epoch, iterations, max_iter, loss, acc, lr = list(map(float, parse.parse(form, line)))
                    n_iter = ((epoch - 1) * max_iter) + iterations
                    self.add(epoch, n_iter, loss, acc, lr)
                    if no_pad_length_set:
                        self.set_pad_length(int(max_iter), self.log_period)
                if 'Validation Results' in line:
                    p = re.compile(r'\d+\.\d+')
                    mAP = float(p.findall(file.readline()).pop())
                    cmc_r1 = float(p.findall(file.readline()).pop())
                    cmc_r5 = float(p.findall(file.readline()).pop())
                    cmc_r10 = float(p.findall(file.readline()).pop())

                    if any(elem is None for elem in [mAP, cmc_r1, cmc_r5, cmc_r10]):
                        self.epoch.pop() # remove incomplete epoch from list
                    else:
                        self.add_valid(mAP, cmc_r1, cmc_r5, cmc_r10)
        # discard lines for incomplete epochs
        max_epoch = int(max(self.epoch))
        if len(self.epoch) != max_epoch * self.pad_len:
            num_elements = (max_epoch - 1) * self.pad_len
            self.epoch = self.epoch[:num_elements]
            self.iter = self.iter[:num_elements]
            self.loss = self.loss[:num_elements]
            self.train_acc = self.train_acc[:num_elements]
            self.lr = self.lr[:num_elements] 