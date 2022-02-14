from typing import Any, List
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import os
from torch import Tensor


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
        self.train_acc.append(Tensor.cpu(train_acc).item())
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
        #self.fig = px.line(
        #    x=self.iter,
        #    y=[self.loss, self.acc, self.lr],
        #    labels={'x': 'N_Iterations',
        #            'y': ['loss', 'accuracy', 'learning rate']},
        #    markers=True)
        
        #self.fig = px.line(
        #    data_frame=self.df,
        #    x='n_iter',
        #    y=['loss', 'training accuracy', 'validation accuracy', 'learning rate', 'mean average precision', 'CMC Rank-1', 'CMC Rank-5','CMC Rank-10'],
        #    markers=True,
        #    title="Training Data")
        cols = self.df.columns.tolist()
        cols.remove('n_iter')
        self.fig = go.Figure()
        for col in cols:
                self.fig.add_trace(go.Line(x=self.df['n_iter'], y=self.df[col]))
    
    def show(self):
        self.fig.show() 
    
    def save(self, dir):
        dateObj = datetime.now()
        timestamp = dateObj.strftime("%d-%b-%Y-%H-%M-%S")
        filename = "train-stat_" + timestamp
        self.fig.write_html(os.path.join(dir, filename))  

    def set_pad_length(self, epoch_length, log_period=50):
        self.pad_len = (epoch_length // log_period)
