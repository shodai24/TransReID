import pandas as pd
import plotly.express as px
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
        self.valid_acc = []
        self.lr = []
    
    def add(self, epoch, n_iter, loss, train_acc, lr):
        self.epoch.append(epoch)
        if len(self.iter) == 0:
            self.iter.append(float(n_iter))
        else:    
            self.iter.append(float(n_iter + self.iter[-1]))
        self.loss.append(loss)
        self.train_acc.append(Tensor.cpu(train_acc).item())
        self.lr.append(lr)
    
    def update_valid_acc(self, valid_acc):
        self.valid_acc.append(Tensor.cpu(valid_acc).item())

    def plot(self):
        self.df = pd.DataFrame({'epoch': self.epoch,
                                'n_iter': self.iter,
                                'loss' : self.loss,
                                'training accuracy' : self.train_acc,
                                'validation accuracy': self.valid_acc,
                                'learning rate' : self.lr})
        #self.fig = px.line(
        #    x=self.iter,
        #    y=[self.loss, self.acc, self.lr],
        #    labels={'x': 'N_Iterations',
        #            'y': ['loss', 'accuracy', 'learning rate']},
        #    markers=True)
        
        self.fig = px.line(self.df, x='n_iter', y=['loss', 'training accuracy', 'validation accuracy', 'learning rate'], markers=True)
    
    def show(self):
        self.fig.show() 
    
    def save(self, dir):
        dateObj = datetime.now()
        timestamp = dateObj.strftime("%d-%b-%Y-%H-%M-%S")
        filename = "train-stat_" + timestamp
        self.fig.write_html(os.path.join(dir, filename))  