# packages
import matplotlib.pyplot as plt
import numpy as np
import torch

from DTC import DTC
from data import getdata
import torch.optim as optim

# features
train_data, test_data = getdata("data/fishing.csv", 100, 0.4, 1000)
sailing = train_data.dataset.dataset.Fs_f[train_data.dataset.dataset.Lab_f == 0]
fishing = train_data.dataset.dataset.Fs_f[train_data.dataset.dataset.Lab_f == 1][:len(sailing)]
data = torch.concat([sailing, fishing], axis=0)

# training
lr = 5e-4
alpha = 1.0
beta = 10
def train(model, optim, epochs, model_name, pre_train=False):
    op = optim.Adam(model.parameters(), lr=lr, weight_decay=1.2)
    if pre_train:
        for epoch in range(100):
            op.zero_grad()
            _, loss,_ = model.loss(data)
            loss.backward()
            op.step()
        model.init_centroids(data)
    
        for epoch in range(epochs):
            op.zero_grad()
            loss1, loss2, loss3 = model.loss(data)
            loss = beta * loss1 + loss2 + beta * loss3
            loss.backward()
            op.step()
    else:
        for epoch in range(epochs):
            op.zero_grad()
            loss = m.fit(data)
            loss.backward()
            op.step()
    torch.save(model.state_dict(), "parameters/" + model_name + ".pkl")
