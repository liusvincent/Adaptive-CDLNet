from archive.model.net import AdaCDLNet_SM, AdaCDLNet_Full
import torch

x = torch.randn(1, 1, 128, 128)
sigma = torch.tensor([25.0])

net_sm = AdaCDLNet_SM(K=3, M=32, P=7, C=1, adaptive=True, init=False)
y_sm, z_sm = net_sm(x, sigma=sigma)
print("AdaCDLNet_SM output:", y_sm.shape, z_sm.shape)

net_full = AdaCDLNet_Full(K=3, M=32, P=7, C=1, adaptive=True, init=False)
y_full, z_full = net_full(x, sigma=sigma)
print("AdaCDLNet_Full output:", y_full.shape, z_full.shape)