import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import unpad, calc_pad_2D

""" This file contains other nets
"""

class DnCNN(nn.Module):
	"""DnCNN implementation taken from github.com/SaoYan/DnCNN-PyTorch"""
	def __init__(self, Co=1, Ci=1, K=17, M=64, P=3):
		super(DnCNN, self).__init__()
		pad = (P-1)//2
		layers = []
		layers.append(nn.Conv2d(Ci, M, P, padding=pad, bias=True))
		layers.append(nn.ReLU(inplace=True))

		for _ in range(K-2):
			layers.append(nn.Conv2d(M, M, P, padding=pad, bias=False))
			layers.append(nn.BatchNorm2d(M))
			layers.append(nn.ReLU(inplace=True))

		layers.append(nn.Conv2d(M, Co, P, padding=pad, bias=True))
		self.dncnn = nn.Sequential(*layers)

	def project(self):
		return

	def forward(self, y, *args, **kwargs):
		n = self.dncnn(y)
		return y-n, n

class FFDNet(DnCNN):
	""" Implementation of FFDNet.
	"""
	def __init__(self, C=1, K=17, M=64, P=3):
		super(FFDNet, self).__init__(Ci=4*C+1, Co=4*C, K=K, M=M, P=P)
	
	def forward(self, y, sigma_n, **kwargs):
		pad = calc_pad_2D(*y.shape[2:], 2)
		yp  = F.pad(y, pad, mode='reflect')
		noise_map = (sigma_n/255.0)*torch.ones(1,1,yp.shape[2]//2,yp.shape[3]//2,device=y.device)
		z = F.pixel_unshuffle(yp, 2)
		z = torch.cat([z, noise_map], dim=1)
		z = self.dncnn(z)
		xhatp = F.pixel_shuffle(z, 2)
		xhat  = unpad(xhatp, pad)
		return xhat, noise_map