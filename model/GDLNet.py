import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .solvers import power_method, uball_project
from .utils import pre_process, post_process, ST

class GDLNet(nn.Module):
    """ Gabor Dictionary Learning Network:"""
    def __init__(self,
                 K = 3,            # num. unrollings
                 M = 64,           # num. filters in each filter bank operation
                 P = 7,            # square filter side length
                 s = 1,            # stride of convolutions
                 C = 1,            # num. input channels
                 t0 = 0,           # initial threshold
                 order = 1,        # mixture of gabor order
                 adaptive = False, # noise-adaptive thresholds
                 shared = "",      # which gabor parameters to share (e.g. "a_psi_w0_alpha")
                 init = True):     # False -> use power-method for weight init
        super(GDLNet, self).__init__()
        
        # -- operator init --
        self.A = nn.ModuleList([ConvAdjoint2dGabor(M, C, P, stride=s, order=order) for _ in range(K)])
        self.B = nn.ModuleList([ConvAdjoint2dGabor(M, C, P, stride=s, order=order) for _ in range(K)])
        self.D = self.B[0]                              # alias D to B[0], otherwise unused as z0 is zero
        self.t = nn.Parameter(t0*torch.ones(K,2,M,1,1)) # learned thresholds

        # set weights 
        alpha = torch.randn(order, M, C, 1, 1)
        a     = torch.randn(order, M, C, 2)
        w0    = torch.randn(order, M, C, 2)
        psi   = torch.randn(order, M, C)

        for k in range(K):
            self.A[k].alpha.data = alpha.clone()
            self.A[k].a.data     = a.clone()
            self.A[k].w0.data    = w0.clone()
            self.A[k].psi.data   = psi.clone()
            self.B[k].alpha.data = alpha.clone()
            self.B[k].a.data     = a.clone()
            self.B[k].w0.data    = w0.clone()
            self.B[k].psi.data   = psi.clone()

            # Gabor parameter sharing
            if k > 0:
                if "alpha" in shared:
                    self.A[k].alpha = self.A[0].alpha
                    # never share alpha (scale) with final dictionary (B[0])
                    if k > 1:
                        self.B[k].alpha = self.B[1].alpha
                if "a_" in shared:
                    self.A[k].a     = self.A[0].a
                    self.B[k].a     = self.B[0].a
                if "w0" in shared:
                    self.A[k].w0    = self.A[0].w0
                    self.B[k].w0    = self.B[0].w0
                if "psi" in shared:
                    self.A[k].psi   = self.A[0].psi
                    self.B[k].psi   = self.B[0].psi

        # Don't bother running code if initializing trained model from state-dict
        if init:
            print("Running power-method on initial dictionary...")
            with torch.no_grad():
                DDt = lambda x: self.D(self.A[0].T(x))
                L = power_method(DDt, torch.rand(1,C,128,128), num_iter=200, verbose=False)[0]
                print(f"Done. L={L:.3e}.")

                if L < 0:
                    print("STOP: something is very very wrong...")
                    sys.exit()

            # spectral normalization (note: D is alised to B[0])
            for k in range(K):
                self.A[k].alpha.data /= np.sqrt(L)
                self.B[k].alpha.data /= np.sqrt(L)
                if "alpha" in shared:
                    self.B[1].alpha.data /= np.sqrt(L)
                    break

        # set parameters
        self.K = K
        self.M = M
        self.P = P
        self.s = s
        self.t0 = t0
        self.order = order
        self.adaptive = adaptive

    @torch.no_grad()
    def project(self):
        r""" \ell_2 ball projection for filters, R_+ projection for thresholds"""
        self.t.clamp_(0.0) 

    def forward(self, y, sigma=None, mask=1):
        """ LISTA + D w/ noise-adaptive thresholds""" 
        yp, params, mask = pre_process(y, self.s, mask=mask)

        # THRESHOLD SCALE-FACTOR c
        c = 0 if sigma is None or not self.adaptive else sigma/255.0

        # LISTA
        z = ST(self.A[0].T(yp), self.t[0,:1] + c*self.t[0,1:2])
        for k in range(1, self.K):
            z = ST(z - self.A[k].T(mask*self.B[k](z) - yp), self.t[k,:1] + c*self.t[k,1:2])

        # DICTIONARY SYNTHESIS
        xphat = self.D(z)
        xhat  = post_process(xphat, params)
        return xhat, z

    def forward_generator(self, y, sigma=None, mask=1):
        """same as forward but yeilds intermediate sparse codes"""
        yp, params, mask = pre_process(y, self.s, mask=mask)
        c = 0 if sigma is None or not self.adaptive else sigma/255.0
        z = ST(self.A[0].T(yp), self.t[0,:1] + c*self.t[0,1:2]); yield z
        for k in range(1, self.K):
            z = ST(z - self.A[k].T(mask*self.B[k](z) - yp), self.t[k,:1] + c*self.t[k,1:2]); yield z
        xphat = self.D(z)
        xhat  = post_process(xphat, params)
        yield xhat

def gabor_kernel(a, w0, psi, ks):
    """ Generate a batch of gabor filterbank via inverse width (a) and frequency (w0) params
    
    a   (precision):   (batch, out_chan, in_chan, 2) 
    w0  (center freq): (batch, out_chan, in_chan, 2)
    psi (phase):       (batch, out_chan, in_chan)
    h   (output):      (batch, out_chan, in_chan, ks, ks)
    """
    a   =  a[:,:,:,None,None,:]
    w0  = w0[:,:,:,None,None,:]
    psi = psi[:,:,:,None,None]

    # x spatial grid
    i = torch.arange(ks).to(a.device)
    x = torch.stack(torch.meshgrid(i,i, indexing='ij'), dim=2)[None,None,...]

    # x0 spatial center
    x0 = torch.tensor([(ks-1)/2,(ks-1)/2], device=a.device)[None,None,None,None,None,:]

    h = torch.exp( -torch.sum((a*(x-x0))**2, dim=-1) ) * \
        torch.cos(torch.sum(w0*(x-x0), dim=-1) + psi)
    return h

class ConvAdjoint2dGabor(nn.Module):
    """ Convolution with a Gabor kernel
    """
    def __init__(self, nic, noc, ks, stride=2, order=1):
        super(ConvAdjoint2dGabor, self).__init__()
        self.alpha = nn.Parameter(torch.randn((order, nic, noc, 1, 1))) 
        self.a     = nn.Parameter(torch.randn((order, nic, noc, 2)))
        self.w0    = nn.Parameter(torch.randn((order, nic, noc, 2)))
        self.psi   = nn.Parameter(torch.randn((order, nic, noc)))
        self.order  = order
        self.stride = stride
        self.ks = ks
        p = (ks-1)//2
        self._pad = (p,p,p,p)
        self._output_padding = nn.ConvTranspose2d(1,1,ks,stride=self.stride)._output_padding
        
    def get_filter(self, transpose=False):
        if transpose:
            w0, psi = -self.w0, -self.psi
        else:
            w0, psi = self.w0, self.psi
        return (self.alpha*gabor_kernel(self.a, w0, psi, self.ks)).sum(dim=0)

    def T(self, x):
        pad_x = F.pad(x, self._pad, mode='constant')
        return F.conv2d(pad_x, self.get_filter(transpose=True), stride=self.stride)

    def forward(self, x):
        output_size = (x.shape[0], x.shape[1], self.stride*x.shape[2], self.stride*x.shape[3])
        op = self._output_padding(x, output_size,
                                  (self.stride, self.stride),
                                  (self._pad[0], self._pad[0]),
                                  (self.ks, self.ks))

        return F.conv_transpose2d(x, self.get_filter(),
                                  padding = self._pad[0],
                                  stride  = self.stride,
                                  output_padding = op)
