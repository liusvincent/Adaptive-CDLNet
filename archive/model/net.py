import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.solvers import power_method, uball_project
from model.utils   import pre_process, post_process, calc_pad_2D, unpad
from model.gabor   import ConvAdjoint2dGabor

def ST(x,t):
    """Shrinkage-thresholding operation."""
    return x.sign()*F.relu(x.abs()-t)

class CDLNet(nn.Module):
    """Convolutional Dictionary Learning Network:
    Interpretable denoising DNN with adaptive thresholds for robustness.
    """
    def __init__(self,
                 K = 3,            # num. unrollings
                 M = 64,           # num. filters in each filter bank operation
                 P = 7,            # square filter side length
                 s = 1,            # stride of convolutions
                 C = 1,            # num. input channels
                 t0 = 0,           # initial threshold
                 adaptive = False, # noise-adaptive thresholds
                 init = True):     # False -> use power-method for weight init
        super(CDLNet, self).__init__()
        
        # -- OPERATOR INIT --
        self.A = nn.ModuleList([nn.Conv2d(C, M, P, stride=s, padding=(P-1)//2, bias=False)  for _ in range(K)])
        self.B = nn.ModuleList([nn.ConvTranspose2d(M, C, P, stride=s, padding=(P-1)//2, output_padding=s-1, bias=False) for _ in range(K)])
        self.D = self.B[0]                              # alias D to B[0], otherwise unused as z0 is zero
        self.t = nn.Parameter(t0*torch.ones(K,2,M,1,1)) # learned thresholds

        # set weights 
        W = torch.randn(M, C, P, P)
        for k in range(K):
            self.A[k].weight.data = W.clone()
            self.B[k].weight.data = W.clone()

        # Don't bother running code if initializing trained model from state-dict
        if init:
            print("Running power-method on initial dictionary...")
            with torch.no_grad():
                DDt = lambda x: self.D(self.A[0](x))
                L = power_method(DDt, torch.rand(1,C,128,128), num_iter=200, verbose=False)[0]
                print(f"Done. L={L:.3e}.")

                if L < 0:
                    print("STOP: something is very very wrong...")
                    sys.exit()

            # spectral normalization (note: D is alised to B[0])
            for k in range(K):
                self.A[k].weight.data /= np.sqrt(L)
                self.B[k].weight.data /= np.sqrt(L)

        # set parameters
        self.K = K
        self.M = M
        self.P = P
        self.s = s
        self.t0 = t0
        self.adaptive = adaptive

    @torch.no_grad()
    def project(self):
        r""" \ell_2 ball projection for filters, R_+ projection for thresholds"""
        self.t.clamp_(0.0) 
        for k in range(self.K):
            self.A[k].weight.data = uball_project(self.A[k].weight.data)
            self.B[k].weight.data = uball_project(self.B[k].weight.data)

    def forward(self, y, sigma=None, mask=1):
        """ LISTA + D w/ noise-adaptive thresholds""" 
        yp, params, mask = pre_process(y, self.s, mask=mask)

        # THRESHOLD SCALE-FACTOR c
        c = 0 if sigma is None or not self.adaptive else sigma/255.0

        # LISTA
        z = ST(self.A[0](yp), self.t[0,:1] + c*self.t[0,1:2])
        for k in range(1, self.K):
            z = ST(z - self.A[k](mask*self.B[k](z) - yp), self.t[k,:1] + c*self.t[k,1:2])

        # DICTIONARY SYNTHESIS
        xphat = self.D(z)
        xhat  = post_process(xphat, params)
        return xhat, z

    def forward_generator(self, y, sigma=None, mask=1):
        """ same as forward but yeilds intermediate sparse codes"""
        yp, params, mask = pre_process(y, self.s, mask=mask)
        c = 0 if sigma is None or not self.adaptive else sigma/255.0
        z = ST(self.A[0](yp), self.t[0,:1] + c*self.t[0,1:2]); yield z
        for k in range(1, self.K):
            z = ST(z - self.A[k](mask*self.B[k](z) - yp), self.t[k,:1] + c*self.t[k,1:2]); yield z
        xphat = self.D(z)
        xhat  = post_process(xphat, params)
        yield xhat

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


###########################################################
#####                  Extra                   ############
###########################################################


class AdaCDLNet_SM(nn.Module):
    """ Convolutional Dictionary Learning Network with Ada-LISTA (Single Matrix)

    Update step:
        z{k+1} = ST(z{k} - D^T W^T{k} (mask * D z{k} - y), tau{k})
        or
        z{k+1} = ST(z{k} + D^T W^T{k} (y - mask * D z{k}), tau{k})
        
        Ada-LISTA — Single Matrix definition inspired from:
        Aberdam et al., "Ada-LISTA: Learned Solvers Adaptive to Varying Models",
        arXiv:2001.08456 (2020).
        https://arxiv.org/abs/2001.08456

    Where:
        D      ...convolutional synthesis dictionary
        DT     ...convolutional analysis dictionary
        W^T{k} ...learned parameter at iteration k (weights)
        tau{k} ...learned threshold at iteration k
        y      ...input image
        z{k}   ...sparse code at iteration k
        z{0}   ...0
        mask   ...image-domain masking operator (use mask = 1 for plain denoising)
    """
    def __init__(self,
                 K = 3,             # num. unrollings
                 M = 64,            # num. filters in each filter bank operation
                 P = 7,             # square filter side length
                 s = 1,             # stride of convolutions
                 C = 1,             # num. input channels
                 t0 = 0,            # initial threshold
                 adaptive = False,  # noise-adaptive thresholds
                 init = True):      # False -> use power-method for weight init
        super(AdaCDLNet_SM, self).__init__()
        
        # -- OPERATOR INIT --
        self.W = nn.ModuleList([nn.ConvTranspose2d(C, C, P, stride=1, padding=(P-1)//2, bias=False)  for _ in range(K)]) # W^T
        self.t = nn.Parameter(t0*torch.ones(K, 2, M, 1, 1)) # learned thresholds
        self.D = nn.Parameter(torch.randn(M, C, P, P)) # conv dictionary filters, define D and D^T operators

        # set weights 
        W_temp = torch.randn(C, C, P, P)
        for k in range(K):
            self.W[k].weight.data = W_temp.clone()

        # set parameters
        self.K = K
        self.M = M
        self.P = P
        self.s = s
        self.t0 = t0
        self.adaptive = adaptive

        # Don't bother running code if initializing trained model from state-dict
        if init:
            with torch.no_grad():
                # initialize W{k} as identity convolutions
                center = P // 2
                for k in range(K):
                    self.W[k].weight.zero_()
                    for c in range(C):
                        self.W[k].weight[c, c, center, center] = 1.0

                print("Running power-method on initial dictionary...")
                DDt = lambda x: self.synthesis(self.analysis(x))
                L = power_method(DDt, torch.rand(1,C,128,128), num_iter=200, verbose=False)[0]
                print(f"Done. L={L:.3e}.")

                if L <= 0:
                    print("STOP: something is very very wrong...")
                    sys.exit()

                # spectral normalization
                self.D.div_(np.sqrt(L)) 

    @torch.no_grad()
    def project(self):
        r""" \ell_2 ball projection for W and D + projection for thresholds"""
        self.t.clamp_(0.0) 
        for k in range(self.K):
            self.W[k].weight.copy_(uball_project(self.W[k].weight))
        self.D.copy_(uball_project(self.D))

    def synthesis(self, z):
        """x = D z"""
        return F.conv_transpose2d(
            z,
            self.D,
            stride = self.s,
            padding=(self.P-1)//2,
            output_padding=self.s - 1
        )
    
    def analysis(self, x):
        """z = D^T x"""
        return F.conv2d(
            x, 
            self.D,
            stride = self.s,
            padding=(self.P-1)//2
        )

    def forward(self, y, sigma=None, mask=1):
        """ AdaLISTA w/noise-adaptive thresholds""" 
        yp, params, mask = pre_process(y, self.s, mask=mask)

        # THRESHOLD SCALE-FACTOR c
        c = 0 if sigma is None or not self.adaptive else sigma/255.0

        # Ada-LISTA
        z = ST(self.analysis(self.W[0](yp)), self.t[0,:1] + c*self.t[0,1:2])
        for k in range(1, self.K):
            z = ST(z - self.analysis(self.W[k](mask * self.synthesis(z) - yp)), self.t[k,:1] + c*self.t[k,1:2])

        # DICTIONARY SYNTHESIS
        xphat = self.synthesis(z)
        xhat  = post_process(xphat, params)
        return xhat, z

    def forward_generator(self, y, sigma=None, mask=1):
        """same as forward but yields reconstructed image"""
        yp, params, mask = pre_process(y, self.s, mask=mask)
        c = 0 if sigma is None or not self.adaptive else sigma/255.0
        z = ST(self.analysis(self.W[0](yp)), self.t[0,:1] + c*self.t[0,1:2])
        for k in range(1, self.K):
            z = ST(z - self.analysis(self.W[k](mask * self.synthesis(z) - yp)), self.t[k,:1] + c*self.t[k,1:2])
        xphat = self.synthesis(z)
        xhat  = post_process(xphat, params)
        yield xhat

    def forward_generator_sparse(self, y, sigma=None, mask=1):
        """same as forward but yields intermediate sparse codes and reconstructed image"""
        yp, params, mask = pre_process(y, self.s, mask=mask)
        c = 0 if sigma is None or not self.adaptive else sigma/255.0
        z = torch.zeros_like(self.analysis(yp))
        yield {"type": "sparsecode", "k": 0, "z": z}

        for k in range(self.K):
            z = ST(z - self.analysis(self.W[k](mask * self.synthesis(z) - yp)), self.t[k,:1] + c*self.t[k,1:2])
            yield {"type": "sparsecode", "k": k+1, "z": z}

        xphat = self.synthesis(z)
        xhat  = post_process(xphat, params)
        yield {"type": "reconimage", "xhat": xhat}
    
class AdaCDLNet_Full(nn.Module):
    """ Convolutional Dictionary Learning Network with Ada-LISTA (Full Version)

    Update step:
        z{k+1} = ST(z{k} - ( gamma{k} D^T W2^T{k} W2{k} mask * D z{k} ) + ( gamma{k} * D^T W1^T{k} y ), tau{k})

        Ada-LISTA definition inspired from:
        Aberdam et al., "Ada-LISTA: Learned Solvers Adaptive to Varying Models",
        arXiv:2001.08456 (2020).
        https://arxiv.org/abs/2001.08456

    Where:
        D         ...convolutional synthesis dictionary
        DT        ...convolutional analysis dictionary
        W1{k}     ...learned parameter at iteration k (weights)
        W2{k}     ...learned parameter at iteration k (weights)
        tau{k}    ...learned threshold at iteration k
        y         ...input image
        z{k}      ...sparse code at iteration k
        z{0}      ...0
        gammas{k} ...learned step-size parameter at iteration k
        mask      ...image-domain masking operator (use mask = 1 for plain denoising)

    """
    def __init__(self,
                 K = 3,             # num. unrollings
                 M = 64,            # num. filters in each filter bank operation
                 P = 7,             # square filter side length
                 s = 1,             # stride of convolutions
                 C = 1,             # num. input channels
                 t0 = 0,            # initial threshold
                 adaptive = False,  # noise-adaptive thresholds
                 init = True):       # False -> use power-method for weight init
        super(AdaCDLNet_Full, self).__init__()

        # -- OPERATOR INIT --
        self.W1 = nn.ModuleList([nn.ConvTranspose2d(C, C, P, stride=1, padding=(P-1)//2, bias=False)  for _ in range(K)]) # W1^T
        self.W2 = nn.ModuleList([nn.Conv2d(C, C, P, stride=1, padding=(P-1)//2, bias=False) for _ in range(K)])
        self.t = nn.Parameter(t0*torch.ones(K, 2, M, 1, 1)) # learned thresholds
        self.D = nn.Parameter(torch.randn(M, C, P, P)) # conv dictionary filters, define D and D^T operators
        self.gammas = nn.Parameter(torch.ones(K, 1, 1, 1)) # gamma for each unrolling (broadcasted)

        # set weights 
        W = torch.randn(C, C, P, P)
        for k in range(K):
            self.W1[k].weight.data = W.clone()
            self.W2[k].weight.data = W.clone()

        # set parameters
        self.K = K
        self.M = M
        self.P = P
        self.s = s
        self.t0 = t0
        self.adaptive = adaptive

        # Don't bother running code if initializing trained model from state-dict
        if init:
            with torch.no_grad():
                # initialize W{k} as identity convolutions
                center = P // 2
                for k in range(K):
                    self.W1[k].weight.zero_()
                    self.W2[k].weight.zero_()
                    for c in range(C):
                        self.W1[k].weight[c, c, center, center] = 1.0
                        self.W2[k].weight[c, c, center, center] = 1.0

                print("Running power-method on initial dictionary...")
                DDt = lambda x: self.synthesis(self.analysis(x))
                L = power_method(DDt, torch.rand(1,C,128,128), num_iter=200, verbose=False)[0]
                print(f"Done. L={L:.3e}.")

                if L <= 0:
                    print("STOP: something is very very wrong...")
                    sys.exit()

                # spectral normalization
                self.D.div_(np.sqrt(L))
                # self.gamma.div_(L)

    @torch.no_grad()
    def project(self):
        r"""\ell_2 ball projection for W and D + projection for thresholds and gamma"""
        self.t.clamp_(0.0) 
        self.gammas.clamp_(0.0)
        for k in range(self.K):
            self.W1[k].weight.copy_(uball_project(self.W1[k].weight))
            self.W2[k].weight.copy_(uball_project(self.W2[k].weight))
        self.D.copy_(uball_project(self.D))

    def synthesis(self, z):
        """
        x = D z
        """
        return F.conv_transpose2d(
            z,
            self.D,
            stride = self.s,
            padding=(self.P-1)//2,
            output_padding=self.s - 1
        )
    
    def analysis(self, x):
        """
        z = D^T x
        """
        return F.conv2d(
            x, 
            self.D,
            stride = self.s,
            padding=(self.P-1)//2
        )
    
    def W2T(self, x, k):
        """
        W2^T
        """
        return F.conv_transpose2d(
            x,
            self.W2[k].weight,
            stride = 1,
            padding=(self.P-1)//2,
            output_padding=0
        )

    def forward(self, y, sigma=None, mask=1):
        """ AdaLISTA w/noise-adaptive thresholds""" 
        yp, params, mask = pre_process(y, self.s, mask=mask)

        # THRESHOLD SCALE-FACTOR c
        c = 0 if sigma is None or not self.adaptive else sigma/255.0

        # Ada-LISTA
        z = torch.zeros_like(self.analysis(yp))
        for k in range(self.K):
            A = self.gammas[k] * self.analysis(self.W2T(self.W2[k](mask * self.synthesis(z)), k))
            B = self.gammas[k] * self.analysis(self.W1[k](yp))
            z = ST(z - A + B, self.t[k,:1] + c*self.t[k,1:2])

        # DICTIONARY SYNTHESIS
        xphat = self.synthesis(z)
        xhat  = post_process(xphat, params)
        return xhat, z

    def forward_generator(self, y, sigma=None, mask=1):
        """same as forward but yields reconstructed image"""
        yp, params, mask = pre_process(y, self.s, mask=mask)
        c = 0 if sigma is None or not self.adaptive else sigma/255.0
        z = torch.zeros_like(self.analysis(yp))
        for k in range(self.K):
            A = self.gammas[k] * self.analysis(self.W2T(self.W2[k](mask * self.synthesis(z)), k))
            B = self.gammas[k] * self.analysis(self.W1[k](yp))
            z = ST(z - A + B, self.t[k,:1] + c*self.t[k,1:2])
        xphat = self.synthesis(z)
        xhat  = post_process(xphat, params)
        return xhat, z

    def forward_generator_sparse(self, y, sigma=None, mask=1):
        """ same as forward but yields intermediate sparse codes and reconstructed image"""
        yp, params, mask = pre_process(y, self.s, mask=mask)
        c = 0 if sigma is None or not self.adaptive else sigma/255.0
        z = torch.zeros_like(self.analysis(yp))
        yield {"type": "sparsecode", "k": 0, "z": z}

        for k in range(self.K):
            A = self.gammas[k] * self.analysis(self.W2T(self.W2[k](mask * self.synthesis(z)), k))
            B = self.gammas[k] * self.analysis(self.W1[k](yp))
            z = ST(z - A + B, self.t[k,:1] + c*self.t[k,1:2])
            yield {"type": "sparsecode", "k": k+1, "z": z}

        xphat = self.synthesis(z)
        xhat  = post_process(xphat, params)
        yield {"type": "reconimage", "xhat": xhat}

# class Online_AdaCDLNet(nn.Module):
#     """ Convolutional Dictionary Learning Network with Ada-LISTA (Full Version) with Mairal inspired dictionary updates

#     Additional Features:
#         Ditionary update with forgetting factor inspired from:
#         Mairal, Julien, Francis Bach, Jean Ponce, and Guillermo Sapiro.
#         "Online Learning for Matrix Factorization and Sparse Coding."
#         Journal of Machine Learning Research 11 (2010): 19-60.
#         https://www.jmlr.org/papers/v11/mairal10a.html
#     """
#     def __init__(self,
#                  K = 3,             # num. unrollings
#                  M = 64,            # num. filters in each filter bank operation
#                  P = 7,             # square filter side length
#                  s = 1,             # stride of convolutions
#                  C = 1,             # num. input channels
#                  t0 = 0,            # initial threshold
#                  adaptive = False,  # noise-adaptive thresholds
#                  init = True):       # False -> use power-method for weight init
#         super(Online_AdaCDLNet, self).__init__()

#         # -- OPERATOR INIT --
#         self.W1 = nn.ModuleList([nn.ConvTranspose2d(C, C, P, stride=1, padding=(P-1)//2, bias=False)  for _ in range(K)]) # W1^T
#         self.W2 = nn.ModuleList([nn.Conv2d(C, C, P, stride=1, padding=(P-1)//2, bias=False) for _ in range(K)])
#         self.t = nn.Parameter(t0*torch.ones(K, 2, M, 1, 1)) # learned thresholds
#         self.D = nn.Parameter(torch.randn(M, C, P, P)) # conv dictionary filters, define D and D^T operators
#         self.gammas = nn.Parameter(torch.ones(K, 1, 1, 1)) # gamma for each unrolling (broadcasted)
        
#         # dictionary update variables
#         self.register_buffer("A_t", torch.zeros(M, M))
#         self.register_buffer("B_t", torch.zeros(M, C, P, P)) # PROBAABLY WRONG
#         self.register_buffer("dict_iter", torch.tensor(0.0))
#         self.rho = 0.0 # forgetting factor (only useful for large datasets n >= 100,000)

#         # set weights 
#         W = torch.randn(C, C, P, P)
#         for k in range(K):
#             self.W1[k].weight.data = W.clone()
#             self.W2[k].weight.data = W.clone()

#         # set parameters
#         self.K = K
#         self.M = M
#         self.P = P
#         self.s = s
#         self.t0 = t0
#         self.adaptive = adaptive

#         # Don't bother running code if initializing trained model from state-dict
#         if init:
#             with torch.no_grad():
#                 # initialize W{k} as identity convolutions
#                 center = P // 2
#                 for k in range(K):
#                     self.W1[k].weight.zero_()
#                     self.W2[k].weight.zero_()
#                     for c in range(C):
#                         self.W1[k].weight[c, c, center, center] = 1.0
#                         self.W2[k].weight[c, c, center, center] = 1.0

#                 print("Running power-method on initial dictionary...")
#                 DDt = lambda x: self.synthesis(self.analysis(x))
#                 L = power_method(DDt, torch.rand(1,C,128,128), num_iter=200, verbose=False)[0]
#                 print(f"Done. L={L:.3e}.")

#                 if L <= 0:
#                     print("STOP: something is very very wrong...")
#                     sys.exit()

#                 # spectral normalization
#                 self.D.div_(np.sqrt(L))
#                 # self.gamma.div_(L)

#     @torch.no_grad()
#     def project(self):
#         r"""\ell_2 ball projection for W and D + projection for thresholds and gamma"""
#         self.t.clamp_(0.0) 
#         self.gammas.clamp_(0.0)
#         for k in range(self.K):
#             self.W1[k].weight.copy_(uball_project(self.W1[k].weight))
#             self.W2[k].weight.copy_(uball_project(self.W2[k].weight))
#         self.D.copy_(uball_project(self.D))

#     def synthesis(self, z):
#         """
#         x = D z
#         """
#         return F.conv_transpose2d(
#             z,
#             self.D,
#             stride = self.s,
#             padding=(self.P-1)//2,
#             output_padding=self.s - 1
#         )
    
#     def analysis(self, x):
#         """
#         z = D^T x
#         """
#         return F.conv2d(
#             x, 
#             self.D,
#             stride = self.s,
#             padding=(self.P-1)//2
#         )
    
#     def W2T(self, x, k):
#         """
#         W2^T
#         """
#         return F.conv_transpose2d(
#             x,
#             self.W2[k].weight,
#             stride = 1,
#             padding=(self.P-1)//2,
#             output_padding=0
#         )
    
#     @torch.no_grad()
#     def accumulate_surrogate(self, x, z, rho=None):
#         if rho is None:
#             rho = self.rho

#         # forgetting factor beta_t = (1 - 1/t)^rho from the paper
#         self.dict_iter += 1
#         iter = float(self.dict_iter.item())
#         beta = (1.0 - 1.0 / max(iter, 1.0)) ** rho if iter > 1 else 0.0
        
#         # sum of z z^T. z has shape (B, M, Hz, Wz)
#         A_new = torch.einsum('bihw, bjhw -> ij', z, z)

#         # sum of x z^T. x has shape (B, C, Hx, Wx).  z has shape (B, M, Hz, Wz)
#         B_new = F.conv_transpose2d(z, x, stride=self.s, padding=(self.P-1)//2, output_padding=self.s-1) # WRONGGWRONGWRONG

#         # accumulate with forgetting factor
#         self.A_t.mul_(beta).add_(A_new)
#         self.B_t.mul_(beta).add_(B_new)

#     @torch.no_grad()
#     def dictionary_update(self, rho=None, eps=1e-8):
#         ...

#     def forward(self, y, sigma=None, mask=1):
#         """AdaLISTA w/noise-adaptive thresholds""" 
#         yp, params, mask = pre_process(y, self.s, mask=mask)

#         # THRESHOLD SCALE-FACTOR c
#         c = 0 if sigma is None or not self.adaptive else sigma/255.0

#         # Ada-LISTA
#         z = torch.zeros_like(self.analysis(yp))
#         for k in range(self.K):
#             A = self.gammas[k] * self.analysis(self.W2T(self.W2[k](mask * self.synthesis(z)), k))
#             B = self.gammas[k] * self.analysis(self.W1[k](yp))
#             z = ST(z - A + B, self.t[k,:1] + c*self.t[k,1:2])

#         # DICTIONARY SYNTHESIS
#         xphat = self.synthesis(z)
#         xhat  = post_process(xphat, params)
#         return xhat, z

#     def forward_generator(self, y, sigma=None, mask=1):
#         """same as forward but yields reconstructed image"""
#         yp, params, mask = pre_process(y, self.s, mask=mask)
#         c = 0 if sigma is None or not self.adaptive else sigma/255.0
#         z = torch.zeros_like(self.analysis(yp))
#         for k in range(self.K):
#             A = self.gammas[k] * self.analysis(self.W2T(self.W2[k](mask * self.synthesis(z)), k))
#             B = self.gammas[k] * self.analysis(self.W1[k](yp))
#             z = ST(z - A + B, self.t[k,:1] + c*self.t[k,1:2])
#         xphat = self.synthesis(z)
#         xhat  = post_process(xphat, params)
#         return xhat, z

#     def forward_generator_sparse(self, y, sigma=None, mask=1):
#         """same as forward but yields intermediate sparse codes and reconstructed image"""
#         yp, params, mask = pre_process(y, self.s, mask=mask)
#         c = 0 if sigma is None or not self.adaptive else sigma/255.0
#         z = torch.zeros_like(self.analysis(yp))
#         yield {"type": "sparsecode", "k": 0, "z": z}

#         for k in range(self.K):
#             A = self.gammas[k] * self.analysis(self.W2T(self.W2[k](mask * self.synthesis(z)), k))
#             B = self.gammas[k] * self.analysis(self.W1[k](yp))
#             z = ST(z - A + B, self.t[k,:1] + c*self.t[k,1:2])
#             yield {"type": "sparsecode", "k": k+1, "z": z}

#         xphat = self.synthesis(z)
#         xhat  = post_process(xphat, params)
#         yield {"type": "reconimage", "xhat": xhat}