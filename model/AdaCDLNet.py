import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .solvers import power_method, uball_project
from .utils import pre_process, post_process, ST

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
        r""" \ell_2 ball projection for W and D + projection for thresholds
        """
        self.t.clamp_(0.0) 
        for k in range(self.K):
            self.W[k].weight.copy_(uball_project(self.W[k].weight))
        self.D.copy_(uball_project(self.D))

    def synthesis(self, z, D=None):
        """ x = D z
        """
        if D is None:
            D = self.D

        return F.conv_transpose2d(
            z,
            D,
            stride = self.s,
            padding=(self.P-1)//2,
            output_padding=self.s - 1
        )
    
    def analysis(self, x, D=None):
        """ z = D^T x
        """
        if D is None:
            D = self.D

        return F.conv2d(
            x, 
            D,
            stride = self.s,
            padding=(self.P-1)//2
        )

    def forward(self, y, sigma=None, mask=1, D=None):
        """ AdaLISTA w/noise-adaptive thresholds
        """ 
        if D is None:
            D = self.D
        else:
            D = D.to(device=y.device, dtype=y.dtype)

        yp, params, mask = pre_process(y, self.s, mask=mask)

        # THRESHOLD SCALE-FACTOR c
        c = 0 if sigma is None or not self.adaptive else sigma/255.0

        # Ada-LISTA
        z = ST(self.analysis(self.W[0](yp), D), self.t[0,:1] + c*self.t[0,1:2])
        for k in range(1, self.K):
            z = ST(z - self.analysis(self.W[k](mask * self.synthesis(z, D) - yp), D), self.t[k,:1] + c*self.t[k,1:2])

        # DICTIONARY SYNTHESIS
        xphat = self.synthesis(z, D)
        xhat  = post_process(xphat, params)
        return xhat, z

    def forward_generator(self, y, sigma=None, mask=1, D=None):
        """ same as forward but yields reconstructed image
        """
        if D is None:
            D = self.D
        else:
            D = D.to(device=y.device, dtype=y.dtype)

        yp, params, mask = pre_process(y, self.s, mask=mask)
        c = 0 if sigma is None or not self.adaptive else sigma/255.0
        z = ST(self.analysis(self.W[0](yp), D), self.t[0,:1] + c*self.t[0,1:2])
        for k in range(1, self.K):
            z = ST(z - self.analysis(self.W[k](mask * self.synthesis(z, D) - yp), D), self.t[k,:1] + c*self.t[k,1:2])
        xphat = self.synthesis(z, D)
        xhat  = post_process(xphat, params)
        yield xhat

    def forward_generator_sparse(self, y, sigma=None, mask=1, D=None):
        """ same as forward but yields intermediate sparse codes and reconstructed image
        """
        if D is None:
            D = self.D
        else:
            D = D.to(device=y.device, dtype=y.dtype)

        yp, params, mask = pre_process(y, self.s, mask=mask)
        c = 0 if sigma is None or not self.adaptive else sigma/255.0
        z = torch.zeros_like(self.analysis(yp, D))
        yield {"type": "sparsecode", "k": 0, "z": z}

        for k in range(self.K):
            z = ST(z - self.analysis(self.W[k](mask * self.synthesis(z, D) - yp), D), self.t[k,:1] + c*self.t[k,1:2])
            yield {"type": "sparsecode", "k": k+1, "z": z}

        xphat = self.synthesis(z, D)
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
        r"""\ell_2 ball projection for W and D + projection for thresholds and gamma
        """
        self.t.clamp_(0.0) 
        self.gammas.clamp_(0.0)
        for k in range(self.K):
            self.W1[k].weight.copy_(uball_project(self.W1[k].weight))
            self.W2[k].weight.copy_(uball_project(self.W2[k].weight))
        self.D.copy_(uball_project(self.D))

    def synthesis(self, z, D=None):
        """ x = D z
        """
        if D is None:
            D = self.D

        return F.conv_transpose2d(
            z,
            D,
            stride = self.s,
            padding=(self.P-1)//2,
            output_padding=self.s - 1
        )
    
    def analysis(self, x, D=None):
        """ z = D^T x
        """
        if D is None:
            D = self.D

        return F.conv2d(
            x, 
            D,
            stride = self.s,
            padding=(self.P-1)//2
        )
    
    def W2T(self, x, k):
        """ W2^T
        """
        return F.conv_transpose2d(
            x,
            self.W2[k].weight,
            stride = 1,
            padding=(self.P-1)//2,
            output_padding=0
        )

    def forward(self, y, sigma=None, mask=1, D=None):
        """ AdaLISTA w/noise-adaptive thresholds
        """ 
        if D is None:
            D = self.D
        else:
            D = D.to(device=y.device, dtype=y.dtype)

        yp, params, mask = pre_process(y, self.s, mask=mask)

        # THRESHOLD SCALE-FACTOR c
        c = 0 if sigma is None or not self.adaptive else sigma/255.0

        # Ada-LISTA
        z = torch.zeros_like(self.analysis(yp, D))
        for k in range(self.K):
            A = self.gammas[k] * self.analysis(self.W2T(self.W2[k](mask * self.synthesis(z, D)), k), D)
            B = self.gammas[k] * self.analysis(self.W1[k](yp), D)
            z = ST(z - A + B, self.t[k,:1] + c*self.t[k,1:2])

        # DICTIONARY SYNTHESIS
        xphat = self.synthesis(z, D)
        xhat  = post_process(xphat, params)
        return xhat, z

    def forward_generator(self, y, sigma=None, mask=1, D=None):
        """same as forward but yields reconstructed image"""
        if D is None:
            D = self.D
        else:
            D = D.to(device=y.device, dtype=y.dtype)

        yp, params, mask = pre_process(y, self.s, mask=mask)
        c = 0 if sigma is None or not self.adaptive else sigma/255.0
        z = torch.zeros_like(self.analysis(yp, D))
        for k in range(self.K):
            A = self.gammas[k] * self.analysis(self.W2T(self.W2[k](mask * self.synthesis(z, D)), k), D)
            B = self.gammas[k] * self.analysis(self.W1[k](yp), D)
            z = ST(z - A + B, self.t[k,:1] + c*self.t[k,1:2])
        xphat = self.synthesis(z, D)
        xhat  = post_process(xphat, params)
        return xhat, z

    def forward_generator_sparse(self, y, sigma=None, mask=1, D=None):
        """ same as forward but yields intermediate sparse codes and reconstructed image"""
        if D is None:
            D = self.D
        else:
            D = D.to(device=y.device, dtype=y.dtype)

        yp, params, mask = pre_process(y, self.s, mask=mask)
        c = 0 if sigma is None or not self.adaptive else sigma/255.0
        z = torch.zeros_like(self.analysis(yp, D))
        yield {"type": "sparsecode", "k": 0, "z": z}

        for k in range(self.K):
            A = self.gammas[k] * self.analysis(self.W2T(self.W2[k](mask * self.synthesis(z, D)), k), D)
            B = self.gammas[k] * self.analysis(self.W1[k](yp), D)
            z = ST(z - A + B, self.t[k,:1] + c*self.t[k,1:2])
            yield {"type": "sparsecode", "k": k+1, "z": z}

        xphat = self.synthesis(z, D)
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