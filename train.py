import os, sys, json
from tqdm import tqdm
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn

from model import *
from data import get_fit_loaders
from utils import awgn, gen_bayer_mask, check_gpu, dictionary_permute, dictionary_noisy, random_dictionary

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--online", action="store_true", help="flag for online setup")

class Trainer:
    """ Universal Trainer Class:
    Accepts model, opt, loaders, sched as prereq
    trains model from image reconstruction loss
    or from MCSURE

    creates:
    ------
    + net.ckpt: last model
    + best_val.ckpt: best validation model
    + 0.ckpt: initial model
    -----
    + train.txt: history of all psnr
    + val.txt: history of all val psnr
    + test.txt: final test psnr 
    + backtrack.txt: history of failed epochs
    """
    def __init__(self, net, opt, loaders, sched=None,
        device=torch.device("cpu"), save_dir=None, 
        clip_grad=1, noise_std=25, demosaic=False,
        verbose=True, val_freq=1, save_freq=1, 
        mcsure=False, backtrack_thresh=1, epochs=6000):
        """ init constructor
        """
        # init variables
        self.net = net
        self.opt = opt
        self.loaders = loaders
        self.sched = sched
        self.device = device
        self.save_dir = save_dir
        self.clip_grad = clip_grad
        self.noise_std = noise_std
        self.demosaic = demosaic
        self.verbose = verbose
        self.val_freq = val_freq
        self.save_freq = save_freq
        self.mcsure = mcsure
        self.backtrack_thresh = backtrack_thresh
        self.epochs = epochs

        if not isinstance(self.noise_std, (list, tuple)):
            self.noise_std = (self.noise_std, self.noise_std)  
        self.curr_epoch = 0

        # vars for backtracking
        self.top_psnr = {"train": 0, "val": 0, "test": 0}

        # vars for early stopping
        self.best_val_psnr = -float("inf")
        self.bad_epochs = 0
        self.patience = 500
        self.min_delta = 0.01

    def getlr(self):
        """ lr getter
        """
        return [pg['lr'] for pg in self.opt.param_groups]
    
    def setlr(self, lr):
        """ lr setter
        """
        # if lr is not a list
        if not isinstance(lr, (list, np.ndarray)):
            lr = [lr for _ in range(len(self.opt.param_groups))]
        # set the new learning rates
        for (i, pg) in enumerate(self.opt.param_groups):
            pg['lr'] = lr[i]

    def run_batch(self, batch, phase, D_used=None):
        """ Run phase on batch
        """
        # prerequisite
        batch = batch.to(self.device)
        mask = gen_bayer_mask(batch) if self.demosaic else 1
        if phase in ["val", "test"]:
            phase_nstd = (self.noise_std[0]+self.noise_std[1])/2.0
        else:
            phase_nstd = self.noise_std

        # add noise to an extra batch
        noisy_batch, sigma_n = awgn(batch, phase_nstd)
        obsrv_batch = mask * noisy_batch

        self.opt.zero_grad() # clear gradients

        with torch.set_grad_enabled(phase == "train"):
            # batch_hat, _ = self.net(obsrv_batch, sigma_n, mask=mask, D_used=D_used) # call net
            if D_used is None:
                batch_hat, _ = self.net(obsrv_batch, sigma_n, mask=mask)
            else:
                batch_hat, _ = self.net(obsrv_batch, sigma_n, mask=mask, D=D_used)
            # Unsupervised (MCSURE) loss
            if self.mcsure and phase == "train":
                h = 1e-3 # tiny perturbation size
                # create a new image of small perturbation
                b = torch.randn_like(obsrv_batch)
                # batch_hat_b, _ = self.net(obsrv_batch + (h*b), sigma_n, mask=mask, D_used=D_used)
                if D_used is None:
                    batch_hat_b, _ = self.net(obsrv_batch + h*b, sigma_n, mask=mask)
                else:
                    batch_hat_b, _ = self.net(obsrv_batch + h*b, sigma_n, mask=mask, D=D_used)

                div = 2.0*torch.mean(((sigma_n/255.0)**2)*b*(batch_hat_b-batch_hat)) / h
                loss = torch.mean((obsrv_batch - batch_hat)**2) + div # (+ div) offset for perturbation
            # supervised typical reconstruction loss
            else: loss = torch.mean((batch - batch_hat)**2)

            # train the net
            if phase == 'train':
                loss.backward()
                if self.clip_grad is not None:
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.clip_grad)
                self.opt.step()
                if hasattr(self.net, "project"): 
                    self.net.project()

        return loss
    
    def progress_bar(self, t, loss_value):
        """ Add info to the progress bar:
        + verbose option
        """
        if self.verbose:
            total_norm = grad_norm(self.net.parameters())
            t.set_postfix_str(f"loss={loss_value:.1e}|gnorm={total_norm:.1e}")

    def run_phase(self, phase, epoch):
        """ Function for each phase in fit function {train, val, test}
        """
        # prerequisite
        self.net.train() if phase == "train" else self.net.eval()
        total_psnr = 0.0
        t = tqdm(self.loaders[phase], desc=f"{phase.upper()}-E{epoch}", dynamic_ncols=True)

        # iterate through batches
        batch_count = 0
        for itern, batch in enumerate(t):
            # calculate loss and psnr
            loss = self.run_batch(batch, phase)
            loss_value = loss.item()
            psnr = -10 * np.log10(loss_value)
            total_psnr += psnr
            batch_count += 1

            self.progress_bar(t, loss_value)

        # output avg_psnr, return it and loss_value
        if batch_count == 0:
            raise ValueError(f"{phase} loader is empty.")
        avg_psnr = total_psnr / batch_count
        print(f"{phase.upper()} PSNR: {avg_psnr:.3f} dB")
        return avg_psnr, loss_value
    
    def fit(self, start_epoch=1):
        """ Function to train net using Dataset:
        Given a model run train, val, test phases
        trains for max 6000 epochs
        val phase occurs per val_freq(uency)
        """
        print(f"fit: using device {self.device}")

        # initial checkpoint for model
        os.makedirs(self.save_dir, exist_ok=True)
        init_path = os.path.join(self.save_dir, "0.ckpt")
        if not os.path.exists(init_path):
            print("Saving initialization to 0.ckpt")
            save_ckpt(init_path, self.net, 0, self.opt, self.sched)

        # epoch phases
        epoch = start_epoch
        while epoch <= self.epochs:
            self.curr_epoch = epoch

            need_backtrack = False
            for phase in ["train", "val", "test"]:

                # skip test except at final epoch
                if phase == "test" and epoch != self.epochs:
                    continue
                # skip val unless epoch matches val_freq
                if phase == "val" and epoch % self.val_freq != 0:
                    continue 
                
                # run each phase {train, val, test}
                psnr, loss = self.run_phase(phase, epoch)
                
                # record each psnr
                with open(os.path.join(self.save_dir, f"{phase}.txt"), "a") as f:
                    f.write(f"{psnr:.3f}, ")

                # update best psnr
                if psnr > self.top_psnr[phase]:
                    self.top_psnr[phase] = psnr
                # early backtracking check, if model diverged
                elif ((psnr + self.backtrack_thresh) < self.top_psnr[phase]
                    or np.isnan(loss) or np.isinf(loss)):
                    need_backtrack = True
                    break
                    
                # validate the epoch
                if phase == "val":
                    # update best validation PSNR so far
                    if psnr > (self.best_val_psnr + self.min_delta):
                        self.best_val_psnr = psnr
                        self.bad_epochs = 0
                        best_path = os.path.join(self.save_dir, "best_val.ckpt")
                        print(f"New best validation PSNR: {psnr:.3f} dB")
                        save_ckpt(best_path, self.net, epoch, self.opt, self.sched)
                    else:
                        self.bad_epochs += self.val_freq
                        print(f"No validation improvement for {self.bad_epochs} epochs")

                    # early stopping
                    if self.bad_epochs >= self.patience:
                        print("Early stopping triggered.")
                        return
            
            # backtracking process
            if need_backtrack:
                # record failed epoch in backtrack.txt
                with open(os.path.join(self.save_dir, f'backtrack.txt'),'a') as psnr_file:
                    psnr_file.write(f'{epoch}  ')

                # find path for checkpoint 0
                if epoch <= self.save_freq:  
                    ckpt_path = os.path.join(self.save_dir, '0.ckpt')
                    epoch = 0
                else:
                    # find path for most recent checkpoint
                    ckpt_path = os.path.join(self.save_dir, 'net.ckpt')
                    # find proper epoch
                    if epoch % self.save_freq == 0:
                        epoch = epoch - self.save_freq
                    else:
                        epoch = epoch - (epoch % self.save_freq)
                print(f"Loss has diverged. Backtracking to {ckpt_path} ...")

                old_lr = np.array(self.getlr())
                # load ckpt model
                self.net, self.opt, self.sched, _ = load_ckpt(ckpt_path, self.net, self.opt, self.sched, self.device)
                self.net.to(self.device)
                new_lr = old_lr * 0.8 # reduce learning rate
                self.setlr(new_lr)
                print("Updated Learning Rate(s):", new_lr)

                epoch = epoch + 1
                continue

            # update scheduler
            if self.sched is not None:
                self.sched.step()
                if hasattr(self.sched, "step_size") and epoch % self.sched.step_size == 0:
                    print("Updated Learning Rate(s): ")
                    print(self.getlr())

            # save model's checkpoint
            if epoch % self.save_freq == 0:
                ckpt_path = os.path.join(self.save_dir, 'net.ckpt')
                print('Checkpoint: ' + ckpt_path)
                save_ckpt(ckpt_path, self.net, epoch, self.opt, self.sched)

            epoch = epoch + 1 # increment epoch
# end Trainer

class AdaTrainer(Trainer):
    """ Adaptive Trainer Class for AdaCDLNet:
    inherits Trainer class
    includes training with perturbed dictionaries
    (more intensive training of Ada-LISTA)
    """
    def __init__(self, clean_prob=0.5, noise_prob=0.3,
                permute_prob=0.15, random_prob=0.05, 
                warmup_frac=0.05, seed=None, **kwargs):
        """ init constructor
        """
        super().__init__(**kwargs)
        self.clean_prob = clean_prob
        self.noise_prob = noise_prob
        self.permute_prob = permute_prob
        self.random_prob = random_prob
        self.warmup_epochs = int(self.epochs * warmup_frac)
        self.last_perturb = "clean"
        self.rng = np.random.default_rng(seed)
        
        prob_sum = clean_prob + noise_prob + permute_prob + random_prob
        if not np.isclose(prob_sum, 1.0):
            raise ValueError(f"Perturbation probabilities must sum to 1. Got {prob_sum}")
        

    def perturb(self):
        """ perturb dictionary based on prob chances
        and no. warmup epochs
        detach() the dictionary if perturb
        """
        r = self.rng.random()
        # if epochs less than warmup epochs use normal dictionary
        if (self.curr_epoch <= self.warmup_epochs or r < self.clean_prob):
            self.last_perturb = "clean"
            return self.net.D
        elif r < self.noise_prob + self.clean_prob:
            self.last_perturb = "noisy"
            return dictionary_noisy(self.net.D.detach()).detach()
        elif r < self.permute_prob + self.noise_prob + self.clean_prob:
            self.last_perturb = "permute"
            return dictionary_permute(self.net.D.detach()).detach()
        else:
            self.last_perturb = "random"
            return random_dictionary(self.net.D.detach()).detach()

    def run_batch(self, batch, phase):
        """ Override run_batch to carry out dict perturbation
        """
        if phase == "train":
            D_used = self.perturb()
        else:
            self.last_perturb = "clean"
            D_used = self.net.D
        return super().run_batch(batch, phase, D_used=D_used)
    
    def progress_bar(self, t, loss_value):
        """ Add info to the progress bar
        + verbose option
        + dict perturbation info
        """
        if self.verbose:
            total_norm = grad_norm(self.net.parameters())
            t.set_postfix_str(f"loss={loss_value:.1e}|gnorm={total_norm:.1e}|dict={self.last_perturb}")
# end AdaTrainer(Trainer)

def grad_norm(params):
    """ Computes norm of mini-batch gradient
    """
    total_norm = 0
    for p in params:
        param_norm = torch.tensor(0)
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
        total_norm = total_norm + param_norm.item()**2
    return total_norm**(.5)

def save_ckpt(path, net=None,epoch=None,opt=None,sched=None):
    """ Save Checkpoint
    Saves net, optimizer, scheduler state dicts and epoch num to path.
    """
    getSD = lambda obj: obj.state_dict() if obj is not None else None
    torch.save({'epoch': epoch,
                'net_state_dict': getSD(net),
                'opt_state_dict':   getSD(opt),
                'sched_state_dict': getSD(sched)
                }, path)

def load_ckpt(path, net=None,opt=None,sched=None, device='cpu'):
    """ Load Checkpoint
    Loads net, optimizer, scheduler and epoch number
    from state dict stored in path.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    def setSD(obj, name):
        if obj is not None and name+"_state_dict" in ckpt:
            print(f"Loading {name} state-dict...")
            obj.load_state_dict(ckpt[name+"_state_dict"])
        return obj

    net = setSD(net, 'net')
    opt   = setSD(opt, 'opt')
    sched = setSD(sched, 'sched')
    return net, opt, sched, ckpt['epoch']

def init_model(model_type, model_args, train_args, paths, device=torch.device("cpu")):
    """ Return model, optimizer, scheduler with optional 
    initialization from checkpoint.
    """
    init = False if paths['ckpt'] is not None else True

    if model_type == "CDLNet":
        net = CDLNet(**model_args, init=init)
    elif model_type == "GDLNet":
        net = GDLNet(**model_args, init=init)
    elif model_type == "AdaCDLNet_SM":
        net = AdaCDLNet_SM(**model_args, init=init)
    elif model_type == "AdaCDLNet_Full":
        net = AdaCDLNet_Full(**model_args, init=init)
    elif model_type == "DnCNN":
        net = DnCNN(**model_args)
    elif model_type == "FFDNet":
        net = FFDNet(**model_args)
    else:
        raise NotImplementedError

    net.to(device)
    opt   = torch.optim.Adam(net.parameters(), **train_args['opt'])     
    sched = torch.optim.lr_scheduler.StepLR(opt, **train_args['sched'])
    ckpt_path = paths['ckpt']

    if ckpt_path is not None:
        print(f"Initializing net from {ckpt_path} ...")
        net, opt, sched, epoch0 = load_ckpt(ckpt_path, net, opt, sched, device)
    else:
        epoch0 = 0

    print("Current Learning Rate(s):")
    for param_group in opt.param_groups:
        print(param_group['lr'])

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total Number of Parameters: {total_params:,}")

    print(f"Using {paths['save']} ...")
    os.makedirs(paths['save'], exist_ok=True)
    return net, opt, sched, epoch0

def save_args(args, ckpt=True):
    """ Write argument dictionary to file,
    with optionally writing the checkpoint.
    """
    save_path = args['paths']['save']
    if ckpt:
        ckpt_path = os.path.join(save_path, f"net.ckpt")
        args['paths']['ckpt'] = ckpt_path
    with open(os.path.join(save_path, "args.json"), "+w") as outfile:
        outfile.write(json.dumps(args, indent=4, sort_keys=True))

def main(args):
    """ Given argument dictionary 
    load data, initialize model, and fit model.
    """
    # prerequisite
    device = check_gpu()
    model_type, model_args, train_args, paths = [args[item] for item in ['type', 'model', 'train', 'paths']]
    loaders = get_fit_loaders(**train_args['loaders'])

    # initialize model
    net, opt, sched, epoch0= init_model(model_type, model_args, train_args, paths, device=device)

    # initialize net.ckpt path in json
    save_args(args, ckpt=True)

    # initialize trainer
    if model_type == "AdaCDLNet_SM" or model_type == "AdaCDLNet_Full":
        trainer = AdaTrainer(net=net, opt=opt, loaders=loaders, sched=sched,
                    device=device, save_dir=paths["save"], **train_args['fit'], 
                    **train_args['dict'])
    else:
        trainer = Trainer(net=net, opt=opt, loaders=loaders, sched=sched,
                    device=device, save_dir=paths["save"], **train_args['fit'])

    try: 
        trainer.fit(start_epoch=epoch0 + 1) # train model
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        ckpt_path = os.path.join(paths["save"], "net.ckpt")
        save_ckpt(ckpt_path, trainer.net, epoch=trainer.curr_epoch - 1, 
                  opt=trainer.opt, sched=trainer.sched)
        print(f"Saved checkpoint at epoch {trainer.curr_epoch} to {ckpt_path}")
    
    # online preparation
    # if trainer.isinstance(AdaTrainer) and ARGS.online:
    #     for p in net.parameters():
    #         p.requires_grad_(False) # disables learning for solver
    #     net.D.requires_grad_(True) # enables learning for dictionary
    #     opt = torch.optim.Adam([net.D], **train_args['opt'])
    #     ckpt_path = os.path.join(paths["save"], "net.ckpt")
    #     save_ckpt(ckpt_path, trainer.net, epoch=trainer.curr_epoch, 
    #               opt=opt, sched=trainer.sched)

    # online_fit(
    #     net,
    #     opt,
    #     dataset_dir="./dataset/Set12",
    #     batch_size=1,
    #     device=device,
    #     save_dir=paths["save"],
    #     noise_std=train_args["fit"].get("noise_std", 25),
    #     demosaic=train_args["fit"].get("demosaic", False),
    #     load_color=train_args["loaders"].get("load_color", False)
    # )

if __name__ == "__main__":
    """ Load arguments dictionary from json file to pass to main.
    """
    if len(sys.argv)<2:
        print('ERROR: usage: train.py [path/to/arg_file.json]')
        sys.exit(1)
    args_file = open(sys.argv[1])
    args = json.load(args_file)
    pprint(args)
    args_file.close()
    main(args)