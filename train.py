import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('model/')
from model.PDCRN import UDC_Arc
from model.DBWN_D import DBWN_D
from model.DBWN_H import DBWN_H
from dataset import Custom_Dataset
from losses import ContrastLoss
import os
import wandb
import pyiqa
import tqdm
from torchsummary import summary
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from utils import save_on_master
class Train():
    def __init__(self,args) -> None:
        self.args = args
        self.init_distributed_mode()
        self.fix_random_seeds(self.args.seed)
        cudnn.benchmark = True
        self.current_epoch = 0
        self.best_psnr = 0
        self.best_ssim = 0
        self.model_intiliaser()
        self.data_intiliaser()
        self.losses_opt_and_metrics_init()
        self.init_summary()
    def model_intiliaser(self):
        if self.args.model_type.endswith("CR"):
            model_name = self.args.model_type.split('_')[0]
        else:
            model_name = self.args.model_type
        print(f'Model Name = {model_name}')
        if model_name=='PDCRN':
            self.model = UDC_Arc('cuda',self.args.in_ch,self.args.num_filters,self.args.dilation_rates,self.args.nPyramidFilters)
        elif model_name=='DBWND':
            self.model = DBWN_D('cuda',num_filters = self.args.num_filters)
        elif model_name=='DBWNH':
            self.model = DBWN_H('cuda',num_filters = self.args.num_filters)
        else:
            print("Enter a valid model name")
        self.model = self.model.cuda()
        summary(self.model,input_size=(3,self.args.im_shape[0],self.args.im_shape[0]))
        self.model = torch.nn.parallel.DistributedDataParallel(self.model,[self.args.gpu])
        return None
    def init_distributed_mode(self):
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.args.rank = int(os.environ["RANK"])
            self.args.world_size = int(os.environ['WORLD_SIZE'])
            self.args.gpu = int(os.environ['LOCAL_RANK'])
        elif torch.cuda.is_available():
            print('Will run the code on one GPU.')
            self.args.rank, self.args.gpu, self.args.world_size = 0, 0, 1
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group(backend="nccl",init_method=self.args.dist_url,world_size=self.args.world_size,rank=self.args.rank)

        torch.cuda.set_device(self.args.gpu)
        print('| distributed init (rank {}): {}'.format(
            self.args.rank, self.args.dist_url), flush=True)
        dist.barrier()
    def fix_random_seeds(self,seed=31):
        """
        Fix random seeds.
        """
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def data_intiliaser(self):
        train_ds = Custom_Dataset(self.args.train_path,im_shape = self.args.im_shape ,is_train=True)
        train_sampler = torch.utils.data.DistributedSampler(train_ds, shuffle=True)
        self.train_dataloader = DataLoader(train_ds,batch_size=self.args.batch_size,num_workers=8,sampler=train_sampler)
        val_ds = Custom_Dataset(self.args.test_path,is_train=False)
        val_sampler = torch.utils.data.DistributedSampler(val_ds, shuffle=False)
        self.val_dataloader = DataLoader(val_ds,batch_size=1,num_workers=8,sampler=val_sampler)
        return None
    def init_summary(self):
        wandb.init(project=f"{self.args.model_type}",name=self.args.log_name)
        return
    def losses_opt_and_metrics_init(self):
        total_count = self.args.epochs*len(self.train_dataloader)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.LR)
        self.scheduler = CosineAnnealingLR(self.optimizer,total_count,self.args.LR*(10**(-4)))
        if self.args.model_type.endswith('CR'):
            self.criterion_CR = ContrastLoss()
        self.criterion = torch.nn.L1Loss().cuda()
        # self.psnr  = PeakSignalNoiseRatio().cuda()
        # self.ssim = StructuralSimilarityIndexMeasure().cuda()
        self.psnr = pyiqa.create_metric('psnr',device = 'cuda')
        self.ssim = pyiqa.create_metric('ssim',device = 'cuda')

    def train_epoch(self):
        self.model.train()
        for count,(inputs, gt) in enumerate(tqdm.tqdm(self.train_dataloader)):
            inputs = inputs.cuda()
            gt = gt.cuda()
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = self.criterion(outputs,gt)
                if self.args.model_type.endswith('CR'):
                    loss+=0.1*self.criterion_CR(outputs,gt,inputs)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
        torch.cuda.synchronize()
        wandb.log({'train_l1_loss':loss.item()})
        wandb.log({'Learning rate':self.optimizer.param_groups[0]['lr']})
        return None
    def save_checkpoint(self,type='last'):
        checkpoint_folder = os.path.join(self.args.checkpoint_folder,self.args.model_type)
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        checkpoint_filename = os.path.join(checkpoint_folder,f'{type}.pth')
        save_data = {
            'step': self.current_epoch,
            f'best_psnr':self.best_psnr,
            f'best_ssim':self.best_ssim,
            'generator_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }
        if save_on_master():
            print('helo')
            torch.save(save_data, checkpoint_filename)

    def load_model_checkpoint_for_training(self,type ='best'):
        checkpoint_folder = os.path.join(self.args.checkpoint_folder,self.args.model_type)
        checkpoint_filename = os.path.join(checkpoint_folder, f'{type}.pth')
        if not os.path.exists(checkpoint_filename):
            print("Couldn't find checkpoint file. Starting training from the beginning.")
            return
        data = torch.load(checkpoint_filename)
        self.current_epoch = data['step']
        self.best_psnr = data['best_psnr']
        self.best_ssim = data['best_ssim']
        self.model.load_state_dict(data['generator_state_dict'],strict=False)
        self.optimizer.load_state_dict(data['optimizer_state_dict'])
        self.scheduler.load_state_dict(data['scheduler_state_dict'])
        print(f"Restored model at epoch {self.current_epoch}.")

    def val_epoch(self):
        self.model.eval()
        psnr_value = []
        ssim_value = []
        for inputs, gt in tqdm.tqdm(self.val_dataloader):
            inputs = inputs.cuda()
            gt = gt.cuda()
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                _ = self.criterion(outputs,gt)
            psnr_value.append(self.psnr(outputs,gt).item())
            ssim_value.append(self.ssim(outputs,gt).item())
        torch.cuda.synchronize()
        wandb.log({'val_psnr':np.mean(psnr_value)})
        wandb.log({'val_ssim':np.mean(ssim_value)})
        val_psnr = np.mean(psnr_value)
        val_ssim = np.mean(ssim_value)

        if val_psnr>self.best_psnr:
            self.best_psnr = val_psnr
            self.save_checkpoint('best')
        if val_ssim>self.best_ssim:
            self.best_ssim = val_ssim
        self.save_checkpoint('last')
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f'Epoch = {self.current_epoch} Val best PSNR = {self.best_psnr},Val current PSNR = {val_psnr},Val best SSIM = {self.best_ssim},Val current SSIM = {val_ssim}, lr ={current_lr}')
        return None
    def run(self):
        self.load_model_checkpoint_for_training()
        for epoch in range(self.current_epoch,self.args.epochs):
            self.current_epoch = epoch
            self.train_dataloader.sampler.set_epoch(self.current_epoch)
            self.val_dataloader.sampler.set_epoch(self.current_epoch)
            self.train_epoch()
            if epoch%10==0:
                self.val_epoch()
        return None



