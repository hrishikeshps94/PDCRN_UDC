import torch
from torch.utils.data import DataLoader
from model import UDC_Arc
from dataset import Custom_Dataset
import os
import wandb
from torchmetrics import PeakSignalNoiseRatio,StructuralSimilarityIndexMeasure
import tqdm
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# from torch.optim.lr_scheduler import CosineAnnealingLR
class Train():
    def __init__(self,args) -> None:
        self.args = args
        self.current_epoch = 0
        self.best_psnr = 0
        self.best_ssim = 0
        self.model_intiliaser()
        self.data_intiliaser()
        self.losses_opt_and_metrics_init()
        self.init_summary()
    def model_intiliaser(self):
        self.model = UDC_Arc(self.args.in_ch,self.args.num_filters,self.args.dilation_rates,self.args.nPyramidFilters)
        summary(self.model)
        self.model = self.model.to(device)
        return None
    def data_intiliaser(self):
        train_ds = Custom_Dataset(self.args.train_path,is_train=True)
        self.train_dataloader = DataLoader(train_ds,batch_size=self.args.batch_size,shuffle=True,num_workers=os.cpu_count())
        val_ds = Custom_Dataset(self.args.test_path,is_train=False)
        self.val_dataloader = DataLoader(val_ds,batch_size=self.args.batch_size,shuffle=False,num_workers=os.cpu_count())
        return None
    def init_summary(self):
        wandb.init(project=f"UDC",name=self.args.log_name)
        return
    def losses_opt_and_metrics_init(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=(self.args.batch_size*0.3)/256, momentum=0.9)
        self.criterion = torch.nn.L1Loss().to(device)
        self.psnr  = PeakSignalNoiseRatio().to(device)
        self.ssim = StructuralSimilarityIndexMeasure().to(device)
    def train_epoch(self):
        self.model.train()
        for count,(inputs, gt) in enumerate(tqdm.tqdm(self.train_dataloader)):
            inputs = inputs.to(device)
            gt = gt.to(device)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = self.model(inputs)
                loss = self.criterion(outputs,gt)
                loss.backward()
                self.optimizer.step()
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
        }
        torch.save(save_data, checkpoint_filename)

    def load_model_checkpoint_for_training(self,type ='last'):
        checkpoint_folder = os.path.join(self.args.checkpoint_folder,self.args.model_type)
        checkpoint_filename = os.path.join(checkpoint_folder, f'{type}.pth')
        if not os.path.exists(checkpoint_filename):
            print("Couldn't find checkpoint file. Starting training from the beginning.")
            return
        data = torch.load(checkpoint_filename)
        self.current_epoch = data['step']
        self.best_psnr = data['best_psnr']
        self.best_ssim = data['best_ssim']
        self.model.load_state_dict(data['generator_state_dict'])
        self.optimizer.load_state_dict(data['optimizer_state_dict'])
        print(f"Restored model at epoch {self.current_epoch}.")

    def val_epoch(self):
        self.model.eval()
        for inputs, gt in tqdm.tqdm(self.val_dataloader):
            inputs = inputs.to(device)
            gt = gt.to(device)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                _ = self.criterion(outputs,gt)
            self.psnr.update(outputs,gt)
            self.ssim.update(outputs,gt)
        wandb.log({'val_psnr':self.psnr.compute().item()})
        wandb.log({'val_ssim':self.ssim.compute().item()})
        val_psnr,val_ssim = self.psnr.compute().item(),self.ssim.compute().item()
        self.psnr.reset()
        self.ssim.reset()
        if val_psnr>self.best_psnr:
            self.best_psnr = val_psnr
            self.save_checkpoint('best')
        else:
            self.save_checkpoint('last')
        if val_ssim>self.best_ssim:
            self.best_ssim = val_ssim
        print(f'Epoch = {self.current_epoch} Val best PSNR = {self.best_psnr},Val best SSIM= {self.best_ssim},\
            Val current PSNR = {val_psnr},Val currentSSIM= {val_ssim}')
        return None
    def run(self):
        self.load_model_checkpoint_for_training()
        for epoch in range(self.current_epoch,self.args.epochs):
            self.current_epoch = epoch
            self.train_epoch()
            self.val_epoch()
        return None



