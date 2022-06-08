import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import PIL.Image as Image
import numpy as np
import os


class Custom_Dataset(Dataset):
    def __init__(self,root_dir, is_train=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.is_train = is_train
        self.hq_im_file_list = []
        self.lq_im_file_list = []
        for dir_path, _, file_names in os.walk(root_dir):
            for f_paths in sorted(file_names):
                if dir_path.endswith('HQ'):
                    self.hq_im_file_list.append(os.path.join(dir_path,f_paths))
                elif dir_path.endswith('LQ'):
                    self.lq_im_file_list.append(os.path.join(dir_path,f_paths))
        self.tensor_convert = T.ToTensor()
        self.train_transform = T.Compose(
    [T.RandomVerticalFlip(p=0.5),T.RandomHorizontalFlip(p=0.5)\
       ,T.RandomAffine((0,360)),T.Resize((256,256)),T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        self.val_transform = T.Compose([T.Resize((256,256)),T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    def __len__(self):
        return len(self.hq_im_file_list)

    def __getitem__(self, idx):
        image_hq_fname = self.hq_im_file_list[idx]
        image_lq_fname = self.lq_im_file_list[idx]
        hq_image = self.tensor_convert(Image.open(image_hq_fname).convert('RGB')).unsqueeze(dim=0)
        lq_image = self.tensor_convert(Image.open(image_lq_fname).convert('RGB')).unsqueeze(dim=0)
        concat_img = torch.cat([hq_image,lq_image],dim=0)
        if self.is_train:
            image = self.train_transform(concat_img)
        else:
            image = self.val_transform(concat_img)
        hq_img,lq_img = image.tensor_split(2)
        return lq_img.squeeze(0),hq_img.squeeze(0)