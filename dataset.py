import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import PIL.Image as Image
import numpy as np
import os


class Custom_Dataset(Dataset):
    def __init__(self,root_dir,im_shape=None, is_train=False):
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
        self.hq_train_files = {}
        self.lq_train_files = {}
        for dir_path, _, file_names in os.walk(root_dir):
            for f_paths in sorted(file_names):
                if dir_path.endswith('GT'):
                    self.hq_im_file_list.append(os.path.join(dir_path,f_paths))
                elif dir_path.endswith('input'):
                    self.lq_im_file_list.append(os.path.join(dir_path,f_paths))
        for im_names in self.hq_im_file_list:
            self.hq_train_files[im_names] = np.load(im_names)
        for im_names in self.lq_im_file_list:
            self.lq_train_files[im_names] = np.load(im_names)
        if is_train:
            self.train_transform = T.Compose([T.RandomCrop((im_shape[0],im_shape[1]))])
    def __len__(self):
        return len(self.hq_im_file_list)
    def tone_transform(self,im,c=0.25):
      mapped_x = im / (im + c)
      return mapped_x
    def __getitem__(self, idx):
        image_hq_fname = self.hq_train_files[self.hq_im_file_list[idx]]
        image_lq_fname = self.lq_train_files[self.lq_im_file_list[idx]]
        hq_image = torch.from_numpy(self.tone_transform(image_hq_fname)).unsqueeze(dim=0).permute(0,3,1,2)
        lq_image = torch.from_numpy(self.tone_transform(image_lq_fname)).unsqueeze(dim=0).permute(0,3,1,2)
        concat_img = torch.cat([hq_image,lq_image],dim=0)
        if self.is_train:
            image = self.train_transform(concat_img)
        else:
            image = concat_img
        hq_img,lq_img = image.tensor_split(2)
        return lq_img.squeeze(0),hq_img.squeeze(0)