import torch
import numpy as np
import cv2
from torch.utils.data import Dataset

class StereoDataset(Dataset):
  def __init__(self, data_dir,
               device,
               dataset_name = 'CATS',
               mode = 'train',
               transform=None
               ):
    super(StereoDataset, self).__init__()
    self.data_dir     = data_dir
    self.dataset_name = dataset_name
    self.mode         = mode
    self.transform    = transform
    self.device       = device
    # Pre-processing
    self.clahe        = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


    cats_data_dict = {
        'train': 'cats_final/filenames1/cats_train.txt',
        'val'  : 'cats_final/filenames1/cats_val.txt',
        'test' : 'cats_final/filenames1/cats_test.txt' 
    }

    cmu_data_dict = {
        'train': 'filenames1/cmu_train.txt',
        'val'  : 'filenames1/cmu_val.txt',
        'test' : 'filenames1/cmu_test.txt' 
    }

    dataset_name_dict = {
        'CATS': cats_data_dict,
        'CMU': cmu_data_dict,
    }

    assert dataset_name in dataset_name_dict.keys()

    self.left_imgs, self.right_imgs, self.gt_imgs = self.get_file_names(dataset_name_dict[dataset_name][mode])

  def get_file_names(self, file_name_list):
    f = open(file_name_list, 'r')
    f = f.readlines()       
    left_arr, right_arr, gt_arr = [], [], []

    for line in f:
      line      = line.split('\n')[0]
      left_img  = self.data_dir + 'left/'  + line
      right_img = self.data_dir + 'right/' + line
      gt_img    = self.data_dir + 'disp/'  + line.split('.')[0] + '.txt'

      gt_arr.append(gt_img)
      left_arr.append(left_img)
      right_arr.append(right_img)

    return left_arr, right_arr, gt_arr

  def __getitem__(self, index):
    left_img_path  = self.left_imgs[index]
    right_img_path = self.right_imgs[index]
    gt_img_path    = self.gt_imgs[index]

    # Read the images
    left_img  = cv2.imread(left_img_path,  0)
    right_img = cv2.imread(right_img_path, 0)
    
    # Apply pre-processing
    left_img   = self.clahe.apply(left_img)
    right_img  = self.clahe.apply(right_img)

    # Perform clipping
    disp_img  = np.genfromtxt(gt_img_path, delimiter=',')
    disp_img  = (disp_img.astype(float))
    disp_img  += 70
    disp_img[disp_img < 0]   = 0.0
    disp_img[disp_img > 192] = 192.0
    
    left_img  = torch.unsqueeze(torch.from_numpy(left_img.astype(float)).float(),dim=0)   #permute(2, 0, 1)
    right_img = torch.unsqueeze(torch.from_numpy(right_img.astype(float)).float(),dim=0)  #permute(2, 0, 1)
    disp_img  = (torch.from_numpy(disp_img).float())

    if self.transform is not None:
      left_img, right_img, disp_img = self.transform(left_img), self.transform(right_img), self.transform(disp_img)
    
    return left_img, right_img, disp_img
  
  def __len__(self):
    return len(self.left_imgs)