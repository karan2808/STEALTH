import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from PIL import Image

# data loader
class StereoDataset(Dataset):
  def __init__(self, data_dir,
               dataset_name = 'CATS',
               mode = 'train',
               transform=None
               ):
    super(StereoDataset, self).__init__()
    self.data_dir     = data_dir
    self.dataset_name = dataset_name
    self.mode         = mode
    self.transform    = transform

    cats_data_dict = {
        'train': 'filenames/cats_train.txt', # filenames folder(in data_dir) containing .txt files
        'val'  : 'filenames/cats_val.txt', 
        'test' : 'filenames/cats_test.txt' 
    }

    cmu_data_dict = {
        'train': 'filenames/cmu_train.txt', # Change path to filenames folder containing .txt files
        'val'  : 'filenames/cmu_val.txt',
        'test' : 'filenames/cmu_test.txt' 
    }

    dataset_name_dict = {
        'CATS': cats_data_dict, # Create a dictionary for each dataset
        'CMU': cmu_data_dict,
    }

    assert dataset_name in dataset_name_dict.keys()

    self.left_imgs, self.right_imgs, self.gt_imgs = self.get_file_names(dataset_name_dict[dataset_name][mode])

  def get_file_names(self, file_name_list):
    f = open(file_name_list, 'r')
    f = f.readlines()       
    left_arr, right_arr, gt_arr = [], [], []

    # Read .png and .txt(Ground Truth) file names
    for line in f:
      line      = line.split('\n')[0]
      left_img  = self.data_dir + 'cats/left/'  + line                            # 'cats' folder should exist in data_dir
      right_img = self.data_dir + 'cats/right/' + line
      gt_img    = self.data_dir + 'cats/disp/'  + line.split('.')[0] + '.txt'

      gt_arr.append(gt_img)
      left_arr.append(left_img)
      right_arr.append(right_img)

    return left_arr, right_arr, gt_arr

  def __getitem__(self, index):
    left_img_path  = self.left_imgs[index]
    right_img_path = self.right_imgs[index]
    gt_img_path    = self.gt_imgs[index]

    # read the images
    left_img  = cv2.imread(left_img_path,  0)
    right_img = cv2.imread(right_img_path, 0)

    left_img = np.uint8(cv2.dct(np.float32(left_img)/255.0)*255.0)
    right_img = np.uint8(cv2.dct(np.float32(right_img)/255.0)*255.0)

    disp_img  = np.genfromtxt(gt_img_path, delimiter=',')
    disp_img  = (disp_img.astype(float))
    disp_img  += 70                      # Add an offset to the ground truth disparity map
    disp_img[disp_img < 0]   = 0.0       # Clip the disparity values that are below 0 or above 192
    disp_img[disp_img > 192] = 192.0
    disp_img = np.array(disp_img)/255.0
    disp_img = np.uint8(cv2.dct(disp_img)*255)
    
  
    if self.transform is not None: # Optional: Can include transforms from torchvision or user-defined classes/functions
      three_ch_img = np.dstack([left_img,right_img,disp_img])
      trans_img = np.array(self.transform(Image.fromarray(three_ch_img)))
      left_img, right_img, disp_img = trans_img[:,:,0], trans_img[:,:,1], trans_img[:,:,2]

    left_img  = torch.tensor(left_img.astype(float)).float()
    right_img = torch.tensor(right_img.astype(float)).float()
    disp_img  = torch.tensor(disp_img.astype(float)).float()
      
    return left_img, right_img, disp_img
  
  def __len__(self):
    return len(self.left_imgs)