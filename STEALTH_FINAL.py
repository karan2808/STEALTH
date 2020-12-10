# -*- coding: utf-8 -*-
"""Super_Res_STEALTH.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BrDkT28DgOWRf2Dwq402qDmxmrdvb3rq
"""

# # download the dataset 
# !gdown https://drive.google.com/uc?id=1Y4HWIEaZc8lQuf1SAYarGT2cBBrH7mnT
# !gdown https://drive.google.com/uc?id=1jg6i8R1MSLMQtvHoU5EhUfzIiSEo2sEW
# !gdown https://drive.google.com/uc?id=1JxhXXqcj0TdVkDAQiLhFWh3TpaPrNMG3
# !unzip cats_expt.zip

# !unzip cats.zip

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# data loader
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
    self.clahe        = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


    cats_data_dict = {
        'train': 'filenames1/cats_train.txt',
        'val'  : 'filenames1/cats_val.txt',
        'test' : 'filenames1/cats_test.txt' 
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
      left_img  = self.data_dir + 'cats/left/'  + line
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

    left_img   = self.clahe.apply(left_img)
    right_img  = self.clahe.apply(right_img)

    disp_img  = np.genfromtxt(gt_img_path, delimiter=',')
    disp_img  = (disp_img.astype(float))
    disp_img  += 70
    disp_img[disp_img < 0]   = 0.0
    disp_img[disp_img > 192] = 192.0

    disp_img_downscl = cv2.resize(disp_img, (160//4, 120//4), interpolation = cv2.INTER_NEAREST) 
    disp_img_upscl   = cv2.resize(disp_img_downscl, (640, 480), interpolation = cv2.INTER_NEAREST) 
    
    left_img  = torch.unsqueeze(torch.from_numpy(left_img.astype(float)).float(),dim=0)   #permute(2, 0, 1)
    right_img = torch.unsqueeze(torch.from_numpy(right_img.astype(float)).float(),dim=0)  #permute(2, 0, 1)
    disp_img  = (torch.from_numpy(disp_img).float())
    disp_img_upscl  = (torch.from_numpy(disp_img_upscl).float())

    if self.transform is not None:
      left_img, right_img, disp_img = self.transform(left_img), self.transform(right_img), self.transform(disp_img)
    
    return left_img, right_img, disp_img, disp_img_upscl
  
  def __len__(self):
    return len(self.left_imgs)

# Basic residual block from Kaimeng He's Resnet Paper:

class residual_block(nn.Module):
  def __init__(self,in_channel,out_channel,kernel,stride,padding,dilation):
    super(residual_block, self).__init__()

    self.in_channels = in_channel
    self.out_channels = out_channel
    self.stride = stride
    self.dilation = dilation

    layers = [nn.Conv2d(in_channel, out_channel, kernel, stride, padding, dilation = dilation, bias = False), nn.BatchNorm2d(out_channel),nn.ReLU(inplace=True),
              nn.Conv2d(out_channel, out_channel, kernel, 1, padding,dilation=dilation, bias = False), nn.BatchNorm2d(out_channel)]

    self.layers = nn.Sequential(*layers)


  def forward(self,x):
    if self.in_channels == self.out_channels: # if-else block deals with the condition when number of channels change across residual/skip connection
      input = x
    else:
      diff = self.out_channels - self.in_channels
      x1 = torch.zeros(x.shape[0],diff,x.shape[2],x.shape[3]).to(device)
      input = torch.cat((x,x1),dim=1)


    if self.stride == 2: # Downsample input tensor when number of channels change across residual connections and stride is 2.
      input = F.max_pool2d(input,kernel_size = 2,stride = 2)

    output = self.layers(x)

    output += input # PSM-Net paper does not use ReLU after skip connection
    return output

# CNN module of PSMNet:
class CNN(nn.Module):
  def __init__(self):
    super(CNN,self).__init__()

    layers_1 = [
              nn.Conv2d(in_channels = 1,out_channels = 32, kernel_size = 3, stride = 2, padding = 1, bias = False), nn.BatchNorm2d(32),nn.ReLU(inplace=True),   # H/2 x W/2 x 32 Conv0_1
              nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm2d(32),nn.ReLU(inplace=True), # H/2 x W/2 x 32 Conv0_2
              nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm2d(32),nn.ReLU(inplace=True), # H/2 x W/2 x 32 Conv0_3
              residual_block(32,32,3,1,1,1), residual_block(32,32,3,1,1,1), residual_block(32,32,3,1,1,1), # H/2 x W/2 x 32 Conv1_ 1-3
              residual_block(32,64,3,2,1,1), residual_block(64,64,3,1,1,1), residual_block(64,64,3,1,1,1), residual_block(64,64,3,1,1,1), # Conv2_ 1-4
              residual_block(64,64,3,1,1,1), residual_block(64,64,3,1,1,1), residual_block(64,64,3,1,1,1), residual_block(64,64,3,1,1,1), # Conv2_ 5-8
              residual_block(64,64,3,1,1,1), residual_block(64,64,3,1,1,1), residual_block(64,64,3,1,1,1), residual_block(64,64,3,1,1,1), # Conv2_ 9-12
              residual_block(64,64,3,1,1,1), residual_block(64,64,3,1,1,1), residual_block(64,64,3,1,1,1), residual_block(64,64,3,1,1,1) # H/4 x W/4 x 64 Conv2_ 13-16
              ]
              
    layers_2 = [      
              residual_block(64,128,3,1,2,2), residual_block(128,128,3,1,2,2), residual_block(128,128,3,1,2,2), # H/4 x W/4 x 64 Conv3_ 1-3
              residual_block(128,128,3,1,4,4), residual_block(128,128,3,1,4,4), residual_block(128,128,3,1,4,4) # H/4 x W/4 x 64 Conv4_ 1-3 
              ]

    # Split sequential block into two lists to extract intermediate outputs for concatentaion in subsequent modules:

    self.layers_1 = nn.Sequential(*layers_1)
    self.layers_2 = nn.Sequential(*layers_2)

  def forward(self,x):
    cnn_concat1 = self.layers_1(x)
    cnn_concat2 = self.layers_2(cnn_concat1)
    return cnn_concat1,cnn_concat2 # Return two outputs(used further in concat operation)

# SPP module of PSMNet:

class SPP(nn.Module):
  def __init__(self):
    super(SPP,self).__init__()

    self.cnn_module = CNN()

    self.branch1 = nn.Sequential(
                  nn.AvgPool2d(32,stride=32),
                  nn.Conv2d(in_channels = 128,out_channels = 32, kernel_size = 1, stride = 1, padding = 0, dilation = 1, bias = False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
                  ) # branch_1

    self.branch2 = nn.Sequential(
                  nn.AvgPool2d(16,stride=16),
                  nn.Conv2d(in_channels = 128,out_channels = 32, kernel_size = 1, stride = 1, padding = 0, dilation = 1, bias = False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
                  ) # branch_2

    self.branch3 = nn.Sequential(
                  nn.AvgPool2d(8,stride=8),
                  nn.Conv2d(in_channels = 128,out_channels = 32, kernel_size = 1, stride = 1, padding = 0, dilation = 1, bias = False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
                  ) # branch_3

    self.branch4 = nn.Sequential(
                  nn.AvgPool2d(4,stride=4),
                  nn.Conv2d(in_channels = 128,out_channels = 32, kernel_size = 1, stride = 1, padding = 0, bias = False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
                  ) # branch_4

    self.branch5 = nn.Sequential(
                  nn.AvgPool2d(2,stride=2),
                  nn.Conv2d(in_channels = 128,out_channels = 32, kernel_size = 1, stride = 1, padding = 0, bias = False), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
                  ) # branch_4

    self.final_conv = nn.Sequential(
                    nn.Conv2d(in_channels = 352, out_channels =160, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm2d(160), nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels = 160, out_channels = 32, kernel_size = 1, stride = 1, bias = False)
                    ) # fusion

  def forward(self,x):
    conv2_16,conv4_3 = self.cnn_module(x)
    branch1 = self.branch1(conv4_3)
    branch2 = self.branch2(conv4_3)
    branch3 = self.branch3(conv4_3)
    branch4 = self.branch4(conv4_3)
    branch5 = self.branch5(conv4_3)
    branch1 = F.interpolate(branch1, (conv4_3.shape[2],conv4_3.shape[3]),mode='bilinear')
    branch2 = F.interpolate(branch2, (conv4_3.shape[2],conv4_3.shape[3]),mode='bilinear')
    branch3 = F.interpolate(branch3, (conv4_3.shape[2],conv4_3.shape[3]),mode='bilinear')
    branch4 = F.interpolate(branch4, (conv4_3.shape[2],conv4_3.shape[3]),mode='bilinear')
    branch5 = F.interpolate(branch5, (conv4_3.shape[2],conv4_3.shape[3]),mode='bilinear')
    concat_tensor = torch.cat((conv2_16,conv4_3,branch1,branch2,branch3,branch4, branch5), dim = 1) # Concat all pooled components (Concat block of SPP)
    spp_output = self.final_conv(concat_tensor)
    return spp_output

# Custom class for disparity regression (Adopted from PSM-Net GitHub repo):

class disp_reg(nn.Module):
  def __init__(self, maxdisp):
    super(disp_reg, self).__init__()
    self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])).cuda(), requires_grad=False)

  def forward(self, x):
    disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
    out  = torch.sum(x*disp,1)
    return out

# PSM-Net model class definition (Trying with simple Basic 3D CNN architecture first!):
# Should take a stereo pair as input and do the rest. 

class PSMNET(nn.Module):
  def __init__(self,max_disp):
    super(PSMNET,self).__init__()

    self.max_disp = max_disp

    self.disp = self.max_disp//2

    self.spp = SPP()

    self.cnn3d_1 = nn.Sequential(nn.Conv3d(in_channels = 64,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                                 nn.Conv3d(in_channels = 32,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm3d(32), nn.ReLU(inplace=True)) # 3DConv_1
    
    self.cnn3d_2 = nn.Sequential(nn.Conv3d(in_channels = 32,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                                 nn.Conv3d(in_channels = 32,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm3d(32)) # 3DConv_2

    self.cnn3d_3 = nn.Sequential(nn.Conv3d(in_channels = 32,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                                 nn.Conv3d(in_channels = 32,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm3d(32)) # 3DConv_3 

    self.cnn3d_4 = nn.Sequential(nn.Conv3d(in_channels = 32,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                                 nn.Conv3d(in_channels = 32,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm3d(32)) # 3DConv_4

    self.cnn3d_5 = nn.Sequential(nn.Conv3d(in_channels = 32,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                                 nn.Conv3d(in_channels = 32,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm3d(32)) # 3DConv_5
    
    self.cnn3d_last = nn.Sequential(nn.Conv3d(in_channels = 32,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = False), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
                                    nn.Conv3d(in_channels = 32,out_channels = 1, kernel_size = 3, stride = 1, padding = 1, bias = False)) # 3DConv_final

    self.disp_reg = disp_reg(self.max_disp)

    self.relu1 = nn.ReLU(inplace=True)
    self.relu2 = nn.ReLU(inplace=True)
    self.relu3 = nn.ReLU(inplace=True)
    self.relu4 = nn.ReLU(inplace=True)
    

  def forward(self,left,right):
    left_feat  = self.spp(left) # Extract feature map from left image
    right_feat = self.spp(right) # Extract feature map from right image

    # The following code snippet has been adopted directly from PSM-Net GitHub repo:
    #
    self.disp_tmp = self.disp // 4
    cost = torch.zeros(left_feat.shape[0], left_feat.shape[1] * 2, self.disp_tmp,  left_feat.shape[2], left_feat.shape[3]).float().cuda()

    for i in range(0, self.disp_tmp, 1):
      if i > 0:
        cost[:, left_feat.shape[1]:, i, :, i:]   = left_feat[:,:,:,i:]
        cost[:, :left_feat.shape[1], i, :, i:]   = right_feat[:,:,:,:-i]
      else:
        cost[:, left_feat.shape[1]:, i, :, i:]   = left_feat
        cost[:, :left_feat.shape[1], i, :, i:]   = right_feat


    cost = self.cnn3d_1(cost)
    cost = self.relu1(self.cnn3d_2(cost) + cost) # Additions represent skip connections across 3D Conv layers
    cost = self.relu2(self.cnn3d_3(cost) + cost)
    cost = self.relu3(self.cnn3d_4(cost) + cost)
    cost = self.relu4(self.cnn3d_3(cost) + cost)
    cost = self.cnn3d_last(cost)

    # Bilinear upsampling to restore image to original size:
    cost = F.interpolate(cost, (self.max_disp,left.shape[2],left.shape[3]), mode = 'trilinear') # GitHub repo uses trilinear for some reason. Trying bilinear for now!

    # Eliminate num_channels dimension(it's 1 anyway):
    cost = torch.squeeze(cost,1)
    
    # Compute softmax across all disparity values:
    cost = F.softmax(-cost,dim=1)

    #Perform disparity regression as defined in paper:
    pred = self.disp_reg(cost)

    return pred

class sup_res_module(nn.Module):
  def __init__(self):
    super(sup_res_module,self).__init__()
    self.branch1 = nn.Sequential(
                  nn.AvgPool2d(8,stride=8),
                  nn.Conv2d(in_channels = 1,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = True), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
                  )

    self.branch2 = nn.Sequential(
                  nn.AvgPool2d(4,stride=4),
                  nn.Conv2d(in_channels = 1,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = True), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
                  )

    self.branch3 = nn.Sequential(
                  nn.AvgPool2d(2,stride=2),
                  nn.Conv2d(in_channels = 1,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = True), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
                  ) 
    
    self.branch4 = nn.Sequential(
                  nn.Conv2d(in_channels = 1,out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = True), nn.BatchNorm2d(32), nn.ReLU(inplace=True)
                  ) 

    self.branch5 = nn.Sequential(
                  nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = 3, stride = 1, padding = 1, bias = True), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                  nn.Conv2d(in_channels = 32, out_channels =  1, kernel_size = 1, stride = 1, bias = True)
                  )
    self.act_f = nn.ReLU(inplace=True)
  
  def forward(self, x):
    res = x
    x  = torch.unsqueeze(x, 1)
    x1 = self.branch1(x)
    x2 = self.branch2(x)
    x3 = self.branch3(x)
    x4 = self.branch4(x)
    x1 = F.interpolate(x1, (x.shape[-2], x.shape[-1]), mode='bilinear')
    x2 = F.interpolate(x2, (x.shape[-2], x.shape[-1]), mode='bilinear')
    x3 = F.interpolate(x3, (x.shape[-2], x.shape[-1]), mode='bilinear')
    x4 = F.interpolate(x4, (x.shape[-2], x.shape[-1]), mode='bilinear')
    cat = torch.cat((x1, x2, x3, x4), 1)
    cat = self.branch5(cat)
    cat = torch.squeeze(cat, 1)
    cat = self.act_f(cat + res)
    return cat

class sup_res(nn.Module):
  def __init__(self):
    super(sup_res, self).__init__()
    self.module_1 = sup_res_module()
    self.module_2 = sup_res_module()
    self.module_3 = sup_res_module()
    self.module_4 = sup_res_module()
    self.module_5 = sup_res_module()

  def forward(self, x):
    x = self.module_1(x)
    x = self.module_2(x)
    x = self.module_3(x)
    x = self.module_4(x)
    x = self.module_5(x)
    return x

Batch_size = 3
train_set  = StereoDataset('cats_final/', device, dataset_name = 'CATS', mode = 'train',transform=None)
valid_set  = StereoDataset('cats_final/', device, dataset_name = 'CATS', mode = 'val',transform=None)

# Loader argument dictionary:
loader_args = dict(shuffle = True, batch_size = Batch_size, num_workers = 6, pin_memory = True) if torch.cuda.is_available()\
                   else dict(shuffle = True, batch_size = 2)

# Creating train and validation loaders:
train_loader = DataLoader(train_set,**loader_args)
valid_loader = DataLoader(valid_set,**loader_args)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def validate(val_dataloader, num_batch, model, loss_function):
  model.eval()
  running_loss, num_sequence = 0, 0

  with torch.no_grad():
    for i, (left_img, right_img, disp_img, disp_ups) in enumerate(val_dataloader):

      # the images go on the gpu
      left_img       = left_img.to(device)
      right_img      = right_img.to(device)
      disp_img_gt    = disp_img.to(device)

      disp_img_pred  = model.forward(left_img, right_img)
      loss           = loss_function(disp_img_pred, disp_img_gt)

      num_sequence, running_loss = num_sequence + 1, running_loss + loss.item()

      # Optimization snippet:
      del left_img,right_img

  print('Validation Loss = ' + str(running_loss/len(valid_set)))
  return (running_loss/len(valid_set))

def train(Batch_size, train_dataloader, val_dataloader):

  # Initialize PSMNET object and load it onto GPU:

  myModel   = PSMNET(max_disp = 192)
  mySup_Res = sup_res()
  # model goes on the gpu
  myModel.to(device)
  mySup_Res.to(device)
  # optimizer, params
  num_batch       = Batch_size
  num_epochs      = 500
  learning_rate   = 1e-3
  v_loss = 10000

  train_list = []
  train_list_super_res = []
  valid_list = []
  epoch_list = []

  # initialize the optimizer
  optimizer     = torch.optim.Adam(myModel.parameters(), lr=learning_rate)
  # initialize the lr scheduler
  scheduler     = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.8, step_size=600, verbose = True)
  # loss function, l1 loss with smoothness penalty
  loss_function = torch.nn.SmoothL1Loss(size_average=True)

  optimizer1     = torch.optim.Adam(mySup_Res.parameters(), lr = learning_rate)
  loss_function1 = torch.nn.MSELoss(size_average = None, reduce = None, reduction = 'mean')
  scheduler1     = torch.optim.lr_scheduler.StepLR(optimizer1, gamma=0.8, step_size=300, verbose = True)

  for epoch in range(num_epochs):

    running_loss, num_sequence, running_loss_supres = 0, 0, 0
    train_loss = 0
    myModel.train(True)
    mySup_Res.train(True)

    for i, (left_img, right_img, disp_img, disp_ups) in enumerate(train_dataloader):

      optimizer.zero_grad()
      optimizer1.zero_grad()

      # the images go on the gpu
      left_img       = left_img.to(device)
      right_img      = right_img.to(device)
      disp_img_gt    = disp_img.to(device)
      disp_ups       = disp_ups.to(device)

      disp_img_pred  = myModel.forward(left_img, right_img)
      loss = loss_function(disp_img_pred, disp_img_gt)
      loss.backward()
      optimizer.step()

      #print(disp_img_gt)
      del left_img
      del right_img

      if epoch < 250:
        disp_img_pred_res = mySup_Res(disp_ups)
      else:
        disp_img_pred_res = mySup_Res(disp_img_pred.detach())
      loss1             = loss_function1(disp_img_pred_res, disp_img_gt)
      loss1.backward()
      optimizer1.step()

      running_loss = running_loss + loss.item()
      running_loss_supres = running_loss_supres + loss1.item()
      torch.cuda.empty_cache()

    plt.imshow(disp_img_gt.cpu().numpy()[0, :, :])
    plt.show()
    plt.imshow(disp_img_pred.detach().cpu().numpy()[0, :, :])
    plt.show()
    plt.imshow(disp_img_pred_res.detach().cpu().numpy()[0, :, :])
    plt.show()

    plt.imshow(disp_ups.detach().cpu().numpy()[0, :, :])
    plt.show()


    print(running_loss/len(train_set))

    print('EPOCH = ' + str(epoch))
    
    print('Training Loss = ' + str(running_loss/len(train_set)))
    train_loss = running_loss/len(train_set)

    print('Training Loss sup_res = ' + str(running_loss_supres/len(train_set)))

    with torch.no_grad():
      val_loss = validate(val_dataloader, num_batch, myModel, loss_function)

    # Plot training and validation loss:
    train_list.append(running_loss/len(train_set))
    valid_list.append(val_loss)
    epoch_list.append(epoch+1)
    # train_list_super_res(running_loss/len(train_set))

    plt.plot(epoch_list,train_list)
    plt.plot(epoch_list,valid_list)
    plt.show()

    scheduler.step()
    scheduler1.step()
    
    if train_loss < v_loss:
      v_loss = train_loss
      torch.save(myModel.state_dict(),'pathname')
      torch.save(mySup_Res.state_dict(),'pathname_super_res')

    torch.cuda.empty_cache()

train(Batch_size, train_loader, valid_loader)

# Commented out IPython magic to ensure Python compatibility.
# %mkdir /content/image_download
# %mkdir /content/image_download/valid
# %mkdir /content/image_download/valid_gt
# %mkdir /content/image_download/train
# %mkdir /content/image_download/train_gt
my_model = PSMNET(max_disp = 192)
my_model.to(device)
state_dict = (torch.load('pathname'))
my_model.load_state_dict(state_dict)
mySup_Res = sup_res()
mySup_Res.to(device)
my_model.load_state_dict(state_dict)
state_dict = (torch.load('pathname_super_res_f'))
mySup_Res.load_state_dict(state_dict)
# test_set = StereoDataset('/content/gdrive/My Drive/CATS_INTERP_FULL/New_Data_Cats/', device, dataset_name = 'CATS', mode = 'val',transform=None)
# # Loader argument dictionary:
# loader_args = dict(shuffle = False, batch_size = Batch_size, num_workers = 6, pin_memory = True) if torch.cuda.is_available()\
#                    else dict(shuffle = True, batch_size = 2)
# test_loader = DataLoader(test_set, **loader_args)

my_model.eval()
mySup_Res.eval()

k = 0
m = 0
f1 = open('filenames1/cats_val.txt', 'r')
f1 = f1.readlines()
f2 = open('filenames1/cats_train.txt', 'r')
f2 = f2.readlines()
with torch.no_grad():
  
  for i, (left_img, right_img, disp_img, disup) in enumerate(valid_loader):
    # the images go on the gpu
    left_img       = left_img.to(device)
    right_img      = right_img.to(device)
    disp_img_gt    = disp_img.to(device)

    disp_img_pred  = my_model(left_img, right_img)
    # disp_img_pred  = mySup_Res(disp_img_gt)

    # Save predictions in a folder
    for j in range(disp_img_pred.shape[0]):
      disp_pred = disp_img_pred[j,:,:].detach().cpu().numpy()
      disp_pred[disp_pred < 0] = 0
      disp_pred[disp_pred > 192.0] = 192.0
      plt.imsave('/content/image_download/valid/' + f1[m].split('\n')[0] ,disp_pred,cmap='gray')
      disp_gt = disp_img_gt[j,:,:].cpu().numpy()
      plt.imsave('/content/image_download/valid_gt/' + f1[m].split('\n')[0] ,disp_gt,cmap='gray')
      m += 1


  for i, (left_img, right_img, disp_img, dispup) in enumerate(train_loader):
    # the images go on the gpu
    left_img       = left_img.to(device)
    right_img      = right_img.to(device)
    disp_img_gt    = disp_img.to(device)

    disp_img_pred  = my_model(left_img, right_img)
    disp_img_pred  = mySup_Res(disp_img_gt)

    
    # Save predictions in a folder
    for j in range(disp_img_pred.shape[0]):
      disp_pred = disp_img_pred[j,:,:].detach().cpu().numpy()
      disp_pred[disp_pred < 0] = 0
      disp_pred[disp_pred > 192.0] = 192.0
      plt.imsave('/content/image_download/train/' + f2[k].split('\n')[0] ,disp_pred,cmap='gray')
      disp_gt = disp_img_gt[j,:,:].cpu().numpy()
      plt.imsave('/content/image_download/train_gt/' + f2[k].split('\n')[0] ,disp_gt,cmap='gray')
      # plt.imsave('/content/image_download/train/' + f2[k].split('\n')[0],disp_img_pred[j,:,:].detach().cpu().numpy(),cmap='gray')
      k += 1

!zip -r thresh_with_superRes_448.zip image_download/