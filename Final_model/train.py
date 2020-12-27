import torch
from torch.utils.data import DataLoader

from model import PSMNET
from dataloader import StereoDataset
from utils import train_valid,validate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define command line arguments:

if __name__ == "__main__":
	max_disp = sys.argv[1] # Max disparity for model object
	batch_size = sys.argv[2] # Batch Size
    num_epochs = sys.argv[3] # Number of epochs
    learning_rate = sys.argv[4] # Learning rate for optimizer
    data_directory = sys.argv[5] # Data directory for the dataset
    dataset_name = sys.argv[6] # Name of the dataset for dataloader
    model_save_path = sys.argv[7] # Path where model needs to be saved
    model_supres_path = sys.argv[8] # Path where super-res model is saved


# Main program:

# Define StereoDatset objects for train and validation sets:
train_set  = StereoDataset(data_dir = data_directory, dataset_name = dataset_name, mode = 'train')
valid_set  = StereoDataset(data_dir = data_directory, dataset_name = dataset_name, mode = 'val')

# Loader argument dictionary:
loader_args = dict(shuffle = True, batch_size = batch_size, num_workers = 6, pin_memory = True) if torch.cuda.is_available()\
                   else dict(shuffle = True, batch_size = 2)

# Creating train and validation loaders:
train_loader = DataLoader(train_set,**loader_args)
valid_loader = DataLoader(valid_set,**loader_args)

# Initialize a PSMNet model object/ load a saved model object:
myModel = PSMNET(max_disp = max_disp)
myModel.to(device)

# Initailize a SR module model object:
mySup_Res = sup_res()
mySup_Res.to(device)

# Define optimizer and scheduler:

# initialize the optimizer
optimizer     = torch.optim.Adam(myModel.parameters(), lr=learning_rate)
# initialize the lr scheduler
scheduler     = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=300, verbose = True)

# Train the model:
# The model with the least validation loss is saved at the specified location.

# Call training function

train_valid(train_loader, valid_loader, num_epochs, myModel, mySup_Res, optimizer, scheduler, model_save_path, model_supres_path)







