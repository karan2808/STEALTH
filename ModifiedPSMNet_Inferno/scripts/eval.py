import torch
from torch.utils.data import DataLoader

from model import PSMNET
from dataloader import StereoDataset
from utils import evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define command line arguments:

if __name__ == "__main__":
    max_disp = sys.argv[1] # Max disparity for model object
    batch_size = sys.argv[2] # Feed in batch size for dataloader
    data_dir = sys.argv[3] # Data directory for the dataset
    dataset_name = sys.argv[4] # Name of the dataset
    saved_model = sys.argv[5] # Path where trained model is saved
    eval_path = sys.argv[6] # Path for storing predictions
    test_filenames = sys.argv[7] # Path to .txt containing test image names


# Main Program:

# Define StereoDatset objects for test set:
test_set  = StereoDataset(data_dir = data_directory, dataset_name = dataset_name, mode = 'test')

# Loader argument dictionary:
loader_args = dict(shuffle = False, batch_size = batch_size, num_workers = 6, pin_memory = True) if torch.cuda.is_available()\
                   else dict(shuffle = False, batch_size = 2)

# Creating train and validation loaders:
test_loader = DataLoader(test_set,**loader_args)

# Load saved model:
myModel = PSMNET(max_disp = max_disp)
myModel.load_state_dict(torch.load(saved_model))
myModel.to(device)

# Call evaluation function
evaluate(test_loader, myModel, test_filenames, eval_path)

    