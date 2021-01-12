import sys
home_env = '../'
sys.path.append(home_env)

import numpy as np
import pandas as pd

import torch as T
import torch.nn as nn
import torchvision as TV
from torch.utils.data import Dataset, DataLoader

class EventsDataset(Dataset):
    def __init__( self, file_name, numrows = None ):
        print( file_name )

        print( "-- converting csv to pandas" )
        file_data = pd.read_csv( file_name, nrows = numrows, dtype=np.float32, header = None )

        print( "-- converting pandas to tensor" )
        self.tensor_data = T.as_tensor( file_data.values.astype(np.float32), dtype = T.float32 )

        del file_data

        ## Defining the transforms
        self.transforms = TV.transforms.Compose([
            TV.transforms.RandomHorizontalFlip(),
            TV.transforms.RandomRotation(10, resample=2),
            TV.transforms.RandomCrop(90),
            TV.transforms.Resize(64, interpolation=2),
            TV.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.01),
            TV.transforms.Normalize((0.4935, 0.4428, 0.3689), (0.2479, 0.2293, 0.2165)),
        ])

        print( "-- done\n" )

    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, idx):
        image_data = self.tensor_data[idx]
        image_data = self.transforms(nn.Unflatten(0, (3, 108, 108))(image_data))
        return ( image_data, 1 )

def load_friend_faces( b_size, n_workers ):
    """ A function which will return the dataloaders required for training on the bro face dataset
    """
    train_set = EventsDataset( "Data/Friend_Faces/Friend_Faces.csv", numrows=64 )
    test_set  = EventsDataset( "Data/Friend_Faces/Friend_Faces.csv", numrows=64 )

    train_loader = DataLoader(train_set, batch_size = b_size, num_workers = n_workers, pin_memory=True, shuffle = True )
    test_loader  = DataLoader(test_set,  batch_size = b_size, num_workers = n_workers, pin_memory=True, shuffle = True )

    unorm_trans = TV.transforms.Normalize( (-0.4935/0.2479, -0.4428/0.2293, -0.3689/0.2165),
                                           (1.0/0.2479, 1.0/0.2293, 1.0/0.2165) )

    return train_loader, test_loader, unorm_trans

def load_mnist_dataset( b_size, n_workers ):
    """ A function which will return the dataloaders required for training on the MNIST dataset
    """

    ## First we define the transforms which are simple converting to torch tensors
    ## This is to allow for extra rotations/crops/warps that we might want to add later
    transform = TV.transforms.Compose([ TV.transforms.ToTensor(),
                                        TV.transforms.Resize(32, interpolation=2),
                                        TV.transforms.Normalize(0.1307, 0.3081) ])
    unorm_trans = TV.transforms.Normalize(-0.1307/0.3081, 1.0/0.3081 )

    ## Now we load the train and test datasets
    train_set = TV.datasets.MNIST( root='Data', train=True,  download=True, transform=transform )
    test_set  = TV.datasets.MNIST( root='Data', train=False, download=True, transform=transform )

    ## Next we create the dataloaders
    train_loader = DataLoader(train_set, batch_size=b_size, num_workers=n_workers, pin_memory=True, shuffle=True )
    test_loader  = DataLoader(test_set,  batch_size=b_size, num_workers=n_workers, pin_memory=True, shuffle=True )

    return train_loader, test_loader, unorm_trans

def load_celeba_dataset( b_size, n_workers ):
    """ A function which will return the dataloaders required for training on the CelebA dataset
    """

    ## First we define the transforms which are simple converting to torch tensors
    ## This is to allow for extra rotations/crops/warps that we might want to add later
    transform = TV.transforms.Compose([ TV.transforms.ToTensor(),
                                        TV.transforms.Resize(64, interpolation=2),
                                        TV.transforms.CenterCrop(64),
                                        TV.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])
    unorm_trans = TV.transforms.Normalize(  (-1,-1,-1), (2,2,2) )

    ## Now we load the train and test datasets
    train_set = TV.datasets.CelebA( root='Data', split="train",  download=False, transform=transform )
    test_set  = TV.datasets.CelebA( root='Data', split="test",   download=False, transform=transform )

    ## Next we create the dataloaders
    train_loader = DataLoader(train_set, batch_size=b_size, num_workers=n_workers, pin_memory=True, shuffle=True )
    test_loader  = DataLoader(test_set,  batch_size=b_size, num_workers=n_workers, pin_memory=True, shuffle=True )

    return train_loader, test_loader, unorm_trans

def load_cifar_dataset( b_size, n_workers ):
    """ A function which will return the dataloaders required for training on the CelebA dataset
    """

    ## First we define the transforms which are simple converting to torch tensors
    ## This is to allow for extra rotations/crops/warps that we might want to add later
    transform = TV.transforms.Compose([ TV.transforms.ToTensor(), TV.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) ])
    unorm_trans = TV.transforms.Normalize( (-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010),
                                           (1.0/0.2023, 1.0/0.1994, 1.0/0.2010) )

    ## Now we load the train and test datasets
    train_set = TV.datasets.CIFAR10( root='Data', train=True,  download=True, transform=transform )
    test_set  = TV.datasets.CIFAR10( root='Data', train=False, download=True, transform=transform )

    ## Next we create the dataloaders
    train_loader = DataLoader(train_set, batch_size=b_size, num_workers=n_workers, pin_memory=True, shuffle=True )
    test_loader  = DataLoader(test_set,  batch_size=b_size, num_workers=n_workers, pin_memory=True, shuffle=True )

    return train_loader, test_loader, unorm_trans

def load_dataset( dataset_name, *args ):
    if dataset_name == "MNIST":
        return load_mnist_dataset( *args )

    if dataset_name == "CelebA":
        return load_celeba_dataset( *args )

    if dataset_name == "CIFAR":
        return load_cifar_dataset( *args )

    if dataset_name == "Friend_Faces":
        return load_friend_faces( *args )
