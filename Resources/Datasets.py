import torch as T
import torchvision as TV
from torch.utils.data import Dataset, DataLoader


def load_mnist_dataset( b_size, n_workers ):
    """ A function which will return the dataloaders required for training on the MNIST dataset
    """

    ## First we define the transforms which are simple converting to torch tensors
    ## This is to allow for extra rotations/crops/warps that we might want to add later
    transform = TV.transforms.Compose([ TV.transforms.ToTensor() ])

    ## Now we load the train and test datasets
    train_set = TV.datasets.MNIST( root='../Data', train=True,  download=True, transform=transform )
    test_set  = TV.datasets.MNIST( root='../Data', train=False, download=True, transform=transform )

    ## Next we create the dataloaders
    train_loader = DataLoader(train_set, batch_size=b_size, num_workers=n_workers, pin_memory=True, shuffle=True )
    test_loader  = DataLoader(test_set,  batch_size=b_size, num_workers=n_workers, pin_memory=True, shuffle=False )

    return train_loader, test_loader

def load_dataset( dataset_name, *args ):
    if dataset_name == "MNIST":
        return load_mnist_dataset( *args )
