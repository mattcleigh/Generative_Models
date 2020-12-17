import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import pandas as pd
import torch as T
import torch.nn as nn
import torchvision as TV
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

to_pil  = TV.transforms.ToPILImage()
unorm = TV.transforms.Normalize( (-0.4945/0.2515, -0.4439/0.2327, -0.3702/0.2201),
                                       (1.0/0.2515, 1.0/0.2327, 1.0/0.2201) )
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
            TV.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.01),
            TV.transforms.Normalize((0.4945, 0.4439, 0.3702), (0.2515, 0.2327, 0.2201)),
        ])

        print( "-- done\n" )

    def __len__(self):
        return len(self.tensor_data)

    def __getitem__(self, idx):
        image_data = self.tensor_data[idx]
        image_data = self.transforms(nn.Unflatten(0, (3, 108, 108))(image_data))
        return ( image_data, 1 )

dataset = EventsDataset( "Data/Friend_Faces/Friend_Faces.csv" )
means = 0.0
vars  = 0.0
i = 0
for image_data, class_data in dataset:
    means += T.mean(image_data, [1,2])
    vars  += T.var(image_data, [1,2])
    i += 1
    print(i)
    # plt.imshow(to_pil(unorm(image_data)))
    # plt.show()
print(means/len(dataset))
print(T.sqrt(vars/len(dataset)))
