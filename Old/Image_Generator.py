import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import pandas as pd
import torch as T
import torch.nn as nn
import torchvision as TV
import torchvision.transforms.functional as tvf
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

to_pil  = TV.transforms.ToPILImage()
to_tensor = TV.transforms.ToTensor()
crop = TV.transforms.CenterCrop(108)

folder =  "/home/matthew/Documents/Generative_Models/Data/Friend_Faces/JPEGs"

data = []
i = 0
for root, dirs, files in os.walk(folder):
   for name in files:
         image = Image.open(os.path.join(root, name))
         image = crop(tvf.rotate(image,-90))
         tens = T.flatten(to_tensor(image)).tolist()
         data.append( tens )
         print(i)
         i += 1
df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv("Friend_Faces.csv", mode='w', index=None, header=None, float_format="%.10e")
