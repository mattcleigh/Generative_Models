import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

old_len = 4
new_len = 2

old = T.arange( old_len*old_len, dtype=T.float32 ).view( (old_len,old_len) )

new = F.interpolate(old.unsqueeze(0).unsqueeze(0), size=[new_len,new_len], mode="nearest" )
new.squeeze_()
print(old)
print(new)
