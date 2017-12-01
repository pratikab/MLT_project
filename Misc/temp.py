# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# # Imports
# import numpy as np
# import tensorflow as tf

# tf.logging.set_verbosity(tf.logging.INFO)
# import tensorflow as tf
# import numpy as np
# import sys
# a=np.array([[1 2 3],[4 5 6]])
# b=np.array([[1 2 3]])
# a[2]=b
# print(a)
import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data_utils 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.sparse import FloatTensor as STensor

batch_size = 4
seq_length = 1
feat_dim = 6

batch_idx = torch.LongTensor([i for i in range(batch_size) for s in range(seq_length)])
seq_idx = torch.LongTensor(list(range(seq_length))*batch_size)
feat_idx = torch.LongTensor([[1],[2],[3],[4]]).view(4,)

my_stack = torch.stack([batch_idx, seq_idx, feat_idx]) # indices must be nDim * nEntries
my_final_array = STensor(my_stack, torch.ones(batch_size * seq_length), 
                         torch.Size([batch_size, seq_length, feat_dim])).to_dense()    

print(my_final_array)



