# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# # Imports
# import numpy as np
# import tensorflow as tf

# tf.logging.set_verbosity(tf.logging.INFO)
import tensorflow as tf
import numpy as np
import sys
a=np.array([[1 2 3],[4 5 6]])
b=np.array([[1 2 3]])
a[2]=b
print(a)
