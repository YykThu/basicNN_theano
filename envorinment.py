import numpy as np
import theano
from theano import tensor as T
import theano.tensor.signal.pool as pool
import os
import glob
import cPickle as pickle
from collections import OrderedDict


floatX = np.float32
