import numpy as np
from copy import deepcopy

def read_npz(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, (list, dict, tuple)):
        res = data
    elif isinstance(data, np.lib.npyio.NpzFile):
        res = dict(data)
        data.close()
    else:
        print("incompatible type of data to unzip...")
        res = None
    return res