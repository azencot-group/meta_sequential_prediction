from collections import OrderedDict

import numpy as np
from datasets import load_dataset

_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']
_NUM_VALUES_PER_FACTOR = OrderedDict({'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                          'scale': 8, 'shape': 4, 'orientation': 15})


def get_index(factors):
    """ Converts factors to indices in range(num_data)
    Args:
    factors: np array shape [6,batch_size].
             factors[i]=factors[i,:] takes integer values in 
             range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[i]]).

    Returns:
    indices: np array shape [batch_size].
    """
    indices = 0
    base = 1
    for factor, name in reversed(list(enumerate(_FACTORS_IN_ORDER))):
        indices += factors[factor] * base
        base *= _NUM_VALUES_PER_FACTOR[name]
    return indices


class ThreeDimShapesDataset(object):
    def __init__(self, root, train=True, T=3, label_velo=False, transforms=None,
                 active_actions=None, force_moving=False, shared_transition=False, rng=None):
        repo_id = 'TalBarami/msd_3dshapes'
        _dataset = load_dataset(repo_id)
        self.dataset = _dataset['train']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        X = data['x']
        X = np.array(X, dtype='float32').squeeze().transpose([0, 3, 1, 2]).astype(np.float32) / 255
        return X
