
import numpy as np
from matplotlib import pyplot as plt

from renaanalysis.multimodal.multimodal import PhysioArray
from scipy.spatial.distance import pdist



def create_discretize_channel_space(parray: PhysioArray):
    assert parray.physio_type == 'eeg'
    assert 'channel_positions' in parray.meta_info.keys()
    channel_locations = parray.meta_info['channel_positions']
    assert channel_locations.shape[-1] == 3

    channel_positions = np.unique(channel_locations, axis=0).reshape(-1, 3)
    min_distance = np.min(pdist(channel_positions))
    voxel_size = min_distance / 2.0

    min_coord, max_coord = np.min(channel_positions, axis=0), np.max(channel_positions, axis=0) + voxel_size
    grid_dim = np.ceil((max_coord - min_coord) / voxel_size).astype(int)
    min_coord_shifted = min_coord - voxel_size / 2.0

    voxel_indices = np.floor((channel_locations - min_coord_shifted) / voxel_size).astype(int)
    voxel_indices_flattened = np.ravel_multi_index(voxel_indices.T, grid_dim)
    parray.meta_info['channel_voxel_indices'] = parray.meta_info_encoded['channel_voxel_indices'] = voxel_indices_flattened.T
    # plt.stem(parray.meta_info['channel_voxel_indices'][0])
    # plt.show()
