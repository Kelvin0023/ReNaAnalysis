import numpy as np
import cv2
import torch


def prepare_image_for_sim_score(img):
    img = np.moveaxis(cv2.resize(img, (64, 64)), -1, 0)
    img = np.expand_dims(img, axis=0)
    img = torch.Tensor(img)
    return img

def temporal_filter_fixation(thresholded_sim_distance, marker_mode='viz', fixation_y_value = -1e-2, verbose=1):
    """
    marker_mode:
        viz: marker value is 1 for the duration of the fixations, nan otherwise
        event: 1 for fixation onsets, 2 for offsets
    """
    fixation_diff = np.diff(np.concatenate([[0], thresholded_sim_distance]))
    fix_onset_indices = np.argwhere(fixation_diff == 1)
    fix_offset_indices = np.argwhere(fixation_diff == -1)
    fix_interval_indices = [(x[0], y[0]) for x, y in zip(fix_onset_indices, fix_offset_indices)]
    fix_interval_indices = [x for x in fix_interval_indices if
                            x[1] - x[0] > 5]  # only keep the fix interval longer than 150 ms == 5 frames
    fix_list_filtered = np.empty(len(thresholded_sim_distance))
    fix_list_filtered[:] = np.nan if marker_mode == 'viz' else 0

    for index_onset, index_offset in fix_interval_indices:
        if marker_mode == 'viz':
            fix_list_filtered[index_onset:index_offset] = fixation_y_value  # for visualization
        else:
            fix_list_filtered[index_onset] = 1
            fix_list_filtered[index_offset] = 2
    if verbose > 0: print('Detected {} fixations from patch similarity'.format(len(fix_onset_indices)))
    return fix_list_filtered