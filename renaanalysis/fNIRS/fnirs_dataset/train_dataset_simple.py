from sklearn.linear_model import LogisticRegression

from renaanalysis.fNIRS.fnirs_dataset.fNIRS_finger_and_foot_tapping_dataset import \
    get_fnirs_finger_and_foot_tapping_dataset
import numpy as np

from renaanalysis.fNIRS.fnirs_dataset.utils import train_logistic_regression

if __name__ == '__main__':
    epoch_t_min = -1.5
    epoch_t_max = 20
    dataset_root_dir = 'D:/HaowenWei/Data/HT_Data/fNIRS/FingerFootTapping'

    # epoch_data_dict = get_fnirs_finger_and_foot_tapping_epoch_dict(dataset_root_dir=dataset_root_dir, epoch_t_min=epoch_t_min, epoch_t_max=epoch_t_max)

    x, y, metadata, event_color, fs = get_fnirs_finger_and_foot_tapping_dataset(dataset_root_dir=dataset_root_dir,
                                                                                epoch_t_min=epoch_t_min,
                                                                                epoch_t_max=epoch_t_max, visualize_participants_epoch=True)
    model = LogisticRegression()

    train_logistic_regression(x, y, model, test_size=0.2)

