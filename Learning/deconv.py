import mne
import scipy
import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def plot_design_matrix(dms, deconv_window, tau, covariates, num=0):
    for i, dm in enumerate(dms):
        plt.imshow(dm)
        plt.xlabel('ERP Window × Number of Covariates')
        plt.ylabel('Deconvolution Window')
        plt.title('Deconv Design Matrix {0}'.format(i))
        plt.yticks(np.linspace(0, dms.shape[1], 4), np.linspace(deconv_window[0], deconv_window[1], 4))
        plt.xticks([(x * tau) + tau/2 for x in range(len(covariates))], covariates)
        plt.show()
        if i == num:
            break


def z_norm_by_channel(X):
    scalers = {}
    X_out = np.empty(X.shape)
    for i in range(X.shape[2]):
        scalers[i] = StandardScaler()
        X_out[:, :, i] = scalers[i].fit_transform(X[:, :, i].reshape(-1, 1)).reshape(data.shape[:2])
        # this_x_ch = X[:, :, i]
        # X_out[:, :, i] = (this_x_ch - np.mean(this_x_ch)) / np.std(this_x_ch)
        # scalers[i] = [ np.mean(this_x_ch),  np.std(this_x_ch)]
    return X_out, scalers

def create_dm_for_cov(dms, tau, cov_index, deconv_window, srate):
    out = np.zeros(dms.shape[1:])
    dm_start_index = int(-deconv_window[0] * srate), cov_index * tau
    for i in range(tau):
        out[dm_start_index[0] + i, dm_start_index[1] + i] = 1
    return out[None, :]

def ridge_regression_gd(X, Y, X_test, Y_test, lamb, learning_rate=1., iterations=400, device='cuda', dtype=torch.float32):
    X = torch.tensor(X, device=device, dtype=dtype)
    Y = torch.tensor(Y, device=device, dtype=dtype)
    X_test = torch.tensor(X_test, device=device, dtype=dtype)
    Y_test = torch.tensor(Y_test, device=device, dtype=dtype)

    weights = torch.randn((X.shape[2], Y.shape[2]), device=device, dtype=dtype, requires_grad=True)
    bias = torch.randn(Y.shape[1:], device=device, dtype=dtype, requires_grad=True)
    losses_train = []
    losses_test = []

    loss_func = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(params=[weights, bias], lr=learning_rate, momentum=0.9)
    # optimizer = torch.optim.SGD(params=[weights], lr=learning_rate, momentum=0.9)

    # main training loop
    for k in range(iterations):
        optimizer.zero_grad()
        # Y_pred = (torch.matmul(X, weights))
        Y_pred = (torch.matmul(X, weights) + bias)
        loss = loss_func(Y, Y_pred)
        l2_norm = sum(p.pow(2.0).sum() for p in weights) + sum(b.pow(2.0).sum() for b in bias)
        loss = loss + lamb * l2_norm
        if np.isnan(loss.item()):
            print("Warning: GD diverged on iteration %i" % k)
            break
        losses_train.append(loss.item())
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            Y_test_pred = (torch.matmul(X_test, weights) + bias)
            # Y_test_pred = (torch.matmul(X_test, weights))
            loss_test = loss_func(Y_test, Y_test_pred)
            losses_test.append(loss_test.item())

        print("Iteration %d, training loss: %.6f, test loss: %.6f"% (k, loss.item(), loss_test.item()))

    return weights, bias, losses_train, losses_test
    # return weights, [], losses_train, losses_test


# path to data and design matrix
covariates = {'Distractor':1, 'Target':2, 'Novelty':3}
color_dict = {'Target': 'red', 'Distractor': 'blue', 'Novelty': 'green'}

# epoch_data_path = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/Subjects-Epochs/epochs_EventLocked_eeg_ica_condition_RSVP_data.npy'
# epoch_dm_path = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/Subjects-Epochs/epochs_EventLocked_eeg_ica_condition_RSVP_DM.npy'
# epoch_label_path = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/Subjects-Epochs/epochs_EventLocked_eeg_ica_condition_RSVP_labels.npy'

# epoch_data_path = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/Subjects-Epochs/epochs_FixationLocked_eeg_ica_condition_RSVP_data.npy'
# epoch_dm_path = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/Subjects-Epochs/epochs_FixationLocked_eeg_ica_condition_RSVP_DM.npy'
# epoch_label_path = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/Subjects-Epochs/epochs_FixationLocked_eeg_ica_condition_RSVP_labels.npy'

epoch_data_path = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/Subjects-Epochs/epochs_FixationLocked_eeg_ica_condition_VS_data.npy'
epoch_dm_path = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/Subjects-Epochs/epochs_FixationLocked_eeg_ica_condition_VS_DM.npy'
epoch_label_path = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/Subjects-Epochs/epochs_FixationLocked_eeg_ica_condition_VS_labels.npy'


srate = 128
deconv_window = (-1.2, 2.4)
frp_window=(0., .8)
viz_window=(0., .8)

manual_seed = 42

viz_index_window = [int(srate * (-deconv_window[0] + viz_window[0])), int(srate * (-deconv_window[0] + viz_window[1]))]
# viz_index_window = [0, int(srate * (-deconv_window[0] + deconv_window[1]))]
frp_index_window = [int(srate * (-deconv_window[0] + frp_window[0])), int(srate * (-deconv_window[0] + frp_window[1]))]

torch.manual_seed(manual_seed)

# load data and change axes
data = np.load(epoch_data_path)
dms = np.load(epoch_dm_path)
labels = np.load(epoch_label_path)
dms = np.swapaxes(dms, 1, 2)
data = np.swapaxes(data, 1, 2)

tau = int(dms.shape[-1] /  len(covariates))

# plot_design_matrix(dms, deconv_window, tau, covariates)

print('Distractor, Target, Novelty prevalence: %f, %f, %f' % (np.sum(labels==1)/len(labels), np.sum(labels==2)/len(labels), np.sum(labels==3)/len(labels)))

# z normalize data along channel
data_znormed, scalars = z_norm_by_channel(data)
dms_train, dms_test, data_znormed_train, data_znormed_test = train_test_split(dms, data_znormed, test_size=0.01, random_state=manual_seed)

beta, error, losses_train, losses_test = ridge_regression_gd(X=dms_train, Y=data_znormed_train, X_test=dms_test, Y_test=data_znormed_test, lamb=1e-1)

eeg_chs = mne.channels.make_standard_montage('biosemi64').ch_names
eeg_ch = 'CPz'
eeg_index = eeg_chs.index(eeg_ch)

plt.plot(losses_train)
plt.plot(losses_test)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# data_recon = torch.matmul(torch.tensor(dms, device='cuda', dtype=torch.float64), beta) + error.expand(data.shape[1], data.shape[2])
# data_recon = data_recon.cpu().detach().numpy()
# data_recon = data_recon[:, viz_deconv_index_window[0]:viz_deconv_index_window[1], eeg_index]
# data_recon = scalars[eeg_index].inverse_transform(data_recon)

# plotting covariate betas
for cov, cov_code in covariates.items():
    dm_cov =  create_dm_for_cov(dms, tau, cov_code-1, deconv_window, srate)  # cov_code - 1 because distractor starts at 1
    # cov_beta = torch.matmul(torch.tensor(dm_cov, device='cuda', dtype=torch.float32), beta) + error.expand(data.shape[0], -1, -1)
    cov_beta = torch.matmul(torch.tensor(dm_cov, device='cuda', dtype=torch.float32), beta)
    cov_beta = cov_beta.cpu().detach().numpy()
    cov_beta = cov_beta[:, viz_index_window[0]:viz_index_window[1], eeg_index]
    cov_beta = scalars[eeg_index].inverse_transform(cov_beta)[0]

    time_vector = np.linspace(viz_window[0], viz_window[1], cov_beta.shape[0])
    plt.plot(time_vector, cov_beta, label='Estimated Beta Coefficient for Cov {0}'.format(cov))
    plt.legend()
    plt.twinx()

    data_orig = data[:, viz_index_window[0]:viz_index_window[1], eeg_index]
    data_orig = data_orig[labels == cov_code]
    data_orig_mean = np.mean(data_orig, axis=0)
    data_orig_upper = data_orig_mean + scipy.stats.sem(data_orig, axis=0)  # this is the upper envelope
    data_orig_lower = data_orig_mean - scipy.stats.sem(data_orig, axis=0)  # this is the lower envelope
    plt.fill_between(time_vector, data_orig_upper, data_orig_lower, where=data_orig_lower <= data_orig_upper, facecolor='red',
                     interpolate=True,
                     alpha=0.5)
    plt.plot(time_vector, data_orig_mean, label='Original Data for Cov {0}'.format(cov), color='red')
    plt.legend()
    plt.title('{0} on Channel {1}'.format(cov, eeg_ch))
    plt.show()

# plot deconv corrected FRP
for cov, cov_code in covariates.items():
    data_orig = data[:, viz_index_window[0]:viz_index_window[1], eeg_index][labels == cov_code]
    dm_orig = dms[:, viz_index_window[0]:viz_index_window[1], :][labels == cov_code]

    data_corrected = []
    for d, dm in zip(data_orig, dm_orig):  # remove the other overlapping signals
        dm_other = dm.copy()
        dm_other[frp_index_window[0]:frp_index_window[1]] = 0
        weights_other = torch.matmul(torch.tensor(dm_other, device='cuda', dtype=torch.float32), beta[:, eeg_index]) + error[viz_index_window[0]:viz_index_window[1], eeg_index]
        weights_other = weights_other.cpu().detach().numpy()
        weights_other_rescaled = scalars[eeg_index].inverse_transform(weights_other.reshape(1, -1))[0]
        data_corrected.append(d - weights_other_rescaled)
    time_vector = np.linspace(viz_window[0], viz_window[1], weights_other.shape[0])
    data_corrected = np.array(data_corrected)
    data_corrected_mean = np.mean(data_corrected, axis=0)
    data_corrected_upper = data_corrected_mean + scipy.stats.sem(data_corrected, axis=0)  # this is the upper envelope
    data_corrected_lower = data_corrected_mean - scipy.stats.sem(data_corrected, axis=0)  # this is the lower envelope
    plt.fill_between(time_vector, data_corrected_upper, data_corrected_lower, where=data_corrected_lower <= data_corrected_upper, facecolor='blue',
                     interpolate=True,
                     alpha=0.5)
    plt.plot(time_vector, data_corrected_mean, label='Deconv corrected Data for Cov {0}'.format(cov), color='blue')

    data_orig_mean = np.mean(data_orig, axis=0)
    data_orig_upper = data_orig_mean + scipy.stats.sem(data_orig, axis=0)  # this is the upper envelope
    data_orig_lower = data_orig_mean - scipy.stats.sem(data_orig, axis=0)  # this is the lower envelope
    plt.fill_between(time_vector, data_orig_upper, data_orig_lower, where=data_orig_lower <= data_orig_upper, facecolor='red',
                     interpolate=True,
                     alpha=0.5)
    plt.plot(time_vector, data_orig_mean, label='Original Data for Cov {0}'.format(cov), color='red')

    plt.legend()
    plt.title('{0} on Channel {1}'.format(cov, eeg_ch))
    plt.show()


dm_copy = dm.copy()
dm_copy[frp_index_window[0]:frp_index_window[1]] = 0
plt.imshow(dm_copy)
plt.show()