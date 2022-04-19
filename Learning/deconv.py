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

def ridge_regression_gd(X, Y, X_test, Y_test, lamb, learning_rate=1., iterations=3000, device='cuda'):
    X = torch.tensor(X, device=device, dtype=torch.float64)
    Y = torch.tensor(Y, device=device, dtype=torch.float64)
    X_test = torch.tensor(X_test, device=device, dtype=torch.float64)
    Y_test = torch.tensor(Y_test, device=device, dtype=torch.float64)

    weights = torch.randn((X.shape[2], Y.shape[2]), device=device, dtype=torch.float64, requires_grad=True)
    bias = torch.randn((Y.shape[1], 1), device=device, dtype=torch.float64, requires_grad=True)
    losses_train = []
    losses_test = []

    loss_func = torch.nn.L1Loss()
    optimizer = torch.optim.SGD(params=[weights, bias], lr=learning_rate, momentum=0.9)

    # main training loop
    for k in range(iterations):
        optimizer.zero_grad()
        Y_pred = (torch.matmul(X, weights) + bias.expand(Y.shape[1], Y.shape[2]))
        loss = loss_func(Y, Y_pred)
        if np.isnan(loss.item()):
            print("Warning: GD diverged on iteration %i" % k)
            break
        losses_train.append(loss.item())
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            Y_test_pred = (torch.matmul(X_test, weights) + bias)
            loss_test = loss_func(Y_test, Y_test_pred)
            losses_test.append(loss_test.item())

        print("Iteration %d, training loss: %.6f, test loss: %.6f"% (k, loss.item(), loss_test.item()))

    return weights, bias, losses_train, losses_test


# path to data and design matrix
covariates = {'Distractor':1, 'Target':2, 'Novelty':3}
color_dict = {'Target': 'red', 'Distractor': 'blue', 'Novelty': 'green'}

epoch_data_path = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/Subjects-Epochs/epochs_eeg_ica_condition_VS_data.npy'
epoch_dm_path = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/Subjects-Epochs/epochs_eeg_ica_condition_VS_DM.npy'
epoch_label_path = 'C:/Users/S-Vec/Dropbox/ReNa/Data/ReNaPilot-2022Spring/Subjects-Epochs/epochs_eeg_ica_condition_VS_labels.npy'

srate = 128
deconv_window = (-1.2, 2.4)
erp_window=(0., .8)
viz_erp_window=(-.1, .8)

manual_seed = 42

viz_deconv_index_window = [int(srate * (-deconv_window[0] + viz_erp_window[0])), int(srate * (-deconv_window[0] + viz_erp_window[1]))]
# viz_deconv_index_window = [0, int(srate * (-deconv_window[0] + deconv_window[1]))]
deconv_index_window = [int(srate * (-deconv_window[0] + erp_window[0])), int(srate * (-deconv_window[0] + erp_window[1]))]

torch.manual_seed(manual_seed)

# load data and change axes
data = np.load(epoch_data_path)
dms = np.load(epoch_dm_path)
labels = np.load(epoch_label_path)
dms = np.swapaxes(dms, 1, 2)
data = np.swapaxes(data, 1, 2)

tau = int(dms.shape[-1] /  len(covariates))

# plot_design_matrix(dms, deconv_window, tau, covariates)

# obtain label array from dm
# labels = []
# for i in range(len(dms)):
#     _temp = [np.sum([dms[i, deconv_index_window[0]+1, tau*j:tau*(j+1)]]) for j, label in enumerate(covariates)]  # find the center event of each dm sample
#     # assert sum(_temp) == 1
#     labels.append( _temp.index(1))
# labels = np.array(labels)
print('Distractor, Target, Novelty prevalence: %f, %f, %f' % (np.sum(labels==1)/len(labels), np.sum(labels==2)/len(labels), np.sum(labels==3)/len(labels)))

# z normalize data along channel
# A = stats.zscore(data, axis=2)

data_znormed, scalars = z_norm_by_channel(data)

dms_train, dms_test, data_znormed_train, data_znormed_test = train_test_split(dms, data_znormed, test_size=0.01, random_state=manual_seed)

# beta, error, losses = ridge_regression_gd(X=dms, Y=data, X_test=dms_test, Y_test=data_znormed_test, lamb=1e-3)
beta, error, losses_train, losses_test = ridge_regression_gd(X=dms_train, Y=data_znormed_train, X_test=dms_test, Y_test=data_znormed_test, lamb=1e-3)

eeg_chs = mne.channels.make_standard_montage('biosemi64').ch_names
eeg_ch = 'CPz'
eeg_index = eeg_chs.index(eeg_ch)

# visualize the betas
# beta_np = beta.cpu().detach().numpy()
# error_np = error.cpu().detach().numpy()
# tau = int(dms.shape[-1] /  len(covariates))

#
# for i, cov in enumerate(covariates):
#     plt.plot(beta_np[i * tau:(i+1)*tau, eeg_index] + error_np, label='{0} Beta'.format(cov))
#     plt.legend()
#     plt.show()
#
plt.plot(losses_train)
plt.plot(losses_test)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# data_recon = torch.matmul(torch.tensor(dms, device='cuda', dtype=torch.float64), beta) + error.expand(data.shape[1], data.shape[2])
# data_recon = data_recon.cpu().detach().numpy()
# data_recon = data_recon[:, viz_deconv_index_window[0]:viz_deconv_index_window[1], eeg_index]
# data_recon = scalars[eeg_index].inverse_transform(data_recon)

for cov, cov_code in covariates.items():
    dm_cov =  create_dm_for_cov(dms, tau, cov_code-1, deconv_window, srate)  # cov_code - 1 because distractor starts at 1
    data_recon = torch.matmul(torch.tensor(dm_cov, device='cuda', dtype=torch.float64), beta) + error.expand(data.shape[1], data.shape[2])
    data_recon = data_recon.cpu().detach().numpy()
    data_recon = data_recon[:, viz_deconv_index_window[0]:viz_deconv_index_window[1], eeg_index]
    data_recon = scalars[eeg_index].inverse_transform(data_recon)[0]

    time_vector = np.linspace(viz_erp_window[0], viz_erp_window[1], data_recon.shape[0])
    plt.plot(time_vector, data_recon, label='Deconv Reconstructed Data')
    plt.twinx()

    data_orig = data[:, viz_deconv_index_window[0]:viz_deconv_index_window[1], eeg_index]
    data_orig = data_orig[labels == cov_code]
    data_orig_mean = np.mean(data_orig, axis=0)
    data_orig_upper = data_orig_mean + scipy.stats.sem(data_orig, axis=0)  # this is the upper envelope
    data_orig_lower = data_orig_mean - scipy.stats.sem(data_orig, axis=0)  # this is the lower envelope
    plt.fill_between(time_vector, data_orig_upper, data_orig_lower, where=data_orig_lower <= data_orig_upper, facecolor='red',
                     interpolate=True,
                     alpha=0.5)
    plt.plot(time_vector, data_orig_mean, label='Original Data', color='red')
    plt.legend()
    plt.title('{0} on Channel {1}'.format(cov, eeg_ch))
    plt.show()
