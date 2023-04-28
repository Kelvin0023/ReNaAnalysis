import contextlib
import math
import os
import pickle

import numpy as np
import torch
from sklearn import preprocessing, metrics
from sklearn.model_selection import StratifiedShuffleSplit, ParameterGrid
from sklearn.preprocessing import LabelEncoder
from torch import nn, autograd
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from torchsummary import summary
from tqdm import tqdm

from renaanalysis.learning.HDCA import hdca, hdca_eeg
from renaanalysis.learning.HT import HierarchicalTransformer, viz_ht
from renaanalysis.learning.HT_viz import ht_viz
from renaanalysis.learning.MutiInputDataset import MultiInputDataset
from renaanalysis.learning.models import EEGPupilCNN, EEGCNN, EEGInceptionNet
from renaanalysis.params.params import epochs, batch_size, model_save_dir, patience, random_seed, \
    export_data_root, num_top_compoenents
from renaanalysis.utils.data_utils import compute_pca_ica, mean_sublists, rebalance_classes, mean_max_sublists, \
    mean_min_sublists, epochs_to_class_samples_rdf, z_norm_by_trial
import matplotlib.pyplot as plt

def eval_lockings(rdf, event_names, locking_name_filters, model_name, exg_resample_rate=200, participant=None, session=None, regenerate_epochs=True, n_folds=10, ht_lr=1e-4, ht_l2=1e-6, ht_output_mode='single'):
    # verify number of event types
    assert np.all(len(event_names) == np.array([len(x) for x in locking_name_filters.values()]))
    locking_performance = {}
    if participant is None:
        participant = 'all'
    if session is None:
        session = 'all'

    for locking_name, locking_filter in locking_name_filters.items():
        test_name = f'Locking-{locking_name}_P-{participant}_S-{session}'
        if regenerate_epochs:
            x, y, _, _ = epochs_to_class_samples_rdf(rdf, event_names, locking_filter, data_type='both', rebalance=False, participant=participant, session=session, plots='full', exg_resample_rate=exg_resample_rate)
            pickle.dump(x, open(os.path.join(export_data_root, f'x_P{participant}_S{session}_L{locking_name}.p'), 'wb'))
            pickle.dump(y, open(os.path.join(export_data_root, f'y_P{participant}_S{session}_L{locking_name}.p'), 'wb'))
        else:
            try:
                x = pickle.load(open(os.path.join(export_data_root, f'x_P{participant}_S{session}_L{locking_name}.p'), 'rb'))
                y = pickle.load(open(os.path.join(export_data_root, f'y_P{participant}_S{session}_L{locking_name}.p'), 'rb'))
            except FileNotFoundError:
                raise Exception(f"Unable to find saved epochs for participant {participant}, session {session}, locking {locking_name}" + ", EEGPupil" if model_name == 'EEGPupil' else "")
        model_performance = eval_model(x[0], x[1], y, event_names, model_name, test_name=test_name, n_folds=n_folds, exg_resample_rate=exg_resample_rate, ht_lr=ht_lr, ht_l2=ht_l2, ht_output_mode=ht_output_mode)
        locking_performance[locking_name] = model_performance
        return locking_performance

def eval_model(x_eeg, x_pupil, y, event_names, model_name, test_name='', n_folds=10, exg_resample_rate=200, ht_lr=1e-4, ht_l2=1e-6, ht_output_mode='multi'):
    x_eeg_znormed = z_norm_by_trial(x_eeg)
    x_pupil_znormed = z_norm_by_trial(x_pupil) if x_pupil is not None else None
    x_eeg_pca_ica, _, _ = compute_pca_ica(x_eeg, num_top_compoenents)

    model_performance = _run_model(model_name, x_eeg_znormed, x_eeg_pca_ica, x_pupil_znormed, y, event_names, test_name, ht_output_mode, ht_lr=ht_lr, ht_l2=ht_l2, n_folds=n_folds, exg_resample_rate=exg_resample_rate)
    return model_performance

def _run_model(model_name, x_eeg, x_eeg_pca_ica, x_pupil, y, event_names, test_name,
               HT_output_mode, ht_lr=1e-4, ht_l2=1e-6,
               n_folds=10, exg_resample_rate=200):
    """
    runs a given model. This funciton is called by eval_model
    @param model_name: name of the model, supports EEGCNN, EEGPupilCNN, EEGInceptionNet, HT, HDCA
    @param x_eeg: znormalized eeg data
    @param x_eeg_pca_ica: pca/ica eeg data
    @param x_pupil: znoramlized pupil data, can be None, in which case only eeg is used, if given model_name is a pupil model, an error is raised
    @param y:
    @param event_names:
    @param test_name:
    @param HT_output_mode: can be 'multi' or 'single', if 'multi', the model's output is a vector of length len(event_names) with softmax activation,
            if 'single', the model's output is a single value with sigmoid activation
    @param ht_lr:
    @param ht_l2:
    @param n_folds:
    @param exg_resample_rate:
    @return:
    """
    # create a test split
    skf = StratifiedShuffleSplit(n_splits=1, random_state=random_seed)
    train, test = [(train, test) for train, test in skf.split(x_eeg, y)][0]
    x_eeg_train, x_eeg_pca_ica_train = x_eeg[train], x_eeg_pca_ica[train]
    x_eeg_test, x_eeg_pca_ica_test = x_eeg[test], x_eeg_pca_ica[test]
    y_train, y_test = y[train], y[test]

    if x_pupil is not None:
        x_pupil_train, x_pupil_test = x_pupil[train], x_pupil[test]
    else:
        x_pupil_train, x_pupil_test = None, None

    note = f"{test_name}_{model_name}"
    performance = {}
    if model_name == 'HDCA':
        hdca_func = hdca if x_pupil is not None else hdca_eeg
        roc_auc_combined, roc_auc_eeg, roc_auc_pupil = hdca_func(x_eeg_train, x_eeg_pca_ica_train, x_pupil_train, y_train, event_names,is_plots=True, exg_srate=exg_resample_rate, notes=note, verbose=0)  # give the original eeg data, no need to apply HDCA again
        performance[f"{note}_EEG"] = {'folds val auc': roc_auc_eeg}
        if x_pupil is not None:
            performance[f"{note}_Pupil"] = {'folds val auc': roc_auc_pupil}
            performance[f"{note}_EEG-Pupil"] = {'folds val auc': roc_auc_combined}
        print(f'{note}: folds EEG AUC {roc_auc_eeg}, folds Pupil AUC: {roc_auc_pupil}, folds EEG-pupil AUC: {roc_auc_combined}')
    else:
        if model_name == 'EEGPupilCNN':  # this model uses PCA-ICA reduced EEG data plus pupil data
            assert x_pupil is not None, "Pupil data is not provided, which is required for EEGPupilCNN model"
            model = EEGPupilCNN(eeg_in_shape=x_eeg_pca_ica.shape, pupil_in_shape=x_pupil.shape, num_classes=2)
            model, training_histories, criterion, label_encoder = train_model([x_eeg_pca_ica_train, x_pupil_train], y_train, model, test_name=test_name, verbose=1, n_folds=n_folds)
            # model, training_histories, criterion, label_encoder = train_model_pupil_eeg([x_eeg_pca_ica_train, x_pupil_train], y_train, model, test_name=test_name, n_folds=n_folds)
        elif model_name == 'HT':  # this model uses un-dimension reduced EEG data
            num_channels, num_timesteps = x_eeg_train.shape[1:]
            model = HierarchicalTransformer(num_timesteps, num_channels, exg_resample_rate, num_classes=2, output=HT_output_mode)
            model, training_histories, criterion, label_encoder = train_model(x_eeg_train, y_train, model, test_name=test_name, verbose=1, lr=ht_lr, l2_weight=ht_l2, n_folds=n_folds)  # use un-dimension reduced EEG data
            ht_viz(model, x, y, event_names, data_root, split_window_eeg, exg_resample_rate, eeg_channel_names,
                   eeg_montage, num_timesteps, num_channels, note='', head_fusion='max', discard_ratio=0.9,
                   load_saved_rollout=False, batch_size=64)
        else:  # these models use PCA-ICA reduced EEG data
            if model_name == 'EEGCNN':
                model = EEGCNN(in_shape=x_eeg_pca_ica.shape, num_classes=2)
            elif model_name == 'EEGInception':
                model = EEGInceptionNet(in_shape=x_eeg_pca_ica.shape, num_classes=2)
            else:
                raise Exception(f"Unknown model name {model_name}")
            model, training_histories, criterion, label_encoder = train_model(x_eeg_pca_ica_train, y_train, model, test_name=test_name, verbose=1, n_folds=n_folds)

        folds_train_acc, folds_val_acc, folds_train_loss, folds_val_loss = mean_max_sublists(training_histories['acc_train']), mean_max_sublists(training_histories['acc_val']), mean_min_sublists(training_histories['loss_val']), mean_min_sublists(training_histories['loss_val'])
        folds_val_auc = mean_max_sublists(training_histories['auc_val'])
        performance[note] = {'folds val auc': folds_val_auc, 'folds val acc': folds_val_acc, 'folds train acc': folds_train_acc, 'folds val loss': folds_val_loss, 'folds trian loss': folds_train_loss}
        print(f'{test_name}: folds val AUC {folds_val_auc}, folds val accuracy: {folds_val_acc}, folds train accuracy: {folds_train_acc}, folds val loss: {folds_val_loss}, folds train loss: {folds_train_loss}')
    return performance

def grid_search_ht(grid_search_params, rdf, event_names, locking_name, locking_filter, exg_resample_rate=128, participant=None, session=None, regenerate_epochs=True):
    # assert np.all(len(event_names) == np.array([len(x) for x in locking_name_filters.values()]))  # verify number of event types
    locking_performance = {}
    test_name = f'Locking-{locking_name}_Model-HT_P-{participant}_S-{session}'
    if regenerate_epochs:
        x, y, _, _ = epochs_to_class_samples_rdf(rdf, event_names, locking_filter, data_type='both', rebalance=False, participant=participant, session=session, plots='full', exg_resample_rate=exg_resample_rate)
        pickle.dump(x, open(os.path.join(export_data_root, f'x_P{participant}_S{session}_L{locking_name}.p'), 'wb'))
        pickle.dump(y, open(os.path.join(export_data_root, f'y_P{participant}_S{session}_L{locking_name}.p'), 'wb'))
    else:
        try:
            x = pickle.load(open(os.path.join(export_data_root, f'x_P{participant}_S{session}_L{locking_name}.p'), 'rb'))
            y = pickle.load(open(os.path.join(export_data_root, f'y_P{participant}_S{session}_L{locking_name}.p'), 'rb'))
        except FileNotFoundError:
            raise Exception(f"Unable to find saved epochs for participant {participant}, session {session}, locking {locking_name}")
    x_eeg = z_norm_by_trial(x[0])
    x_pupil = z_norm_by_trial(x[1])
    x_eeg_pca_ica, _, _ = compute_pca_ica(x[0], num_top_compoenents)
    num_channels, num_timesteps = x_eeg.shape[1:]

    param_grid = ParameterGrid(grid_search_params)
    training_histories = {}
    models = {}
    for params in param_grid:
        print(f"Grid search params: {params}. Searching {len(training_histories) + 1} of {len(param_grid)}")
        model = HierarchicalTransformer(num_timesteps, num_channels, exg_resample_rate, num_classes=2,
                                        depth=params['depth'], num_heads=params['num_heads'], feedforward_mlp_dim=params['feedforward_mlp_dim'],
                                        pool=params['pool'], patch_embed_dim=params['patch_embed_dim'],
                                        dim_head=params['dim_head'], emb_dropout=params['emb_dropout'], attn_dropout=params['attn_dropout'], output=params['output'])
        model, training_histories, criterion, label_encoder = train_model(x_eeg, y, model, test_name=test_name, verbose=1, lr=params['lr'], l2_weight=params['l2_weight'])  # use un-dimension reduced EEG data
        folds_train_acc, folds_val_acc, folds_train_loss, folds_val_loss = mean_max_sublists(training_histories['acc_train']), mean_max_sublists(training_histories['acc_val']), mean_min_sublists(training_histories['loss_val']), mean_min_sublists(training_histories['loss_val'])
        folds_val_auc = mean_max_sublists(training_histories['auc_val'])
        print(f'{test_name} with param {params}: folds val AUC {folds_val_auc}, folds val accuracy: {folds_val_acc}, folds train accuracy: {folds_train_acc}, folds val loss: {folds_val_loss}, folds train loss: {folds_train_loss}')

        hashable_params = tuple(params.items())
        locking_performance[hashable_params] = {'folds val auc': folds_val_auc, 'folds val acc': folds_val_acc, 'folds train acc': folds_train_acc, 'folds val loss': folds_val_loss,'folds trian loss': folds_train_loss}
        training_histories[hashable_params] = training_histories
        models[hashable_params] = model
    return locking_performance, training_histories, models


def train_model(X, Y, model, test_name="CNN", n_folds=10, lr=1e-3, verbose=1, l2_weight=1e-6, lr_scheduler_type='exponential', plot_histories=False):
    """

    @param X: can be a list of inputs
    @param Y:
    @param model:
    @param test_name:
    @param n_folds:
    @param lr:
    @param verbose:
    @param l2_weight:
    @param lr_scheduler: can be 'exponential' or 'cosine' or None
    @return:
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)

    if isinstance(X, list):
        # create dummy random input for each input
        rand_input = []
        for x in X:
            input_shape = x.shape[1:]
            rand_input.append(torch.randn(1, *input_shape).to(device))
        dataset_class = MultiInputDataset
    else:
        # check the model's output shape
        input_shape = X.shape[1:]
        rand_input = torch.randn(1, *input_shape).to(device)
        dataset_class = TensorDataset

    with torch.no_grad():
        model.eval()
        output_shape = model(rand_input).shape[1]

    if output_shape == 1:
        assert len(np.unique(Y)) == 2, "Model only has one output node. But given Y has more than two classes. Binary classification model should have 2 classes"
        label_encoder = LabelEncoder()
        label_encoder.fit(Y)
        _encoder = lambda y: label_encoder.transform(y)
        criterion = nn.BCELoss(reduction='mean')
        last_activation = nn.Sigmoid()
    else:
        label_encoder = preprocessing.OneHotEncoder()
        label_encoder.fit(Y.reshape(-1, 1))
        _encoder = lambda y: label_encoder.transform(y.reshape(-1, 1)).toarray()
        criterion = nn.CrossEntropyLoss()
        last_activation = nn.Softmax(dim=1)

    X = model.prepare_data(X)

    skf = StratifiedShuffleSplit(n_splits=n_folds, random_state=random_seed)
    train_losses_folds = []
    train_accs_folds = []
    val_losses_folds = []
    val_accs_folds = []
    val_aucs_folds = []

    for f_index, (train, val) in enumerate(skf.split(X[0] if isinstance(X, list) else X, Y)):
        if isinstance(X, list):
            x_train = []
            x_val = []
            y_train, y_val = Y[train], Y[val]

            rebalanced_labels = []
            for this_x in X:
                this_x_train, this_y_train = rebalance_classes(this_x[train], y_train)
                x_train.append(torch.Tensor(this_x_train).to(device))
                rebalanced_labels.append(this_y_train)
                x_val.append(torch.Tensor(this_x[val]).to(device))
            assert np.all([label_set == rebalanced_labels[0] for label_set in rebalanced_labels])
            y_train = rebalanced_labels[0]
        else:
            x_train, x_val, y_train, y_val = X[train], X[val], Y[train], Y[val]
            x_train, y_train = rebalance_classes(x_train, y_train)  # rebalance by class
            x_train = torch.Tensor(x_train).to(device)
            x_val = torch.Tensor(x_val).to(device)

        y_train_encoded = _encoder(y_train)
        y_val_encoded = _encoder(y_val)

        y_train_encoded = torch.Tensor(y_train_encoded)
        y_val_encoded = torch.Tensor(y_val_encoded)

        train_dataset = dataset_class(x_train, y_train_encoded)
        val_dataset = dataset_class(x_val, y_val_encoded)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        val_aucs = []
        best_loss = np.inf
        patience_counter = 0

        for epoch in range(epochs):

            test_auc, train_loss, train_accuracy = _run_one_epoch(model, train_dataloader, label_encoder, criterion, last_activation, optimizer, mode='train', device=device, l2_weight=l2_weight, test_name=test_name, verbose=verbose)
            scheduler.step()
            val_auc, val_loss, val_accuracy = _run_one_epoch(model, val_dataloader, label_encoder, criterion, last_activation, optimizer, mode='val', device=device, l2_weight=l2_weight, test_name=test_name, verbose=verbose)

            train_losses.append(train_loss)
            train_accs.append(train_accuracy)
            val_aucs.append(val_auc)
            val_losses.append(val_loss)
            val_accs.append(val_accuracy)

            if verbose >= 1:
                print("Fold {}, Epoch {}: val auc = {:.8f}, train accuracy = {:.8f}, train loss={:.8f}; val accuracy = {:.8f}, val loss={:.8f}, patience left {}".format(f_index, epoch, np.max(val_aucs), train_accs[-1], train_losses[-1], val_accs[-1],val_losses[-1], patience - patience_counter))
            # Save training histories after every epoch
            training_histories = {'loss_train': train_losses, 'acc_train': train_accs, 'loss_val': val_losses, 'acc_val': val_accs}
            pickle.dump(training_histories, open(os.path.join(model_save_dir, 'training_histories.pickle'), 'wb'))
            if val_losses[-1] < best_loss:
                torch.save(model.state_dict(), os.path.join(model_save_dir, test_name))
                if verbose >= 1: print('Best model loss improved from {} to {}, saved best model to {}'.format(best_loss, val_losses[-1], model_save_dir))
                best_loss = val_losses[-1]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > patience:
                    if verbose >= 1: print(f'Fold {f_index}: Terminated terminated by patience, validation loss has not improved in {patience} epochs')
                    break
        train_accs_folds.append(train_accs)
        train_losses_folds.append(train_losses)
        val_accs_folds.append(val_accs)
        val_losses_folds.append(val_losses)
        val_aucs_folds.append(val_aucs)

    training_histories_folds = {'loss_train': train_losses_folds, 'acc_train': train_accs_folds, 'loss_val': val_losses_folds, 'acc_val': val_accs_folds, 'auc_val': val_aucs_folds}
    if plot_histories:
        plt.plot(training_histories_folds['acc_train'])
        plt.plot(training_histories_folds['acc_val'])
        plt.title(f"Accuracy, {test_name}")
        plt.tight_layout()
        plt.show()

        plt.plot(training_histories_folds['loss_train'])
        plt.plot(training_histories_folds['loss_val'])
        plt.title(f"Loss, {test_name}")
        plt.tight_layout()
        plt.show()
    return model, training_histories_folds, criterion, label_encoder


def eval(model, x, y, criterion, label_encoder):

    y = label_encoder.fit_transform(np.array(y).reshape(-1, 1)).toarray()
    x = torch.Tensor(x).to(device)  # transform to torch tensor

    test_dataset = TensorDataset(x, y)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        y_pred = F.softmax(y_pred, dim=1)

        loss = criterion(y, y_pred)
        num_correct_preds = torch.sum(torch.argmax(y, dim=1) == torch.argmax(y_pred, dim=1)).item()
        accuracy = num_correct_preds / len(x)

    return loss, accuracy

def _run_one_epoch(model, dataloader, label_encoder, criterion, last_activation, optimizer, mode, l2_weight=1e-5, device=None, test_name='', verbose=1):
    """

    @param model:
    @param dataloader: contains x and y, y should be onehot encoded
    @param label_encoder:
    @param criterion:
    @param device:
    @param mode: can be train or val
    @param test_name:
    @param verbose:
    @return:
    """
    if device is None:
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
    if mode == 'train':
        model.train()
    elif mode == 'val':
        model.eval()
    else:
        raise ValueError('mode must be train or eval')
    grad_norms = []
    context_manager = torch.no_grad() if mode == 'test' else contextlib.nullcontext()
    mini_batch_i = 0

    if verbose >= 1:
        pbar = tqdm(total=math.ceil(len(dataloader.dataset) / dataloader.batch_size), desc=f'{mode} {test_name}')
        pbar.update(mini_batch_i := 0)
    batch_losses = []
    num_correct_preds = 0
    y_all = None
    y_all_pred = None
    for x, y in dataloader:
        if mode == 'train': optimizer.zero_grad()

        mini_batch_i += 1
        if verbose >= 1:
            pbar.update(1)

        with context_manager:
            y_pred = model(x)

            y_tensor = y.to(device)
            y_pred = last_activation(y_pred)
            classification_loss = criterion(y_pred, y_tensor)

        if mode == 'train' and l2_weight > 0:
            l2_penalty = l2_weight * sum([(p ** 2).sum() for p in model.parameters()])
        else:
            l2_penalty = 0
        loss = classification_loss + l2_penalty
        if mode == 'train':
            loss.backward()
            grad_norms.append([torch.mean(param.grad.norm()).item() for _, param in model.named_parameters() if  param.grad is not None])
            nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()

        y_all = np.concatenate([y_all, y.detach().cpu().numpy()]) if y_all is not None else y.detach().cpu().numpy()
        y_all_pred = np.concatenate([y_all_pred, y_pred.detach().cpu().numpy()]) if y_all_pred is not None else y_pred.detach().cpu().numpy()

        batch_losses.append(loss.item())
        if y_pred.shape[1] == 1:
            predicted_labels = (y_pred > .5).int()
            true_label = y_tensor
        else:
            predicted_labels = torch.argmax(y_pred, dim=1)
            true_label = torch.argmax(y_tensor, dim=1)
        num_correct_preds += torch.sum(true_label == predicted_labels).item()
        if verbose >= 1: pbar.set_description('Validating [{}]: loss:{:.8f}'.format(mini_batch_i, loss.item()))

    if verbose >= 1: pbar.close()
    return metrics.roc_auc_score(y_all, y_all_pred), np.mean(batch_losses), num_correct_preds / len(dataloader.dataset)


def train_model_pupil_eeg(X, Y, model, test_name="CNN-EEG-Pupil", n_folds=10, lr=1e-3, l2_weight=1e-5, verbose=1):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model.to(device)
    # create the train dataset
    label_encoder = preprocessing.OneHotEncoder()
    y = label_encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()
    X = model.prepare_data(X)

    skf = StratifiedShuffleSplit(n_splits=n_folds, random_state=random_seed)
    train_losses_folds = []
    train_accs_folds = []
    val_losses_folds = []
    val_accs_folds = []
    val_aucs_folds = []
    criterion = nn.CrossEntropyLoss()

    for f_index, (train, test) in enumerate(skf.split(X[0], Y)):
        x_eeg_train, x_eeg_test, y_train, y_test = X[0][train], X[0][test], Y[train], Y[test]
        x_pupil_train, x_pupil_test, y_train, y_test = X[1][train], X[1][test], Y[train], Y[test]

        x_eeg_train, y_train_e = rebalance_classes(x_eeg_train, y_train)  # rebalance by class
        x_pupil_train, y_train_p = rebalance_classes(x_pupil_train, y_train)  # rebalance by class
        assert np.all(y_train_e == y_train_p)

        y_train = label_encoder.transform(np.array(y_train_e).reshape(-1, 1)).toarray()
        y_test = label_encoder.transform(np.array(y_test).reshape(-1, 1)).toarray()

        train_size, val_size = len(x_eeg_train), len(x_eeg_test)
        x_eeg_train = torch.Tensor(x_eeg_train)  # transform to torch tensor
        x_eeg_test = torch.Tensor(x_eeg_test)

        x_pupil_train = torch.Tensor(x_pupil_train)  # transform to torch tensor
        x_pupil_test = torch.Tensor(x_pupil_test)

        y_train = torch.Tensor(y_train)
        y_test = torch.Tensor(y_test)

        train_dataset_eeg = TensorDataset(x_eeg_train, y_train)
        train_dataloader_eeg = DataLoader(train_dataset_eeg, batch_size=batch_size)

        train_dataset_pupil = TensorDataset(x_pupil_train, y_train)
        train_dataloader_pupil = DataLoader(train_dataset_pupil, batch_size=batch_size)

        val_dataset_eeg = TensorDataset(x_eeg_test, y_test)
        val_dataloader_eeg = DataLoader(val_dataset_eeg, batch_size=batch_size)

        val_dataset_pupil = TensorDataset(x_pupil_test, y_test)
        val_dataloader_pupil = DataLoader(val_dataset_pupil, batch_size=batch_size)

        # print("Model Summary: ")
        # summary(model, input_size=x_train.shape[1:])

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        val_aucs = []
        best_loss = np.inf
        patience_counter = []

        for epoch in range(epochs):
            mini_batch_i = 0
            batch_losses = []
            num_correct_preds = 0
            if verbose >= 1:
                pbar = tqdm(total=math.ceil(len(train_dataloader_eeg.dataset) / train_dataloader_eeg.batch_size),
                            desc='Training {}'.format(test_name))
                pbar.update(mini_batch_i)

            model.train()  # set the model in training model (dropout and batchnormal behaves differently in train vs. eval)
            for (x_eeg, y), (x_pupil, y) in zip(train_dataloader_eeg, train_dataloader_pupil):
                optimizer.zero_grad()

                mini_batch_i += 1
                if verbose >= 1:pbar.update(1)

                y_pred = model([x_eeg.to(device), x_pupil.to(device)])
                y_pred = F.softmax(y_pred, dim=1)
                # y_tensor = F.one_hot(y, num_classes=2).to(torch.float32).to(device)
                y_tensor = y.to(device)

                l2_penalty = l2_weight * sum([(p ** 2).sum() for p in model.parameters()]) if l2_weight > 0 else 0
                loss = criterion(y_pred, y_tensor) + l2_penalty
                # loss = criterion(y_tensor, y_pred)
                loss.backward()
                optimizer.step()

                # measure accuracy
                num_correct_preds += torch.sum(torch.argmax(y_tensor, dim=1) == torch.argmax(y_pred, dim=1)).item()

                if verbose >= 1: pbar.set_description('Training [{}]: loss:{:.8f}'.format(mini_batch_i, loss.item()))
                batch_losses.append(loss.item())
            train_losses.append(np.mean(batch_losses))
            train_accs.append(num_correct_preds / train_size)
            if verbose >= 1: pbar.close()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                if verbose >= 1:
                    pbar = tqdm(total=math.ceil(len(val_dataloader_eeg.dataset) / val_dataloader_eeg.batch_size),
                                desc='Validating {}'.format(test_name))
                    pbar.update(mini_batch_i := 0)
                batch_losses = []
                # batch_aucs =[]
                num_correct_preds = 0
                y_val = np.empty((0, 2))
                y_val_pred = np.empty((0, 2))
                for (x_eeg, y), (x_pupil, y) in zip(val_dataloader_eeg, val_dataloader_pupil):
                    mini_batch_i += 1
                    if verbose >= 1: pbar.update(1)
                    y_pred = model([x_eeg.to(device), x_pupil.to(device)])
                    y_pred = F.softmax(y_pred, dim=1)
                    # y_tensor = F.one_hot(y, num_classes=2).to(torch.float32).to(device)
                    y_tensor = y.to(device)
                    loss = criterion(y_pred, y_tensor)
                    # roc_auc = metrics.roc_auc_score(y, y_pred.detach().cpu().numpy())
                    # batch_aucs.append(roc_auc)

                    y_val = np.concatenate([y_val, y.detach().cpu().numpy()])
                    y_val_pred = np.concatenate([y_val_pred, y_pred.detach().cpu().numpy()])

                    batch_losses.append(loss.item())
                    num_correct_preds += torch.sum(torch.argmax(y_tensor, dim=1) == torch.argmax(y_pred, dim=1)).item()
                    if verbose >= 1:pbar.set_description('Validating [{}]: loss:{:.8f}'.format(mini_batch_i, loss.item()))

                val_aucs.append(metrics.roc_auc_score(y_val, y_val_pred))
                val_losses.append(np.mean(batch_losses))
                val_accs.append(num_correct_preds / val_size)
                if verbose >= 1:pbar.close()
            if verbose >= 1:
                print(
                "Epoch {}: train accuracy = {:.8f}, train loss={:.8f}; val accuracy = "
                "{:.8f}, val loss={:.8f}".format(epoch, train_accs[-1], train_losses[-1], val_accs[-1],val_losses[-1]))
            # Save training histories after every epoch
            training_histories = {'loss_train': train_losses, 'acc_train': train_accs, 'loss_val': val_losses,
                                  'acc_val': val_accs}
            pickle.dump(training_histories, open(os.path.join(model_save_dir, 'training_histories.pickle'), 'wb'))
            if val_losses[-1] < best_loss:
                torch.save(model.state_dict(), os.path.join(model_save_dir, test_name))
                if verbose >= 1:
                    print(
                        'Best model loss improved from {} to {}, saved best model to {}'.format(best_loss, val_losses[-1],
                                                                                                model_save_dir))
                best_loss = val_losses[-1]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > patience:
                    if verbose >= 1:
                        print(f'Terminated terminated by patience, validation loss has not improved in {patience} epochs')
                    break
        train_accs_folds.append(train_accs)
        train_losses_folds.append(train_losses)
        val_accs_folds.append(val_accs)
        val_losses_folds.append(val_losses)
        val_aucs_folds.append(val_aucs)
    training_histories_folds = {'loss_train': train_losses_folds, 'acc_train': train_accs_folds,
                                'loss_val': val_losses_folds,
                                'acc_val': val_accs_folds, 'auc_val': val_aucs_folds}
        # plt.plot(training_histories['acc_train'])
        # plt.plot(training_histories['acc_val'])
        # plt.title(f"Accuracy, {test_name}")
        # plt.show()
        #
        # plt.plot(training_histories['loss_train'])
        # plt.plot(training_histories['loss_val'])
        # plt.title(f"Loss, {test_name}")
        # plt.show()

    return model, training_histories_folds, criterion, label_encoder



def train_model_pupil_eeg_no_folds(X, Y, model, num_epochs=5000, test_name="CNN-EEG-Pupil", lr=1e-3, l2_weight=1e-5, verbose=1):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model.to(device)
    # create the train dataset
    label_encoder = preprocessing.OneHotEncoder()
    y = label_encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()
    y_train = torch.Tensor(y)

    X = model.prepare_data(X)

    criterion = nn.CrossEntropyLoss()

    x_eeg_train = torch.Tensor(X[0])  # transform to torch tensor
    x_pupil_train = torch.Tensor(X[1])  # transform to torch tensor
    train_size = len(x_eeg_train)


    train_dataset_eeg = TensorDataset(x_eeg_train, y_train)
    train_dataloader_eeg = DataLoader(train_dataset_eeg, batch_size=batch_size)

    train_dataset_pupil = TensorDataset(x_pupil_train, y_train)
    train_dataloader_pupil = DataLoader(train_dataset_pupil, batch_size=batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    train_losses = []
    train_accs = []
    best_loss = np.inf
    patience_counter = []

    for epoch in range(num_epochs):
        mini_batch_i = 0
        batch_losses = []
        num_correct_preds = 0
        if verbose >= 1:
            pbar = tqdm(total=math.ceil(len(train_dataloader_eeg.dataset) / train_dataloader_eeg.batch_size),
                        desc='Training {}'.format(test_name))
            pbar.update(mini_batch_i)

        model.train()  # set the model in training model (dropout and batchnormal behaves differently in train vs. eval)
        for (x_eeg, y), (x_pupil, y) in zip(train_dataloader_eeg, train_dataloader_pupil):
            optimizer.zero_grad()

            mini_batch_i += 1
            if verbose >= 1:pbar.update(1)

            y_pred = model([x_eeg.to(device), x_pupil.to(device)])
            y_pred = F.softmax(y_pred, dim=1)
            # y_tensor = F.one_hot(y, num_classes=2).to(torch.float32).to(device)
            y_tensor = y.to(device)

            l2_penalty = l2_weight * sum([(p ** 2).sum() for p in model.parameters()]) if l2_weight > 0 else 0
            loss = criterion(y_tensor, y_pred) + l2_penalty
            # loss = criterion(y_tensor, y_pred)
            loss.backward()
            optimizer.step()

            # measure accuracy
            num_correct_preds += torch.sum(torch.argmax(y_tensor, dim=1) == torch.argmax(y_pred, dim=1)).item()

            if verbose >= 1: pbar.set_description('Training [{}]: loss:{:.8f}'.format(mini_batch_i, loss.item()))
            batch_losses.append(loss.item())
        train_losses.append(np.mean(batch_losses))
        train_accs.append(num_correct_preds / train_size)
        if verbose >= 1: pbar.close()
        scheduler.step()

        if verbose >= 1:
            print("Epoch {}: train accuracy = {:.8f}, train loss={:.8f};".format(epoch, train_accs[-1], train_losses[-1]))
        # Save training histories after every epoch
        training_histories = {'loss_train': train_losses, 'acc_train': train_accs}
        pickle.dump(training_histories, open(os.path.join(model_save_dir, 'training_histories.pickle'), 'wb'))
        if train_losses[-1] < best_loss:
            torch.save(model.state_dict(), os.path.join(model_save_dir, test_name))
            if verbose >= 1:
                print('Best model loss improved from {} to {}, saved best model to {}'.format(best_loss, train_losses[-1], model_save_dir))
            best_loss = train_losses[-1]
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > patience:
                if verbose >= 1:
                    print(f'Terminated terminated by patience, validation loss has not improved in {patience} epochs')
                break

        # plt.plot(training_histories['acc_train'])
        # plt.plot(training_histories['acc_val'])
        # plt.title(f"Accuracy, {test_name}")
        # plt.show()
        #
        # plt.plot(training_histories['loss_train'])
        # plt.plot(training_histories['loss_val'])
        # plt.title(f"Loss, {test_name}")
        # plt.show()

    return model, {'train accs': train_accs, 'train losses': train_losses}, criterion, label_encoder

def prepare_sample_label(rdf, event_names, event_filters, data_type='eeg', picks=None, tmin_eeg=-0.1, tmax_eeg=1.0, participant=None, session=None ):
    assert len(event_names) == len(event_filters) == 2
    x, y, _, _ = epochs_to_class_samples_rdf(rdf, event_names, event_filters, data_type=data_type, picks=picks, tmin_eeg=tmin_eeg, tmax_eeg=tmax_eeg, participant=participant, session=session)
    return x, y
