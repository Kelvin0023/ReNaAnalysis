import contextlib
import copy
import math
import os
import pickle

import mne
import numpy as np
import torch
from sklearn import preprocessing, metrics
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from renaanalysis.learning.Conformer import interaug
from renaanalysis.learning.HT import ContrastiveLoss

from tqdm import tqdm

from renaanalysis.learning.HDCA import HDCA
from renaanalysis.learning.HT import HierarchicalTransformer
from renaanalysis.learning.HT_viz import ht_viz
from renaanalysis.learning.MutiInputDataset import MultiInputDataset
from renaanalysis.learning.models import EEGPupilCNN, EEGCNN, EEGInceptionNet
from renaanalysis.learning.preprocess import preprocess_samples_eeg_pupil
from renaanalysis.params.params import epochs, batch_size, model_save_dir, patience, random_seed, \
    export_data_root, TaskName
from renaanalysis.utils.training_utils import count_standard_error, count_target_error
from renaanalysis.utils.data_utils import rebalance_classes, mean_max_sublists, \
    mean_min_sublists
from renaanalysis.utils.rdf_utils import rena_epochs_to_class_samples_rdf
from renaanalysis.utils.viz_utils import viz_confusion_matrix, visualize_eeg_samples, plot_training_history


def eval_model(x_eeg, x_pupil, y, event_names, model_name, eeg_montage,
               test_name='', task_name=TaskName.TrainClassifier, n_folds=10, exg_resample_rate=200, ht_lr=1e-4, ht_l2=1e-6, ht_output_mode='multi',
               x_eeg_znormed=None, x_eeg_pca_ica=None, x_pupil_znormed=None, n_top_components=20, viz_rebalance=False, pca=None, ica=None, is_plot_conf_matrix=False):
    if x_pupil is None:
        if x_eeg_znormed is None or x_eeg_pca_ica is None:
            x_eeg_znormed, x_eeg_pca_ica, x_pupil_znormed, pca, ica = preprocess_samples_eeg_pupil(x_eeg, x_pupil, n_top_components)
    else:
        if x_eeg_znormed is None or x_eeg_pca_ica is None or x_pupil_znormed is None:
            x_eeg_znormed, x_eeg_pca_ica, x_pupil_znormed, pca, ica = preprocess_samples_eeg_pupil(x_eeg, x_pupil, n_top_components)
        else:
            assert pca is not None and ica is not None, Exception("pca and ica must not be None if x_eeg_znormed and x_eeg_pca_ica are given")


    model_performance, training_histories = _run_model(model_name, x_eeg_znormed, x_eeg_pca_ica, x_pupil_znormed, y, event_names, test_name, task_name, ht_output_mode, eeg_montage, ht_lr=ht_lr, ht_l2=ht_l2, n_folds=n_folds, exg_resample_rate=exg_resample_rate, pca=pca, ica=ica, viz_rebalance=viz_rebalance, is_plot_conf_matrix=is_plot_conf_matrix)
    return model_performance, training_histories

def _run_model(model_name, x_eeg, x_eeg_pca_ica, x_pupil, y, event_names, test_name, task_name,
               ht_output_mode, eeg_montage, pca, ica, ht_lr=1e-4, ht_l2=1e-6,
               n_folds=10, exg_resample_rate=200, viz_rebalance=False, is_plot_conf_matrix=False):
    """
    runs a given model. This funciton is called by eval_model
    @param model_name: name of the model, supports EEGCNN, EEGPupilCNN, EEGInceptionNet, HT, HDCA
    @param x_eeg: znormalized eeg data
    @param x_eeg_pca_ica: pca/ica eeg data
    @param x_pupil: znoramlized pupil data, can be None, in which case only eeg is used, if given model_name is a pupil model, an error is raised
    @param y:
    @param event_names:
    @param test_name:
    @param ht_output_mode: can be 'multi' or 'single', if 'multi', the model's output is a vector of length len(event_names) with softmax activation,
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
    assert np.all(np.unique(y_test) == np.unique(y_train) ), "train and test labels are not the same"
    assert len(np.unique(y_test)) == len(event_names), "number of unique labels is not the same as number of event names"

    if x_pupil is not None:
        x_pupil_train, x_pupil_test = x_pupil[train], x_pupil[test]
    else:
        x_pupil_train, x_pupil_test = None, None

    note = f"{test_name}"
    performance = {}
    training_histories = None
    if model_name == 'HDCA':
        # hdca_func = hdca if x_pupil is not None else hdca_eeg
        # roc_auc_combined, roc_auc_eeg, roc_auc_pupil = hdca_func(x_eeg_train, x_eeg_pca_ica_train, x_pupil_train, y_train, event_names,is_plots=True, exg_srate=exg_resample_rate, notes=note, verbose=0)  # give the original eeg data, no need to apply HDCA again

        hdca_model = HDCA(event_names)
        roc_auc_combined, roc_auc_eeg, roc_auc_pupil = hdca_model.fit(x_eeg_train, x_eeg_pca_ica_train, x_pupil_train, y_train, is_plots=True, exg_srate=exg_resample_rate, notes=note, verbose=0)  # give the original eeg data, no need to apply HDCA again

        y_pred, roc_auc_eeg_pupil, roc_auc_eeg, roc_auc_pupil = hdca_model.eval(x_eeg_test, x_eeg_pca_ica_test, x_pupil_test, y_test, notes=note)
        if x_pupil is not None:
            performance[f"{model_name}_Pupil"] = {'test auc': roc_auc_pupil, 'folds val auc': roc_auc_pupil}
            performance[f"{model_name}_EEG-Pupil"] = {'test auc': roc_auc_eeg_pupil, 'folds val auc': roc_auc_combined}
        performance[f"{model_name}_EEG"] = {'test auc': roc_auc_eeg, 'folds val auc': roc_auc_combined}
        print(f'{note}: test auc combined {roc_auc_combined}, test auc combined {roc_auc_eeg}, test auc pupil {roc_auc_pupil}\n'
              f'folds EEG AUC {roc_auc_eeg}, folds Pupil AUC: {roc_auc_pupil}, folds EEG-pupil AUC: {roc_auc_combined}\n')
    else:
        if model_name == 'EEGPupilCNN':  # this model uses PCA-ICA reduced EEG data plus pupil data
            assert x_pupil is not None, "Pupil data is not provided, which is required for EEGPupilCNN model"
            model = EEGPupilCNN(eeg_in_shape=x_eeg_pca_ica.shape, pupil_in_shape=x_pupil.shape, num_classes=2)
            model, training_histories, criterion, last_activation, _encoder = cv_train_test_model([x_eeg_pca_ica_train, x_pupil_train], y_train, model, test_name=test_name, verbose=1, n_folds=n_folds)
            # model, training_histories, criterion, label_encoder = train_model_pupil_eeg([x_eeg_pca_ica_train, x_pupil_train], y_train, model, test_name=test_name, n_folds=n_folds)
            test_auc, test_loss, test_acc = eval_test(model, [x_eeg_pca_ica_test, x_pupil_test], y_test, criterion, last_activation, _encoder, test_name='', verbose=1)

        elif model_name == 'HT' or model_name == 'HT-pca-ica':  # this model uses un-dimension reduced EEG data
            num_timesteps = x_eeg_train.shape[2]
            num_channels = x_eeg_pca_ica_train.shape[1] if model_name == 'HT-pca-ica' else x_eeg_train.shape[1]

            model = HierarchicalTransformer(num_timesteps, num_channels, exg_resample_rate, num_classes=2, output=ht_output_mode)
            training_data = x_eeg_pca_ica_train if model_name == 'HT-pca-ica' else x_eeg_train
            test_data = x_eeg_pca_ica_test if model_name == 'HT-pca-ica' else x_eeg_test
            models, training_histories, criterion, last_activation, _encoder, test_auc, test_loss, test_acc = cv_train_test_model(training_data, y_train, model, is_plot_conf_matrix=is_plot_conf_matrix, X_test=test_data, Y_test=y_test, test_name=test_name, task_name=task_name, verbose=1, lr=ht_lr, l2_weight=ht_l2, n_folds=n_folds, viz_rebalance=viz_rebalance)  # use un-dimension reduced EEG data
            rollout_data_root = f'{note}'
            if not os.path.exists(rollout_data_root):
                os.mkdir(rollout_data_root)
            for i in range(n_folds):
                ht_viz(models[i], x_eeg_test, y_test, _encoder, event_names, rollout_data_root, models[i].window_duration, exg_resample_rate,
                       eeg_montage, num_timesteps, num_channels, note='', head_fusion='max', discard_ratio=0.9, load_saved_rollout=False, batch_size=64, X_pca_ica=test_data if model_name == 'HT-pca-ica' else None, pca=pca, ica=ica)
        elif model_name == 'HT-sesup' or 'HT-pca-ica-sesup':
            num_timesteps = x_eeg_train.shape[2]
            num_channels = x_eeg_pca_ica_train.shape[1] if model_name == 'HT-pca-ica-sesup' else x_eeg_train.shape[1]
            model = HierarchicalTransformer(num_timesteps, num_channels, exg_resample_rate, num_classes=2,
                                            output=ht_output_mode)
            training_data = x_eeg_pca_ica_train if model_name == 'HT-pca-ica-sesup' else x_eeg_train
            test_data = x_eeg_pca_ica_test if model_name == 'HT-pca-ica-sesup' else x_eeg_test
            models, training_histories, criterion, last_activation, _encoder = cv_train_test_model(
                training_data, y_train, model, is_plot_conf_matrix=False, X_test=test_data, Y_test=y_test,
                test_name=test_name, verbose=1, lr=ht_lr, l2_weight=ht_l2, n_folds=n_folds)  # use un-dimension reduced EEG data
            rollout_data_root = f'{note}'
            if not os.path.exists(rollout_data_root):
                os.mkdir(rollout_data_root)
            for i in range(n_folds):
                ht_viz(models[i], x_eeg_test, y_test, _encoder, event_names, rollout_data_root,
                       models[i].window_duration, exg_resample_rate,
                       eeg_montage, num_timesteps, num_channels, note='', head_fusion='max', discard_ratio=0.9,
                       load_saved_rollout=False, batch_size=64,
                       X_pca_ica=test_data if model_name == 'HT-pca-ica' else None)
        else:  # these models use PCA-ICA reduced EEG data
            if model_name == 'EEGCNN':
                model = EEGCNN(in_shape=x_eeg_pca_ica.shape, num_classes=2)
            elif model_name == 'EEGInception':
                model = EEGInceptionNet(in_shape=x_eeg_pca_ica.shape, num_classes=2)
            else:
                raise Exception(f"Unknown model name {model_name}")
            model, training_histories, criterion, last_activation, _encoder = cv_train_test_model(x_eeg_pca_ica_train, y_train, model, test_name=test_name, verbose=1, n_folds=n_folds)
            test_auc, test_loss, test_acc = eval_test(model, x_eeg_pca_ica_test, y_test, criterion, last_activation, _encoder, test_name='', verbose=1)

        folds_train_acc, folds_val_acc, folds_train_loss, folds_val_loss = mean_max_sublists(training_histories['acc_train']), mean_max_sublists(training_histories['acc_val']), mean_min_sublists(training_histories['loss_val']), mean_min_sublists(training_histories['loss_val'])
        folds_val_auc = mean_max_sublists(training_histories['auc_val'])
        performance[model_name] = {'test auc': test_auc, 'folds val auc': folds_val_auc, 'folds val acc': folds_val_acc, 'folds train acc': folds_train_acc, 'folds val loss': folds_val_loss, 'folds trian loss': folds_train_loss}
        print(f'{test_name}: test AUC {test_auc}, test accuracy: {test_acc}, test loss: {test_loss}\n'
              f'folds val AUC {folds_val_auc}, folds val accuracy: {folds_val_acc}, folds train accuracy: {folds_train_acc}, folds val loss: {folds_val_loss}, folds train loss: {folds_train_loss}\n')
    print("#" * 100)
    return performance, training_histories


def eval_test(model, X, Y, criterion, last_activation, _encoder, task_name=TaskName.TrainClassifier, verbose=1):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if isinstance(X, list):
        X = [torch.Tensor(this_x).to(device) for this_x in X]
        dataset_class = MultiInputDataset
    else:
        X = torch.Tensor(X).to(device)
        dataset_class = TensorDataset

    if task_name == TaskName.PreTrain:
        test_dataset = dataset_class(X)
    elif task_name == TaskName.TrainClassifier or task_name == TaskName.PretrainedClassifierFineTune:
        Y = _encoder(Y)
        Y = torch.Tensor(Y).to(device)
        test_dataset = dataset_class(X, Y)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    if task_name == TaskName.PreTrain:
        return _run_one_epoch_self_sup(model, test_dataloader, criterion, optimizer=None, mode='val', device=device, task_name=task_name, verbose=verbose)
    elif task_name == TaskName.TrainClassifier or task_name == TaskName.PretrainedClassifierFineTune:
        return _run_one_epoch_classification(model, test_dataloader, criterion, last_activation, optimizer=None, mode='val', device=device, task_name=task_name, verbose=verbose)

def eval_test_augmented(model, X, Y, criterion, last_activation, _encoder, task_name=TaskName.TrainClassifier, verbose=1):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if isinstance(X, list):
        X = [torch.Tensor(this_x).to(device) for this_x in X]
        dataset_class = MultiInputDataset
    else:
        X = torch.Tensor(X).to(device)
        dataset_class = TensorDataset

    if task_name == TaskName.PreTrain:
        test_dataset = dataset_class(X)
    elif task_name == TaskName.TrainClassifier or task_name == TaskName.PretrainedClassifierFineTune:
        Y = _encoder(Y)
        Y = torch.Tensor(Y).to(device)
        test_dataset = dataset_class(X, Y)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    if task_name == TaskName.PreTrain:
        return _run_one_epoch_self_sup(model, test_dataloader, criterion, optimizer=None, mode='val', device=device, task_name=task_name, verbose=verbose)
    elif task_name == TaskName.TrainClassifier or task_name == TaskName.PretrainedClassifierFineTune:
        return _run_one_epoch_classification_augmented(model, test_dataloader, criterion, last_activation, optimizer=None, mode='val', device=device, task_name=task_name, verbose=verbose)

def cv_train_test_model(X, Y, model, test_name="", task_name=TaskName.TrainClassifier, n_folds=10, lr=1e-4, verbose=1, l2_weight=1e-6, lr_scheduler_type='exponential', is_plot_conf_matrix=False, is_by_channel=False, rebalance_method='SMOT', X_test=None, Y_test=None, plot_histories=True, viz_rebalance=False):
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
    # determine training mode

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
    if rebalance_method == 'class weight':
        #compute class proportion
        unique_classes, counts = np.unique(Y, return_counts=True)
        num_unique_classes = len(unique_classes)
        class_proportions = counts / len(Y)
        class_weights = 1/class_proportions
        class_weights = torch.tensor(class_weights).to(device)

    with torch.no_grad():
        model.eval()
        output_shape = model.to(device)(rand_input).shape[1]

    if output_shape == 1:
        assert len(np.unique(Y)) == 2, "Model only has one output node. But given Y has more than two classes. Binary classification model should have 2 classes"
        label_encoder = LabelEncoder()
        label_encoder.fit(Y)
        _encoder = lambda y: label_encoder.transform(y).reshape(-1, 1)
        # _decoder = lambda y: label_encoder.inverse_transform(y.reshape(-1, 1))
        criterion = nn.BCELoss(reduction='mean')
        last_activation = nn.Sigmoid()
    else:
        label_encoder = preprocessing.OneHotEncoder()
        label_encoder.fit(Y.reshape(-1, 1))
        _encoder = lambda y: label_encoder.transform(y.reshape(-1, 1)).toarray()
        # _decoder = lambda y: label_encoder.inverse_transform(y.reshape(-1, 1))
        if rebalance_method == 'SMOT':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        last_activation = nn.Softmax(dim=1)
    with open(os.path.join(export_data_root, 'label_encoder.p'), 'wb') as f:
        pickle.dump(label_encoder, f)

    X = model.prepare_data(X)

    skf = StratifiedShuffleSplit(n_splits=n_folds, random_state=random_seed)
    train_losses_folds = []
    train_accs_folds = []
    val_losses_folds = []
    val_accs_folds = []
    val_aucs_folds = []
    models = []

    model_copy = None
    test_auc = []
    test_acc = []
    test_loss = []
    for f_index, (train, val) in enumerate(skf.split(X[0] if isinstance(X, list) else X, Y)):
        model_copy = copy.deepcopy(model)
        # model_copy = HierarchicalTransformer(180, 20, 200, num_classes=2,
        #                                 output='multi')
        model_copy = model_copy.to(device)
        # model = model.to(device)
        # rollout = VITAttentionRollout(model_copy, device, attention_layer_class=Attention,
        #                               token_shape=model_copy.grid_dims,
        #                               discard_ratio=0.9, head_fusion='max')

        if isinstance(X, list):
            x_train = []
            x_val = []
            y_train, y_val = Y[train], Y[val]

            rebalanced_labels = []
            for this_x in X:
                this_x_train, this_y_train = rebalance_classes(this_x[train], y_train, by_channel=is_by_channel) if rebalance_method == 'SMOT' else zip(this_x[train], y_train)
                x_train.append(torch.Tensor(this_x_train).to(device))
                rebalanced_labels.append(this_y_train)
                x_val.append(torch.Tensor(this_x[val]).to(device))
            assert np.all([label_set == rebalanced_labels[0] for label_set in rebalanced_labels])
            y_train = rebalanced_labels[0]
        else:
            x_train, x_val, y_train, y_val = X[train], X[val], Y[train], Y[val]
            if rebalance_method == 'SMOT':
                x_train, y_train = rebalance_classes(x_train, y_train, by_channel=is_by_channel)  # rebalance by class
            if viz_rebalance:
                colors = {0:'red', 6:'blue'}
                eeg_picks = mne.channels.make_standard_montage('biosemi64').ch_names
                visualize_eeg_samples(x_train, y_train, colors, eeg_picks, out_dir=f'renaanalysis/learning/saved_images/by_channel_{is_by_channel}')
            x_train = torch.Tensor(x_train).to(device)
            x_val = torch.Tensor(x_val).to(device)

        y_train_encoded = _encoder(y_train)
        y_val_encoded = _encoder(y_val)

        y_train_encoded = torch.Tensor(y_train_encoded)
        y_val_encoded = torch.Tensor(y_val_encoded)

        train_dataset = dataset_class(x_train, y_train_encoded)
        val_dataset = dataset_class(x_val, y_val_encoded)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        optimizer = torch.optim.Adam(model_copy.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(model_copy.parameters(), lr=lr, momentum=0.9)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        patience_counter = 0

        num_train_standard_errors = []
        num_train_target_errors = []
        num_val_standard_errors = []
        num_val_target_errors = []
        best_auc = 0
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        val_aucs = []
        for epoch in range(epochs):
        # prev_para = []
        # for param in model_copy.parameters():
        #     prev_para.append(param.cpu().detach().numpy())
            train_auc, train_loss, train_accuracy, num_train_standard_error, num_train_target_error, train_y_all, train_y_all_pred = _run_one_epoch_classification(model_copy, train_dataloader, criterion, last_activation, optimizer, mode='train', device=device, l2_weight=l2_weight, test_name=test_name, task_name=task_name, verbose=verbose)
            if is_plot_conf_matrix:
                train_predicted_labels_all = np.argmax(train_y_all_pred, axis=1)
                train_true_label_all = np.argmax(train_y_all, axis=1)
                num_train_standard_errors.append(num_train_standard_error)
                num_train_target_errors.append(num_train_target_error)
                viz_confusion_matrix(train_true_label_all, train_predicted_labels_all, epoch, f_index, 'train')
            scheduler.step()
            # ht_viz_training(X, Y, model_copy, rollout, _encoder, device, epoch)
            val_auc, val_loss, val_accuracy, num_val_standard_error, num_val_target_error, val_y_all, val_y_all_pred = _run_one_epoch_classification(model_copy, val_dataloader, criterion, last_activation, optimizer, mode='val', device=device, l2_weight=l2_weight, test_name=test_name, task_name=task_name, verbose=verbose)
            if is_plot_conf_matrix:
                val_predicted_labels_all = np.argmax(val_y_all_pred, axis=1)
                val_true_label_all = np.argmax(val_y_all, axis=1)
                num_val_standard_errors.append(num_val_standard_error)
                num_val_target_errors.append(num_val_target_error)
                viz_confusion_matrix(val_true_label_all, val_predicted_labels_all, epoch, f_index, 'val')

            train_losses.append(train_loss)
            train_accs.append(train_accuracy)
            val_aucs.append(val_auc)
            val_losses.append(val_loss)
            val_accs.append(val_accuracy)

            if verbose >= 1:
                print("Fold {}, Epoch {}: val auc = {:.16f}, train accuracy = {:.16f}, train loss={:.16f}; val accuracy = {:.16f}, val loss={:.16f}, patience left {}".format(f_index, epoch, np.max(val_aucs), train_accs[-1], train_losses[-1], val_accs[-1],val_losses[-1], patience - patience_counter))
            if val_auc > best_auc:
                if verbose >= 1: print('Best validation auc improved from {} to {}'.format(best_auc, val_auc))
                # best_loss = val_losses[-1]
                best_auc = val_auc
                patience_counter = 0
                best_model = copy.deepcopy(model_copy)
            else:
                patience_counter += 1
                if patience_counter > patience:
                    if verbose >= 1: print(f'Fold {f_index}: Terminated terminated by patience, validation loss has not improved in {patience} epochs')
                    break
        # viz_class_error(num_train_standard_errors, num_train_target_errors, 'train')
        # viz_class_error(num_val_standard_errors, num_val_target_errors, 'validation')
        train_accs_folds.append(train_accs)
        train_losses_folds.append(train_losses)
        val_accs_folds.append(val_accs)
        val_losses_folds.append(val_losses)
        val_aucs_folds.append(val_aucs)
        test_auc_model, test_loss_model, test_acc_model, num_test_standard_error, num_test_target_error, test_y_all, test_y_all_pred = eval_test(best_model, X_test, Y_test, criterion, last_activation, _encoder,
                                         task_name=task_name, verbose=1)
        if verbose >= 1:
            print("Tested Fold {}: test auc = {:.8f}, test loss = {:.8f}, test acc = {:.8f}".format(f_index, test_auc_model, test_loss_model, test_acc_model))
        test_auc.append(test_auc_model)
        test_loss.append(test_loss_model)
        test_acc.append(test_acc_model)
        models.append(best_model)

    training_histories_folds = {'loss_train': train_losses_folds, 'acc_train': train_accs_folds, 'loss_val': val_losses_folds, 'acc_val': val_accs_folds, 'auc_val': val_aucs_folds, 'auc_test': test_auc, 'acc_test': test_acc, 'loss_test': test_loss}
    if plot_histories:
        for i in range(n_folds):
            history = {'loss_train': training_histories_folds['loss_train'][i],
                       'acc_train': training_histories_folds['acc_train'][i],
                       'loss_val': training_histories_folds['loss_val'][i],
                       'acc_val': training_histories_folds['acc_val'][i],
                       'auc_val': training_histories_folds['auc_val'][i],
                       'auc_test': training_histories_folds['auc_test'][i],
                       'acc_test': training_histories_folds['acc_test'][i],
                       'loss_test': training_histories_folds['loss_test'][i]}
            seached_params = None
            plot_training_history(history, seached_params, i)


    return models, training_histories_folds, criterion, last_activation, _encoder, test_auc, test_loss, test_acc

def self_supervised_pretrain(X, model, test_name="CNN", task_name=TaskName.PreTrain, n_folds=10, lr=1e-4, verbose=1, l2_weight=1e-6,
                            lr_scheduler_type='exponential', temperature=1, n_neg=20, is_plot_conf_matrix=False,
                            X_test=None, plot_histories=True):

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

    criterion = ContrastiveLoss(temperature, n_neg)
    last_activation = None

    X = model.prepare_data(X)

    sf = ShuffleSplit(n_splits=n_folds, random_state=random_seed)
    train_losses_folds = []
    val_losses_folds = []
    models = []

    model_copy = None
    test_loss = []
    for f_index, (train, val) in enumerate(sf.split(X[0] if isinstance(X, list) else X)):
        model_copy = copy.deepcopy(model)
        # model_copy = HierarchicalTransformer(180, 20, 200, num_classes=2,
        #                                 output='multi')
        model_copy = model_copy.to(device)
        # model = model.to(device)
        # rollout = VITAttentionRollout(model_copy, device, attention_layer_class=Attention,
        #                               token_shape=model_copy.grid_dims,
        #                               discard_ratio=0.9, head_fusion='max')

        if isinstance(X, list):
            x_train = []
            x_val = []

            rebalanced_labels = []
            for this_x in X:
                this_x_train, this_y_train = rebalance_classes(this_x[train], y_train, by_channel=is_by_channel) if rebalance_method == 'SMOT' else zip(this_x[train], y_train)
                x_train.append(torch.Tensor(this_x_train).to(device))
                rebalanced_labels.append(this_y_train)
                x_val.append(torch.Tensor(this_x[val]).to(device))
            assert np.all([label_set == rebalanced_labels[0] for label_set in rebalanced_labels])
            y_train = rebalanced_labels[0]
        else:
            x_train, x_val = X[train], X[val]
            x_train = torch.Tensor(x_train).to(device)
            x_val = torch.Tensor(x_val).to(device)

        train_dataset = dataset_class(x_train)
        val_dataset = dataset_class(x_val)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        optimizer = torch.optim.Adam(model_copy.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(model_copy.parameters(), lr=lr, momentum=0.9)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        patience_counter = 0

        train_mean_losses = []
        val_mean_losses = []
        best_loss = np.inf
        for epoch in range(epochs):
            # prev_para = []
            # for param in model_copy.parameters():
            #     prev_para.append(param.cpu().detach().numpy())
            train_batch_losses, train_mean_loss = _run_one_epoch_self_sup(
                model_copy, train_dataloader, criterion, optimizer, mode='train', device=device,
                l2_weight=l2_weight, test_name=test_name, task_name=task_name, verbose=verbose)
            scheduler.step()
            val_batch_losses, val_mean_loss = _run_one_epoch_self_sup(
                model_copy, val_dataloader, criterion, optimizer, mode='val', device=device,
                l2_weight=l2_weight, test_name=test_name, task_name=task_name, verbose=verbose)

            train_mean_losses.append(train_mean_loss)
            val_mean_losses.append(val_mean_loss)

            if verbose >= 1:
                print(
                    "Fold {}, Epoch {}: val loss = {:.16f}, train loss = {:.16f}, patience left {}".format(
                        f_index, epoch, val_mean_loss, train_mean_loss, patience - patience_counter))
            if val_mean_loss < best_loss:
                if verbose >= 1: print(
                    'Best validation loss improved from {} to {}'.format(best_loss, val_mean_loss))
                # best_loss = val_losses[-1]
                best_loss = val_mean_loss
                patience_counter = 0
                best_model = copy.deepcopy(model_copy)
            else:
                patience_counter += 1
                if patience_counter > patience:
                    if verbose >= 1: print(
                        f'Fold {f_index}: Terminated terminated by patience, validation loss has not improved in {patience} epochs')
                    break
        train_losses_folds.append(train_mean_losses)
        val_losses_folds.append(val_mean_losses)
        test_batch_losses_model, test_mean_loss_model = eval_test(
            best_model, X_test, None, criterion, None, None,
            task_name=task_name, verbose=1)
        if verbose >= 1:
            print("Tested Fold {}: test mean loss = {:.8f}".format(f_index, test_mean_loss_model))
        test_loss.append(test_mean_loss_model)
        models.append(best_model)

    training_histories_folds = {'loss_train': train_losses_folds, 'loss_val': val_losses_folds, 'loss_test': test_loss}


    return models, training_histories_folds, criterion, last_activation

def _run_one_epoch_classification(model, dataloader, criterion, last_activation, optimizer, mode, l2_weight=1e-5, device=None, test_name='', task_name=TaskName.TrainClassifier, verbose=1, check_param=1):
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
        # determine which layer to require grad
        if task_name == TaskName.PretrainedClassifierFineTune:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.transformer.parameters():
                param.requires_grad = True
            for param in model.mlp_head.parameters():
                param.requires_grad = True
            model.cls_token.requires_grad = True
    elif mode == 'val':
        model.eval()
    else:
        raise ValueError('mode must be train or eval')
    grad_norms = []
    context_manager = torch.no_grad() if mode == 'val' else contextlib.nullcontext()
    mini_batch_i = 0

    if verbose >= 1:
        pbar = tqdm(total=len(dataloader), desc=f'{mode} {test_name}', unit='batch')
        pbar.update(mini_batch_i := 0)
    batch_losses = []
    num_correct_preds = 0
    y_all = None
    y_all_pred = None
    num_standard_errors = 0
    num_target_errors = 0
    for x, y in dataloader:
        if mode == 'train': optimizer.zero_grad()

        mini_batch_i += 1
        if verbose >= 1:
            pbar.update(1)

        # if check_param:
        #     for key1, value1 in a.items():
        #         value2 = b[key1]
        #         # Perform tensor comparison
        #         if torch.equal(value1, value2):
        #             print(f"Tensor {key1} is equal in both models.")
        #         else:
        #             print(f"Tensor {key1} is not equal in both models.")

        with context_manager:
            x = x if isinstance(x, tuple) else (x,)
            y_pred = model(*x)

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
        num_standard_errors += count_standard_error(true_label, predicted_labels)
        num_target_errors += count_target_error(true_label, predicted_labels)
        if verbose >= 1: pbar.set_description('{} [{}]: loss:{:.8f}'.format(mode, mini_batch_i, loss.item()))

    if verbose >= 1: pbar.close()
    return metrics.roc_auc_score(y_all, y_all_pred), np.mean(batch_losses), num_correct_preds / len(dataloader.dataset), num_standard_errors, num_target_errors, y_all, y_all_pred

def _run_one_epoch_classification_augmented(model, dataloader, criterion, last_activation, optimizer, mode, l2_weight=1e-5, device=None, test_name='', task_name=TaskName.TrainClassifier, verbose=1, check_param=1):
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
        # determine which layer to require grad
        if task_name == TaskName.PretrainedClassifierFineTune:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.transformer.parameters():
                param.requires_grad = True
            for param in model.mlp_head.parameters():
                param.requires_grad = True
            model.cls_token.requires_grad = True
    elif mode == 'val':
        model.eval()
    else:
        raise ValueError('mode must be train or eval')
    grad_norms = []
    context_manager = torch.no_grad() if mode == 'val' else contextlib.nullcontext()
    mini_batch_i = 0

    if verbose >= 1:
        pbar = tqdm(total=math.ceil(len(dataloader.dataset) / dataloader.batch_size), desc=f'{mode} {test_name}')
        pbar.update(mini_batch_i := 0)
    batch_losses = []
    num_correct_preds = 0
    y_all = None
    y_all_pred = None
    num_standard_errors = 0
    num_target_errors = 0
    for x, y in dataloader:
        aug_data, aug_labels = interaug(x, y)
        aug_data = aug_data.to(device)
        aug_labels = aug_labels.to(device)
        x = torch.cat((x, aug_data), dim=0)
        y = torch.cat((y, aug_labels), dim=0)
        if mode == 'train': optimizer.zero_grad()

        mini_batch_i += 1
        if verbose >= 1:
            pbar.update(1)

        # if check_param:
        #     for key1, value1 in a.items():
        #         value2 = b[key1]
        #         # Perform tensor comparison
        #         if torch.equal(value1, value2):
        #             print(f"Tensor {key1} is equal in both models.")
        #         else:
        #             print(f"Tensor {key1} is not equal in both models.")

        with context_manager:
            x = x if isinstance(x, tuple) else (x,)
            y_pred = model(*x)

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
        num_standard_errors += count_standard_error(true_label, predicted_labels)
        num_target_errors += count_target_error(true_label, predicted_labels)
        if verbose >= 1: pbar.set_description('{} [{}]: loss:{:.8f}'.format(mode, mini_batch_i, loss.item()))

    if verbose >= 1: pbar.close()
    return metrics.roc_auc_score(y_all, y_all_pred), np.mean(batch_losses), num_correct_preds / len(dataloader.dataset), num_standard_errors, num_target_errors, y_all, y_all_pred

def _run_one_epoch_self_sup(model, dataloader, criterion, optimizer, mode, l2_weight=1e-5, device=None, test_name='', task_name=TaskName.PreTrain, verbose=1, check_param=1):
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
    context_manager = torch.no_grad() if mode == 'val' else contextlib.nullcontext()
    mini_batch_i = 0

    if verbose >= 1:
        pbar = tqdm(total=math.ceil(len(dataloader.dataset) / dataloader.batch_size), desc=f'{mode} {test_name}')
        pbar.update(mini_batch_i := 0)
    batch_losses = []
    for x in dataloader:
        if mode == 'train': optimizer.zero_grad()

        mini_batch_i += 1
        if verbose >= 1:
            pbar.update(1)

        # if check_param:
        #     for key1, value1 in a.items():
        #         value2 = b[key1]
        #         # Perform tensor comparison
        #         if torch.equal(value1, value2):
        #             print(f"Tensor {key1} is equal in both models.")
        #         else:
        #             print(f"Tensor {key1} is not equal in both models.")

        with context_manager:
            x = x if isinstance(x[0], tuple) else (x[0],)
            pred_tokens, orig_tokens, mask_t, mask_c = model(*x)
            # y_tensor = y.to(device)
            classification_loss = criterion(pred_tokens, orig_tokens)

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

        batch_losses.append(loss.item())
        if verbose >= 1: pbar.set_description('{} [{}]: loss:{:.8f}'.format(mode, mini_batch_i, loss.item()))

    if verbose >= 1: pbar.close()
    return batch_losses, np.mean(batch_losses)

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
    x, y, _, _ = rena_epochs_to_class_samples_rdf(rdf, event_names, event_filters, data_type=data_type, picks=picks, tmin_eeg=tmin_eeg, tmax_eeg=tmax_eeg, participant=participant, session=session)
    return x, y
