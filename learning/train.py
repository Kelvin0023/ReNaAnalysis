import math
import os
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from torchsummary import summary
from tqdm import tqdm

from learning.HDCA import hdca
from learning.models import EEGPupilCNN, EEGCNN, EEGInceptionNet
from params import lr, epochs, batch_size, model_save_dir, patience, eeg_montage, l2_weight, random_seed, \
    export_data_root, num_top_compoenents
from utils.data_utils import compute_pca_ica, mean_sublists, rebalance_classes, mean_max_sublists, mean_min_sublists


def eval_lockings(rdf, event_names, locking_name_filters,  model_name, participant=None, session=None, regenerate_epochs=True, reduce_dim=False):
    # verify number of event types
    assert np.all(len(event_names) == np.array([len(x) for x in locking_name_filters.values()]))
    locking_performance = {}
    is_using_pupil = 'pupil' in str.lower(model_name)
    for locking_name, locking_filter in locking_name_filters.items():
        test_name = f'Locking-{locking_name}_Model-{model_name}_P-{participant}_S-{session}_Visual-Search'
        if regenerate_epochs:
            # x, y, _ = prepare_sample_label(rdf, event_names, locking_filter, participant=participant, session=session)  # pick all EEG channels
            x, y, _, _ = epochs_to_class_samples(rdf, event_names, locking_filter, data_type='both', rebalance=False, participant=participant, session=session)
            pickle.dump(x, open(os.path.join(export_data_root, f'x_P{participant}_S{session}_L{locking_name}.p'), 'wb'))
            pickle.dump(y, open(os.path.join(export_data_root, f'y_P{participant}_S{session}_L{locking_name}.p'), 'wb'))
        else:
            try:
                x = pickle.load(open(os.path.join(export_data_root, f'x_P{participant}_S{session}_L{locking_name}.p'), 'rb'))
                y = pickle.load(open(os.path.join(export_data_root, f'y_P{participant}_S{session}_L{locking_name}.p'), 'rb'))
            except FileNotFoundError:
                raise Exception(f"Unable to find saved epochs for participant {participant}, session {session}, locking {locking_name}" + ", EEGPupil" if model_name == 'EEGPupil' else "")
        x_eeg = np.copy(x[0])
        if reduce_dim:
            x[0] = compute_pca_ica(x[0], num_top_compoenents)

        if model_name == 'HDCA':
            roc_auc_combined, roc_auc_eeg, roc_auc_pupil = hdca([x_eeg, x[1]], y, event_names, is_plots=True, notes=test_name + '\n')  # give the original eeg data
            locking_performance[locking_name, 'HDCA EEG'] = {'folds val auc': roc_auc_eeg}
            locking_performance[locking_name, 'HDCA Pupil'] = {'folds val auc': roc_auc_pupil}
            locking_performance[locking_name, 'HDCA EEG-Pupil'] = {'folds val auc': roc_auc_combined}
            print(f'{test_name}: folds EEG AUC {roc_auc_eeg}, folds Pupil AUC: {roc_auc_pupil}, folds EEG-pupil AUC: {roc_auc_combined}')

        else:
            if model_name == 'EEGPupilCNN':
                model = EEGPupilCNN(eeg_in_shape=x[0].shape, pupil_in_shape=x[1].shape, num_classes=2,  eeg_in_channels=20 if reduce_dim else 64)
                model, training_histories, criterion, label_encoder = train_model_pupil_eeg(x, y, model, test_name=test_name, verbose=0)
            else:
                if model_name == 'EEGCNN':
                    model = EEGCNN(in_shape=x[0].shape, num_classes=2, in_channels=20 if reduce_dim else 64)
                elif model_name == 'EEGInception':
                    model = EEGInceptionNet(in_shape=x[0].shape, num_classes=2)
                model, training_histories, criterion, label_encoder = train_model(x[0], y, model, test_name=test_name, verbose=0)
            folds_train_acc, folds_val_acc, folds_train_loss, folds_val_loss = mean_max_sublists(training_histories['acc_train']), mean_max_sublists(training_histories['acc_val']), mean_min_sublists(training_histories['loss_val']), mean_min_sublists(training_histories['loss_val'])
            folds_val_auc = mean_max_sublists(training_histories['auc_val'])
            print(f'{test_name}: folds val AUC {folds_val_auc}, folds val accuracy: {folds_val_acc}, folds train accuracy: {folds_train_acc}, folds val loss: {folds_val_loss}, folds train loss: {folds_train_loss}')
            locking_performance[locking_name, model] = {'folds val auc': folds_val_auc, 'folds val acc': folds_val_acc, 'folds train acc': folds_train_acc, 'folds val loss': folds_val_loss, 'folds trian loss': folds_train_loss}
    return locking_performance

def eval_lockings_models(rdf, event_names, locking_name_filters, participant, session, models=('EEGCNN', 'EEGInception', 'EEGPupil'), regenerate_epochs=True, reduce_dim=False):
    # verify number of event types
    assert np.all(len(event_names) == np.array([len(x) for x in locking_name_filters.values()]))
    performance = {}
    for locking_name, locking_filter in locking_name_filters.items():
        test_name = f'L {locking_name}, P {participant}, S {session}, Visaul Search'
        if regenerate_epochs:
            x, y, _, _ = epochs_to_class_samples(rdf, event_names, locking_filter, data_type='both', rebalance=False, participant=participant, session=session)
            pickle.dump(x, open(os.path.join(export_data_root, f'x_P{participant}_S{session}_L{locking_name}_PupilEEG.p'), 'wb'))
            pickle.dump(y, open(os.path.join(export_data_root, f'y_P{participant}_S{session}_L{locking_name}_PupilEEG.p'), 'wb'))
        else:
            x = pickle.load(open(os.path.join(export_data_root, f'x_P{participant}_S{session}_L{locking_name}_PupilEEG.p'), 'rb'))
            y = pickle.load(open(os.path.join(export_data_root, f'y_P{participant}_S{session}_L{locking_name}_PupilEEG.p'), 'rb'))
        if reduce_dim:
            x[0] = compute_pca_ica(x[0], num_top_compoenents)  # reduce dimension of eeg data at index 0

        # data is ready for this locking from above, now iterate over the models
        for m in models:
            if m == 'EEGPupil':
                model = EEGPupilCNN(eeg_in_shape=x[0].shape, pupil_in_shape=x[1].shape, num_classes=2,  eeg_in_channels=20 if reduce_dim else 64)
                model, training_histories, criterion, label_encoder = train_model_pupil_eeg(x, y, model, test_name=test_name, verbose=0)
            else:
                if m == 'EEGCNN':
                    model = EEGCNN(in_shape=x.shape, num_classes=2, in_channels=20 if reduce_dim else 64)
                elif m == 'EEGInception':
                    model = EEGInceptionNet(in_shape=x.shape, num_classes=2)
                model, training_histories, criterion, label_encoder = train_model(x, y, model, test_name=test_name, verbose=0)
            best_train_acc, best_val_acc, best_train_loss, best_val_loss = mean_sublists(training_histories['acc_train']), mean_sublists(training_histories['acc_val']), mean_sublists(training_histories['loss_val']), mean_sublists(training_histories['loss_val'])
            best_val_auc = mean_sublists(training_histories['auc_val'])
            print(f'{test_name}: average val AUC {best_val_auc}, average val accuracy: {best_val_acc}, average train accuracy: {best_train_acc}, average val loss: {best_val_loss}, average train loss: {best_train_loss}')
            performance[m, locking_name] = {'average val auc': best_val_auc, 'average val acc': best_val_acc, 'average train acc': best_train_acc, 'average val loss': best_val_loss, 'average trian loss': best_train_loss}
    return performance

def train_model(X, Y, model, num_folds=10, test_name="CNN", verbose=1):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)
    # create the train dataset
    label_encoder = preprocessing.OneHotEncoder()
    label_encoder = label_encoder.fit(np.array(Y).reshape(-1, 1))
    X = model.prepare_data(X)

    skf = StratifiedShuffleSplit(n_splits=10, random_state=random_seed)
    train_losses_folds = []
    train_accs_folds = []
    val_losses_folds = []
    val_accs_folds = []
    val_aucs_folds = []
    criterion = nn.CrossEntropyLoss()

    for f_index, (train, test) in enumerate(skf.split(X, Y)):
        x_train, x_test, y_train, y_test = X[train], X[test], Y[train], Y[test]

        x_train, y_train = rebalance_classes(x_train, y_train)  # rebalance by class
        y_train = label_encoder.transform(np.array(y_train).reshape(-1, 1)).toarray()
        y_test = label_encoder.transform(np.array(y_test).reshape(-1, 1)).toarray()

        # x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_ratio, random_state=random_seed)
        train_size, val_size = len(x_train), len(x_test)
        x_train = torch.Tensor(x_train)  # transform to torch tensor
        x_test = torch.Tensor(x_test)
        y_train = torch.Tensor(y_train)
        y_test = torch.Tensor(y_test)

        train_dataset = TensorDataset(x_train, y_train)  # create your datset
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

        val_dataset = TensorDataset(x_test, y_test)  # create your datset
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        if verbose >= 1:
            print("Model Summary: ")
            summary(model, input_size=x_train.shape[1:])

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
                pbar = tqdm(total=math.ceil(len(train_dataloader.dataset) / train_dataloader.batch_size),
                            desc='Training {}'.format(test_name))
                pbar.update(mini_batch_i)

            model.train()  # set the model in training model (dropout and batchnormal behaves differently in train vs. eval)
            for x, y in train_dataloader:
                optimizer.zero_grad()

                mini_batch_i += 1
                if verbose >= 1: pbar.update(1)

                y_pred = model(x.to(device))
                y_pred = F.softmax(y_pred, dim=1)
                # y_tensor = F.one_hot(y, num_classes=2).to(torch.float32).to(device)
                y_tensor = y.to(device)

                l2_penalty = l2_weight * sum([(p ** 2).sum() for p in model.parameters()])
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

            model.eval()
            with torch.no_grad():
                if verbose >= 1:
                    pbar = tqdm(total=math.ceil(len(val_dataloader.dataset) / val_dataloader.batch_size),
                            desc='Validating {}'.format(test_name))
                    pbar.update(mini_batch_i := 0)
                batch_losses = []
                # batch_aucs =[]
                num_correct_preds = 0
                y_val = np.empty((0, 2))
                y_val_pred = np.empty((0, 2))
                for x, y in val_dataloader:
                    mini_batch_i += 1
                    if verbose >= 1: pbar.update(1)
                    y_pred = model(x.to(device))
                    y_pred = F.softmax(y_pred, dim=1)
                    # y_tensor = F.one_hot(y, num_classes=2).to(torch.float32).to(device)
                    y_tensor = y.to(device)
                    loss = criterion(y_tensor, y_pred)
                    # fpr, tpr, thresholds = metrics.roc_curve(y, y_pred.detach().cpu().numpy())
                    # roc_auc = metrics.roc_auc_score(y, y_pred.detach().cpu().numpy())
                    y_val = np.concatenate([y_val, y.detach().cpu().numpy()])
                    y_val_pred = np.concatenate([y_val_pred, y_pred.detach().cpu().numpy()])

                    # batch_aucs.append(roc_auc)
                    batch_losses.append(loss.item())
                    num_correct_preds += torch.sum(torch.argmax(y_tensor, dim=1) == torch.argmax(y_pred, dim=1)).item()
                    if verbose >= 1: pbar.set_description('Validating [{}]: loss:{:.8f}'.format(mini_batch_i, loss.item()))

                val_aucs.append(metrics.roc_auc_score(y_val, y_val_pred))
                val_losses.append(np.mean(batch_losses))
                val_accs.append(num_correct_preds / val_size)
                if verbose >= 1: pbar.close()
            if verbose >= 1:
                print(
                "Fold {}, Epoch {}: train accuracy = {:.8f}, train loss={:.8f}; val accuracy = "
                "{:.8f}, val loss={:.8f}".format(f_index, epoch, train_accs[-1], train_losses[-1], val_accs[-1],val_losses[-1]))
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
                        print(f'Fold {f_index}: Terminated terminated by patience, validation loss has not improved in {patience} epochs')
                    break
        train_accs_folds.append(train_accs)
        train_losses_folds.append(train_losses)
        val_accs_folds.append(val_accs)
        val_losses_folds.append(val_losses)
        val_aucs_folds.append(val_aucs)
    training_histories_folds = {'loss_train': train_losses_folds, 'acc_train': train_accs_folds, 'loss_val': val_losses_folds,
                          'acc_val': val_accs_folds, 'auc_val': val_aucs_folds}
    # plt.plot(training_histories['acc_train'])
    # plt.plot(training_histories['acc_val'])
    # plt.title(f"Accuracy, {test_name}")
    # plt.tight_layout()
    # plt.show()
    #
    # plt.plot(training_histories['loss_train'])
    # plt.plot(training_histories['loss_val'])
    # plt.title(f"Loss, {test_name}")
    # plt.tight_layout()
    # plt.show()
    if verbose: print(f"Average AUC for {num_folds} folds is {np.mean([np.max(x) for x in val_aucs_folds])}")
    return model, training_histories_folds, criterion, label_encoder

def eval_model(model, x, y, criterion, label_encoder):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    y = label_encoder.fit_transform(np.array(y).reshape(-1, 1)).toarray()
    x = torch.Tensor(x).to(device)  # transform to torch tensor
    y = torch.Tensor(y).to(device)
    model.eval()
    with torch.no_grad():
        y_pred = model(x)
        y_pred = F.softmax(y_pred, dim=1)

        loss = criterion(y, y_pred)
        num_correct_preds = torch.sum(torch.argmax(y, dim=1) == torch.argmax(y_pred, dim=1)).item()
        accuracy = num_correct_preds / len(x)

    return loss, accuracy

def epochs_to_class_samples(rdf, event_names, event_filters, rebalance=False, participant=None, session=None, picks=None, data_type='eeg', tmin_eeg=-0.1, tmax_eeg=0.8):
    """
    script will always z norm along channels for the input
    @param: data_type: can be eeg, pupil or mixed
    """
    if data_type == 'both':
        epochs_eeg, event_ids, ar_log, ps_group_eeg = rdf.get_eeg_epochs(event_names, event_filters, tmin=tmin_eeg, tmax=tmax_eeg, participant=participant, session=session)
        epochs_pupil, event_ids, ps_group_pupil = rdf.get_pupil_epochs(event_names, event_filters, participant=participant, session=session)
        epochs_pupil = epochs_pupil[np.logical_not(ar_log.bad_epochs)]
        ps_group_pupil = np.array(ps_group_pupil)[np.logical_not(ar_log.bad_epochs)]
        assert np.all(ps_group_pupil == ps_group_eeg)
        y = []
        x_eeg = [epochs_eeg[event_name].get_data(picks=picks) for event_name, _ in event_ids.items()]
        x_pupil = [epochs_pupil[event_name].get_data(picks=picks) for event_name, _ in event_ids.items()]
        x_eeg = np.concatenate(x_eeg, axis=0)
        x_pupil = np.concatenate(x_pupil, axis=0)

        for event_name, event_class in event_ids.items():
            y += [event_class] * len(epochs_pupil[event_name].get_data())
        if np.min(y) == 1:
            y = np.array(y) - 1
        if rebalance:
            x_eeg, y_eeg = rebalance_classes(x_eeg, y)
            x_pupil, y_pupil = rebalance_classes(x_pupil, y)
            assert np.all(y_eeg == y_pupil)
            y = y_eeg
        sanity_check_eeg(x_eeg, y, picks)
        sanity_check_pupil(x_pupil, y)
        return [x_eeg, x_pupil], y, [epochs_eeg, epochs_pupil], event_ids

    if data_type == 'eeg':
        epochs, event_ids, _, ps_group_eeg = rdf.get_eeg_epochs(event_names, event_filters, tmin=tmin_eeg, tmax=tmax_eeg, participant=participant, session=session)
    elif data_type == 'pupil':
        epochs, event_ids, ps_group_eeg = rdf.get_pupil_epochs(event_names, event_filters, participant=participant, session=session)
    else:
        raise NotImplementedError(f'data type {data_type} is not implemented')

    x = []
    y = []
    for event_name, event_class in event_ids.items():
        x.append(epochs[event_name].get_data(picks=picks))
        y += [event_class] * len(epochs[event_name].get_data())
    x = np.concatenate(x, axis=0)

    if np.min(y) == 1:
        y = np.array(y) - 1

    if rebalance:
        x, y = rebalance_classes(x, y)

    x = (x - np.mean(x, axis=(0, 2), keepdims=True)) / np.std(x, axis=(0, 2), keepdims=True)  # z normalize x

    if data_type == 'eeg':
        sanity_check_eeg(x, y, picks)
    elif data_type == 'pupil':
        sanity_check_pupil(x, y)

    # return x, y, epochs, event_ids, ps_group_eeg
    return x, y, epochs, event_ids


def sanity_check_eeg(x, y, picks):
    coi = picks.index('CPz') if picks else eeg_montage.ch_names.index('CPz')
    x_distractors = x[:, coi, :][y == 0]
    x_targets = x[:, coi, :][y == 1]
    x_distractors = np.mean(x_distractors, axis=0)
    x_targets = np.mean(x_targets, axis=0)
    plt.plot(x_distractors, label='distractor')
    plt.plot(x_targets, label='target')
    plt.title('EEG sample sanity check')
    plt.legend()
    plt.show()

def sanity_check_pupil(x, y):
    x_distractors = x[y == 0]
    x_targets = x[y == 1]
    x_distractors = np.mean(x_distractors, axis=(0, 1))  # also average left and right
    x_targets = np.mean(x_targets, axis=(0, 1))

    plt.plot(x_distractors, label='distractor')
    plt.plot(x_targets, label='target')
    plt.title('Pupil sample sanity check')
    plt.legend()
    plt.show()


def train_model_pupil_eeg(X, Y, model, test_name="CNN-EEG-Pupil", verbose=1):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model.to(device)
    # create the train dataset
    label_encoder = preprocessing.OneHotEncoder()
    y = label_encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()
    X = model.prepare_data(X)


    skf = StratifiedShuffleSplit(n_splits=10, random_state=random_seed)
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

                l2_penalty = l2_weight * sum([(p ** 2).sum() for p in model.parameters()])
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

            model.eval()
            with torch.no_grad():
                if verbose >= 1:
                    pbar = tqdm(total=math.ceil(len(val_dataloader_eeg.dataset) / val_dataloader_eeg.batch_size),
                                desc='Validating {}'.format(test_name))
                    pbar.update(mini_batch_i := 0)
                batch_losses = []
                batch_aucs =[]
                num_correct_preds = 0

                for (x_eeg, y), (x_pupil, y) in zip(val_dataloader_eeg, val_dataloader_pupil):
                    mini_batch_i += 1
                    if verbose >= 1: pbar.update(1)
                    y_pred = model([x_eeg.to(device), x_pupil.to(device)])
                    y_pred = F.softmax(y_pred, dim=1)
                    # y_tensor = F.one_hot(y, num_classes=2).to(torch.float32).to(device)
                    y_tensor = y.to(device)
                    loss = criterion(y_tensor, y_pred)
                    roc_auc = metrics.roc_auc_score(y, y_pred.detach().cpu().numpy())
                    batch_aucs.append(roc_auc)
                    batch_losses.append(loss.item())
                    num_correct_preds += torch.sum(torch.argmax(y_tensor, dim=1) == torch.argmax(y_pred, dim=1)).item()
                    if verbose >= 1:pbar.set_description('Validating [{}]: loss:{:.8f}, loss:{:.8f}'.format(mini_batch_i, loss.item(),roc_auc))

                val_aucs.append(np.mean(batch_aucs))
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


def prepare_sample_label(rdf, event_names, event_filters, data_type='eeg', picks=None, tmin_eeg=-0.1, tmax_eeg=1.0, participant=None, session=None ):
    assert len(event_names) == len(event_filters) == 2
    x, y, _, _ = epochs_to_class_samples(rdf, event_names, event_filters, data_type=data_type, picks=picks, tmin_eeg=tmin_eeg, tmax_eeg=tmax_eeg, participant=participant, session=session)
    return x, y
