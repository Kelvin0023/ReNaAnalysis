import math
import os
import pickle

import numpy as np
import torch
from sklearn import preprocessing, metrics
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from renaanalysis.params.params import random_seed, batch_size, epochs, model_save_dir, patience
from renaanalysis.utils.data_utils import rebalance_classes


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
