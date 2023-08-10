import copy
import warnings

import numpy as np
import torch
from torch.optim import lr_scheduler
from torch.utils.data import TensorDataset, DataLoader

from renaanalysis.learning.HT import HierarchicalTransformerContrastivePretrain, SimularityLoss, ContrastiveLoss, \
    ReconstructionLoss, HierarchicalTransformerAutoEncoderPretrain
from renaanalysis.learning.HATC import HierarchicalAutoTranscoderPretrain
from renaanalysis.learning.train import _run_one_epoch_classification, eval_test, _run_one_epoch_self_sup, \
    _run_one_epoch_classification_augmented
from renaanalysis.params.params import batch_size, epochs, patience, TaskName
from renaanalysis.utils.viz_utils import viz_confusion_matrix, plot_training_history


def train_test_classifier_multimodal(mmarray, model, test_name="", task_name=TaskName.TrainClassifier,
                                     n_folds=10, lr=1e-4, verbose=1, l2_weight=1e-6, val_size=0.1, test_size=0.1,
                                     lr_scheduler_type='exponential', is_plot_conf_matrix=False, plot_histories=True, random_seed=None, epochs=5000, patience=30, batch_size=16,
                                     use_ordered=False):
    """

    """
    if random_seed is None:
        warnings.warn("train_test_classifier_multimodal: random_seed is None, which means the results are not reproducible.")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    criterion, last_activation = mmarray.get_label_encoder_criterion_for_model(model, device)


    # X = model.prepare_data(X)

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

    if use_ordered:
        mmarray.training_val_test_split_ordered_by_subject_run(n_folds, batch_size=batch_size, val_size=val_size, test_size=0.1, random_seed=random_seed)
        test_dataloader = mmarray.get_test_ordered_batch_iterator(device=device, shuffle_within_batches=True)
        train_val_func = mmarray.get_train_val_ordered_batch_iterator_fold
    else:
        mmarray.test_train_val_split(n_folds, test_size=test_size, val_size=val_size, random_seed=random_seed)
        test_dataloader = mmarray.get_test_dataloader(batch_size=batch_size, device=device)
        train_val_func = mmarray.get_dataloader_fold

    for f_index in range(n_folds):
        model_copy = copy.deepcopy(model)
        model_copy = model_copy.to(device)

        train_dataloader, val_dataloader = train_val_func(f_index, batch_size=batch_size, is_rebalance_training=True, random_seed=random_seed, device=device, shuffle_within_batches=True)
        # train_dataloader, val_dataloader = mmarray.get_train_val_ordered_batch_iterator_fold(f_index, device=device, return_metainfo=True, shuffle_within_batches=True)

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
            train_auc, train_loss, train_accuracy, num_train_standard_error, num_train_target_error, train_y_all, train_y_all_pred = _run_one_epoch_classification(model_copy, train_dataloader, criterion, last_activation, optimizer, rebalance_method=mmarray.rebalance_method, mode='train', device=device, l2_weight=l2_weight, test_name=test_name, task_name=task_name, verbose=verbose)
            if is_plot_conf_matrix:
                train_predicted_labels_all = np.argmax(train_y_all_pred, axis=1)
                train_true_label_all = np.argmax(train_y_all, axis=1)
                num_train_standard_errors.append(num_train_standard_error)
                num_train_target_errors.append(num_train_target_error)
                viz_confusion_matrix(train_true_label_all, train_predicted_labels_all, epoch, f_index, 'train')
            scheduler.step()
            # ht_viz_training(X, Y, model_copy, rollout, _encoder, device, epoch)
            val_auc, val_loss, val_accuracy, num_val_standard_error, num_val_target_error, val_y_all, val_y_all_pred = _run_one_epoch_classification(model_copy, val_dataloader, criterion, last_activation, optimizer, rebalance_method=mmarray.rebalance_method, mode='val', device=device, l2_weight=l2_weight, test_name=test_name, task_name=task_name, verbose=verbose)
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
        # test_auc_model, test_loss_model, test_acc_model, num_test_standard_error, num_test_target_error, test_y_all, test_y_all_pred = eval_test(best_model, X_test, Y_test, criterion, last_activation,
        #                                  _encoder=mmarray.get_encoder_function(), task_name=task_name, verbose=1)

        test_auc_model, test_loss_model, test_acc_model, num_test_standard_error, num_test_target_error, test_y_all, test_y_all_pred =\
            _run_one_epoch_classification(best_model, test_dataloader, criterion, last_activation, rebalance_method=mmarray.rebalance_method, optimizer=None, mode='val', device=device, task_name=task_name, verbose=verbose)

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


    return models, training_histories_folds, criterion, last_activation, test_auc, test_loss, test_acc

def self_supervised_pretrain_multimodal(mmarray, model, test_name="", task_name=TaskName.PreTrain, n_folds=10, lr=1e-4, verbose=1, l2_weight=1e-6,
                            lr_scheduler_type='exponential', temperature=1, n_neg=20, test_size=0.1, val_size=0.1, is_plot_conf_matrix=False,
                            plot_histories=True, random_seed=None):

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
    if random_seed is None:
        warnings.warn("self_supervised_pretrain_multimodal: random_seed is None, which means the results are not reproducible.")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    assert isinstance(model, HierarchicalTransformerContrastivePretrain) or isinstance(model, HierarchicalAutoTranscoderPretrain) or isinstance(model, HierarchicalTransformerAutoEncoderPretrain), "self_supervised_pretrain_multimodal: model must be a HierarchicalTransformerContrastivePretrain instance"
    # criterion = ReconstructionLoss()
    criterion = ContrastiveLoss(temperature, n_neg)
    mmarray.train_test_split(test_size=test_size, random_seed=random_seed)
    # X_test, _ = mmarray.get_test_set()
    test_dataloader = mmarray.get_test_dataloader(batch_size=batch_size, encode_y=True, return_metainfo=True,
                                                  device=device)

    last_activation = None

    train_losses_folds = []
    val_losses_folds = []
    models = []

    model_copy = None
    test_loss = []
    mmarray.training_val_split(n_folds, val_size=val_size, random_seed=random_seed)
    for f_index in range(n_folds):
        model_copy = copy.deepcopy(model)
        model_copy = model_copy.to(device)

        train_dataloader, val_dataloader = mmarray.get_dataloader_fold(f_index, batch_size=batch_size, is_rebalance_training=False, random_seed=random_seed, return_metainfo=True, device=device, task_name=task_name)

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
        test_batch_losses, test_mean_loss = _run_one_epoch_self_sup(best_model, test_dataloader, criterion, optimizer=optimizer,
                                      mode='val',
                                      device=device, task_name=task_name, verbose=verbose)

        if verbose >= 1:
            print("Tested Fold {}: test mean loss = {:.8f}".format(f_index, test_mean_loss))
        test_loss.append(test_mean_loss)
        models.append(best_model)

    training_histories_folds = {'loss_train': train_losses_folds, 'loss_val': val_losses_folds, 'loss_test': test_loss}


    return models, training_histories_folds, criterion, last_activation

def train_test_classifier_multimodal_ordered_batches(mmarray, model, test_name="", task_name=TaskName.TrainClassifier,
                                     n_folds=10, lr=1e-4, verbose=1, l2_weight=1e-6,
                                     lr_scheduler_type='exponential', is_plot_conf_matrix=False, plot_histories=True, random_seed=None,
                                                     epochs=5000, patience=30, batch_size=16):
    """

    """
    if random_seed is None:
        warnings.warn("train_test_classifier_multimodal: random_seed is None, which means the results are not reproducible.")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    criterion, last_activation = mmarray.get_label_encoder_criterion_for_model(model, device)  # reset the memory of the recurrent model
    model.reset()  # reset model memories

    train_losses_folds = []
    train_accs_folds = []
    val_losses_folds = []
    val_accs_folds = []
    val_aucs_folds = []
    best_models_from_folds = []

    model_copy = None
    best_model_from_training = None
    best_model_folds = None
    best_test_auc = - np.inf
    test_auc_folds = []
    test_acc_folds = []
    test_loss_folds = []
    # test_iterator = mmarray.get_test_dataloader(batch_size=batch_size, encode_y=True, return_metainfo=True, device=device)
    test_iterator = mmarray.get_test_ordered_batch_iterator(device=device)

    for f_index in range(n_folds):
        model_copy = copy.deepcopy(model)
        model_copy = model_copy.to(device)
        # train_iterator, val_iterator = mmarray.get_dataloader_fold(f_index, batch_size=batch_size, is_rebalance_training=False, random_seed=random_seed, device=device, task_name=task_name, return_metainfo=True)
        train_iterator, val_iterator = mmarray.get_train_val_ordered_batch_iterator_fold(f_index, device=device, return_metainfo=True)

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
            train_auc, train_loss, train_accuracy, num_train_standard_error, num_train_target_error, train_y_all, train_y_all_pred = _run_one_epoch_classification(model_copy, train_iterator, criterion, last_activation, optimizer, rebalance_method=mmarray.rebalance_method, mode='train', device=device, l2_weight=l2_weight, test_name=test_name, task_name=task_name, verbose=verbose)
            model_copy.reset()
            if is_plot_conf_matrix:
                train_predicted_labels_all = np.argmax(train_y_all_pred, axis=1)
                train_true_label_all = np.argmax(train_y_all, axis=1)
                num_train_standard_errors.append(num_train_standard_error)
                num_train_target_errors.append(num_train_target_error)
                viz_confusion_matrix(train_true_label_all, train_predicted_labels_all, epoch, f_index, 'train')
            scheduler.step()
            # ht_viz_training(X, Y, model_copy, rollout, _encoder, device, epoch)
            val_auc, val_loss, val_accuracy, num_val_standard_error, num_val_target_error, val_y_all, val_y_all_pred = _run_one_epoch_classification(model_copy, val_iterator, criterion, last_activation, optimizer, rebalance_method=mmarray.rebalance_method, mode='val', device=device, l2_weight=l2_weight, test_name=test_name, task_name=task_name, verbose=verbose)
            model_copy.reset()

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
                best_model_from_training = copy.deepcopy(model_copy)
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

        test_auc, test_loss, test_acc, num_test_standard_error, num_test_target_error, test_y_all, test_y_all_pred =\
            _run_one_epoch_classification(best_model_from_training, test_iterator, criterion, last_activation, rebalance_method=mmarray.rebalance_method, optimizer=None, mode='val', device=device, task_name=task_name, verbose=verbose)

        if verbose >= 1:
            print("Tested Fold {}: test auc = {:.8f}, test loss = {:.8f}, test acc = {:.8f}".format(f_index, test_auc, test_loss, test_acc))
        test_auc_folds.append(test_auc)
        test_loss_folds.append(test_loss)
        test_acc_folds.append(test_acc)
        best_models_from_folds.append(best_model_from_training)
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            best_model_folds = copy.deepcopy(best_model_from_training)

    training_histories_folds = {'loss_train': train_losses_folds, 'acc_train': train_accs_folds, 'loss_val': val_losses_folds, 'acc_val': val_accs_folds, 'auc_val': val_aucs_folds, 'auc_test': test_auc_folds, 'acc_test': test_acc_folds, 'loss_test': test_loss_folds}
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

    return best_model_folds, best_models_from_folds, training_histories_folds, criterion, last_activation, test_auc_folds, test_loss_folds, test_acc_folds

def train_test_augmented(mmarray, model, test_name="", task_name=TaskName.TrainClassifier,
                                     n_folds=10, lr=1e-4, verbose=1, l2_weight=1e-6, val_size=0.1,
                                     lr_scheduler_type='exponential', is_plot_conf_matrix=False, plot_histories=True, random_seed=None):
    """

    """
    if random_seed is None:
        warnings.warn("train_test_classifier_multimodal: random_seed is None, which means the results are not reproducible.")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    criterion, last_activation = mmarray.get_label_encoder_criterion_for_model(model, device)
    # X_test, Y_test = mmarray.get_test_set()
    # X = model.prepare_data(X)

    train_losses_folds = []
    train_accs_folds = []
    val_losses_folds = []
    val_accs_folds = []
    models = []

    model_copy = None
    train_indices = []
    val_indices = []
    for i in range(n_folds):
        subject_indices = np.where(mmarray['eeg'].meta_info['subject_id'] == i+1)[0]

        train_indices.append(subject_indices[subject_indices < 2592])
        val_indices.append(subject_indices[subject_indices >= 2592])
        # train_indices.append()
    mmarray.set_training_val_set(train_indices=train_indices, val_indices=val_indices)
    for f_index in range(n_folds):
        model_copy = copy.deepcopy(model)
        model_copy = model_copy.to(device)
        # train_dataloader, val_dataloader = mmarray.get_dataloader_fold(f_index, batch_size=batch_size, is_rebalance_training=False, random_seed=random_seed, device=device)
        assert mmarray._encoder is not None, 'get_label_encoder_criterion_for_model must be called before get_rebalanced_dataloader_fold'
        training_indices, val_indices = mmarray.training_val_split_indices[f_index]
        x_train = []
        x_val = []
        y_train = mmarray.labels_array[training_indices]
        y_val = mmarray.labels_array[val_indices]

        labels = []
        for parray in mmarray.physio_arrays:
            this_x_train, this_y_train = parray[training_indices], y_train
            x_train.append(torch.Tensor(this_x_train).to(device))
            x_val.append(torch.Tensor(parray[val_indices]).to(device))

            labels.append(this_y_train)  # just for assertion

        assert np.all([label_set == labels[0] for label_set in labels])
        y_train = labels[0]
        y_train_encoded = mmarray._encoder(y_train)
        y_val_encoded = mmarray._encoder(y_val)
        y_train_encoded = torch.Tensor(y_train_encoded)
        y_val_encoded = torch.Tensor(y_val_encoded)
        train_dataset = TensorDataset(*x_train, y_train_encoded)
        val_dataset = TensorDataset(*x_val, y_val_encoded)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = torch.optim.Adam(model_copy.parameters(), lr=lr, betas=(0.5, 0.999))
        # optimizer = torch.optim.SGD(model_copy.parameters(), lr=lr, momentum=0.9)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        patience_counter = 0

        num_train_standard_errors = []
        num_train_target_errors = []
        num_val_standard_errors = []
        num_val_target_errors = []
        best_acc = 0
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        for epoch in range(epochs):
        # prev_para = []
        # for param in model_copy.parameters():
        #     prev_para.append(param.cpu().detach().numpy())
            train_loss, train_accuracy, num_train_standard_error, num_train_target_error, train_y_all, train_y_all_pred = _run_one_epoch_classification_augmented(model_copy, train_dataloader, mmarray.get_encoder_function(), criterion, last_activation, optimizer, mode='train', device=device, l2_weight=l2_weight, test_name=test_name, task_name=task_name, verbose=verbose)
            if is_plot_conf_matrix:
                train_predicted_labels_all = np.argmax(train_y_all_pred, axis=1)
                train_true_label_all = np.argmax(train_y_all, axis=1)
                num_train_standard_errors.append(num_train_standard_error)
                num_train_target_errors.append(num_train_target_error)
                viz_confusion_matrix(train_true_label_all, train_predicted_labels_all, epoch, f_index, 'train')
            scheduler.step()
            # ht_viz_training(X, Y, model_copy, rollout, _encoder, device, epoch)
            val_loss, val_accuracy, num_val_standard_error, num_val_target_error, val_y_all, val_y_all_pred = _run_one_epoch_classification_augmented(model_copy, val_dataloader, mmarray.get_encoder_function(), criterion, last_activation, optimizer, mode='val', device=device, l2_weight=l2_weight, test_name=test_name, task_name=task_name, verbose=verbose)
            if is_plot_conf_matrix:
                val_predicted_labels_all = np.argmax(val_y_all_pred, axis=1)
                val_true_label_all = np.argmax(val_y_all, axis=1)
                num_val_standard_errors.append(num_val_standard_error)
                num_val_target_errors.append(num_val_target_error)
                viz_confusion_matrix(val_true_label_all, val_predicted_labels_all, epoch, f_index, 'val')

            train_losses.append(train_loss)
            train_accs.append(train_accuracy)
            val_losses.append(val_loss)
            val_accs.append(val_accuracy)

            if verbose >= 1:
                print("Fold {}, Epoch {}: train accuracy = {:.16f}, train loss={:.16f}; val accuracy = {:.16f}, val loss={:.16f}, patience left {}".format(f_index, epoch, train_accs[-1], train_losses[-1], val_accs[-1],val_losses[-1], patience - patience_counter))
            if val_accuracy > best_acc:
                if verbose >= 1: print('Best validation auc improved from {} to {}'.format(best_acc, val_accuracy))
                # best_loss = val_losses[-1]
                best_acc = val_accuracy
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
        models.append(best_model)

    training_histories_folds = {'loss_train': train_losses_folds, 'acc_train': train_accs_folds, 'loss_val': val_losses_folds, 'acc_val': val_accs_folds}
    if plot_histories:
        for i in range(n_folds):
            history = {'loss_train': training_histories_folds['loss_train'][i],
                       'acc_train': training_histories_folds['acc_train'][i],
                       'loss_val': training_histories_folds['loss_val'][i],
                       'acc_val': training_histories_folds['acc_val'][i]}
            seached_params = None
            plot_training_history(history, seached_params, i, is_plot_auc=False)


    return models, training_histories_folds, criterion, last_activation