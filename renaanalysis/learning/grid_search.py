import os

import torch
from sklearn.model_selection import ParameterGrid

from renaanalysis.learning.HT import HierarchicalTransformer, HierarchicalTransformerContrastivePretrain
from renaanalysis.learning.RHT import RecurrentHierarchicalTransformer
from renaanalysis.learning.train import cv_train_test_model, self_supervised_pretrain
from renaanalysis.multimodal.train_multimodal import train_test_classifier_multimodal, \
    train_test_classifier_multimodal_ordered_batches
from renaanalysis.params.params import TaskName, eeg_name, model_save_dir
from renaanalysis.utils.data_utils import mean_min_sublists, mean_max_sublists
from renaanalysis.multimodal.multimodal import MultiModalArrays

def get_grid_search_test_name(grid_search_params):
    """
    only search parameter with more than 1 value will be included in the test name
    """
    test_name = 'GridSearch_'
    for key, value in grid_search_params.items():
        if len(value) > 1:
            test_name += f"{key}={value}_"
    return test_name

def grid_search_ht_eeg(grid_search_params, mmarray: MultiModalArrays, n_folds: int,
                       test_size=0.1,
                       is_pca_ica=False,
                       task_name=TaskName.PreTrain, is_plot_confusion_matrix=False, random_seed=None):
    """

    @param grid_search_params:
    @param x_eeg:
    @param x_eeg_pca_ica:
    @param y:
    @param event_names:
    @param n_folds:
    @param test_name:
    @param task_name:
    @param eeg_fs:
    @param is_plot_confusion_matrix:
    @param is_eeg_rebalance_by_channel:
    @param is_plot_rebalanced_eeg: if true,
    @return:
    """
    assert eeg_name in mmarray.keys(), f"grid_search_ht_eeg: {eeg_name} is not in x {mmarray.keys()} , please check the input dataset has EEG data"
    test_name = get_grid_search_test_name(grid_search_params)

    # assert np.all(len(event_names) == np.array([len(x) for x in locking_name_filters.values()]))  # verify number of event types
    # for physio_array in mmarray.physio_arrays:
    #     if physio_array.physio_type == 'eeg':
    #         x_eeg = physio_array.array
    #     elif physio_array.physio_type == 'pupil':
    #         x_pupil = physio_array.array
    # x_eeg = mmarray

    # if task_name == TaskName.TrainClassifier or task_name == TaskName.PretrainedClassifierFineTune:
        # skf = StratifiedShuffleSplit(n_splits=1, random_state=random_seed)
        # train, test = [(train, test) for train, test in skf.split(x_eeg, y)][0]
        # for physio_array in mmarray.physio_arrays:
        #     if physio_array.physio_type == 'eeg':
        #         x_eeg_train = physio_array.array[train]
        #         x_eeg_test = physio_array.array[test]
        #     elif physio_array.physio_type == 'pupil':
        #         x_pupil_train = physio_array.array[train]
        #         x_pupil_test = physio_array.array[test]
        # y_train, y_test = mmarray.labels_array[train], mmarray.labels_array[test]
        # assert np.all(np.unique(y_test) == np.unique(y_train)), "train and test labels are not the same"
        # assert len(np.unique(y_test)) == len(event_names), "number of unique labels is not the same as number of event names"
        # mmarray.train_test_split(test_size=test_size, random_seed=random_seed)
    # elif task_name == TaskName.PreTrain:
        # x_eeg_train, x_eeg_test = train_test_split(x_eeg, test_size=0.1, random_state=random_seed)
        # x_eeg_pca_ica_train, x_eeg_pca_ica_test = train_test_split(x_eeg_pca_ica, test_size=0.1, random_state=random_seed)
        # if not reload_saved_samples:
        #     with open(os.path.join(export_data_root, 'x_eeg_pca_ica_test.p'), 'wb') as f:
        #         pickle.dump(x_eeg_pca_ica_test, f)
        #     with open(os.path.join(export_data_root, 'x_eeg_test.p'), 'wb') as f:
        #         pickle.dump(x_eeg_test, f)
    mmarray.train_test_split(test_size=test_size, random_seed=random_seed)
    eeg_num_channels, eeg_num_timesteps = mmarray['eeg'].get_pca_ica_array().shape[1:] if is_pca_ica else mmarray['eeg'].array.shape[1:]
    eeg_fs = mmarray['eeg'].sampling_rate
    param_grid = ParameterGrid(grid_search_params)

    total_training_histories = {}
    models_param = {}
    locking_performance = {}

    for params in param_grid:
        print(f"Grid search params: {params}. Searching {len(total_training_histories) + 1} of {len(param_grid)}")
        if task_name == TaskName.TrainClassifier or task_name == TaskName.PretrainedClassifierFineTune:
            model = HierarchicalTransformer(eeg_num_timesteps, eeg_num_channels, eeg_fs, num_classes=2,
                                            depth=params['depth'], num_heads=params['num_heads'], feedforward_mlp_dim=params['feedforward_mlp_dim'],
                                            pool=params['pool'], patch_embed_dim=params['patch_embed_dim'],
                                            dim_head=params['dim_head'], emb_dropout=params['emb_dropout'], attn_dropout=params['attn_dropout'], output=params['output'])
            models, training_histories, criterion, _, test_auc, test_loss, test_acc = train_test_classifier_multimodal(
                                                                                            mmarray, model, test_name, task_name=task_name, n_folds=n_folds,
                                                                                            is_plot_conf_matrix=is_plot_confusion_matrix,
                                                                                             verbose=1, lr=params['lr'], l2_weight=params['l2_weight'], random_seed=random_seed)

        elif task_name == TaskName.PreTrain:
            model = HierarchicalTransformerContrastivePretrain(eeg_num_timesteps, eeg_num_channels, eeg_fs, num_classes=2,
                                                               depth=params['depth'], num_heads=params['num_heads'], feedforward_mlp_dim=params['feedforward_mlp_dim'],
                                                               pool=params['pool'], patch_embed_dim=params['patch_embed_dim'],
                                                               dim_head=params['dim_head'], emb_dropout=params['emb_dropout'], attn_dropout=params['attn_dropout'], output=params['output'],
                                                               p_t=params['p_t'], p_c=params['p_c'], mask_t_span=params['mask_t_span'], mask_c_span=params['mask_c_span'])
            models, training_histories, criterion, last_activation, _encoder = self_supervised_pretrain(x_eeg_pca_ica_train if is_pca_ica else x_eeg_train, model, temperature=params['temperature'], n_neg=params['n_neg'], is_plot_conf_matrix=is_plot_confusion_matrix, X_test=x_eeg_pca_ica_test if is_pca_ica else x_eeg_test, n_folds=n_folds, test_name=test_name, task_name=task_name, verbose=1, lr=params['lr'], l2_weight=params['l2_weight'])  # use un-dimension reduced EEG data
        if task_name == TaskName.PreTrain:
            folds_train_loss, folds_val_loss =  mean_min_sublists(training_histories['loss_train']), mean_min_sublists(training_histories['loss_val'])
            print(f'{test_name} with param {params}: folds val loss: {folds_val_loss}, folds train loss: {folds_train_loss} ')

            hashable_params = tuple(params.items())
            locking_performance[hashable_params] = {'folds val loss': folds_val_loss,
                                                    'folds trian loss': folds_train_loss}
            total_training_histories[hashable_params] = training_histories
            models_param[hashable_params] = models
            if not os.path.exists('HT_grid_pretrain'):
                os.mkdir('HT_grid_pretrain')
            for i in range(n_folds):
                torch.save(models[i], os.path.join(model_save_dir, test_name + f"_lr_{params['lr']}_dimhead_{params['dim_head']}_feeddim_{params['feedforward_mlp_dim']}_numheads_{params['num_heads']}_patchdim_{params['patch_embed_dim']}_fold_{i}_pca_{is_pca_ica}.pt"))
        else:
            folds_train_acc, folds_val_acc, folds_train_loss, folds_val_loss = mean_max_sublists(training_histories['acc_train']), mean_max_sublists(training_histories['acc_val']), mean_min_sublists(training_histories['loss_train']), mean_min_sublists(training_histories['loss_val'])
            folds_val_auc = mean_max_sublists(training_histories['auc_val'])
            print(f'{test_name} with param {params}: folds val AUC {folds_val_auc}, folds val accuracy: {folds_val_acc}, folds train accuracy: {folds_train_acc}, folds val loss: {folds_val_loss}, folds train loss: {folds_train_loss}, ')

            hashable_params = tuple(params.items())
            locking_performance[hashable_params] = {'folds val auc': folds_val_auc, 'folds val acc': folds_val_acc, 'folds train acc': folds_train_acc, 'folds val loss': folds_val_loss,'folds trian loss': folds_train_loss, 'folds test auc': training_histories['auc_test']}
            total_training_histories[hashable_params] = training_histories
            models_param[hashable_params] = models
            if not os.path.exists('HT_grid'):
                os.mkdir('HT_grid')
            for i in range(n_folds):
                torch.save(models[i], f"HT_grid/lr_{params['lr']}_dimhead_{params['dim_head']}_feeddim_{params['feedforward_mlp_dim']}_numheads_{params['num_heads']}_patchdim_{params['patch_embed_dim']}_fold_{i}_pca_{is_pca_ica}.pt")
    return locking_performance, total_training_histories, models_param


def grid_search_rht_eeg(grid_search_params, mmarray: MultiModalArrays, n_folds: int, results_path, is_pca_ica=False,
                       task_name=TaskName.PreTrain, is_plot_confusion_matrix=False, random_seed=None):
    """

    @param grid_search_params:
    @param x_eeg:
    @param x_eeg_pca_ica:
    @param y:
    @param event_names:
    @param n_folds:
    @param test_name:
    @param task_name:
    @param eeg_fs:
    @param is_plot_confusion_matrix:
    @param is_eeg_rebalance_by_channel:
    @param is_plot_rebalanced_eeg: if true,
    @return:
    """
    assert eeg_name in mmarray.keys(), f"grid_search_rht_eeg: {eeg_name} is not in x {mmarray.keys()} , please check the input dataset has EEG data"
    test_name = get_grid_search_test_name(grid_search_params)

    # mmarray.train_test_split(test_size=test_size, random_seed=random_seed)  # in ordered split, the test set is generated along with the val set
    eeg_num_channels, eeg_num_timesteps = mmarray['eeg'].get_pca_ica_array().shape[1:] if is_pca_ica else mmarray['eeg'].array.shape[1:]
    eeg_fs = mmarray['eeg'].sampling_rate
    param_grid = ParameterGrid(grid_search_params)

    total_training_histories = {}
    models_param = {}
    param_performance = {}

    for params in param_grid:
        print(f"Grid search params: {params}. Searching {len(total_training_histories) + 1} of {len(param_grid)}")
        if task_name == TaskName.TrainClassifier or task_name == TaskName.PretrainedClassifierFineTune:
            model = RecurrentHierarchicalTransformer(eeg_num_timesteps, eeg_num_channels, eeg_fs, num_classes=2,
                                            depth=params['depth'], num_heads=params['num_heads'], feedforward_mlp_dim=params['feedforward_mlp_dim'],
                                            pool=params['pool'], patch_embed_dim=params['patch_embed_dim'],
                                            dim_head=params['dim_head'], emb_dropout=params['emb_dropout'], attn_dropout=params['attn_dropout'], output=params['output'])
            best_models_folds, training_histories, criterion, _, test_auc, test_loss, test_acc = train_test_classifier_multimodal_ordered_batches(
                                                                                            mmarray, model, test_name, task_name=task_name, n_folds=n_folds,
                                                                                            is_plot_conf_matrix=is_plot_confusion_matrix,
                                                                                             verbose=1, lr=params['lr'], l2_weight=params['l2_weight'], random_seed=random_seed)
            folds_train_acc, folds_val_acc, folds_train_loss, folds_val_loss = mean_max_sublists(training_histories['acc_train']), mean_max_sublists(training_histories['acc_val']), mean_min_sublists(training_histories['loss_train']), mean_min_sublists(training_histories['loss_val'])
            folds_val_auc = mean_max_sublists(training_histories['auc_val'])
            print(f'{test_name} with param {params}: folds val AUC {folds_val_auc}, folds val accuracy: {folds_val_acc}, folds train accuracy: {folds_train_acc}, folds val loss: {folds_val_loss}, folds train loss: {folds_train_loss}, ')

            hashable_params = tuple(params.items())
            param_performance[hashable_params] = {'folds val auc': folds_val_auc, 'folds val acc': folds_val_acc, 'folds train acc': folds_train_acc, 'folds val loss': folds_val_loss,'folds trian loss': folds_train_loss, 'folds test auc': training_histories['auc_test']}
            total_training_histories[hashable_params] = training_histories
            models_param[hashable_params] = best_models_folds

            for i in range(n_folds):
                model_name = 'rht' + '_'.join([f'{key}_{value}' for key, value in params.items()]) + f'_fold_{i}.pt'
                torch.save(best_models_folds[i], os.path.join(results_path, model_name))
        elif task_name == TaskName.PreTrain:
            raise NotImplementedError
            # model = HierarchicalTransformerContrastivePretrain(eeg_num_timesteps, eeg_num_channels, eeg_fs, num_classes=2,
            #                                                    depth=params['depth'], num_heads=params['num_heads'], feedforward_mlp_dim=params['feedforward_mlp_dim'],
            #                                                    pool=params['pool'], patch_embed_dim=params['patch_embed_dim'],
            #                                                    dim_head=params['dim_head'], emb_dropout=params['emb_dropout'], attn_dropout=params['attn_dropout'], output=params['output'],
            #                                                    p_t=params['p_t'], p_c=params['p_c'], mask_t_span=params['mask_t_span'], mask_c_span=params['mask_c_span'])
            # models, training_histories, criterion, last_activation, _encoder = self_supervised_pretrain(x_eeg_pca_ica_train if is_pca_ica else x_eeg_train, model, temperature=params['temperature'], n_neg=params['n_neg'], is_plot_conf_matrix=is_plot_confusion_matrix, X_test=x_eeg_pca_ica_test if is_pca_ica else x_eeg_test, n_folds=n_folds, test_name=test_name, task_name=task_name, verbose=1, lr=params['lr'], l2_weight=params['l2_weight'])  # use un-dimension reduced EEG data
            # folds_train_loss, folds_val_loss =  mean_min_sublists(training_histories['loss_train']), mean_min_sublists(training_histories['loss_val'])
            # print(f'{test_name} with param {params}: folds val loss: {folds_val_loss}, folds train loss: {folds_train_loss} ')
            #
            # hashable_params = tuple(params.items())
            # locking_performance[hashable_params] = {'folds val loss': folds_val_loss,
            #                                         'folds trian loss': folds_train_loss}
            # total_training_histories[hashable_params] = training_histories
            # models_param[hashable_params] = models
            # if not os.path.exists('HT_grid_pretrain'):
            #     os.mkdir('HT_grid_pretrain')
            # for i in range(n_folds):
            #     torch.save(models[i], os.path.join(model_save_dir, test_name + f"_lr_{params['lr']}_dimhead_{params['dim_head']}_feeddim_{params['feedforward_mlp_dim']}_numheads_{params['num_heads']}_patchdim_{params['patch_embed_dim']}_fold_{i}_pca_{is_pca_ica}.pt"))

    return param_performance, total_training_histories, models_param