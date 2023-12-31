import os
import pickle

import numpy as np
import torch
from sklearn.model_selection import ParameterGrid

from renaanalysis.learning.HT import HierarchicalTransformer
from renaanalysis.learning.train import self_supervised_pretrain
from renaanalysis.multimodal.multimodal import MultiModalArrays
from renaanalysis.multimodal.train_multimodal import train_test_classifier_multimodal, \
    self_supervised_pretrain_multimodal
from renaanalysis.params.params import TaskName, eeg_name, model_save_dir
from renaanalysis.utils.data_utils import mean_min_sublists, mean_max_sublists


def get_grid_search_test_name(grid_search_params):
    """
    only search parameter with more than 1 value will be included in the test name
    """
    test_name = 'GridSearch_'
    for key, value in grid_search_params.items():
        if len(value) > 1:
            test_name += f"{key}={value}_"
    return test_name

# def grid_search_eeg(grid_search_params, mmarray: MultiModalArrays, model_class, n_folds: int,
#                        test_size=0.1,
#                        is_pca_ica=False,
#                        task_name=TaskName.PreTrain, is_plot_confusion_matrix=False, random_seed=None):
#     """
#
#     @param grid_search_params:
#     @param x_eeg:
#     @param x_eeg_pca_ica:
#     @param y:
#     @param event_names:
#     @param n_folds:
#     @param test_name:
#     @param task_name:
#     @param eeg_fs:
#     @param is_plot_confusion_matrix:
#     @param is_eeg_rebalance_by_channel:
#     @param is_plot_rebalanced_eeg: if true,
#     @return:
#     """
#     assert eeg_name in mmarray.keys(), f"grid_search_ht_eeg: {eeg_name} is not in x {mmarray.keys()} , please check the input dataset has EEG data"
#     test_name = get_grid_search_test_name(grid_search_params)
#
#     mmarray.train_test_split(test_size=test_size, random_seed=random_seed)
#     eeg_num_channels, eeg_num_timesteps = mmarray['eeg'].get_pca_ica_array().shape[1:] if is_pca_ica else mmarray['eeg'].array.shape[1:]
#     eeg_fs = mmarray['eeg'].sampling_rate
#     param_grid = ParameterGrid(grid_search_params)
#
#     total_training_histories = {}
#     models_param = {}
#     locking_performance = {}
#
#     for params in param_grid:
#         print(f"Grid search params: {params}. Searching {len(total_training_histories) + 1} of {len(param_grid)}")
#         if task_name == TaskName.TrainClassifier or task_name == TaskName.PretrainedClassifierFineTune:
#             models, training_histories, criterion, _, test_auc, test_loss, test_acc = train_test_classifier_multimodal(
#                                                                                             mmarray, model_class, test_name, task_name=task_name, n_folds=n_folds,
#                                                                                             is_plot_conf_matrix=is_plot_confusion_matrix,
#                                                                                              verbose=1, lr=params['lr'], l2_weight=params['l2_weight'], random_seed=random_seed)
#
#         elif task_name == TaskName.PreTrain:
#             models, training_histories, criterion, last_activation, _encoder = self_supervised_pretrain(x_eeg_pca_ica_train if is_pca_ica else x_eeg_train, model_class, temperature=params['temperature'], n_neg=params['n_neg'], is_plot_conf_matrix=is_plot_confusion_matrix, X_test=x_eeg_pca_ica_test if is_pca_ica else x_eeg_test, n_folds=n_folds, test_name=test_name, task_name=task_name, verbose=1, lr=params['lr'], l2_weight=params['l2_weight'])  # use un-dimension reduced EEG data
#         if task_name == TaskName.PreTrain:
#             folds_train_loss, folds_val_loss =  mean_min_sublists(training_histories['loss_train']), mean_min_sublists(training_histories['loss_val'])
#             print(f'{test_name} with param {params}: folds val loss: {folds_val_loss}, folds train loss: {folds_train_loss} ')
#
#             hashable_params = tuple(params.items())
#             locking_performance[hashable_params] = {'folds val loss': folds_val_loss,
#                                                     'folds trian loss': folds_train_loss}
#             total_training_histories[hashable_params] = training_histories
#             models_param[hashable_params] = models
#             if not os.path.exists('HT_grid_pretrain'):
#                 os.mkdir('HT_grid_pretrain')
#             for i in range(n_folds):
#                 torch.save(models[i], os.path.join(model_save_dir, test_name + f"_lr_{params['lr']}_dimhead_{params['dim_head']}_feeddim_{params['feedforward_mlp_dim']}_numheads_{params['num_heads']}_patchdim_{params['patch_embed_dim']}_fold_{i}_pca_{is_pca_ica}.pt"))
#         else:
#             folds_train_acc, folds_val_acc, folds_train_loss, folds_val_loss = mean_max_sublists(training_histories['acc_train']), mean_max_sublists(training_histories['acc_val']), mean_min_sublists(training_histories['loss_train']), mean_min_sublists(training_histories['loss_val'])
#             folds_val_auc = mean_max_sublists(training_histories['auc_val'])
#             print(f'{test_name} with param {params}: folds val AUC {folds_val_auc}, folds val accuracy: {folds_val_acc}, folds train accuracy: {folds_train_acc}, folds val loss: {folds_val_loss}, folds train loss: {folds_train_loss}, ')
#
#             hashable_params = tuple(params.items())
#             locking_performance[hashable_params] = {'folds val auc': folds_val_auc, 'folds val acc': folds_val_acc, 'folds train acc': folds_train_acc, 'folds val loss': folds_val_loss,'folds trian loss': folds_train_loss, 'folds test auc': training_histories['auc_test']}
#             total_training_histories[hashable_params] = training_histories
#             models_param[hashable_params] = models
#             if not os.path.exists('HT_grid'):
#                 os.mkdir('HT_grid')
#             for i in range(n_folds):
#                 torch.save(models[i], f"HT_grid/lr_{params['lr']}_dimhead_{params['dim_head']}_feeddim_{params['feedforward_mlp_dim']}_numheads_{params['num_heads']}_patchdim_{params['patch_embed_dim']}_fold_{i}_pca_{is_pca_ica}.pt")
#     return locking_performance, total_training_histories, models_param

def grid_search_ht_eeg(grid_search_params, mmarray: MultiModalArrays, n_folds: int, model_class=HierarchicalTransformer,
                        training_results_path="",
                       physio_type=eeg_name,
                       is_pca_ica=False,
                       task_name=TaskName.PreTrain, is_plot_confusion_matrix=False, random_seed=None,
                       picks_sbj_run=None, *args, **kwargs):
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
    assert os.path.exists(training_results_path), f"grid_search_ht_eeg: {training_results_path} does not exist"
    model_check_points_dir = os.path.join(training_results_path, 'model_check_points')
    if not os.path.exists(model_check_points_dir):
        os.makedirs(model_check_points_dir, exist_ok=True)
    assert physio_type in mmarray.keys(), f"grid_search_ht: {physio_type} is not in x {mmarray.keys()} , please check the input dataset has {physio_type} data"
    test_name = get_grid_search_test_name(grid_search_params)
    num_classes = len(np.unique(mmarray.labels_array))

    # get meta information about the training data
    num_channels, num_timesteps = mmarray[physio_type].get_pca_ica_array().shape[1:] if is_pca_ica else mmarray[physio_type].array.shape[1:]
    fs = mmarray[physio_type].sampling_rate

    param_grid = ParameterGrid(grid_search_params)  # parameter grid for grid search

    # variables to save the results
    total_training_histories = {}
    models_param = {}
    param_performance = {}

    # decide the training function based on the task name
    if task_name == TaskName.TrainClassifier:
        train_func = train_test_classifier_multimodal
    elif task_name == TaskName.PreTrain:  # Pretrain does not use ordered batch at the moment
        train_func = self_supervised_pretrain_multimodal
    elif task_name == TaskName.PretrainedClassifierFineTune:
        raise NotImplementedError("PretrainedClassifierFineTune is not implemented yet")

    train_params = {**{
        'mmarray': mmarray,
        'test_name': test_name,
        'task_name': task_name,
        'n_folds': n_folds,
        'is_plot_conf_matrix': is_plot_confusion_matrix,
        'random_seed': random_seed,
        'picks_sbj_run': picks_sbj_run
    }, **kwargs}

    # create split for mmarray
    mmarray.create_split(**train_params)
    mmarray.save_to_path(os.path.join(training_results_path, "mmarray.p"))  # save the mmarray with the split

    for model_params in param_grid:
        print(f"Grid search params: {model_params}. Searching {len(total_training_histories) + 1} of {len(param_grid)}")
        train_params['model'] = model_class(num_timesteps, num_channels, fs, num_classes=num_classes, physio_type=physio_type, **model_params)
        results = train_func(**train_params, **model_params)

        folds_metrics = {}
        if task_name == TaskName.TrainClassifier:
            models, training_histories, criterion, _, test_auc, test_loss, test_acc = results
            folds_metrics['folds_train_acc'], folds_metrics['folds_val_acc'], folds_metrics['folds_val_auc'] = mean_max_sublists(training_histories['acc_train']), mean_max_sublists(training_histories['acc_val']), mean_max_sublists(training_histories['auc_val'])
            folds_metrics['folds_test_acc'], folds_metrics['folds_test_auc'], folds_metrics['folds_test_loss'] = test_acc, test_auc, test_loss
        elif task_name == TaskName.PreTrain:
            models, training_histories, criterion, last_activation = results
        folds_metrics['folds_train_loss'], folds_metrics['folds_val_loss'] = mean_min_sublists(training_histories['loss_train']), mean_min_sublists(training_histories['loss_val'])

        hashable_params = tuple(model_params.items())
        total_training_histories[hashable_params] = training_histories
        models_param[hashable_params] = models

        print(f"{test_name} with param {model_params}: {[ f'{key}: {value}' for key, value in folds_metrics.items()]}")
        param_performance[hashable_params] = folds_metrics

        for i in range(n_folds):
            torch.save(models[i], os.path.join(model_save_dir, test_name + f"_lr_{model_params['lr']}_dimhead_{model_params['dim_head']}_feeddim_{model_params['feedforward_mlp_dim']}_numheads_{model_params['num_heads']}_patchdim_{model_params['patch_embed_dim']}_fold_{i}_pca_{is_pca_ica}.pt"))
        pickle.dump(total_training_histories, open(os.path.join(training_results_path, f'training_histories.p'), 'wb'))
        pickle.dump(param_performance, open(os.path.join(training_results_path, f'model_performances.p'), 'wb'))
        pickle.dump(models_param, open(os.path.join(training_results_path, f'models_with_params.p'), 'wb'))
    return param_performance, total_training_histories, models_param


# def grid_search_ht_fnirs(grid_search_params, mmarray: MultiModalArrays, n_folds: int,
#                        test_size=0.1,
#                        is_pca_ica=False,
#                        task_name=TaskName.PreTrain, is_plot_confusion_matrix=False, random_seed=None):
#     """
#
#     @param grid_search_params:
#     @param x_eeg:
#     @param x_eeg_pca_ica:
#     @param y:
#     @param event_names:
#     @param n_folds:
#     @param test_name:
#     @param task_name:
#     @param fnirs_fs:
#     @param is_plot_confusion_matrix:
#     @param is_eeg_rebalance_by_channel:
#     @param is_plot_rebalanced_eeg: if true,
#     @return:
#     """
#     assert fnirs_name in mmarray.keys(), f"grid_search_ht_fnirs: {fnirs_name} is not in x {mmarray.keys()} , please check the input dataset has EEG data"
#     test_name = get_grid_search_test_name(grid_search_params)
#
#     fnirs_num_channels, fnirs_num_timesteps = mmarray['fnirs'].get_pca_ica_array().shape[1:] if is_pca_ica else mmarray['fnirs'].array.shape[1:]
#     fnirs_fs = mmarray['fnirs'].sampling_rate
#     param_grid = ParameterGrid(grid_search_params)
#
#     total_training_histories = {}
#     models_param = {}
#     locking_performance = {}
#
#     for params in param_grid:
#         print(f"Grid search params: {params}. Searching {len(total_training_histories) + 1} of {len(param_grid)}")
#         if task_name == TaskName.TrainClassifier:
#             model = HierarchicalTransformer(fnirs_num_timesteps, fnirs_num_channels, fnirs_fs, num_classes=2,
#                                             depth=params['depth'], num_heads=params['num_heads'], feedforward_mlp_dim=params['feedforward_mlp_dim'],
#                                             pool=params['pool'], patch_embed_dim=params['patch_embed_dim'], pos_embed_mode=params['pos_embed_mode'],
#                                             dim_head=params['dim_head'], emb_dropout=params['emb_dropout'], attn_dropout=params['attn_dropout'], output=params['output'])
#             models, training_histories, criterion, _, test_auc, test_loss, test_acc = train_test_classifier_multimodal(
#                                                                                             mmarray, model, test_name, task_name=task_name, n_folds=n_folds,
#                                                                                             is_plot_conf_matrix=is_plot_confusion_matrix, test_size=test_size,
#                                                                                              verbose=1, lr=params['lr'], l2_weight=params['l2_weight'], random_seed=random_seed)
#
#         elif task_name == TaskName.PreTrain:
#             model = HierarchicalTransformerAutoEncoderPretrain(fnirs_num_timesteps, fnirs_num_channels, fnirs_fs, num_classes=2,
#                                                                depth=params['depth'], num_heads=params['num_heads'], feedforward_mlp_dim=params['feedforward_mlp_dim'],
#                                                                pool=params['pool'], patch_embed_dim=params['patch_embed_dim'], pos_embed_mode=params['pos_embed_mode'],
#                                                                dim_head=params['dim_head'], emb_dropout=params['emb_dropout'], attn_dropout=params['attn_dropout'], output=params['output'],
#                                                                p_t=params['p_t'], p_c=params['p_c'], mask_t_span=params['mask_t_span'], mask_c_span=params['mask_c_span'])
#             models, training_histories, criterion, last_activation = self_supervised_pretrain_multimodal(mmarray, model, temperature=params['temperature'], n_neg=params['n_neg'], is_plot_conf_matrix=is_plot_confusion_matrix, n_folds=n_folds, test_name=test_name, task_name=task_name, verbose=1, lr=params['lr'], l2_weight=params['l2_weight'], random_seed=random_seed)  # use un-dimension reduced EEG data
#         if task_name == TaskName.PreTrain:
#             folds_train_loss, folds_val_loss = mean_min_sublists(training_histories['loss_train']), mean_min_sublists(training_histories['loss_val'])
#             print(f'{test_name} with param {params}: folds val loss: {folds_val_loss}, folds train loss: {folds_train_loss} ')
#
#             hashable_params = tuple(params.items())
#             locking_performance[hashable_params] = {'folds val loss': folds_val_loss,
#                                                     'folds trian loss': folds_train_loss,
#                                                     'folds test loss': training_histories['loss_test']}
#             total_training_histories[hashable_params] = training_histories
#             models_param[hashable_params] = models
#             if not os.path.exists('HT_grid_pretrain'):
#                 os.mkdir('HT_grid_pretrain')
#             for i in range(n_folds):
#                 torch.save(models[i], os.path.join(model_save_dir, test_name + f"_lr_{params['lr']}_dimhead_{params['dim_head']}_feeddim_{params['feedforward_mlp_dim']}_numheads_{params['num_heads']}_patchdim_{params['patch_embed_dim']}_fold_{i}_pca_{is_pca_ica}.pt"))
#         else:
#             folds_train_acc, folds_val_acc, folds_train_loss, folds_val_loss = mean_max_sublists(training_histories['acc_train']), mean_max_sublists(training_histories['acc_val']), mean_min_sublists(training_histories['loss_train']), mean_min_sublists(training_histories['loss_val'])
#             folds_val_auc = mean_max_sublists(training_histories['auc_val'])
#             print(f'{test_name} with param {params}: folds val AUC {folds_val_auc}, folds val accuracy: {folds_val_acc}, folds train accuracy: {folds_train_acc}, folds val loss: {folds_val_loss}, folds train loss: {folds_train_loss}, ')
#
#             hashable_params = tuple(params.items())
#             locking_performance[hashable_params] = {'folds val auc': folds_val_auc, 'folds val acc': folds_val_acc, 'folds train acc': folds_train_acc, 'folds val loss': folds_val_loss,'folds train loss': folds_train_loss, 'folds test auc': training_histories['auc_test']}
#             total_training_histories[hashable_params] = training_histories
#             models_param[hashable_params] = models
#             if not os.path.exists('HT_grid'):
#                 os.mkdir('HT_grid')
#             for i in range(n_folds):
#                 torch.save(models[i], f"HT_grid/lr_{params['lr']}_dimhead_{params['dim_head']}_feeddim_{params['feedforward_mlp_dim']}_numheads_{params['num_heads']}_patchdim_{params['patch_embed_dim']}_fold_{i}_pca_{is_pca_ica}.pt")
#     return locking_performance, total_training_histories, models_param




# def grid_search_rht_eeg(grid_search_params, mmarray: MultiModalArrays, n_folds: int, results_path, is_pca_ica=False,
#                        task_name=TaskName.PreTrain, is_plot_confusion_matrix=False, random_seed=None, val_size = 0.1, test_size=0.1, batch_size=16, epochs=5000, patience=30):
#     """
#
#     @param grid_search_params:
#     @param x_eeg:
#     @param x_eeg_pca_ica:
#     @param y:
#     @param event_names:
#     @param n_folds:
#     @param test_name:
#     @param task_name:
#     @param eeg_fs:
#     @param is_plot_confusion_matrix:
#     @param is_eeg_rebalance_by_channel:
#     @param is_plot_rebalanced_eeg: if true,
#     @return:
#     """
#     assert eeg_name in mmarray.keys(), f"grid_search_rht_eeg: {eeg_name} is not in x {mmarray.keys()} , please check the input dataset has EEG data"
#     test_name = get_grid_search_test_name(grid_search_params)
#
#     # mmarray.train_test_split(test_size=test_size, random_seed=random_seed)  # in ordered split, the test set is generated along with the val set
#     eeg_num_channels, eeg_num_timesteps = mmarray['eeg'].get_pca_ica_array().shape[1:] if is_pca_ica else mmarray['eeg'].array.shape[1:]
#     eeg_fs = mmarray['eeg'].sampling_rate
#     param_grid = ParameterGrid(grid_search_params)
#
#     param_training_histories = {}
#     models_param = {}
#     param_performance = {}
#
#     mmarray.training_val_test_split_ordered_by_subject_run(n_folds, batch_size=batch_size, val_size=val_size, test_size=test_size, random_seed=random_seed)
#     mmarray.train_test_split(test_size=0.1, random_seed=random_seed)
#     mmarray.training_val_split(n_folds, val_size=0.1, random_seed=random_seed)
#
#     model_param_dict = {}
#     for param_index, params in enumerate(param_grid):
#         print(f"Grid search params: {params}. Searching {len(param_training_histories) + 1} of {len(param_grid)}")
#         if task_name == TaskName.TrainClassifier or task_name == TaskName.PretrainedClassifierFineTune:
#             model = HierarchicalTransformer(num_timesteps, num_channels, fs, num_classes=num_classes, physio_type=physio_type, **params)
#             model = RecurrentHierarchicalTransformer(eeg_num_timesteps, eeg_num_channels, eeg_fs, num_classes=2, **params)
#             best_model_folds, best_models_from_folds, training_histories, criterion, _, test_auc, test_loss, test_acc = train_test_classifier_multimodal_ordered_batches(
#                                                                                             mmarray, model, test_name, task_name=task_name, n_folds=n_folds,
#                                                                                             is_plot_conf_matrix=is_plot_confusion_matrix,
#                                                                                              verbose=1, lr=params['lr'], l2_weight=params['l2_weight'], random_seed=random_seed, epochs=epochs, patience=patience)
#             folds_train_acc, folds_val_acc, folds_train_loss, folds_val_loss = mean_max_sublists(training_histories['acc_train']), mean_max_sublists(training_histories['acc_val']), mean_min_sublists(training_histories['loss_train']), mean_min_sublists(training_histories['loss_val'])
#             folds_val_auc = mean_max_sublists(training_histories['auc_val'])
#             print(f'{test_name} with param {params}: folds val AUC {folds_val_auc}, folds val accuracy: {folds_val_acc}, folds train accuracy: {folds_train_acc}, folds val loss: {folds_val_loss}, folds train loss: {folds_train_loss}, ')
#
#             hashable_params = tuple(params.items())
#             param_performance[hashable_params] = {'folds val auc': folds_val_auc, 'folds val acc': folds_val_acc, 'folds train acc': folds_train_acc, 'folds val loss': folds_val_loss,'folds train loss': folds_train_loss, 'folds test auc': training_histories['auc_test']}
#             param_training_histories[hashable_params] = training_histories
#             models_param[hashable_params] = best_models_from_folds
#
#             model_param_dict[json.dumps(params)] = param_index, param_performance[hashable_params], training_histories, best_models_from_folds
#             torch.save(best_model_folds, os.path.join(results_path, f'{param_index}.pt'))
#             pickle.dump(model_param_dict, open(os.path.join(results_path, 'model_param_dict.p'), 'wb'))
#
#         elif task_name == TaskName.PreTrain:
#             model = RecurrentHierarchicalTransformerAutoEncoderPretrain(eeg_num_timesteps, eeg_num_channels, eeg_fs, num_classes=2,
#                                                                depth=params['depth'], num_heads=params['num_heads'], feedforward_mlp_dim=params['feedforward_mlp_dim'],
#                                                                pool=params['pool'], patch_embed_dim=params['patch_embed_dim'],
#                                                                dim_head=params['dim_head'], emb_dropout=params['emb_dropout'], attn_dropout=params['attn_dropout'], output=params['output'],
#                                                                p_t=params['p_t'], p_c=params['p_c'])
#             models, training_histories, criterion, last_activation = self_supervised_pretrain_multimodal(mmarray, model, temperature=params['temperature'], n_neg=params['n_neg'], is_plot_conf_matrix=is_plot_confusion_matrix, n_folds=n_folds, test_name=test_name, task_name=task_name, verbose=1, lr=params['lr'], l2_weight=params['l2_weight'], random_seed=random_seed, use_ordered=True)
#             folds_train_loss, folds_val_loss =  mean_min_sublists(training_histories['loss_train']), mean_min_sublists(training_histories['loss_val'])
#             print(f'{test_name} with param {params}: folds val loss: {folds_val_loss}, folds train loss: {folds_train_loss} ')
#
#             hashable_params = tuple(params.items())
#             param_performance[hashable_params] = {'folds val loss': folds_val_loss,
#                                                     'folds trian loss': folds_train_loss,
#                                                     'folds test loss': training_histories['loss_test']}
#             param_training_histories[hashable_params] = training_histories
#             models_param[hashable_params] = models
#     # mmarray.save_to_path(os.path.join(results_path, 'mmarray.p'))
#     return param_performance, param_training_histories, models_param