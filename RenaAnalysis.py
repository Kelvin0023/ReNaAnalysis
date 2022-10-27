def rena_analysis(is_data_preloaded, is_epochs_preloaded, is_save_loaded_data, is_regenerate_ica: bool,
                  preloaded_dats_path = 'data/participant_session_dict.p',
                  preloaded_epoch_path = 'data/participant_condition_epoch_dict.p',
                  preloaded_block_path = 'data/participant_condition_block_dict.p'):
    """

    @param is_data_preloaded:
    @param is_epochs_preloaded:
    @param is_save_loaded_data:
    @param is_regenerate_ica: whether to regenerate ica for the EEG data, if yes, the script calculates the ica components
    while processing the EEG data. The generated ica weights will be save to the data path, so when running the script
    the next time and if the EEG data is not changed, you can set this to false to skip recalculating ica to save time
    """
    pass

