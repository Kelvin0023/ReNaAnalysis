import re

import mne


def is_standard_10_20_name(channel_name):
    return channel_name in mne.channels.make_standard_montage('standard_1020').ch_names