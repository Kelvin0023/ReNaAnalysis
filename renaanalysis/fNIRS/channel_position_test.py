import mne

#
# # Load an example montage (you can choose the one that fits your data)
# montage_name = 'standard_1005'  # For example, using the 10-20 electrode system
# montage = mne.channels.make_standard_montage(montage_name)
# montage.plot()
# # Get channel positions from the montage
# channel_positions = montage.get_positions()
#
# fnirs_channel_names = ['AFF5h', 'F5', 'F3', 'FFC5h', 'FC5', 'FC3', 'FCC5h', 'C5', 'C3', 'CCP5h', 'AFF6h', 'F4', 'F6',
#                        'FFC6h', 'FC4', 'FC6', 'FCC6h', 'C4', 'C6', 'CCP6h']
#
# # Print channel positions
# for channel, position in channel_positions.items():
#     print(f"Channel {channel}: Position {position}")


def create_montage(channel_names, standard_montage_name='standard_1005'):
    default_montage = mne.channels.make_standard_montage(standard_montage_name)
    default_montage_channel_positions = default_montage.get_positions()['ch_pos']
    new_channel_positions = {}
    for channel_name in channel_names:
        if channel_name in default_montage_channel_positions.keys():
            new_channel_positions[channel_name] = default_montage_channel_positions[channel_name]
        else:
            print(f"Channel {channel_name} not found in {standard_montage_name} montage.")
    new_montage = mne.channels.make_dig_montage(ch_pos=new_channel_positions, coord_frame='head')
    return new_montage



if __name__ == '__main__':
    fnirs_channel_names = ['AFF5h', 'F5', 'F3', 'FFC5h', 'FC5', 'FC3', 'FCC5h', 'C5', 'C3', 'CCP5h',
                           'AFF6h', 'F4','F6', 'FFC6h', 'FC4', 'FC6', 'FCC6h', 'C4', 'C6', 'CCP6h']
    new_montage = create_montage(fnirs_channel_names, standard_montage_name='standard_1005')
    new_montage.plot()

# ['standard_1005',
#  'standard_1020',
#  'standard_alphabetic',
#  'standard_postfixed',
#  'standard_prefixed',
#  'standard_primed',
#  'biosemi16',
#  'biosemi32',
#  'biosemi64',
#  'biosemi128',
#  'biosemi160',
#  'biosemi256',
#  'easycap-M1',
#  'easycap-M10',
#  'EGI_256',
#  'GSN-HydroCel-32',
#  'GSN-HydroCel-64_1.0',
#  'GSN-HydroCel-65_1.0',
#  'GSN-HydroCel-128',
#  'GSN-HydroCel-129',
#  'GSN-HydroCel-256',
#  'GSN-HydroCel-257',
#  'mgh60',
#  'mgh70',
#  'artinis-octamon',
#  'artinis-brite23',
#  'brainproducts-RNP-BA-128']
