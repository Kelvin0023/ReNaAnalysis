import pickle

from utils.cv_utils import generate_video
from utils.fs_utils import convert_dats_in_folder_to_p

data_path = "D:/Dropbox/Dropbox/ReNa/data/ReNaGame-2022Fall/11-18-2022/11_18_2022_20_52_00-Exp_RenaGameBeatSaber-Sbj_zl-Ssn_0-CountryRound-0.p"

data = pickle.load(open(data_path, 'rb'))

generate_video(data, video_stream_name='monitor1')