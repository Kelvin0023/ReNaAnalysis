import os
import pickle
import time

import torch

import os
import time

import numpy as np

from renaanalysis.params.params import random_seed, export_data_root
from renaanalysis.utils.viz_utils import visualize_block_gaze_event
from RenaAnalysis import get_rdf

torch.manual_seed(random_seed)
np.random.seed(random_seed)

# start of the main block ######################################################

start_time = time.time()  # record the start time of the analysis

base_root = "D:/Dropbox/Dropbox/ReNa/data/RenaPipeline-2023Spring"
# rdf = get_rdf(base_root, ocular_artifact_mode='proxy')
rdf = pickle.load(open(os.path.join(export_data_root, 'rdf.p'), 'rb'))
pickle.dump(rdf, open(os.path.join(export_data_root, 'rdf.p'), 'wb'))  # dump to the SSD c drive
print(f"Saving/loading RDF complete, took {time.time() - start_time} seconds")
# discriminant test  ####################################################################################################

visualize_block_gaze_event(rdf, participant='0', session=0, block_id=7, generate_video=True, video_fix_alg='I-DT')
visualize_block_gaze_event(rdf, participant='0', session=0, block_id=7, generate_video=True, video_fix_alg='I-VT')
# visualize_block_gaze_event(rdf, participant='0', session=0, block_id=1, generate_video=True, video_fix_alg='I-DT')
# visualize_block_gaze_event(rdf, participant='0', session=0, block_id=1, generate_video=True, video_fix_alg='I-VT')
# visualize_block_gaze_event(rdf, participant='0', session=0, block_id=2, generate_video=True, video_fix_alg='I-DT')
# visualize_block_gaze_event(rdf, participant='0', session=0, block_id=2, generate_video=True, video_fix_alg='I-VT')
# visualize_block_gaze_event(rdf, participant='0', session=0, block_id=2, generate_video=True, video_fix_alg='Patch-Sim')

# visualize_block_gaze_event(rdf, participant='1', session=2, block_id=7, generate_video=True, video_fix_alg=None)
# visualize_block_gaze_event(rdf, participant='1', session=2, block_id=7, generate_video=True, video_fix_alg='I-VT')
# visualize_block_gaze_event(rdf, participant='1', session=2, block_id=7, generate_video=True, video_fix_alg='I-VT-Head')
# visualize_block_gaze_event(rdf, participant='1', session=2, block_id=7, generate_video=True, video_fix_alg='Patch-Sim')
#
# visualize_block_gaze_event(rdf, participant='1', session=2, block_id=1, generate_video=True, video_fix_alg=None)
# visualize_block_gaze_event(rdf, participant='1', session=2, block_id=1, generate_video=True, video_fix_alg='I-VT')
# visualize_block_gaze_event(rdf, participant='1', session=2, block_id=1, generate_video=True, video_fix_alg='I-VT-Head')
# visualize_block_gaze_event(rdf, participant='1', session=2, block_id=1, generate_video=True, video_fix_alg='Patch-Sim')
#
# visualize_block_gaze_event(rdf, participant='1', session=2, block_id=2, generate_video=True, video_fix_alg=None)
# visualize_block_gaze_event(rdf, participant='1', session=2, block_id=2, generate_video=True, video_fix_alg='I-VT')
# visualize_block_gaze_event(rdf, participant='1', session=2, block_id=2, generate_video=True, video_fix_alg='I-VT-Head')
# visualize_block_gaze_event(rdf, participant='1', session=2, block_id=2, generate_video=True, video_fix_alg='Patch-Sim')

