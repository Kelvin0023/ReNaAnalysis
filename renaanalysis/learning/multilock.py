import numpy as np

from renaanalysis.eye.eyetracking import GazeRayIntersect
from renaanalysis.params.params import conditions, dtnn_types
from renaanalysis.utils.RenaDataFrame import RenaDataFrame
from renaanalysis.utils.data_utils import epochs_to_class_samples_rdf


def eval_multi_locking_model(rdf, event_names, regenerate_epochs=True, exg_resample_rate=200):
    # locking_name_filters = [
    #     lambda x: type(x) == GazeRayIntersect and x.block_condition == conditions['VS'] and x.dtn == dtnn_types[ "Distractor"],
    #     lambda x: type(x) == GazeRayIntersect and x.block_condition == conditions['VS'] and x.dtn == dtnn_types[ "Target"]]

    ps_vs_block_ids = rdf.get_condition_block_ids(conditions['VS'])
    samples = []
    labels = []
    for (p, s), block_ids in ps_vs_block_ids.items():
        for block_i, block_id in enumerate(block_ids):
            print(f"Processing block {block_id} ({block_i + 1} of {len(block_ids)}) for participant {p} session {s}", flush=True)
            item_ids_dtn_in_this_block = rdf.get_block_item_ids(block_id, participant=p, session=s, return_dtn_type=True)[p, s]
            item_ids_in_this_block = item_ids_dtn_in_this_block[:, 0]
            item_id_dtn_dict = dict(item_ids_dtn_in_this_block)
            event_filters = [(lambda x, i=i_id: type(x) == GazeRayIntersect and x.block_id == block_id and x.item_id == i) for i_id in item_ids_in_this_block]
            item_samples, y, item_epochs, event_ids = epochs_to_class_samples_rdf(rdf, [str(x) for x in item_ids_in_this_block], event_filters, participant=p, session=s, data_type='both', reject=None)

            for item_id in item_ids_in_this_block:
                same_item_samples = [x[(y + 1) == event_ids[str(item_id)]] for x in item_samples]
                samples.append(same_item_samples)
                labels.append(item_id_dtn_dict[item_id])
    print()
