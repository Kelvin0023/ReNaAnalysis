import numpy as np


class Event:
    def __init__(self, timestamp, *args, **kwargs):
        self.timestamp = timestamp

        self.meta_block= kwargs['meta_block'] if 'meta_block' in kwargs.keys() else None
        self.block_condition = kwargs['block_condition'] if 'block_condition' in kwargs.keys() else None
        self.block_id = kwargs['block_id'] if 'block_id' in kwargs.keys() else None

        self.dtn = kwargs['dtn'] if 'dtn' in kwargs.keys() else None
        self.dtn_onffset = kwargs['dtn_onffset'] if 'dtn_onffset' in kwargs.keys() else None

        self.item_id = kwargs['item_id'] if 'itemID' in kwargs.keys() else None
        self.obj_dist = kwargs['obj_dist'] if 'objDistFromPlayer' in kwargs.keys() else None
        self.carousel_speed = kwargs['carousel_speed'] if 'CarouselSpeed' in kwargs.keys() else None
        self.carousel_angle = kwargs['carousel_angle'] if 'CarouselAngle' in kwargs.keys() else None

        self.is_block_start = kwargs['is_block_start'] if 'is_block_start' in kwargs.keys() else None
        self.is_block_end = kwargs['is_block_end'] if 'is_block_end' in kwargs.keys() else None
        self.is_new_cp = kwargs['is_new_cp'] if 'is_new_cp' in kwargs.keys() else None

        self.gaze_intersect = kwargs['gaze_intersect'] if 'gaze_intersect' in kwargs.keys() else None

        self.likert = kwargs['Likert'] if 'Likert' in kwargs.keys() else None
        self.is_practice = kwargs['is_practice'] if 'is_practice' in kwargs.keys() else None


def add_event_meta_info(event, events):
    """
    return the meta event info from events for a specific event, including its condition, is or is not practice and block ID
    :param event:
    :param events:
    :return:
    """
    event.condition = get_closest_event(events, event.timestamp, 'block_condition',
                                  event_filter=lambda e: e.is_block_start)  # must be a block start event
    event.is_practice = get_closest_event(events, event.timestamp, 'is_practice',
                                    event_filter=lambda e: e.is_block_start)  # must be a block start event
    event.block_id = get_closest_event(events, event.timestamp, 'block_id',
                                 event_filter=lambda e: e.is_block_start)  # must be a block start event
    return event


def get_closest_event(events, timestamp, attribute, event_filter: callable):
    filter_events = np.array([e for e in events if event_filter(e)])
    events_timestamps = np.array([e.timestamp for e in filter_events])

    closest_event: Event = filter_events[np.argmax(events_timestamps[events_timestamps < timestamp])]
    return closest_event.__getattribute__(attribute)

def get_events_between(start_time, end_time, events, event_filter: callable):
    filter_events = np.array([e for e in events if event_filter(e)])
    events_timestamps = np.array([e.timestamp for e in filter_events])

    rtn_events = filter_events[np.logical_and(events_timestamps > start_time, events_timestamps < end_time)]
    return rtn_events

def add_event_to_data(data_array, data_timestamp, event_filter: callable):
    events = np.zeros(data_timestamp.shape)
    # TODO

def get_indices_from_transfer_timestamps(target_timestamps, source_timestamps):
    """
    reutnr the indices in target timestamps that are closest to the source timestamps
    @rtype: object
    """
    rtn = []
    for s_timestamps in source_timestamps:
        rtn.append(np.argmin(np.abs(target_timestamps - s_timestamps)))
    return rtn
