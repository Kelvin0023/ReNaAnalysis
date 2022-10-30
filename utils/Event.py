import numpy as np


class Event:
    def __init__(self, timestamp, block_id, block_condition, *args, **kwargs):
        self.timestamp = timestamp
        self.block_condition = block_condition
        self.block_id = block_id

        self.dtn = kwargs['dtn'] if 'dtn' in kwargs.keys() else None
        self.item_id = kwargs['item_id'] if 'itemID' in kwargs.keys() else None
        self.obj_dist = kwargs['obj_dist'] if 'objDistFromPlayer' in kwargs.keys() else None
        self.carousel_speed = kwargs['carousel_speed'] if 'CarouselSpeed' in kwargs.keys() else None
        self.carousel_angle = kwargs['carousel_angle'] if 'CarouselAngle' in kwargs.keys() else None

        self.is_block_start = kwargs['is_block_start'] if 'is_block_start' in kwargs.keys() else None
        self.is_block_end = kwargs['is_block_end'] if 'is_block_end' in kwargs.keys() else None

        self.likert = kwargs['Likert'] if 'Likert' in kwargs.keys() else None
        self.is_practice = kwargs['is_practice'] if 'is_practice' in kwargs.keys() else None


def get_closest_event(events, timestamp, attribute, event_filter: callable):
    events_timestamps = np.array([e.timestamp for e in events if event_filter(e)])
    closest_event: Event = events[np.argmax(events_timestamps[events_timestamps < timestamp])]
    return closest_event.__getattribute__(attribute)

def add_event_to_data(data_array, data_timestamp, event_filter: callable):
    pass