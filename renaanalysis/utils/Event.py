import inspect
from typing import Union

import numpy as np
from mne.io import RawArray

from renaanalysis.params.params import *

class Event:
    def __init__(self, timestamp, *args, **kwargs):
        """
        dtn can be
        @param timestamp:
        @param args:
        @param kwargs:
        """
        self.timestamp = timestamp

        # block information
        self.meta_block= kwargs['meta_block'] if 'meta_block' in kwargs.keys() else None
        self.block_condition = kwargs['block_condition'] if 'block_condition' in kwargs.keys() else None
        self.block_id = kwargs['block_id'] if 'block_id' in kwargs.keys() else None
        self.is_practice = kwargs['is_practice'] if 'is_practice' in kwargs.keys() else None

        self.is_block_start = kwargs['is_block_start'] if 'is_block_start' in kwargs.keys() else None
        self.is_block_end = kwargs['is_block_end'] if 'is_block_end' in kwargs.keys() else None
        self.is_new_cp = kwargs['is_new_cp'] if 'is_new_cp' in kwargs.keys() else None

        # distractor, target, novelty or null
        self.dtn = kwargs['dtn'] if 'dtn' in kwargs.keys() else None
        self.dtn_onffset = kwargs['dtn_onffset'] if 'dtn_onffset' in kwargs.keys() else None  # only event marker will have this field

        # object related markers
        self.item_index = kwargs['item_index'] if 'item_index' in kwargs.keys() else None
        self.item_id = kwargs['item_id'] if 'itemID' in kwargs.keys() else None
        self.obj_dist = kwargs['obj_dist'] if 'objDistFromPlayer' in kwargs.keys() else None
        self.carousel_speed = kwargs['carousel_speed'] if 'CarouselSpeed' in kwargs.keys() else None
        self.carousel_angle = kwargs['carousel_angle'] if 'CarouselAngle' in kwargs.keys() else None

        self.likert = kwargs['Likert'] if 'Likert' in kwargs.keys() else None  # TODO to be added

    def __str__(self):
        rtn = ''
        for i in inspect.getmembers(self):
            if not i[0].startswith('_'):  # To remove other methods that
                if not inspect.ismethod(i[1]):
                    rtn += f'{i}, '
        return rtn

class FoveateAngleCrossing(Event):
    def __init__(self, timestamp, onset_time, offset_time, threshold, *args, **kwargs):
        super().__init__(timestamp, *args, **kwargs)
        self.onset_time = onset_time
        self.offset_time = offset_time
        self.threshold = threshold

def add_event_meta_info(event, events):
    """
    return the meta event info from events for a specific event, including its condition, is or is not practice and block ID
    :param event:
    :param events:
    :return:
    """
    event.block_condition = get_closest_event_attribute_before(events, event.timestamp, 'block_condition',
                                                         event_filter=lambda e: e.is_block_start)  # must be a block start event
    event.is_practice = get_closest_event_attribute_before(events, event.timestamp, 'is_practice',
                                                           event_filter=lambda e: e.is_block_start)  # must be a block start event
    event.block_id = get_closest_event_attribute_before(events, event.timestamp, 'block_id',
                                                        event_filter=lambda e: e.is_block_start)  # must be a block start event
    return event


def get_closest_event_attribute_before(events, timestamp, attribute, event_filter: callable):
    filter_events = np.array([e for e in events if event_filter(e)])
    events_timestamps = np.array([e.timestamp for e in filter_events])

    if np.sum(events_timestamps < timestamp) == 0:
        raise ValueError("No event is found before the given event")
    closest_event: Event = filter_events[np.argmax(events_timestamps[events_timestamps < timestamp])]
    return closest_event.__getattribute__(attribute)


def get_closest_event_before(events, timestamp, event_filter: callable):
    filter_events = np.array([e for e in events if event_filter(e)])
    events_timestamps = np.array([e.timestamp for e in filter_events])

    if np.sum(events_timestamps < timestamp) == 0:
        raise ValueError("No event is found before the given event")
    closest_event: Event = filter_events[np.argmax(events_timestamps[events_timestamps < timestamp])]
    return closest_event


def get_events_between(start_time, end_time, events, event_filter: callable):
    filter_events = np.array([e for e in events if event_filter(e)])
    events_timestamps = np.array([e.timestamp for e in filter_events])

    rtn_events = list(filter_events[np.logical_and(events_timestamps > start_time, events_timestamps < end_time)])
    return rtn_events

def get_overlapping_events(start_time, end_time, events, event_filter: callable=None):
    """
    given events must have onset and offset in fields
    @return:
    """
    if event_filter is None:
        event_filter = lambda x: x  # use identity lambda

    filter_events = np.array([e for e in events if event_filter(e)])
    onset_times = np.array([e.onset_time for e in filter_events])
    offset_times = np.array([e.offset_time for e in filter_events])

    after_start_event_mask = np.logical_and(onset_times >= start_time, onset_times <= end_time)
    before_start_event_mask = np.logical_and(offset_times <= end_time, offset_times >= start_time)
    return filter_events[np.logical_or(after_start_event_mask, before_start_event_mask)]

def get_overlapping_events_single_target(target_time, events, event_filter: callable=None):
    """
    given events must have onset and offset in fields
    @return:
    """
    if event_filter is None:
        event_filter = lambda x: x  # use identity lambda
    filter_events = np.array([e for e in events if event_filter(e)])
    onset_times = np.array([e.onset_time for e in filter_events])
    offset_times = np.array([e.offset_time for e in filter_events])

    mask = np.logical_and(onset_times <= target_time, offset_times >= target_time)
    return filter_events[mask]

def add_events_to_data(data_array: Union[np.ndarray, RawArray], data_timestamp, events, event_names, event_filters, deviate=25e-2):
    event_array = np.zeros(data_timestamp.shape)
    event_ids = {}
    deviant = 0
    for i, e_filter in enumerate(event_filters):
        filtered_events = np.array([e for e in events if e_filter(e)])
        event_ts = [e.timestamp for e in filtered_events]


        event_data_indices = [np.argmin(np.abs(data_timestamp - t)) for t in event_ts if np.min(np.abs(data_timestamp - t)) < deviate]

        if len(event_data_indices) > 0:
            deviate_event_count = len(event_ts) - len(event_data_indices)
            if deviate_event_count > 0: print("Removing {} deviate events".format(deviate_event_count))
            deviant += deviate_event_count

            event_array[event_data_indices] = i + 1
            event_ids[event_names[i]] = i + 1
        else:
            print(f'Unable to find event with name {event_names[i]}, skipping')
    if type(data_array) is np.ndarray:
        rtn = np.concatenate([data_array, np.expand_dims(event_array, axis=1)], axis=1)
    elif type(data_array) is RawArray:
        print()
        stim_index = data_array.ch_names.index('stim')
        rtn = data_array.copy()
        rtn._data[stim_index, :] = event_array
    else:
        raise Exception(f'Unsupported data type {type(data_array)}')
    return rtn, event_ids, deviant

def get_indices_from_transfer_timestamps(target_timestamps, source_timestamps):
    """
    reutnr the indices in target timestamps that are closest to the source timestamps
    @rtype: object
    """
    rtn = []
    for s_timestamps in source_timestamps:
        rtn.append(np.argmin(np.abs(target_timestamps - s_timestamps)))
    return rtn

def get_block_startend_times(events):
    block_start_timestamps = [e.timestamp for e in events if e.is_block_start]
    block_end_timestamps = [e.timestamp for e in events if e.is_block_end]
    return block_start_timestamps, block_end_timestamps

def is_event_in_block(event, events):
    block_start_timestamps, block_end_timestamps = get_block_startend_times(events)
    for start_time, end_time in zip(block_start_timestamps, block_end_timestamps):
        if start_time < event.timestamp < end_time:
            return True
    return False

def copy_item_info(dest_event, source_event):
    dest_event.block_condition = source_event.block_condition
    dest_event.block_id = source_event.block_id

    dest_event.dtn = source_event.dtn
    dest_event.dtn_onffset = source_event.dtn_onffset
    dest_event.item_id = source_event.item_id
    dest_event.item_index = source_event.item_index
    dest_event.obj_dist = source_event.obj_dist
    dest_event.carousel_speed = source_event.carousel_speed
    dest_event.carousel_angle = source_event.carousel_angle
    return dest_event

def get_block_start_event(block_id, events):
    filter_events = np.array([e for e in events if e.is_block_start and e.block_id==block_id])
    assert len(filter_events) == 1
    return filter_events[0]

def check_contraint_block_counts(events, epoch_count):
    contraint_block_events = [e for e in events if e.is_block_start == True and (e.block_condition == conditions['Carousel'] or e.block_condition == conditions['RSVP'])]
    assert epoch_count == len(contraint_block_events) * num_items_per_constrainted_block

def get_last_block_end_time(events):
    filter_events = [e for e in events if e.is_block_end]
    return filter_events[-1].timestamp

