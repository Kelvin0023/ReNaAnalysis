import os

import cv2
import lpips
import matplotlib
import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
from moviepy.video.io import ImageSequenceClip
from sklearn.metrics import confusion_matrix, roc_curve, auc

from renaanalysis.eye.EyeUtils import temporal_filter_fixation
from renaanalysis.eye.eyetracking import Fixation, Saccade, GazeRayIntersect
from renaanalysis.utils.Event import get_events_between, get_block_start_event, get_overlapping_events_single_at_time
from renaanalysis.params.params import *
from renaanalysis.utils.TorchUtils import prepare_image_for_sim_score
from renaanalysis.utils.utils import visualize_pupil_epochs, visualize_eeg_epochs, remove_value


def visualiza_session(events):
    plt.rcParams["figure.figsize"] = [40, 3.5]

    meta_block_timestamps = [e.timestamp for e in events if e.meta_block]
    meta_blocks = [e.meta_block for e in events if e.meta_block]

    block_start_timestamps = [e.timestamp for e in events if e.is_block_start]
    block_conditions = [e.block_condition for e in events if e.is_block_start]

    dtn_timestamps = [e.timestamp for e in events if e.dtn]
    dtn_conditions = [e.block_condition for e in events if e.dtn]

    (markers, stemlines, baseline) = plt.stem(block_start_timestamps, block_conditions, label='block start conditions')
    (markers, stemlines, baseline) = plt.stem(dtn_timestamps, dtn_conditions, linefmt='orange', markerfmt='D', label='DTN conditions')
    plt.setp(markers, marker='D', markersize=2, markeredgecolor="orange", markeredgewidth=2)

    (markers, stemlines, baseline) = plt.stem(meta_block_timestamps, meta_blocks, linefmt='cyan', markerfmt='D', label='meta blocks')
    plt.setp(markers, marker='D', markersize=2, markeredgecolor="cyan", markeredgewidth=2)

    plt.legend()
    plt.title('Session Conditions')
    plt.show()

def visualize_dtn(events, block_id):
    plt.rcParams["figure.figsize"] = [40, 5]

    block_start_timestamps = [e.timestamp for e in events if e.is_block_start]
    block_conditions = np.array([e.block_condition for e in events if e.is_block_start])

    plt.stem(block_start_timestamps, block_conditions, label='block start conditions')

    if block_id:
        block_start_timestamp = [e.timestamp for e in events if e.is_block_start and e.block_id==block_id][0]
        block_end_timestamp = [e.timestamp for e in events if e.is_block_end and e.block_id==block_id][0]

        # also plot the dtns
        dtn_onsets_ts = np.array([e.timestamp for e in events if e.dtn_onffset and e.block_id==block_id])
        dtn_offsets_ts = np.array([e.timestamp for e in events if e.dtn_onffset == False and e.block_id==block_id])
        dtn_type = np.array([e.dtn for e in events if e.block_id==block_id])
        [plt.axvspan(onset, offset, alpha=0.5, color='red' if dtn==2 else 'blue') for onset, offset, dtn in zip(dtn_onsets_ts, dtn_offsets_ts, dtn_type)]
        plt.xlim(block_start_timestamp, block_end_timestamp)
    plt.legend()
    plt.show()


def visualize_gazeray(events, block_id=None):
    plt.rcParams["figure.figsize"] = [40, 5]

    block_start_timestamps = [e.timestamp for e in events if e.is_block_start]
    block_conditions = np.array([e.block_condition for e in events if e.is_block_start])
    gaze_intersects_dtn_timestamps = np.array([e.timestamp for e in events if e.gaze_intersect])
    gaze_intersects_dtn = np.array([e.dtn for e in events if e.gaze_intersect])

    (markers, stemlines, baseline) = plt.stem(block_start_timestamps, block_conditions, label='block start conditions')

    (markers, stemlines, baseline) = plt.stem(gaze_intersects_dtn_timestamps, gaze_intersects_dtn, label='gaze intersect DTN')
    plt.setp(markers, marker='D', markersize=2, markeredgecolor="orange", markeredgewidth=2)

    if block_id:
        block_start_timestamp = [e.timestamp for e in events if e.is_block_start and e.block_id==block_id][0]
        block_end_timestamp = [e.timestamp for e in events if e.is_block_end and e.block_id==block_id][0]

        # also plot the dtns
        gaze_intersects_dtn_timestamps = np.array([e.timestamp for e in events if e.gaze_intersect])
        dtn_onsets_ts = np.array([e.timestamp for e in events if e.dtn_onffset and e.block_id==block_id])
        dtn_offsets_ts = np.array([e.timestamp for e in events if e.dtn_onffset == False and e.block_id==block_id])
        dtn_type = np.array([e.dtn for e in events if e.block_id==block_id])
        [plt.axvspan(onset, offset, alpha=0.5, color='red' if dtn==2 else 'blue') for onset, offset, dtn in zip(dtn_onsets_ts, dtn_offsets_ts, dtn_type)]
        plt.xlim(block_start_timestamp, block_end_timestamp)
    plt.legend()
    plt.show()


def visualize_block_gaze_event(rdf, participant, session, block_id=None, only_long_gaze=False, generate_video=True, video_fix_alg='I-VT'):
    events = rdf.get_event(participant, session)
    generate_video = rdf.participant_session_videos[participant, session] if generate_video else None
    visualize_gaze_events(events, block_id, only_long_gaze=only_long_gaze, generate_video=generate_video, video_fix_alg=video_fix_alg)

def visualize_gaze_events(events, block_id=None, gaze_intersect_y=0.1, IDT_fix_head_y=.5, IVT_fix_head_y=1., pathSim_fix_y = 1.5, only_long_gaze=False, generate_video=None, video_fix_alg='I-VT'):
    f, ax = plt.subplots(figsize=[40, 5])

    block_start_timestamps = [e.timestamp for e in events if e.is_block_start]
    block_conditions = np.array([e.block_condition for e in events if e.is_block_start])

    (markers, stemlines, baseline) = ax.stem(block_start_timestamps, block_conditions, label='block start conditions')

    # ax.scatter(gaze_intersects_dtn_timestamps, len(gaze_intersects_dtn_timestamps) * [gaze_intersect_y], marker='D', c=[dtn_color_dict[x] for x in gaze_intersects_dtn], label='gaze intersect DTN')

    if block_id is not None:
        block_start_timestamp = [e.timestamp for e in events if e.is_block_start and e.block_id==block_id][0]
        block_end_timestamp = [e.timestamp for e in events if e.is_block_end and e.block_id==block_id][0]

        dtn_onsets_ts = np.array([e.timestamp for e in events if e.dtn_onffset and e.block_id==block_id])
        dtn_offsets_ts = np.array([e.timestamp for e in events if e.dtn_onffset == False and e.block_id==block_id])
        dtn_type = np.array([e.dtn for e in events if e.block_id == block_id and e.dtn_onffset])
        [plt.axvspan(onset, offset, alpha=0.25, color='red' if dtn==2 else 'blue') for onset, offset, dtn in zip(dtn_onsets_ts, dtn_offsets_ts, dtn_type)]

        if only_long_gaze:
            gaze_ray_intersects = draw_fixations(ax, events, lambda x: type(x) == GazeRayIntersect and block_start_timestamp < x.timestamp < block_end_timestamp and x.is_first_long_gaze, gaze_intersect_y)
        else:
            gaze_ray_intersects = draw_fixations(ax, events, lambda x: type(x) == GazeRayIntersect and block_start_timestamp < x.timestamp < block_end_timestamp, gaze_intersect_y)
        # fix_ivt = draw_fixations(ax, events, lambda x: type(x) == Fixation and x.detection_alg == 'I-VT' and block_start_timestamp < x.timestamp < block_end_timestamp, IDT_fix_y)
        fix_idt_head = draw_fixations(ax, events, lambda x: type(x) == Fixation and x.detection_alg == 'I-DT-Head' and block_start_timestamp < x.timestamp < block_end_timestamp, IDT_fix_head_y)
        fix_ivt_head = draw_fixations(ax, events, lambda x: type(x) == Fixation and x.detection_alg == 'I-VT-Head' and block_start_timestamp < x.timestamp < block_end_timestamp, IVT_fix_head_y)
        fix_patch_sim = draw_fixations(ax, events, lambda x: type(x) == Fixation and x.detection_alg == 'Patch-Sim' and block_start_timestamp < x.timestamp < block_end_timestamp, pathSim_fix_y)

        ax.set_xlim(block_start_timestamp, block_end_timestamp)
        ax.set_title("Block ID {}, condition {}".format(block_id, get_block_start_event(block_id, events).block_condition))

    ax.set_yticks([gaze_intersect_y, IDT_fix_head_y, IVT_fix_head_y, pathSim_fix_y])
    ax.set_yticklabels(['Gaze Ray Intersect', 'I-DT-Head', 'I-VT-Head', 'Patch-Sim'])
    ax.legend()
    ax.set_xlabel('Time (sec)')
    plt.show()

    if generate_video is not None and block_id is not None:
        generate_block_video(image_folder=generate_video, block_id=block_id, block_start_time=block_start_timestamp,
                             block_end_time=block_end_timestamp, gaze_ray_intersects=gaze_ray_intersects, fix_idt_head=fix_idt_head, fix_ivt_head=fix_ivt_head, fix_patch_sim=fix_patch_sim, video_fix_alg=video_fix_alg)


def add_bounding_box(a, x, y, width, height, color):
    copy = np.copy(a)
    image_height = a.shape[0]
    image_width = a.shape[1]
    bounding_box = (np.max([0, x - int(width/2)]), np.max([0, y - int(height/2)]), width, height)

    copy[bounding_box[1], bounding_box[0]:bounding_box[0] + bounding_box[2]] = color
    copy[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]] = color

    copy[np.min([image_height-1, bounding_box[1] + bounding_box[3]]), bounding_box[0]:bounding_box[0] + bounding_box[2]] = color
    copy[bounding_box[1]:bounding_box[1] + bounding_box[3], np.min([image_width-1, bounding_box[0] + bounding_box[2]])] = color
    return copy

def draw_fixations(ax, events, event_filter, fix_y, include_item_index=True):
    filtered_events = [e for e in events if event_filter(e)]
    fix_onset_times = [e.onset_time for e in filtered_events]
    fix_offset_times = [e.offset_time for e in filtered_events]
    fix_dtn = [e.dtn for e in filtered_events]
    fix_item_indices = [e.item_index for e in filtered_events]
    for f_onset_ts, f_offset_ts, f_dtn, f_item_index in zip(fix_onset_times, fix_offset_times, fix_dtn, fix_item_indices):
        ax.hlines(y=fix_y, xmin=f_onset_ts, xmax=f_offset_ts, linewidth=4, colors=dtn_color_dict[f_dtn])
        if f_item_index is not None and include_item_index:
            ax.text((f_onset_ts + f_offset_ts) / 2, fix_y, f'{f_item_index}')
    return filtered_events

def generate_block_video(image_folder, block_id, block_start_time, block_end_time, gaze_ray_intersects, fix_idt_head, fix_ivt_head, fix_patch_sim, video_fix_alg, is_add_patch_sim=False, is_flipping_y=False):
    """
    When video_fix_alg is None, it will generate a video with all the fixations marked with circles. It will
    add the circle if there is only one fix event in this frame, and the fix event has a dtn type, in other
    words, only fixation on distractor/target/novelty will be marked with the video
    @param image_folder:
    @param block_id:
    @param block_start_time:
    @param block_end_time:
    @param gaze_ray_intersects:
    @param fix_idt_head:
    @param fix_ivt_head:
    @param fix_patch_sim:
    @param video_fix_alg:
    @param is_add_patch_sim:
    @param is_flipping_y:
    @return:
    """
    gaze_info_file = os.path.join(image_folder, 'GazeInfo.csv')
    video_name = f'BlockVideo_{block_id}-FixationAlgorithm_{video_fix_alg}.mp4'

    #defining parameters ##############################

    similarity_threshold = .02

    fixation_y_value = -1e-2
    video_fps = 30

    patch_size = 63, 111  # width, height
    fovs = 115, 90  # horizontal, vertical, in degrees
    central_fov = 13  # fov of the fovea
    near_peripheral_fov = 30  # physio fov
    mid_perpheral_fov = 60  # physio fov

    # for drawing the fovea box
    patch_color = (255, 255, 0)

    center_color_dict = {-1: (128, 128, 128), 1: (0, 0, 255), 2: (255, 0, 0), 3: (255, 165, 0)}

    center_no_intersect_size = 5
    center_intersect_size = 10

    cmap = matplotlib.cm.get_cmap('cool')
    fix_circle_size = 20
    fix_color = 255 * np.array(cmap(1))
    # ivt_size = 20
    # ivt_color = 255 * np.array(cmap(1/3))
    # ivt_head_size = 25
    # ivt_head_color = 255 * np.array(cmap(2/3))
    # patch_sim_size = 30
    # patch_sim_color = 255 * np.array(cmap(1))
    fix_dict = {'I-DT': fix_idt_head, 'I-VT': fix_ivt_head, 'Patch-Sim': fix_patch_sim}

    cmap = matplotlib.cm.get_cmap('summer')
    fovea_color = 255 * np.array(cmap(1/3))
    parafovea_color = 255 * np.array(cmap(2/3))
    peripheri_color = 255 * np.array(cmap(1))

    y_axis = 0

    # end of parameters ###############################################################

    # read the gaze info csv
    gaze_info = pd.read_csv(gaze_info_file)

    timestamps = gaze_info['LocalClock']

    video_start_frame = np.argmin(np.abs(timestamps - block_start_time))
    video_frame_count = np.argmin(np.abs(timestamps - block_end_time)) - video_start_frame
    block_duration = block_end_time - block_start_time
    assert block_duration - 50e-3 < timestamps[video_start_frame + video_frame_count] - timestamps[video_start_frame] < block_duration + 50e-3
    assert video_frame_count > 1

    # get the video frames
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    image_size = cv2.imread(os.path.join(image_folder, images[0])).shape[:2]
    ppds = image_size[0] / fovs[0], image_size[1] / fovs[1]  # horizontal, vertical, calculate pixel per degree of FOV

    images.sort(key=lambda x: int(x.strip('.png')))  # sort the image files
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width,height))
    loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
    images_with_bb = []
    previous_img_patch = None
    distance_list = []
    patch_boundaries = []
    if video_frame_count is None: video_frame_count = len(images[video_start_frame:])
    for i, image in enumerate(images[video_start_frame:]):  # iterate through the images
        print('Processing {0} of {1} images'.format(i + 1, video_frame_count), end='\r', flush=True)
        img = cv2.imread(os.path.join(image_folder, image))  # read in the image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert from BGR to RGB
        img_modified = img.copy()
        gaze_coordinate = gaze_info.iloc[video_start_frame + i, :].values  # get the gaze coordinate for this image
        gaze_x, gaze_y, timestamp = int(gaze_coordinate[2]), int(gaze_coordinate[3]), gaze_coordinate[4]  # the gaze coordinate

        if is_flipping_y: gaze_y = image_size[y_axis] - gaze_y  # because CV's y zero is at the bottom of the screen
        center = gaze_x, gaze_y

        gaze_intersect_this_frame = get_overlapping_events_single_at_time(timestamp, gaze_ray_intersects)
        if len(gaze_intersect_this_frame) == 1:
            e = gaze_intersect_this_frame[0]
            intersect_index = e.item_index
            center_color = center_color_dict[e.dtn]
            center_radius = center_intersect_size
            center_thickness = 2
        elif len(gaze_intersect_this_frame) == 0:
            intersect_index = None
            center_radius = center_no_intersect_size
            center_color = center_color_dict[-1]
            center_thickness = 1
        else:
            raise Exception("There can only be at most one gaze ray intersect at a eyetracking frame")
        cv2.circle(img_modified, center, center_radius, center_color, center_thickness)

        img_patch_x_min = int(np.min([np.max([0, gaze_x - patch_size[0] / 2]), image_size[0] - patch_size[0]]))
        img_patch_x_max = int(np.max([np.min([image_size[0], gaze_x + patch_size[0] / 2]), patch_size[0]]))
        img_patch_y_min = int(np.min([np.max([0, gaze_y - patch_size[1] / 2]), image_size[1] - patch_size[1]]))
        img_patch_y_max = int(np.max([np.min([image_size[1], gaze_y + patch_size[1] / 2]), patch_size[1]]))
        patch_boundaries.append((img_patch_x_min, img_patch_y_min, img_patch_x_max, img_patch_y_max))

        img_patch = img[img_patch_x_min: img_patch_x_max, img_patch_y_min: img_patch_y_max]

        # draw the center circle
       # get similarity score
        if previous_img_patch is not None:
            img_tensor, previous_img_tensor = prepare_image_for_sim_score(img_patch), prepare_image_for_sim_score(
                previous_img_patch)
            distance = loss_fn_alex(img_tensor, previous_img_tensor).item()
            # img_modified = cv2.putText(img_modified, "%.2f" % distance, center, cv2.FONT_HERSHEY_SIMPLEX, 1, center_color, 2, cv2.LINE_AA)
            distance_list.append(distance)
        previous_img_patch = img_patch

        axis = (int(central_fov * ppds[0]), int(central_fov * ppds[1]))
        cv2.ellipse(img_modified, center, axis, 0, 0, 360, fovea_color, thickness=3)
        axis = (int(near_peripheral_fov * ppds[0]), int(near_peripheral_fov * ppds[1]))
        cv2.ellipse(img_modified, center, axis, 0, 0, 360, parafovea_color, thickness=3)
        axis = (int(1.25 * mid_perpheral_fov * ppds[0]), int(mid_perpheral_fov * ppds[1]))
        cv2.ellipse(img_modified, center, axis, 0, 0, 360, peripheri_color, thickness=3)

        if intersect_index is not None: img_modified = cv2.putText(img_modified, f'GR:ItemIdx:{intersect_index}',
                                                                   center + np.array([15, 30]),
                                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, center_color, 1,
                                                                   cv2.LINE_AA)
        if video_fix_alg is not None:
            img_modified = add_fix_detection_circle(img_modified, center, timestamp, fix_dict[video_fix_alg], center_color_dict, fix_circle_size, video_fix_alg)


        images_with_bb.append(img_modified)
        if i + 1 == video_frame_count:
            break

    distance_list = np.array(distance_list)
    thresholded_sim_distance = np.ones(len(distance_list))
    thresholded_sim_distance[distance_list > similarity_threshold] = 0
    # duration thresholding
    fix_list_filtered = temporal_filter_fixation(thresholded_sim_distance, marker_mode='viz')

    if is_add_patch_sim:
        for i, (fix, img, patch_boundary) in enumerate(zip(fix_list_filtered, images_with_bb, patch_boundaries)):
            img_modified = img.copy()
            if fix == fixation_y_value:
                shapes = np.zeros_like(img_modified, np.uint8)
                alpha = 0.3
                cv2.rectangle(shapes, patch_boundary[:2], patch_boundary[2:], patch_color, thickness=-1)
                mask = shapes.astype(bool)
                img_modified[mask] = cv2.addWeighted(img_modified, alpha, shapes, 1 - alpha, 0)[mask]
            else:
                cv2.rectangle(img_modified, patch_boundary[:2], patch_boundary[2:], patch_color, thickness=2)
            images_with_bb[i] = img_modified

    clip = ImageSequenceClip.ImageSequenceClip(images_with_bb, fps=video_fps)
    clip.write_videofile(os.path.join(image_folder, video_name))

    fig = plt.gcf()
    fig.set_size_inches(30, 10.5)
    plt.rcParams['font.size'] = '24'
    plt.plot(np.linspace(0, block_duration, len(distance_list)), distance_list,
             linewidth=5, label='Fovea Patch Distance')
    plt.plot(np.linspace(0, block_duration, len(distance_list)),
             fix_list_filtered, linewidth=10, label='Fixation')
    plt.title('Example similarity distance sequence')
    plt.ylabel("Similarity distance between previous and this frame")
    plt.xlabel("Time (seconds)")
    plt.show()

def add_fix_detection_circle(img_modified, center, timestamp, fix_events, center_color_dict, marker_radius, alg):
    """
    will only add the circle if there is only one fix event in this frame, and the fix event has a dtn type, in other
    words, only fixation on distractor/target/novelty will be marked with the video
    @param img_modified:
    @param center:
    @param timestamp:
    @param fix_events:
    @param center_color_dict:
    @param marker_radius:
    @param alg:
    @return:
    """
    fix_this_frame = get_overlapping_events_single_at_time(timestamp, fix_events)
    if len(fix_this_frame) == 1:
        fix_event = fix_this_frame[0]
        if fix_event.dtn is not None:
            center_color = center_color_dict[fix_event.dtn]
            img_modified = cv2.circle(img_modified, center, marker_radius, center_color, 2)
            img_modified = cv2.putText(img_modified, f'Fix:{alg}',
                                       center + np.array([15, -30]),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, center_color, 1,
                                       cv2.LINE_AA)
    elif len(fix_this_frame) == 0:
        pass
    else:
        raise Exception(
            "There can only be at most one fixation event with a certain detection algorithm at a eyetracking frame")
    return img_modified


def viz_pupil_epochs(rdf, event_names, event_filters, colors, title='', eyetracking_resample_srate=20, participant=None, session=None, tmin=-1., tmax=3., n_jobs=1):
    pupil_epochs, pupil_event_ids, _ = rdf.get_pupil_epochs(event_names, event_filters, eyetracking_resample_srate, participant, session, tmin=tmin, tmax=tmax, n_jobs=n_jobs)
    visualize_pupil_epochs(pupil_epochs, pupil_event_ids, colors, title=title)


def viz_eeg_epochs(rdf, event_names, event_filters, colors, title='', participant=None, session=None, tmin=-0.1, tmax=0.8, n_jobs=1):
    eeg_epochs, eeg_event_ids, _, _ = rdf.get_eeg_epochs(event_names, event_filters, tmin, tmax, participant, session, n_jobs=n_jobs)
    visualize_eeg_epochs(eeg_epochs, eeg_event_ids, colors, title=title)

def viz_class_error(standard, target, type):
    plt.plot(standard, label='Standard')
    plt.plot(target, label='Target')

    # Adding labels and title
    plt.xlabel('Epochs')
    plt.ylabel('num_errors')
    plt.title(f'{type} Errors')

    # Adding legend
    plt.legend()

    # Display the plot
    plt.show()

def viz_confusion_matrix(true_label, pred_label, class_names, epoch, fold, type):
    cm = confusion_matrix(true_label, pred_label)

    fig, ax = plt.subplots()

    # Plot the confusion matrix as an image
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Set axis labels and title
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xlabel='Predicted label',
           ylabel='True label',
           title=f'{type} Confusion Matrix of epoch {epoch} in fold {fold}')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Display the tick labels on the x and y axes
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)

    # Loop over data dimensions and create text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='red')

    plt.savefig(f'renaanalysis/learning/saved_images/confusion_matrices/{fold}_{epoch}_{type}.png')
    # Show the plot
    plt.show()

def viz_binary_roc(true_label, pred_score, param_list, fold, target_label=1):
    '''

    @param true_label:
    @param pred_score:
    @param fold:
    @param target_label: the index of the target in the label if multiclass is presented. For example,
    if there are two classes, and target_label = 1, that means the true_label = [0, 1] is a target, and
    [1. 0] being non-target

    '''

    if true_label.ndim == 2:
        true_label = true_label.take([target_label], axis=1).ravel()
    if pred_score.ndim == 2:
        pred_score = pred_score.take([target_label], axis=1).ravel()
    fpr, tpr, thresholds = roc_curve(true_label, pred_score)
    # Compute AUC (Area Under the ROC Curve)
    auc_score = auc(fpr, tpr)
    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve of param {param_list} in fold {fold}')
    plt.legend()
    if not os.path.exists('renaanalysis/learning/saved_images/roc'):
        os.mkdir('renaanalysis/learning/saved_images/roc')
    plt.savefig(f'renaanalysis/learning/saved_images/roc/{param_list}_{fold}.png')

    plt.show()

def reorganize_folds_histories(total_training_histories):
    """
    reorganize the histories from the folds
    from
        param: tuple of tuples (hashable dict) -> metric (dict) -> metric name (str) -> list of folds -> list of epoch history for this fold
    to
        param: tuple of tuples (hashable dict) -> list of folds ->  metric name (str) -> list of epoch history for this fold
    """
    reorganized_histories = {}
    n_folds = len(list(list(total_training_histories.values())[0].values())[0])
    for param, metric_dict in total_training_histories.items():
        reorganized_histories[param] = []
        for fold_i in range(n_folds):
            fold_metrics = {metric_name: metric_folds[fold_i] for metric_name, metric_folds in metric_dict.items()}
            reorganized_histories[param].append(fold_metrics)
    return reorganized_histories


def plot_training_history_folds(total_training_histories):
    training_history_reorganized = reorganize_folds_histories(total_training_histories)
    for param, histories in training_history_reorganized.items():
        for fold_i, fold_hist in enumerate(histories):
            plot_training_history(fold_hist, dict(param)['mem_len'], fold_i)


def plot_training_history(history, param_list, fold, is_plot_auc=True):
    # Extract the training history
    train_loss = history['loss_train']
    val_loss = history['loss_val']
    train_acc = history['acc_train']
    val_acc = history['acc_val']
    if is_plot_auc:
        val_auc = history['auc_val']
        test_auc = history['auc_test']
    test_acc = history['acc_test']
    test_loss = history['loss_test']

    # Plot the training and validation loss
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    if param_list is not None:
        if is_plot_auc:
            fig.suptitle(f'Training history of param {param_list} in fold {fold}, test auc = {test_auc}')
        else:
            fig.suptitle(f'Training history of param {param_list} in fold {fold}, test acc = {test_acc}')
    else:
        if is_plot_auc:
            fig.suptitle(f'Training history of fold {fold}, test auc = {test_auc}')
        else:
            fig.suptitle(f'Training history of fold {fold}, test acc = {test_acc}')
    axs[0].plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    axs[0].plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title(f'Training and Validation Loss')
    axs[0].legend()

    # Plot the training and validation accuracy
    axs[1].plot(range(1, len(train_acc) + 1), train_acc, label='Training Accuracy')
    axs[1].plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy')
    if is_plot_auc:
        axs[1].plot(range(1, len(val_auc) + 1), val_auc, label='Validation AUC')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy / Area')
    axs[1].set_title('Training and Validation Accuracy')
    axs[1].legend()

    # Display the plot
    plt.tight_layout()
    if param_list is not None:
        plt.savefig(f'{param_list}_{fold}.png')
    plt.show()

def plot_training_loss_history(history, param_list, fold):
    # Extract the training history
    train_loss = history['loss_train']
    val_loss = history['loss_val']
    test_loss = history['loss_test']

    # Plot the training and validation loss
    fig, axs = plt.subplots(1, 1, figsize=(9, 6))
    if param_list is not None:
        fig.suptitle(f'Training history of param {param_list} in fold {fold}, test loss = {test_loss}')
    else:
        fig.suptitle(f'Training history of fold {fold}, test loss = {test_loss}')
    axs.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    axs.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
    axs.set_xlabel('Epochs')
    axs.set_ylabel('Loss')
    axs.set_title(f'Training and Validation Loss')
    axs.legend()

    # Display the plot
    plt.tight_layout()
    if param_list is not None:
        plt.savefig(f'renaanalysis/learning/saved_images/training_history/{param_list}_{fold}.png')
    plt.show()


def compare_epochs():
    pass

def visualize_eeg_samples(x, y, colors, eeg_picks, title='', out_dir=None, verbose='INFO', fig_size=(12.8, 7.2),
                          is_plot_timeseries=True):

    # Set the figure size for the plot
    plt.rcParams["figure.figsize"] = fig_size
    event_ids = set(y)
    x = np.swapaxes(x, 0, 1)
    # Plot each EEG channel for each event type
    if is_plot_timeseries:
        for idx, ch in enumerate(eeg_picks):
            for event in event_ids:
                event_idx = np.where(y == event)[0]
                x_mean = np.mean(x[idx][event_idx], axis=0)
                x1 = x_mean + scipy.stats.sem(x[idx][event_idx], axis=0)  # this is the upper envelope
                x2 = x_mean - scipy.stats.sem(x[idx][event_idx], axis=0)
                time_vector = np.linspace(-0.1, 0.8, x[idx][event_idx].shape[-1])
                # Plot the EEG data as a shaded area
                plt.fill_between(time_vector, x1, x2, where=x2 <= x1, facecolor=colors[event], interpolate=True, alpha=0.5)
                plt.plot(time_vector, x_mean, c=colors[event], label='{0}, N={1}'.format(event, x[idx][event_idx].shape[0]))

                # Set the labels and title for the plot
            plt.xlabel('Time (sec)')
            plt.ylabel('BioSemi Channel {0} (μV), shades are SEM'.format(ch))
            plt.legend()
            plt.title('{0} - Channel {1}'.format(title, ch))

            # Save or show the plot
            if out_dir:
                plt.savefig(os.path.join(out_dir, '{0} - Channel {1}.png'.format(title, ch)))
                plt.clf()
            else:
                plt.show()


def get_line_styles(num_styles):
    """
    Get a list of different line styles for a line chart.

    Parameters:
        num_styles (int): The number of styles to generate.

    Returns:
        list: A list of line styles.
    """
    line_styles = ['-', '--', '-.', ':', (0, (1, 1)), (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (3, 5, 1, 5)), (0, (3, 5, 3, 5, 1, 5))]
    return line_styles[:num_styles]

def grid_search_bars(model_performance, grid_search_params, metric, title="Grid Search"):
    # plot performance for different params
    plot_params = [param_name for param_name, param_values in grid_search_params.items() if len(param_values) > 1]

    grouped_results = {}
    for key, value in model_performance.items():
        params = [dict(key)[x] for x in plot_params]
        grouped_results[tuple(params)] = sum(value[metric]) / len(value[metric]) if isinstance(value[metric], list) else value[metric]

    unique_params = dict([(param_name, np.unique([key[i] for key in grouped_results.keys()])) for i, param_name in enumerate(plot_params)])

    # Create subplots for each parameter
    fig, axes = plt.subplots(nrows=1, ncols=len(plot_params), figsize=(16, 5))

    # Plot the bar charts for each parameter
    for i, (param_name, param_values) in enumerate(unique_params.items()):  # iterate over the hyperparameter types
        axis = axes[i] if len(plot_params) > 1 else axes
        labels = []
        auc_values = []

        common_keys = []  # find the intersection of keys for this parameter to avoid biasing the results (needed when the grid search is incomplete)
        for j, param_val in enumerate(param_values):  # iterate over the values of the parameter
            other_keys = [list(key) for key, value in grouped_results.items() if key[i] == param_val]
            [x.remove(param_val) for x in other_keys]
            other_keys = [tuple(x) for x in other_keys]
            common_keys.append(other_keys)
        common_keys = set(common_keys[0]).intersection(*common_keys[1:])

        for j, param_val in enumerate(param_values):  # iterate over the values of the parameter
            # metric_values = [value for key, value in grouped_results.items() if key[i] == param_val and remove_value(key, param_val) in common_keys]
            metric_values = []
            for key, value in grouped_results.items():
                if key[i] == param_val and tuple(remove_value(key, param_val)) in common_keys:
                    metric_values.append(value)

            auc_values.append(np.mean(metric_values))
            labels.append(param_val)

        xticks = np.arange(len(labels))
        axis.bar(xticks, auc_values)
        # put text of the auc values
        for j, auc_value in enumerate(auc_values):
            axis.text(xticks[j], auc_value, f'{auc_value:.6f}', ha='center', va='bottom', fontsize=12)
        axis.set_xticks(xticks, labels=labels, rotation=45)
        axis.set_xlabel(param_name)
        axis.set_ylabel(metric)
        axis.set_title(title)

    # Adjust the layout of subplots and show the figure
    fig.tight_layout()
    plt.show()

