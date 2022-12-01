import cv2
import os

# Read the video from specified path

def video_file_to_frame_array(path):
    # Path to video file
    video = cv2.VideoCapture(path)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    success = 1
    rtn = []
    while success:
        print("Extracting video frame {0}/{1}".format(count, length), end='\r')
        success, image = video.read()
        rtn.append(image)
        count += 1
    return rtn

def generate_video(data, video_stream_name, output_path=''):
    """
    if output path is not specified, the output video will be place in the same directory as the
    stream .dats file with a tag to its stream name
    :param stream_name:
    :param output_path:
    """
    print('Load video stream...')
    video_frame_stream = data[video_stream_name][0]
    frame_count = video_frame_stream.shape[-1]

    timestamp_stream = data[video_stream_name][1]
    frate = len(timestamp_stream) / (timestamp_stream[-1] - timestamp_stream[0])
    try:
        assert len(video_frame_stream.shape) == 4 and video_frame_stream.shape[2] == 3
    except AssertionError:
        raise Exception('target stream is not a video stream. It does not have 4 dims (height, width, color, time)'
                        'and/or the number of its color channel does not equal 3.')
    frame_size = (data[video_stream_name][0].shape[1], data[video_stream_name][0].shape[0])
    output_path = os.path.join(output_path, '{0}.avi'.format(video_stream_name))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'),frate, frame_size)

    for i in range(frame_count):
        print('Creating video progress {}%'.format(str(round(100 * i / frame_count, 2))), sep=' ', end='\r',
              flush=True)
        img = video_frame_stream[:, :, :, i]
        # img = np.reshape(img, newshape=list(frame_size) + [-1,])
        out.write(img)

    out.release()