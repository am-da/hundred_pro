import os
import cv2
import threading
from queue import Queue

"""
Given individual video files (mp4, webm) on disk, creates a folder for
every video file and saves the video's RGB frames as jpeg files in that
folder.

It can be used to turn SomethingSomethingV2, which comes as 
many ".webm" files, into an RGB folder for each ".webm" file.
Uses multithreading to extract frames faster.

Modify the two filepaths at the bottom and then run this script.
"""


def video_to_rgb(video_filename, out_dir, resize_shape):
    file_template = 'frame_{0:012d}.jpg'
    reader = cv2.VideoCapture(video_filename)
    success, frame, = reader.read()  # read first frame

    count = 0
    while success:
        out_filepath = os.path.join(out_dir, file_template.format(count))
        frame = cv2.resize(frame, resize_shape)
        cv2.imwrite(out_filepath, frame)
        success, frame = reader.read()
        count += 1


def process_videofile(video_filename, video_path, class_name, rgb_out_path, file_extension: str = '.mp4'):
    filepath = os.path.join(video_path, class_name, video_filename)
    video_filename = video_filename.replace(file_extension, '')
    print(class_name)
    out_dir = os.path.join(rgb_out_path, class_name, video_filename)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    video_to_rgb(filepath, out_dir, resize_shape=(224, 224))


def thread_job(queue, video_path, class_name, rgb_out_path, file_extension='.mp4'):
    while not queue.empty():
        video_filename = queue.get()
        process_videofile(video_filename, video_path, class_name, rgb_out_path, file_extension=file_extension)
        queue.task_done()


if __name__ == '__main__':
    # the path to the folder which contains all video files (mp4, webm, or other)
    video_path = 'video'
    # the root output path where RGB frame folders should be created
    rgb_out_path = 'rgb_classes'
    # the file extension that the videos have
    file_extension = '.mp4'

    # video_filenames = os.listdir(video_path)
    video_classnames = os.listdir(video_path)
    print(video_classnames)
    for class_name in video_classnames:
        class_path = os.path.join(video_path, class_name)

        video_filenames = os.listdir(class_path)
        print(video_filenames)

        queue = Queue()
        [queue.put(video_filename) for video_filename in video_filenames]

        NUM_THREADS = 8
        for i in range(NUM_THREADS):
            worker = threading.Thread(target=thread_job,
                                      args=(queue, video_path, class_name, rgb_out_path, file_extension))
            worker.start()

    print('waiting for all videos to be completed.', queue.qsize(), 'videos')
    print('This can take an hour or two depending on dataset size')
    queue.join()
    print('all done')
