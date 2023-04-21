#! /usr/bin/env python3

import os

rgb_path = "rgb_classes"
#data_path = "data"
output_file = rgb_path + "/annotations.txt"

classes = os.listdir(rgb_path)
decoder = {}
for i in range(len(classes)):
    decoder[classes[i]] = i
encoder = {}
for i in range(len(classes)):
    encoder[i] = classes[i]

print(decoder)

with open(output_file, 'w') as f:
    for class_id, class_name in enumerate(classes):
        class_path = os.path.join(rgb_path, class_name)
        if os.path.isdir(class_path):
            video_names = os.listdir(class_path)
            for video_name in video_names:
                file_path = os.path.join(class_name, video_name)
                print(file_path)
                file_path_from_rgb = os.path.join(rgb_path, file_path)
                frame_end = len(os.listdir(file_path_from_rgb)) - 1
                print(frame_end)
                frame_start = 0

                #write annotations.txt
                annotation_string = "{} {} {} {}\n".format(file_path, frame_start, frame_end, class_id)
                f.write(annotation_string)

