#! wget https://raw.githubusercontent.com/pytorch/vision/6de158c473b83cf43344a0651d7c01128c7850e6/references/video_classification/transforms.py
import os
import time
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torch
import transforms as T
from movinets.models import MoViNet
from movinets.config import _C

from video_dataset import VideoFrameDataset, ImglistToTensor
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from tqdm import tqdm
import numpy as np

import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

torch.manual_seed(97)
num_frames = 8  # 16
clip_steps = 2

Bs_Train = 2
Bs_Test = 1

img_size = 172
cpu_num = os.cpu_count()

model_path = 'model/model.pth'

videos_root = os.path.join(os.getcwd(), 'rgb_classes')

annotation_file = os.path.join(videos_root, 'annotations.txt')


def normalize(input_tensor):
    normalize_ = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize_(input_tensor)


model = MoViNet(_C.MODEL.MoViNetA2, causal=True, pretrained=True)
if not os.path.isfile(model_path):
    print("model from Network")
    model.classifier[3] = torch.nn.Conv3d(2048, 2, (1, 1, 1))

else:
    print("model from local")
    # model = MoViNet(_C.MODEL.MoViNetA2, causal = True, pretrained = False, tf_like=False)
    model.classifier[3] = torch.nn.Conv3d(2048, 2, (1, 1, 1))

    model.load_state_dict(torch.load(model_path))

model.eval()
# model.to(device)
csamp = 0
tloss = 0
sb = 0
cap = cv2.VideoCapture(0)

with torch.no_grad():
    while True:
        if sb % num_frames == 0:
            model.clean_activation_buffers()

        ret, frame = cap.read()
        video = cv2.resize(frame, (img_size, img_size))
        video = video.reshape(3, img_size, img_size)

        video = normalize(torch.FloatTensor(video) / 255.)
        video = video.reshape([1, 3, 1, img_size, img_size])
        # output = F.log_softmax(model(video), dim=1)
        output = model(video)
        _, pred = torch.max(output, dim=1)
        print("pred:", pred)
        print("outpu:", output.shape)

        sb += 1

        frame = cv2.putText(frame, "pred:" + str(pred), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                            cv2.LINE_4)

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    csamp += pred.eq(target).sum()

cap.release()
cv2.destroyAllWindows()
