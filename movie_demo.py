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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

torch.manual_seed(97)
num_frames = 8  # 16
clip_steps = 2

Bs_Train = 2
Bs_Test = 1

N_EPOCHS = 5

cpu_num = os.cpu_count()

model_path = 'model/model.pth'
onnx_path = 'model/model.onnx'
tf_path = 'model/tf_model'
tflite_path = 'model/model.tflite'

videos_root = os.path.join(os.getcwd(), 'rgb_classes')
annotation_file = os.path.join(videos_root, 'annotations.txt')

""" DEMO 3 WITH TRANSFORMS """


# As of torchvision 0.8.0, torchvision transforms support batches of images
# of size (BATCH x CHANNELS x HEIGHT x WIDTH) and apply deterministic or random
# transformations on the batch identically on all images of the batch. Any torchvision
# transform for image augmentation can thus also be used  for video augmentation.
# preprocess = transforms.Compose([
#     ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
#     transforms.Resize(299),  # image batch, resize smaller edge to 299
#     transforms.CenterCrop(299),  # image batch, center crop to square 299x299
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])


def plot_video(rows, cols, frame_list, plot_width, plot_height, title: str):
    fig = plt.figure(figsize=(plot_width, plot_height))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     )

    for index, (ax, im) in enumerate(zip(grid, frame_list)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_title(index)
    plt.suptitle(title)
    plt.show()


def denormalize(video_tensor):
    """
        Undoes mean/standard deviation normalization, zero to one scaling,
        and channel rearrangement for a batch of images.
        args:
            video_tensor: a (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
    """
    inverse_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    return (inverse_normalize(video_tensor) * 255.).type(torch.uint8).permute(0, 2, 3, 1).numpy()


transform = transforms.Compose([
    ImglistToTensor(),
    T.Resize((200, 200)),
    T.RandomCrop((172, 172)),
    T.RandomHorizontalFlip(),
    # transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((200, 200)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    T.CenterCrop((172, 172))])

dataset = VideoFrameDataset(
    root_path=videos_root,
    annotationfile_path=annotation_file,
    num_segments=clip_steps,
    frames_per_segment=num_frames,
    imagefile_template='frame_{:012d}.jpg',
    transform=transform,
    test_mode=False
)

sample = dataset[2]
frame_tensor = sample[0]  # tensor of shape (NUM_SEGMENTS*FRAMES_PER_SEGMENT) x CHANNELS x HEIGHT x WIDTH
label = sample[1]  # integer label

print('Video Tensor Size:', frame_tensor.size())

# frame_tensor = denormalize(frame_tensor)
# plot_video(rows=1, cols=8, frame_list=frame_tensor, plot_width=15., plot_height=3.,
#            title='Evenly Sampled Frames, + Video Transform')


# """ DEMO 3 CONTINUED: DATALOADER """
# dataloader = DataLoader(
#     dataset=dataset,
#     batch_size=8,
#     shuffle=False,
#     num_workers=cpu_num,
#     pin_memory=True
# )

# for epoch in range(10):
#     for video_batch, labels in dataloader:
#         """
#         Insert Training Code Here
#         """
#         print("labels:", labels)
#         print("\nVideo Batch Tensor Size:", video_batch.size())
#         print("Batch Labels Size:", labels.size())
#         break
#     break

# 学習データ、検証データに 8:2 の割合で分割する。
train_size = int(0.8 * len(dataset))
print(len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=Bs_Train, num_workers=cpu_num, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=Bs_Test, num_workers=cpu_num, shuffle=False, pin_memory=True)

dataloaders = {'train': train_loader, 'test': test_loader}
print("the num of train dataset: {}, the num of val dataset: {}".format(len(train_dataset), len(test_dataset)))


def train_iter_stream(model, optimz, data_load, loss_val, n_clips=2, n_clip_frames=8, img_size=172):
    """
    In causal mode with stream buffer a single video is fed to the network
pp    using subclips of lenght n_clip_frames.
    n_clips*n_clip_frames should be equal to the total number of frames presents
    in the video.

    n_clips : number of clips that are used
    n_clip_frames : number of frame contained in each clip
    """
    # clean the buffer of activations
    samples = len(data_load.dataset)
    model.to(device)
    model.train()
    model.clean_activation_buffers()
    optimz.zero_grad()

    for i, (data, target) in enumerate(data_load):
        data = data.view(-1, 3, n_clip_frames * n_clips, img_size, img_size)
        data = data.to(device)
        target = target.to(device)
        l_batch = 0
        # backward pass for each clip
        for j in range(n_clips):
            output = F.log_softmax(model(data[:, :, (n_clip_frames) * (j):(n_clip_frames) * (j + 1)]), dim=1)
            loss = F.nll_loss(output, target)
            _, pred = torch.max(output, dim=1)
            loss = F.nll_loss(output, target) / n_clips
            loss.backward()
        l_batch += loss.item() * n_clips
        optimz.step()
        optimz.zero_grad()

        # clean the buffer of activations
        model.clean_activation_buffers()
        if i % 2 == 0:
            print('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(l_batch))
            loss_val.append(l_batch)


def evaluate_stream(model, data_load, best_accuracy, loss_val, n_clips=2, n_clip_frames=8, img_size=172):
    model.eval()
    model.to(device)
    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0
    with torch.no_grad():
        for data, target in tqdm(data_load):
            data = data.view(-1, 3, n_clip_frames * n_clips, img_size, img_size)

            data = data.to(device)
            target = target.to(device)
            model.clean_activation_buffers()
            for j in range(n_clips):
                output = F.log_softmax(model(data[:, :, (n_clip_frames) * (j):(n_clip_frames) * (j + 1)]), dim=1)
                loss = F.nll_loss(output, target)
            _, pred = torch.max(output, dim=1)
            tloss += loss.item()
            csamp += pred.eq(target).sum()
    if csamp > best_accuracy:
        torch.save(model.state_dict(), model_path)
        best_accuracy = csamp

    aloss = tloss / len(data_load)
    loss_val.append(aloss)
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * csamp / samples) + '%)\n')


model = MoViNet(_C.MODEL.MoViNetA2, causal=True, pretrained=True)
if not os.path.isfile(model_path):
    print("model from Network")
    model.classifier[3] = torch.nn.Conv3d(2048, 2, (1, 1, 1))
else:
    print("model from local")
    # model = MoViNet(_C.MODEL.MoViNetA2, causal = True, pretrained = False, tf_like=False)
    model.classifier[3] = torch.nn.Conv3d(2048, 2, (1, 1, 1))
    model.load_state_dict(torch.load(model_path))
start_time = time.time()

trloss_val, tsloss_val = [], []
# model.classifier[3] = torch.nn.Conv3d(2048, 51, (1,1,1))

optimz = optim.Adam(model.parameters(), lr=0.00005)

best_accuracy = 0

for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:', epoch)
    train_iter_stream(model, optimz, train_loader, trloss_val, n_clips=clip_steps, n_clip_frames=num_frames)
    evaluate_stream(model, test_loader, best_accuracy, tsloss_val, n_clips=clip_steps, n_clip_frames=num_frames)
print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')

