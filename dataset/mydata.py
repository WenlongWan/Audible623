# coding=utf-8
import os
import glob
import cv2
import pandas as pd
import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from dataset.data_augment import create_my_augment, flip
from dataset.op_flow import load_flow_to_numpy

def get_videos_dict(csv_path, select_class):
    df = pd.read_csv(csv_path, header=0)
    filenames = df["video_id"]
    length = len(filenames)

    dataset = []
    for k in range(length):
        row = df.loc[k, :]
        video_id = row["video_id"]
        label_str = row["sound_label"]
        class_name = row["class"]
        if video_id.startswith('v_'):
            class_name = 'ucf50'

        gt = list(map(int, label_str.split(',')))
        if (select_class is not None) and (select_class not in "all"):
            if (class_name not in select_class):
                continue
        # if (class_name in 'battlerope'):
        #     continue
        if len(gt) > 0 and gt[0] != -1 and gt[0] != -2:
            video_path = class_name + '/' + video_id
            data = {"video": video_id, "video_class": class_name, "label": gt}
            dataset.append(data)
    return dataset

def label_preprocess(gt_label, video_length):
    frame_label = np.zeros(video_length)
    for i in range(video_length):
        if i in gt_label:
            frame_label[i] = 1
    frame_label = torch.from_numpy(frame_label).float()
    return frame_label

def read_video(video_path, imsize):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    if cap.isOpened():
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (imsize, imsize))
            frame = transforms.ToTensor()(frame_rgb)
            frames.append(frame)
    frame_tensor = torch.as_tensor(np.stack(frames)) 
    return frame_tensor, fps

def read_flow(datapath, imsize, bwd = False):
    """Read flow from .flo file."""
    if bwd:
        flow_files = glob.glob(os.path.join(datapath, '*flow_bwd.png'))
    else:
        flow_files = glob.glob(os.path.join(datapath, '*flow.png'))
    flow_files.sort()
    flows = []
    for file_path in flow_files:
        flow_np = cv2.imread(file_path)
        flow_np = cv2.cvtColor(flow_np, cv2.COLOR_BGR2RGB)
        flow_np = cv2.resize(flow_np, (imsize, imsize))
        flow = transforms.ToTensor()(flow_np)
        flows.append(flow)
    flows = torch.as_tensor(np.stack(flows)) 
    return flows

# Read Data From Split Frame(Image)
class MyData(torch.utils.data.Dataset):
    def __init__(self, root_path, video_path, label_path, num_frame, img_size, aug=False):
        self.label_path = os.path.join(root_path, label_path)
        self.data_path = os.path.join(root_path, video_path)
        self.dataset = get_videos_dict(self.label_path, None)
        self.clip = True if 'train' in label_path else False
        self.aug = aug

        self.num_frame = num_frame
        self.img_size = img_size
        self.max_len = 300
        self.augment = create_my_augment()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        video_id, video_class = self.dataset[index]["video"], self.dataset[index]["video_class"]
        gt = self.dataset[index]["label"]
        video_file = os.path.join(self.data_path, video_class, video_id)
        flow_file = os.path.join(self.data_path, video_id).replace('video', 'flow')

        video, fps = read_video(video_file + '.mp4', self.img_size)    # [n, 3, 224, 224]
        video = video[0: self.max_len]
        label = label_preprocess(gt, len(video))
        
        flow = read_flow(flow_file, self.img_size)
        bwd_flow = read_flow(flow_file, self.img_size, bwd = True)
        video, diff, d2f, label = self.cal_diff(video, flow, bwd_flow, label)
        if self.clip:
            video_clip, diff_clip, d2f_clip, label_clip = self.batch_select(video, diff, d2f, label, self.num_frame)
            return video_clip, diff_clip, d2f_clip, label_clip
        return video, diff, d2f, label

    def batch_select(self, video, diff, d2f, label, num_frame):
        max_len = len(video)
        gt = random.choice(label.nonzero())
        lidx = random.randint(max(0, gt - num_frame + 1), gt)   # +1
        ridx = min(lidx + num_frame, max_len)

        seq_len = ridx - lidx
        video_clip = video[lidx: ridx]
        diff_clip = diff[lidx: ridx]
        d2f_clip = d2f[lidx: ridx]
        label_clip = label[lidx: ridx]

        if (ridx - lidx) < num_frame:
            video_clip = torch.cat((video_clip, video[-1:].repeat(num_frame - seq_len, 1, 1, 1)), dim = 0)
            diff_clip = torch.cat((diff_clip, diff[-1:].repeat(num_frame - seq_len, 1, 1, 1)), dim = 0)
            d2f_clip = torch.cat((d2f_clip, d2f[-1:].repeat(num_frame - seq_len, 1, 1, 1)), dim = 0)
            label_clip = torch.cat((label_clip, label[-1:].repeat(num_frame - seq_len)), dim = 0)
        return video_clip, diff_clip, d2f_clip, label_clip

    def cal_diff(self, video, flow_fwd, flow_bwd, label):
        st_diff = torch.cat((flow_fwd[1:].contiguous(), flow_bwd[:-1].contiguous()), dim = 1)  # [diff[:-1], 1-diff[1:]]
        flow_fwd_diff = (flow_fwd[1:] - flow_fwd[:-1])  # (-1~1) [F-2]
        flow_bwd_diff = (flow_bwd[:-1] - flow_bwd[1:])
        st_d2f = torch.cat((flow_fwd_diff[2:].contiguous(), flow_bwd_diff[:-2].contiguous()), dim = 1)
        return video[2:-2], st_diff[1:-1], st_d2f, label[2:-2]
