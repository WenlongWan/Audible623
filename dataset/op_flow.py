import cv2
import os
import numpy as np
import glob
import multiprocessing
from matplotlib.colors import hsv_to_rgb

###### 使用frames帧进行 flow光流计算
video_root = 'video_list.txt'
root = 'frames'
out_root = 'flow'


def cal_for_frames(video_path):
    # print(video_path)
    frames = glob.glob(os.path.join(video_path, '*.jpg'))
    frames.sort()

    flow = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(frames[1:]):
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr

    return flow


def compute_TVL1(prev, curr, bound=15):
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)

    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow


def save_flow(video_flows, flow_path):
    if not os.path.exists(flow_path):
        os.mkdir(os.path.join(flow_path))
    for i, flow in enumerate(video_flows):
        cv2.imwrite(os.path.join(flow_path, str(i) + '_x.jpg'), flow[:, :, 0])
        cv2.imwrite(os.path.join(flow_path, str(i) + '_y.jpg'), flow[:, :, 1])


def process(video_path, flow_path):
    flow = cal_for_frames(video_path)
    save_flow(flow, flow_path)


def extract_flow(root, out_root):
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    # dir_list = os.listdir(root)
    dir_list = []
    ### 读取txt中视频信息
    with open(video_root, 'r') as f:
        for id, line in enumerate(f):
            video_name = line.strip().split()
            preffix = video_name[0].split('.')[0]
            dir_list.append(preffix)

    pool = multiprocessing.Pool(processes=4)
    for dir_name in dir_list:
        video_path = os.path.join(root, dir_name)
        flow_path = os.path.join(out_root, dir_name)

        # flow = cal_for_frames(video_path)
        # save_flow(flow,flow_path)
        # print('save flow data: ',flow_path)
        # process(video_path,flow_path)
        pool.apply_async(process, args=(video_path, flow_path))

    pool.close()
    pool.join()

def load_flow_to_numpy(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape testdata into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

def flow_to_image(flow, max_flow=256):
    if max_flow is not None:
        max_flow = max(max_flow, 1.)
    else:
        max_flow = np.max(flow)

    n = 8
    u, v = flow[:, :, 0], flow[:, :, 1]
    mag = np.sqrt(np.square(u) + np.square(v))
    angle = np.arctan2(v, u)
    im_h = np.mod(angle / (2 * np.pi) + 1, 1)
    im_s = np.clip(mag * n / max_flow, a_min=0, a_max=1)
    im_v = np.clip(n - im_s, a_min=0, a_max=1)
    im = hsv_to_rgb(np.stack([im_h, im_s, im_v], 2))
    return (im * 255).astype(np.uint8)

if __name__ == '__main__':
    extract_flow(root, out_root)
    print("finish")