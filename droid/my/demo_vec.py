import multiprocessing as mp
import torch.multiprocessing as tmp

for ctx in (mp, tmp):          # stdlib and Torch wrapper
    try:
        ctx.set_start_method("spawn", force=True)
    except RuntimeError:
        pass                   # start-method was already set


import sys
import numpy as np
import open3d as o3d
from tqdm import tqdm
import torch
from scipy.spatial.transform import Rotation as R
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import cv2
import os

import time
import argparse
from droid.my.vector_droid import VectorDroid




def main():
    args = get_args()
    if args.asynchronous:
        raise NotImplementedError

    args.stereo = False
    args.disable_vis = False
    # torch.multiprocessing.set_start_method('fork', force=True)
    torch.autograd.set_grad_enabled(False)

    num_instances = 5
    droidvec = VectorDroid(num_instances, args, None)

    poses = []

    from gui_viewer import start_viewer
    # viewer = start_viewer()

    for (t, image, intrinsics) in tqdm(image_stream(args.imagedir, args.calib, args.stride)):
        images = image.repeat((len(droidvec),1,1,1))
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(images[0])

        if droidvec.intrinsics is None:
            droidvec.intrinsics = intrinsics

        poses, points = droidvec.track(t, images)
        pcd = points[0]
        # if poses is not None and len(poses[0]) > 0 and t > 12:
        #     viewer.update(points[0], poses[0])

        if t % 10:
            print(f'[{t}] lengths:::', len(points[0]))

        if t == 200:
            from vispy import scene, app
            app.run()
            np.save('bench.npy', pcd)

    t0 = time.time()
    # traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride))
    t1 = time.time()
    duration = t1 - t0
    print('duration of factor graph optimization:', duration)
    # traj_est = droid.terminate()


def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, calib, stride):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]
    print('loading images', len(image_list))

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        yield t, image[None], intrinsics


def save_reconstruction(droid, save_path):

    if hasattr(droid, "video2"):
        video = droid.video2
    else:
        video = droid.video

    t = video.counter
    save_data = {
        "tstamps": video.tstamp[:t].cpu(),
        "images": video.images[:t].cpu(),
        "disps": video.disps_up[:t].cpu(),
        "poses": video.poses[:t].cpu(),
        "intrinsics": video.intrinsics[:t].cpu()
    }

    torch.save(save_data, save_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--path", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true", default=False)

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4,
                        help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0,
                        help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--asynchronous", action="store_true")
    parser.add_argument("--frontend_device", type=str, default="cuda")
    parser.add_argument("--backend_device", type=str, default="cuda")

    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    return parser.parse_args()


if __name__ == '__main__':
    main()
