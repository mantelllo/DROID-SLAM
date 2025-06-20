from typing import Optional, Tuple

import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import lietorch
import droid_backends
import numpy as np

from droid.droid_net import DroidNet
from droid.depth_video import DepthVideo
from droid.motion_filter import MotionFilter
from droid.droid_frontend import DroidFrontend
from droid.droid_backend import DroidBackend
from droid.trajectory_filler import PoseTrajectoryFiller

from collections import OrderedDict
from torch.multiprocessing import Process


class Droid:
    def __init__(self, droidnet: DroidNet, args, worker_idx: int = None):
        super(Droid, self).__init__()
        self.net = droidnet
        self.args = args
        self.disable_vis = bool(worker_idx > 0) or args.disable_vis

        # store images, depth, poses, intrinsics (shared between processes)
        self.video = DepthVideo(args.image_size, args.buffer, stereo=args.stereo)

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)
        
        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)

        # visualizer
        if not self.disable_vis:
            from .visualizer.droid_visualizer import visualization_fn
            self.visualizer = Process(
                target=visualization_fn, args=(self.video, None),)
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)

        self.activated = False

    def track(self, tstamp, image, depth=None, intrinsics=None) -> Tuple[Optional[lietorch.SE3], Optional[torch.Tensor]]:
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            image = image.to(self.filterx.device)
            new_keyframe = self.filterx.track(tstamp, image, depth, intrinsics)
            if new_keyframe:
                # local bundle adjustment
                self.frontend()

        N = self.video.counter
        if N == 0:
            return None, None

        poses = self.get_poses()
        points = self.get_points()

        self.activated = True

        return poses, points

    def get_poses(self):
        N = self.video.counter
        return self.video.poses[:N]

    def terminate(self, stream=None):
        """ terminate the visualization process, return poses [t, q] """

        del self.frontend

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(7)

        torch.cuda.empty_cache()
        print("#" * 32)
        self.backend(12)

        camera_trajectory = self.traj_filler(stream)
        return camera_trajectory.inv().data.cpu().numpy()

    def get_points(self):
        return self.video.get_points()
