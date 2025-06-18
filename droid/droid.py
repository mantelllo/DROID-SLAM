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
    def __init__(self, droidnet: DroidNet, args):
        super(Droid, self).__init__()
        self.net = droidnet
        self.args = args
        self.disable_vis = args.disable_vis

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
            self.visualizer = Process(target=visualization_fn, args=(self.video, None))
            self.visualizer.start()

        # post processor - fill in poses for non-keyframes
        self.traj_filler = PoseTrajectoryFiller(self.net, self.video)

    def track(self, tstamp, image, depth=None, intrinsics=None) -> Tuple[Optional[lietorch.SE3], Optional[torch.Tensor]]:
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            image = image.to(self.filterx.device)
            new_keyframe = self.filterx.track(tstamp, image, depth, intrinsics)
            if new_keyframe:
                # local bundle adjustment
                self.frontend()

        N = self.video.counter.value
        if N == 0:
            return None, None

        poses = self.get_poses()
        points = self.points()
        return poses, points

    def get_poses(self):
        N = self.video.counter.value
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

    def points(self):
        t = self.video.counter.value
        # if t > 15:
        #     print()

        poses = self.video.poses[:t].contiguous().cuda()
        images = self.video.images[:t, :, 4::8, 4::8].contiguous().cuda()
        disps = self.video.disps[:t].contiguous().cuda()
        intrinsics = self.video.intrinsics.contiguous().cuda()

        points = droid_backends.iproj(lietorch.SE3(poses).inv().data, disps, intrinsics[0]).reshape(-1, 3)
        colors = (images[:, [0,1,2]].permute(0, 2, 3, 1) / 255.0).reshape(-1, 3)

        filter_threshold = 0.01
        filter_count = 2

        index = torch.arange(t, device="cuda")
        thresh = filter_threshold * torch.ones_like(disps.mean(dim=[1, 2]))

        counts = droid_backends.depth_filter(
            poses, disps, intrinsics[0], index, thresh
        )
        mask = (counts >= filter_count) & (disps > 0.25 * disps.mean())
        valid = mask.flatten()

        # print('Points from ME:', points[17])

        points = torch.cat([points, colors], dim=1)
        return points[valid].contiguous()
