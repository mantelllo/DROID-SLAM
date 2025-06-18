from functools import partial
from typing import List, Optional, Tuple

import lietorch
import numpy as np
import torch

from droid.droid import Droid
from droid.droid_net import DroidNet

if torch.__version__.startswith("2"):
    autocast = partial(torch.autocast, device_type="cuda")
else:
    autocast = torch.cuda.amp.autocast



class VectorDroid:
    num_instances: int = None
    net: DroidNet = None
    droids: List[Droid] = None
    K = None
    intrinsics = None

    def __init__(self, num_instances: int, args, intrinsics):
        self.num_instances = num_instances
        self.args = args
        self.net = DroidNet.load(args.path)
        self.device = args.backend_device
        self.droids = []
        self.intrinsics = intrinsics  # camera's intrinsics: [fx, fy, cx, cy]

    @autocast(enabled=True)
    def track(self, tstamp, images) -> Tuple[List[lietorch.SE3], List[torch.Tensor]]:
        import gc; gc.collect()
        torch.cuda.empty_cache()

        images = images.to(self.device)
        assert images.ndim == 4  # B C H W

        if images.shape[-1] == 3:  # ensure C is 2nd
            images = images.permute(0, 3, 1, 2)

        # prep
        w, h = self.args.image_size = [images.shape[2], images.shape[3]]
        if not self.droids:
            self.droids = [Droid(self.net, self.args) for _ in range(self.num_instances)]
        [d.filterx.prepare_for_image_size(w, h, self.intrinsics) for d in self.droids]

        filterx = self.droids[0].filterx

        # torch.Size([2, 1, 3, 320, 480])
        inputs = filterx.prepare_images(images)

        with torch.no_grad():
            fmaps1 = self.net.fnet(inputs).half()
            nets1, inps1 = self._encode_context(inputs)

        # iterating over images instead of droid instances allows having many droid instances, but keep active less
        it = partial(range, 0, len(images))
        filterx = lambda i: self.droids[i].filterx

        corrs = torch.stack([filterx(i).corr(fmaps1[i]) for i in it()]).squeeze(1).half()
        nets0 = torch.stack([getattr(filterx(i), 'net', nets1[i]) for i in it()]).half()
        inps0 = torch.stack([getattr(filterx(i), 'inp', inps1[i]) for i in it()]).half()

        with torch.no_grad():
            assert nets0.ndim == 5
            _, deltas, weights = self.net.update(nets0, inps0, corrs)

        for i in range(len(images)):
            if tstamp == 7 and i == 0:
                print(' inside: ', i)
            feats = fmaps1[i], nets1[i], inps1[i]
            corr_vars = deltas[i], weights[i]
            image = images[i]

            filterx = self.droids[i].filterx
            added = filterx.add_keyframe_if_meets_condition(image, tstamp, *feats, *corr_vars)
            if added:
                with torch.no_grad():
                    self.droids[i].frontend()

        poses = [self.droids[i].get_poses() for i in it()]
        points = [self.droids[i].points() for i in it()]
        return poses, points

    def reset_terminated_envs(self, termination_tensor):
        if any(termination_tensor):
            termination_tensor = termination_tensor.cpu().numpy().astype(bool)
            for i, terminated in enumerate(termination_tensor):
                if not terminated:
                    continue

                if self.droids and self.droids[i] is not None:
                    print(f'RESETTING env #{i}')
                    self.droids[i] = Droid(self.net, self.args)

    def _encode_context(self, inputs):
        net, inp = self.net.cnet(inputs).split([128,128], dim=2)
        return net.tanh().half(), inp.relu().half()

    def ba(self, steps=3):
        for droid in self.droids:
            droid.backend(steps)

    def terminate(self):
        raise NotImplementedError()

    def __len__(self):
        return self.num_instances


def extract_intrinsics_from_proj_matrix(P, width, height):
    fx = P[0, 0] * (width / 2)
    fy = P[1, 1] * (height / 2)
    cx = (P[0, 2] + 1) * (width / 2)
    cy = (P[1, 2] + 1) * (height / 2)
    return torch.as_tensor([fx, fy, cx, cy])

