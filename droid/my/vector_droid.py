import threading
from functools import partial
from typing import List, Optional, Tuple

import lietorch
import numpy as np
import torch
import time

import torch.multiprocessing as mp
from torch.multiprocessing.pool import Pool

from droid.depth_video import DepthVideo
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
        self.duration_total_async = 0
        self.duration_frontend = 0
        self.pool = None

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
            self.droids = [Droid(self.net, self.args, idx) for idx in range(self.num_instances)]
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

        for i in it():
            self.droids[i].activated = True

        corrs = torch.stack([filterx(i).corr(fmaps1[i]) for i in it()]).squeeze(1).half()
        nets0 = torch.stack([getattr(filterx(i), 'net', nets1[i]) for i in it()]).half()
        inps0 = torch.stack([getattr(filterx(i), 'inp', inps1[i]) for i in it()]).half()

        with torch.no_grad():
            assert nets0.ndim == 5
            _, deltas, weights = self.net.update(nets0, inps0, corrs)

        t0 = time.time()
        added = self.add_keyframes_if_meets_conditions(images, tstamp, fmaps1, nets1, inps1, deltas, weights)
        t1 = time.time()
        self.frontend(added)
        t2 = time.time()
        poses, points = self.collect_status()
        t3 = time.time()

        duration_frontend = t2 - t1
        duration_total_async = t3 - t0
        self.duration_frontend += duration_frontend
        self.duration_total_async += duration_total_async

        print(f'Track durations::\n'
              f' - duration_frontend {duration_frontend:.2f}s [cum {self.duration_frontend:.2f}s]]\n'
              # f' - duration_total_async {duration_total_async:.2f}s [cum {self.duration_total_async:.2f}s]'
              )

        log_mem(tstamp)

        return poses, points

    def collect_status(self):
        poses = []
        points = []

        for d in self.droids:
            if not d.activated:
                continue

            poses.append(d.get_poses())
            points.append(d.get_points())

        return poses, points

    def add_keyframes_if_meets_conditions(self, images, tstamp, fmaps1, nets1, inps1, deltas, weights):
        addeds = []
        for i in range(len(images)):
            filterx = self.droids[i].filterx
            feats = fmaps1[i], nets1[i], inps1[i]
            corr_vars = deltas[i], weights[i]
            added = filterx.add_keyframe_if_meets_condition(images[i], tstamp, *feats, *corr_vars)
            addeds.append(added)

        return addeds

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

    def frontend(self, was_frame_added__vector):
        frontends = [d.frontend for d in self.droids if d.activated]
        data = list(enumerate(was_frame_added__vector))

        mode = 'sequential'

        if mode == 'sequential':
            for item in data:
                idx, added = item
                if added:
                    with torch.no_grad():
                        frontends[idx]()

        if mode == 'threads':
            streams = [torch.cuda.Stream(self.device) for _ in range(10)]
            results = [None] * 10

            def worker(idx: int, added: bool):
                print(f'HELLO from thread #{idx}')
                if added:
                    # print(f'WORKING #{idx}')
                    with torch.no_grad(), torch.cuda.stream(streams[idx]):
                        results[idx] = self.droids[idx].frontend()
                torch.cuda.empty_cache()
                # print(f'FINISHED from thread #{idx}')

            # launch 10 threads on 10 streams
            threads = []
            for item in data:
                t = threading.Thread(target=worker, args=(*item,), daemon=True)
                t.start()
                threads.append(t)
            for t in threads:
                t.join()

        if mode == 'mp':
            print()
            if self.pool is None:
                ctx = mp.get_context("spawn")  # always spawn
                self.pool = ctx.Pool(
                    processes=len(frontends),
                    initializer=init__frontend,     # ← no lambda
                    initargs=(frontends,)           # sent only once
            )

            self.pool.map(worker__frontend, data)


__frontends = []
def init__frontend(frontends):
    global __frontends
    __frontends = frontends


def worker__frontend(item):
    idx, added = item
    if added:
        with torch.no_grad():
            __frontends[idx]()          # uses its own CUDA buffers
        # torch.cuda.empty_cache()        # free scratch space


def log_mem(step: int):
    device = 'cuda:0'
    torch.cuda.synchronize(device)   # wait for all streams/kernels
    used = torch.cuda.memory_allocated(device) / 2**30
    resv = torch.cuda.memory_reserved(device)  / 2**30
    print(f"[Step {step:2d}]  allocated={used:5.2f} GiB   reserved={resv:5.2f} GiB")


def extract_intrinsics_from_proj_matrix(P, width, height):
    fx = P[0, 0] * (width / 2)
    fy = P[1, 1] * (height / 2)
    cx = (P[0, 2] + 1) * (width / 2)
    cy = (P[1, 2] + 1) * (height / 2)
    return torch.as_tensor([fx, fy, cx, cy])

