import numpy as np
import torch
import lietorch
import droid_backends
from hashlib import sha256


from droid.droid_net import cvx_upsample
from .geom import projective_ops as pops

class DepthVideo:
    def __init__(self, image_size=[480, 640], buffer=1024, stereo=False, device="cuda:0"):
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]
        self.buffer  = buffer
        self.stereo  = stereo
        self.device  = device

        # current keyframe count
        self._counter = torch.zeros(1, dtype=torch.int32, device='cpu')
        self._ready   = torch.zeros(1, dtype=torch.int32, device='cpu')

        self.ht_div8 = self.special_div8(ht)
        self.wd_div8 = self.special_div8(wd)

        ### state attributes ###
        self.tstamp = torch.zeros(buffer, device=device, dtype=torch.float).share_memory_()
        self.images = torch.zeros(buffer, 3, ht, wd, device=device, dtype=torch.uint8).share_memory_()
        self.dirty = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()
        self.red = torch.zeros(buffer, device=device, dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(buffer, 7, device=device, dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht//8, wd//8, device=device, dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(buffer, ht//8, wd//8, device=device, dtype=torch.float).share_memory_()
        # self.disps_up = torch.zeros(buffer, ht, wd, device=device, dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device=device, dtype=torch.float).share_memory_()

        self.stereo = stereo
        c = 1 if not self.stereo else 2

        self.fmaps = torch.zeros(buffer, c, 128, self.ht_div8, self.wd_div8, dtype=torch.half, device=device).share_memory_()
        self.nets = torch.zeros(buffer, 128, self.ht_div8, self.wd_div8, dtype=torch.half, device=device).share_memory_()
        self.inps = torch.zeros(buffer, 128, self.ht_div8, self.wd_div8, dtype=torch.half, device=device).share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device=device).share_memory_()

    def special_div8(self, v):
        return v // 8

    def to(self, device="cuda"):
        self.tstamp = self.tstamp.to(device=device)
        self.images = self.images.to(device=device)
        self.dirty = self.dirty.to(device=device)
        self.red = self.red.to(device=device)
        self.poses = self.poses.to(device=device)
        self.disps = self.disps.to(device=device)
        self.disps_sens = self.disps_sens.to(device=device)
        self.disps_up = self.disps_up.to(device=device)
        self.intrinsics = self.intrinsics.to(device=device)

        self.fmaps = self.fmaps.to(device=device)
        self.nets = self.nets.to(device=device)
        self.inps = self.inps.to(device=device)

        return self

    def __del__(self):
        # delete all tensors
        if hasattr(self, "tstamp"):
            del self.tstamp
            del self.images
            del self.dirty
            del self.red
            del self.poses
            del self.disps
            del self.disps_sens
            # del self.disps_up
            del self.intrinsics
            del self.fmaps
            del self.nets
            del self.inps

    # def get_lock(self):
    #     return self.lock

    def __item_setter(self, index, item):
        item = list(item)

        # if index == 5:
        #     repr_ = [sha256(item_.cpu().numpy().tobytes()).hexdigest()
        #              if isinstance(item_, torch.Tensor) else item_ for item_ in item]

        if item[7].ndim == 4:
            if item[7].shape[0] != 1:
                raise ValueError
            item[7] = item[7].squeeze(0)

        if item[8].ndim == 4:
            if item[8].shape[0] != 1:
                raise ValueError
            item[8] = item[8].squeeze(0)

        assert item[1].shape == self.images.shape[1:]
        assert item[2] is None or item[2].shape == self.poses.shape[1:]
        assert item[3] is None or isinstance(item[3], float) or item[3].shape == self.disps.shape[1:]
        assert item[4] is None or item[4].shape == self.disps_sens.shape[1:]
        assert item[5].shape == self.intrinsics.shape[1:]
        assert item[6].shape == self.fmaps.shape[1:]
        assert item[7].shape == self.nets.shape[1:]
        assert item[8].shape == self.inps.shape[1:]

        if isinstance(index, int) and index >= self.counter:
            self.counter = index + 1

        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter:
            self.counter = index.max().item() + 1

        # self.dirty[index] = True
        self.tstamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            depth = item[4][3::8,3::8].cuda()
            self.disps_sens[index] = torch.where(depth>0, 1.0/depth, depth)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            self.nets[index] = item[7]

        if len(item) > 8:
            self.inps[index] = item[8]

    def __setitem__(self, index, item):
        # with self.get_lock():
        self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        # with self.get_lock():

        # support negative indexing
        if isinstance(index, int) and index < 0:
            index = self.counter + index

        item = (
            self.poses[index],
            self.disps[index],
            self.intrinsics[index],
            self.fmaps[index],
            self.nets[index],
            self.inps[index])

        return item

    def append(self, *item):
        # with self.get_lock():
        self.__item_setter(self.counter, item)
        return self.counter

    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def normalize(self):
        """ normalize depth and poses """

        with self.get_lock():
            s = self.disps[:self.counter].mean()
            self.disps[:self.counter] /= s
            self.poses[:self.counter,:3] *= s
            self.dirty[:self.counter] = True

    def remove(self, ix):
        t = self.counter
        # with self.video.get_lock():
        self.images[ix : t - 1] = self.images[ix + 1 : t].clone()
        self.poses[ix : t - 1] = self.poses[ix + 1 : t].clone()
        self.disps[ix : t - 1] = self.disps[ix + 1 : t].clone()
        self.disps_sens[ix : t - 1] = self.disps_sens[ix + 1 : t].clone()
        self.intrinsics[ix : t - 1] = self.intrinsics[ix + 1 : t].clone()

        self.nets[ix : t - 1] = self.nets[ix + 1 : t].clone()
        self.inps[ix : t - 1] = self.inps[ix + 1 : t].clone()
        self.fmaps[ix : t - 1] = self.fmaps[ix + 1 : t].clone()
        self.tstamp[ix: t - 1] = self.tstamp[ix + 1 : t].clone()

    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj)

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
        
        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False):
        """ dense bundle adjustment (DBA) """
        # [t0, t1] window of bundle adjustment optimization
        if t1 is None:
            t1 = max(ii.max().item(), jj.max().item()) + 1

        droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.disps_sens,
            target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only)

        self.disps.clamp_(min=0.001)

    @property
    def counter(self):
        return self._counter[0].item()

    @counter.setter
    def counter(self, value):
        self._counter[0] = value

    @property
    def ready(self):
        return self._counter[0].item()

    @ready.setter
    def ready(self, value):
        self._ready[0] = value

    def get_poses(self):
        N = self.counter
        return self.poses[:N].clone()  # meant to be used by lib client

    def get_points(self):
        t = self.counter
        # if t > 15:
        #     print()

        poses = self.poses[:t].contiguous().cuda()
        images = self.images[:t, :, 4::8, 4::8].contiguous().cuda()
        disps = self.disps[:t].contiguous().cuda()
        intrinsics = self.intrinsics.contiguous().cuda()

        points = droid_backends.iproj(lietorch.SE3(poses).inv().data, disps, intrinsics[0]).reshape(-1, 3)
        colors = (images[:, [0, 1, 2]].permute(0, 2, 3, 1) / 255.0).reshape(-1, 3)

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
        return points[valid].contiguous().clone() # meant to be used by lib client
