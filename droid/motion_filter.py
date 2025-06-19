import cv2
import torch
import lietorch

from collections import OrderedDict

from droid.depth_video import DepthVideo

from .geom import projective_ops as pops
from .modules.corr import CorrBlock

from functools import partial

if torch.__version__.startswith("2"):
    autocast = partial(torch.autocast, device_type="cuda")
else:
    autocast = torch.cuda.amp.autocast


class MotionFilter:
    """ This class is used to filter incoming frames and extract features """

    def __init__(self, net, video: DepthVideo, thresh=2.5, device="cuda"):
        # split net modules
        self.cnet = net.cnet
        self.fnet = net.fnet
        self.update = net.update

        self.video = video
        self.thresh = thresh
        self.device = device

        self.count = 0

        # mean, std for image normalization
        self.MEAN = torch.as_tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        self.STDV = torch.as_tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]

        self.added = 0
        self.coords0 = None

    @autocast(enabled=True)
    def __context_encoder(self, image):
        """ context features """
        net, inp = self.cnet(image).split([128,128], dim=2)
        return net.tanh().squeeze(0), inp.relu().squeeze(0)

    @autocast(enabled=True)
    def __feature_encoder(self, image):
        """ features for correlation volume """
        return self.fnet(image).squeeze(0)

    @autocast(enabled=True)
    @torch.no_grad()
    def prepare_image(self, image):
        if len(image.shape) == 3:
            inputs = image[None, None, [2, 1, 0]]  # B S C H W
        elif len(image.shape) == 4:
            inputs = image[None, :, [2,1,0]]  # B S C H W

        # normalize images
        inputs = inputs / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)
        return inputs

    @autocast(enabled=True)
    @torch.no_grad()
    def prepare_images(self, images):
        inputs = images[:, None, [2, 1, 0]]  # B S C H W
        # normalize images
        inputs = inputs / 255.0
        inputs = inputs.sub_(self.MEAN).div_(self.STDV)
        return inputs

    @autocast(enabled=True)
    @torch.no_grad()
    def corr(self, fmap1):
        fmap0 = self.fmap if hasattr(self, 'fmap') else fmap1
        return CorrBlock(fmap0[None,[0]], fmap1[None,[0]])(self.coords0)

    @autocast(enabled=True)
    @torch.no_grad()
    def prepare_for_image_size(self, w, h, intrinsics):
        if self.coords0 is None:
            ht = h // 8
            wd = w // 8
            self.coords0 = pops.coords_grid(ht, wd, device=self.device)[None, None]
        self.intrinsics = intrinsics

    @autocast(enabled=True)
    @torch.no_grad()
    def add_keyframe_if_meets_condition(self, image, tstamp, fmap1, net1, inp1, delta, weight):
        if self.video.counter == 0:
            self.net, self.inp, self.fmap = net1, inp1, fmap1
            Id = lietorch.SE3.Identity(1, ).data.squeeze()
            i = self.video.append(tstamp, image, Id, 1.0, None,
                                  self.intrinsics / 8.0, fmap1, net1, inp1)
            self.added += 1
            return True

        if delta.norm(dim=-1).mean().item() <= self.thresh:
            self.count += 1
            return False

        self.count = 0
        self.net, self.inp, self.fmap = net1, inp1, fmap1
        self.video.append(tstamp, image, None, None, None,
                          self.intrinsics / 8.0, fmap1, net1, inp1)
        self.added += 1
        return True

    @autocast(enabled=True)
    @torch.no_grad()
    def track(self, tstamp, image: torch.Tensor, depth=None, intrinsics=None) -> bool:
        """ main update operation - run on every frame in video """
        # returns True if frame was added
        if image.shape[-2]%8 > 0 or image.shape[-1]%8 > 0:
            raise ValueError('Image dimension must be divisible by 8')
        assert image.get_device() == 0, 'image tensor must be on device cuda:0'

        inputs = self.prepare_image(image)

        # extract features
        fmap1 = self.__feature_encoder(inputs)

        ### always add first frame ###
        if self.video.counter == 0:
            net1, inp1 = self.__context_encoder(inputs[:,[0]])
            self.net, self.inp, self.fmap = net1, inp1, fmap1
            Id = lietorch.SE3.Identity(1, ).data.squeeze().to(self.device)
            i = self.video.append(tstamp, image[0], Id, 1.0, depth, intrinsics / 8.0, fmap1, net1[0], inp1[0])
            self.added += 1
            return True

        ### only add new frame if there is enough of motion ###
        # index correlation volume

        corr = self.corr(fmap1)

        # approximate flow magnitude using 1 update iteration
        _, delta, weight = self.update(self.net[None], self.inp[None], corr)

        # check motion magnitue / add new frame to video
        if delta.norm(dim=-1).mean().item() <= self.thresh:
            self.count += 1
            return False

        self.count = 0
        net1, inp1 = self.__context_encoder(inputs[:,[0]])
        self.net, self.inp, self.fmap = net1, inp1, fmap1
        self.video.append(tstamp, image[0], None, None, depth, intrinsics / 8.0, fmap1, net1[0], inp1[0])
        self.added += 1
        return True
