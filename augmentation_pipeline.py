import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from numpy.random import RandomState


class Flip(nn.Module):

    def __init__(self):

        super(Flip, self).__init__()

    def forward(self, im_data, data):

        boxes = data[1].clone().numpy()
        oldx1 = boxes[:, :, 0].copy()
        oldx2 = boxes[:, :, 2].copy()

        bs, h, w, _ = im_data.size()
        boxes[:, :, 0] = w - oldx2 - 1
        boxes[:, :, 2] = w - oldx1 - 1
        assert (boxes[:, :, 2] >= boxes[:, :, 0]).all()

        boxes = torch.from_numpy(boxes)

        return torch.flip(im_data, [3]), [data[0], boxes, data[2]]


class RandomCutout(nn.Module):

    def __init__(self):

        self.length = 60
        super(RandomCutout, self).__init__()

    def forward(self, im_data, data):

        bs, c, h, w = im_data.size()

        mask = np.ones((bs, c, h, w), np.float32)
        prng = RandomState()

        for n in range(bs):

            y = prng.randint(h)
            x = prng.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[n, :, y1: y2, x1: x2] = 0.0

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(im_data).cuda().float()

        im_data = im_data*mask
        mask.detach().cpu()

        return im_data, data


class GaussianNoise(nn.Module):

    def __init__(self):

        self.mu = 0.0
        self.sigma = 0.1
        super(GaussianNoise, self).__init__()

    def forward(self, im_data, data):

        bs, c, h, w = im_data.size()

        prng = RandomState()
        noise = prng.normal(self.mu, self.sigma, (bs, c, h, w))

        noise = torch.from_numpy(noise)
        noise = noise.expand_as(im_data).cuda().float()
        im_data = im_data + noise
        noise.detach().cpu()

        return im_data, data


class Jitter(nn.Module):

    def __init__(self):

        self.range = [-0.1, 0.1]
        super(Jitter, self).__init__()

    def forward(self, im_data, data):

        bs, c, h, w = im_data.size()

        prng = RandomState()
        value = prng.uniform(self.range[0], self.range[1], (bs, 1, 1, 1))

        value = torch.from_numpy(value)
        value = value.repeat(1,c,h,w)
        value = value.cuda().float()
        im_data = im_data + value
        value.detach().cpu()

        return im_data, data


class AvengersAssemble(nn.Module):

    def __init__(self):

        super(AvengersAssemble, self).__init__()

        self.flip = Flip()
        self.random_cutout = RandomCutout()
        self.gaussian_noise = GaussianNoise()
        self.jitter = Jitter()

    def forward(self, im_data, data):

        prng = RandomState()
        use_flip = np.random.choice([True, False], 1, 0.5)

        if use_flip:
            im_data, data = self.flip(im_data, data)

        prng = RandomState()
        use_cutout = np.random.choice([True, False], 1, [0.4, 0.6])

        if use_cutout:
            im_data, data = self.random_cutout(im_data, data)

        prng = RandomState()
        use_jitter = np.random.choice([True, False], 1, [0.4, 0.6])

        if use_jitter:
            im_data, data = self.jitter(im_data, data)

        prng = RandomState()
        use_noise = np.random.choice([True, False], 1, [0.4, 0.6])

        if use_noise:
            im_data, data = self.gaussian_noise(im_data, data)

        return im_data, data

class FixedCutout(nn.Module):

    def __init__(self):

        self.length = 60
        super(FixedCutout, self).__init__()

    def get_data(self, data):

        h = data[0].size(1)
        w = data[0].size(2)
        bs = data[0].size(0)

        mask = np.ones((bs, c, h, w), np.float32)
        prng = RandomState()

        for n in range(bs):

            y = prng.randint(h)
            x = prng.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[n, :, y1: y2, x1: x2] = 0.0

        mask = torch.from_numpy(mask)
        self.mask = mask.expand_as(data[0])

        new_names = [name + "_" + "fixcut" for name in data[-1]]

        return [data[0]*self.mask, data[1], data[2], data[3], new_names]

    def forward(self, im_data):

        return self.mask*im_data
