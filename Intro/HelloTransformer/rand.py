#rand.py
import random
import numpy as np
from torchvision import transforms


class RandomRotation_z(object):
    def __call__(self, pointcloud):
        theta = random.random() * 2. * np.pi
        rot_matrix = np.array([[ np.cos(theta), -np.sin(theta),      0],
                               [ np.sin(theta),  np.cos(theta),      0],
                               [0,                               0,      1]])
        rot_pointcloud = rot_matrix.dot(pointcloud.T).T
        return rot_pointcloud

class RandomNoise(object):
    def __call__(self, pointcloud):
        noise = np.random.normal(0, 0.02, (pointcloud.shape))
        noisy_pointcloud = pointcloud + noise
        return noisy_pointcloud

class ShufflePoints(object):
    def __call__(self, pointcloud):
        np.random.shuffle(pointcloud)
        return pointcloud

def default_transforms():
    return transforms.Compose([RandomRotation_z(), RandomNoise()])


