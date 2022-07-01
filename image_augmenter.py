from time import time
from imgaug import augmenters as iaa
from torch import dtype
import numpy as np
class image_augmenter:
    '''augments images before storing them'''
    def __init__(self):
        st = lambda aug: iaa.Sometimes(0.4, aug)
        oc = lambda aug: iaa.Sometimes(0.3, aug)
        rl = lambda aug: iaa.Sometimes(0.09, aug)
        self.augment = iaa.Sequential([

            rl(iaa.GaussianBlur((0, 1.5))),  # blur images with a sigma between 0 and 1.5
            rl(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05), per_channel=0.5)),  # add gaussian noise to images
            oc(iaa.Dropout((0.0, 0.10), per_channel=0.5)),  # randomly remove up to X% of the pixels
            oc(iaa.CoarseDropout((0.0, 0.10), size_percent=(0.08, 0.2), per_channel=0.5)),
            # randomly remove up to X% of the pixels
            oc(iaa.Add((-40, 40), per_channel=0.5)),  # change brightness of images (by -X to Y of original value)
            st(iaa.Multiply((0.10, 2.5), per_channel=0.2)),  # change brightness of images (X-Y% of original value)
            rl(iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5)),  # improve or worsen the contrast
            rl(iaa.Grayscale((0.0, 1))),  # put grayscale

        ],
            random_order=True  # do all of the above in random order
        )
    def aug(self, images):
        #a list of 3d images in the form of (h,w,channels)
        images_aug = self.augment(images=images)
        return images_aug
    def normalise_samples(self,  images, speeds, commands, grayscale=False):
       
        normalised_cmds = []
        
        images = np.array(images) / 255
        speeds = np.array(speeds) / TARGET_SPEED
        left_cmd = [0,0,1]
        right_cmd = [1,0,0]
        straight_cmd = [0,1,0]
        left = 0
        right = 0
        straight = 0
        normalised_cmds = [left_cmd if cmd == "left" else straight_cmd if cmd == "straight" else right_cmd for cmd in commands]
        return images, speeds, np.array(normalised_cmds)
