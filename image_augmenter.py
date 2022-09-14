from imgaug import augmenters as iaa
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
    