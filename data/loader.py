"""

"""


# Built-in
import os
import math

# Libs
import cv2
import imageio
import numpy as np

# Pytorch
import torch
from torch.utils import data

# Own modules


class TransmissionDataLoader(data.Dataset):
    def __init__(self, parent_path, file_list, transforms=None):
        """
        A data reader for the remote sensing dataset
        The dataset storage structure should be like
        /parent_path
            /patches
                img0.png
                img1.png
            file_list.txt
        Normally the downloaded remote sensing dataset needs to be preprocessed
        :param parent_path: path to a preprocessed remote sensing dataset
        :param file_list: a text file where each row contains rgb and gt files separated by space
        :param transforms: albumentation transforms
        """
        with open(file_list, 'r') as f:
            self.file_list = f.readlines()[:-1]
        self.parent_path = parent_path
        self.transforms = transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        rgb_filename, lbl_filename, vec_filename = [os.path.join(self.parent_path, a)
                                                    for a in self.file_list[index].strip().split(' ')]
        rgb = imageio.imread(rgb_filename)
        lbl = imageio.imread(lbl_filename)
        vec = imageio.imread(vec_filename)
        if self.transforms:
            for tsfm in self.transforms:
                tsfm_image = tsfm(image=rgb, masks=[lbl, vec])
                rgb = tsfm_image['image']
                lbl, vec = tsfm_image['masks']

        lbl, vec = torch.from_numpy(np.expand_dims(lbl, 0)).float(), torch.from_numpy(np.expand_dims(vec, 0)).float()
        return rgb, lbl, vec


if __name__ == '__main__':
    import albumentations as A
    from albumentations.pytorch import ToTensorV2


    def visualize(rgb, gt, vec, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.255)):
        """
        Visualize a given pair of image and mask normalized tensors
        :param rgb: the image tensor with shape [c, h, w]
        :param gt: the mask tensor with shape [1, h, w]
        :param mean: the mean used to normalize the input
        :param std: the std used to normalize the input
        :return:
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from torchvision import transforms
        def change_channel_order(data, to_channel_last=True):
            """
            Switch the image type from channel first to channel last
            :param data: the data to switch the channels
            :param to_channel_last: if True, switch the first channel to the last
            :return: the channel switched data
            """
            if to_channel_last:
                if len(data.shape) == 3:
                    return np.rollaxis(data, 0, 3)
                else:
                    return np.rollaxis(data, 1, 4)
            else:
                if len(data.shape) == 3:
                    return np.rollaxis(data, 2, 0)
                else:
                    return np.rollaxis(data, 3, 1)
        mean = [-a / b for a, b in zip(mean, std)]
        std = [1 / a for a in std]
        inv_normalize = transforms.Normalize(
            mean=mean,
            std=std
        )
        rgb = inv_normalize(rgb)
        rgb, gt, vec = rgb.numpy(), gt.numpy(), vec.numpy()
        rgb = change_channel_order(rgb, True)
        gt = change_channel_order(gt, True)
        vec = change_channel_order(vec, True)
        plt.figure(figsize=(12, 6))
        plt.subplot(131)
        plt.imshow((rgb * 255).astype(np.uint8))
        plt.subplot(132)
        plt.imshow(gt[:, :, 0].astype(np.uint8))
        plt.subplot(133)
        plt.imshow(vec[:, :, 0].astype(np.uint8))
        plt.tight_layout()
        plt.show()

    tsfm = A.Compose([
        A.Flip(),
        A.RandomRotate90(),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])
    ds = TransmissionDataLoader(r'/hdd/pgm/nz_mtl_ps512_ol0/patches', r'/hdd/pgm/nz_mtl_ps512_ol0/file_list_valid.txt', transforms=tsfm)
    for cnt, (rgb, gt, vec) in enumerate(ds):
        visualize(rgb, gt, vec)

        if cnt == 10:
            break
