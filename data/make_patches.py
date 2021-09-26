"""

"""


# Built-in
import os
from glob import glob

# Libs
import imageio
import numpy as np
from tqdm import tqdm
from natsort import natsorted

# Own modules
from data import data_utils


def make_grid(tile_size, patch_size, overlap):
    """
    Extract patches at fixed locations. Output coordinates for Y,X as a list (not two lists)
    :param tile_size: size of the tile (input image)
    :param patch_size: size of the output patch
    :param overlap: #overlapping pixels
    :return:
    """
    max_h = tile_size[0] - patch_size[0]
    max_w = tile_size[1] - patch_size[1]
    if max_h > 0 and max_w > 0:
        h_step = np.ceil(tile_size[0] / (patch_size[0] - overlap))
        w_step = np.ceil(tile_size[1] / (patch_size[1] - overlap))
    else:
        h_step = 1
        w_step = 1
    patch_grid_h = np.floor(np.linspace(0, max_h, h_step)).astype(np.int32)
    patch_grid_w = np.floor(np.linspace(0, max_w, w_step)).astype(np.int32)

    y, x = np.meshgrid(patch_grid_h, patch_grid_w)
    return list(zip(y.flatten(), x.flatten()))


def crop_image(img, y, x, h, w):
    """
    Crop the image with given top-left anchor and corresponding width & height
    :param img: image to be cropped
    :param y: height of anchor
    :param x: width of anchor
    :param h: height of the patch
    :param w: width of the patch
    :return:
    """
    if len(img.shape) == 2:
        return img[y:y+w, x:x+h]
    else:
        return img[y:y+w, x:x+h, :]


def patch_tile(rgb, gt, vec, patch_size, overlap=0):
    """
    Extract the given rgb and gt tiles into patches
    :param rgb_file: path to the rgb file
    :param gt_file: path to the gt file
    :param patch_size: size of the patches, should be a tuple of (h, w)
    :param pad: #pixels to be padded around each tile, should be either one element or four elements
    :param overlap: #overlapping pixels between two patches in both vertical and horizontal direction
    :return: rgb and gt patches as well as coordinates
    """
    np.testing.assert_array_equal(rgb.shape[:2], gt.shape)
    grid_list = make_grid(np.array(rgb.shape[:2]), patch_size, overlap)
    for y, x in grid_list:
        rgb_patch = crop_image(rgb, y, x, patch_size[0], patch_size[1])
        gt_patch = crop_image(gt, y, x, patch_size[0], patch_size[1])
        vec_patch = crop_image(vec, y, x, patch_size[0], patch_size[1])
        yield rgb_patch, gt_patch, vec_patch, y, x


def make_patches(data_dir, gt_dir, vec_dir, save_dir, patch_size, overlap=0):
    """
    Preprocess the standard inria dataset
    :param data_dir: path to the original inria dataset
    :param save_dir: directory to save the extracted patches
    :param patch_size: size of the patches, should be a tuple of (h, w)
    :param pad: #pixels to be padded around each tile, should be either one element or four elements
    :param overlap: #overlapping pixels between two patches in both vertical and horizontal direction
    :return:
    """
    # create folders and files
    patch_dir = os.path.join(save_dir, 'patches')
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)
    record_file_train = open(os.path.join(save_dir, 'file_list_train.txt'), 'w+')
    record_file_valid = open(os.path.join(save_dir, 'file_list_valid.txt'), 'w+')
    # get rgb and gt files
    gt_files = natsorted(glob(os.path.join(gt_dir, '*.csv')))
    for gt_file in tqdm(gt_files):
        # only do NZ image
        if 'NZ' not in gt_file:
            continue

        # parse ground truth
        file_name = os.path.splitext(os.path.basename(gt_file))[0]
        boxes, lines = data_utils.csv_to_annotation(gt_file)
        tile_id = int(''.join([a for a in file_name if a.isdigit()]))

        # read rgb image
        rgb_file = os.path.join(data_dir, '{}.tif'.format(file_name))
        rgb = imageio.imread(rgb_file)[..., :3]

        # make line map
        line_map, line_pts = data_utils.render_line_graph(rgb.shape[:2], boxes, lines)

        # read vec image
        vec_file = os.path.join(vec_dir, '{}_angle.png'.format(file_name))
        vec = imageio.imread(vec_file)

        for rgb_patch, gt_patch, vec_patch, y, x, in patch_tile(rgb, line_map, vec, patch_size, overlap):
            rgb_patchname = '{}_y{}x{}.jpg'.format(file_name.replace(' ', '_'), int(y), int(x))
            gt_patchname = '{}_y{}x{}.png'.format(file_name.replace(' ', '_'), int(y), int(x))
            vec_patchname = '{}_y{}x{}_angle.png'.format(file_name.replace(' ', '_'), int(y), int(x))
            imageio.imsave(os.path.join(patch_dir, rgb_patchname), rgb_patch.astype(np.uint8))
            imageio.imsave(os.path.join(patch_dir, gt_patchname), gt_patch.astype(np.uint8))
            imageio.imsave(os.path.join(patch_dir, vec_patchname), vec_patch.astype(np.uint8))
            if tile_id <= 3:
                record_file_valid.write('{} {} {}\n'.format(rgb_patchname, gt_patchname, vec_patchname))
            else:
                record_file_train.write('{} {} {}\n'.format(rgb_patchname, gt_patchname, vec_patchname))
    record_file_train.close()
    record_file_valid.close()


if __name__ == '__main__':
    make_patches(
        data_dir=r'~/Documents/bohao/data/transmission_line/raw2',
        gt_dir=r'/media/ei-edl01/data/remote_sensing_data/transmission_line/parsed_annotation',
        vec_dir=r'/hdd/pgm/angle',
        save_dir=r'/hdd/pgm/patches_mtl_nz',
        patch_size=(512, 512),
    )
