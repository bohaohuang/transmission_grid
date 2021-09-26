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


def create_angle(data_dir, gt_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    gt_files = natsorted(glob(os.path.join(gt_dir, '*.csv')))
    for gt_file in tqdm(gt_files):
        # parse ground truth
        file_name = os.path.splitext(os.path.basename(gt_file))[0]
        boxes, lines = data_utils.csv_to_annotation(gt_file)

        # read rgb image
        rgb_file = os.path.join(data_dir, '{}.tif'.format(file_name))
        rgb = imageio.imread(rgb_file)

        # make line map
        line_map, line_pts = data_utils.render_line_graph(rgb.shape[:2], boxes, lines)

        # make angle map
        orient_map, angle_map = data_utils.render_angle_map(line_map, boxes, lines, line_pts)

        # save image
        '''save_name = os.path.join(save_dir, '{}_angle.png'.format(file_name))
        imageio.imsave(save_name, angle_map.astype(np.uint8))
        save_name = os.path.join(save_dir, '{}_vecmap.png'.format(file_name))
        imageio.imsave(save_name, orient_map.astype(np.uint8))'''


if __name__ == '__main__':
    create_angle(
        data_dir=r'/hdd6/data/transmission_line/raw2',
        gt_dir=r'/media/ei-edl01/data/remote_sensing_data/transmission_line/parsed_annotation',
        save_dir=r'/hdd/pgm/angle',
    )

    '''img = imageio.imread(os.path.join(r'~/Documents/bohao/data/transmission_line/raw2', 'NZ_Gisborne_3_resize.tif'))[..., :3]
    boxes, lines = data_utils.csv_to_annotation(os.path.join(
        r'/media/ei-edl01/data/remote_sensing_data/transmission_line/parsed_annotation',
        'NZ_Gisborne_3_resize.csv'
    ))
    line_map, line_pts = data_utils.render_line_graph(img.shape[:2], boxes, lines)
    orient_map, angle_map = data_utils.render_angle_map(line_map, boxes, lines, line_pts)

    def plotVecMap(vectorMap, image, figsize_=(6, 6), name=''):
        from numpy import ma
        # fig = plt.figure(figsize=figsize_)
        plt.axis('off')
        plt.imshow(image)

        U = vectorMap[:, :, 0] * -1
        V = vectorMap[:, :, 1]
        X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))

        print(U.shape, vectorMap.shape)
        print(X.shape, Y.shape)

        M = np.zeros(U.shape, dtype='bool')
        M[U ** 2 + V ** 2 < 0.5 * 0.5] = True
        U = ma.masked_array(U, mask=M)
        V = ma.masked_array(V, mask=M)

        s = 15
        Q = plt.quiver(X[::s, ::s], Y[::s, ::s], U[::s, ::s], V[::s, ::s], scale=50, headaxislength=5, headwidth=5,
                       width=0.01, alpha=.8, color='y')
        # fig = plt.gcf()
        # plt.show()

    import matplotlib.pyplot as plt
    # plotVecMap(orient_map, img)
    plt.figure(figsize=(15, 5))
    plt.subplot(141)
    plt.axis('off')
    plt.imshow(img[5200:6100, 1400:2000, :])
    plt.subplot(142)
    plt.axis('off')
    plt.imshow(line_map[5200:6100, 1400:2000])
    plt.subplot(143)
    plotVecMap(orient_map[5200:6100, 1400:2000, :], img[5200:6100, 1400:2000, :])
    plt.subplot(144)
    plt.axis('off')
    plt.imshow(angle_map[5200:6100, 1400:2000])
    # plt.colorbar()
    plt.tight_layout()
    plt.show()'''
