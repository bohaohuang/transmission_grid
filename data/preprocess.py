"""
This file preprocess transmission line dataset
"""


# Built-in
import os
from glob import glob

# Libs
import imageio
import numpy as np
from natsort import natsorted

# Own modules


def csv_to_annotation(file_name):
    node_info = []
    adj_flag = False
    with open(file_name, 'r') as f:
        line = f.readline()
        while line:
            if adj_flag:
                pass
            else:
                box = tuple([int(a) for a in line.strip().split(',')[:-1]])
                node_info.append(box)
            line = f.readline()
            if line == '\n':
                adj_flag = True
                line = f.readline()
    return node_info


def allocate_box(box, patch_size, h_offsets, w_offsets):
    def iou(box_1, box_2):
        inter_h_top = max([box_1[0], box_2[0]])
        inter_h_bot = min([box_1[2], box_2[2]])
        inter_w_lef = max([box_1[1], box_2[1]])
        inter_w_rht = min([box_1[3], box_2[3]])
        inter = max(0, inter_h_bot - inter_h_top + 1) * max(0, inter_w_rht - inter_w_lef + 1)

        union_1 = (box_1[2] - box_1[0] + 1) * (box_1[3] - box_1[1] + 1)
        union_2 = (box_2[2] - box_2[0] + 1) * (box_2[3] - box_2[1] + 1)
        union = union_1 + union_2 - inter

        return inter / union

    max_iou = 0
    record_box = (0, 0)
    for h in h_offsets:
        for w in w_offsets:
            curr_iou = iou(box, (h, w, h+patch_size[0], w+patch_size[1]))
            if curr_iou > max_iou:
                max_iou = curr_iou
                record_box = (h, w)
    return record_box


def get_boxes_range(boxes):
    h_min, w_min, h_max, w_max = boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]
    if len(boxes) > 1:
        for cnt in range(1, len(boxes)):
            if boxes[cnt][0] < h_min:
                h_min = boxes[cnt][0]
            if boxes[cnt][1] < w_min:
                w_min = boxes[cnt][1]
            if boxes[cnt][2] > h_max:
                h_max = boxes[cnt][2]
            if boxes[cnt][3] > w_max:
                w_max = boxes[cnt][3]
    return h_min, w_min, h_max, w_max


def adjust_patch_range(h, w, patch_size, box_region):
    adjust_flag = [0, 0, 0, 0]      # top, left, bottom, right
    if box_region[0] < h:
        adjust_flag[0] = 1
    if box_region[1] < w:
        adjust_flag[1] = 1
    if box_region[2] > h + patch_size[0]:
        adjust_flag[2] = 1
    if box_region[3] > w + patch_size[1]:
        adjust_flag[3] = 1

    if adjust_flag[0] + adjust_flag[2] > 1 or adjust_flag[1] + adjust_flag[3] > 1:
        print('Warning')
        return h, w

    if adjust_flag[0]:
        h = box_region[0]       # move box up
    if adjust_flag[1]:
        w = box_region[1]       # move box left
    if adjust_flag[2]:
        h = h + (box_region[2] - h - patch_size[0])
    if adjust_flag[3]:
        w = w + (box_region[3] - w - patch_size[1])
    return h, w


def plot_patch_with_boxes(ax, h_offset, w_offset, box):
    import matplotlib.patches as patches
    rect = patches.Rectangle((box[1]-w_offset, box[0]-h_offset), box[3]-box[1], box[2]-box[0], linewidth=1,
                             facecolor='none', edgecolor='r')
    ax.add_patch(rect)


def make_patches(img, patch_size, boxes):
    tile_size = img.shape[:2]
    max_h = tile_size[0] - patch_size[0]
    max_w = tile_size[1] - patch_size[1]
    h_steps = np.ceil(tile_size[0] / patch_size[0]).astype(np.int32)
    w_steps = np.ceil(tile_size[1] / patch_size[1]).astype(np.int32)
    h_offsets = np.floor(np.linspace(0, max_h, h_steps)).astype(np.int32)
    w_offsets = np.floor(np.linspace(0, max_w, w_steps)).astype(np.int32)

    grid_info = {}
    for h_cnt in range(h_steps):
        grid_info[h_offsets[h_cnt]] = {}
        for w_cnt in range(w_steps):
            grid_info[h_offsets[h_cnt]][w_offsets[w_cnt]] = {'h': h_offsets[h_cnt], 'w': w_offsets[w_cnt],
                                                             'boxes': []}

    # assign boxes to patches
    for box in boxes:
        h, w = allocate_box(box, patch_size, h_offsets, w_offsets)
        grid_info[h][w]['boxes'].append(box)

    # adjust boxes
    for h_cnt in range(h_steps):
        h = h_offsets[h_cnt]
        for w_cnt in range(w_steps):
            w = w_offsets[w_cnt]
            if len(grid_info[h][w]['boxes']) > 0:
                # there are towers inside the patch
                box_region = get_boxes_range(grid_info[h][w]['boxes'])
                new_h, new_w = adjust_patch_range(h, w, patch_size, box_region)

                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1)
                ax.imshow(img[new_h:new_h+patch_size[0], new_w:new_w+patch_size[1], :3])
                plot_patch_with_boxes(ax, new_h, new_w, box_region)
                for b in grid_info[h][w]['boxes']:
                    plot_patch_with_boxes(ax, new_h, new_w, b)
                plt.tight_layout()
                plt.show()


def preprocess(data_dir, gt_dir):
    gt_files = natsorted(glob(os.path.join(gt_dir, '*.csv')))
    for gt_file in gt_files:
        # parse ground truth
        file_name = os.path.splitext(os.path.basename(gt_file))[0]
        boxes = csv_to_annotation(gt_file)

        # read rgb image
        rgb_file = os.path.join(data_dir, '{}.tif'.format(file_name))
        rgb = imageio.imread(rgb_file)

        # make boxes
        make_patches(rgb, (500, 500), boxes)


if __name__ == '__main__':
    preprocess(
        data_dir=r'~/Documents/bohao/data/transmission_line/raw2',
        gt_dir=r'/media/ei-edl01/data/remote_sensing_data/transmission_line/parsed_annotation',
    )
