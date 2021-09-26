"""
1. Run export_all_models()
2. Run infer_images.py to get all predictions
"""


# Built-in
import os
import pickle
from glob import glob

# Libs
import imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from natsort import natsorted

# Own modules
from eval import infer_images


def parse_preds(pred_dir=r'/media/ei-edl01/data/transmission/eccv/pred'):
    model_names = sorted(os.listdir(pred_dir))
    for model_name in model_names:
        lbl_dir = os.path.join(pred_dir, model_name, 'gt')
        pre_dir = os.path.join(pred_dir, model_name, 'pred')


def rename(pred_dir=r'/media/ei-edl01/data/transmission/eccv/pred'):
    test_imgs, test_lbls, test_masks = infer_images.get_test_images(r'/media/ei-edl01/data/transmission/eccv')
    lut = {}
    for image_path, lbl_path, mask_path in zip(test_imgs, test_lbls, test_masks):
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        print('Building lut for {}...'.format(file_name))
        img = imageio.imread(image_path)
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        with open(lbl_path, 'rb') as f:
            data = pickle.load(f)
        node_info, adj_mat, polygons_list = data['node_info'], data['adj_mat'], data['polygons_list']
        lbl = [a for a in node_info if a['label'] != 2]
        grid = infer_images.make_cell_grid(img, lbl, (500, 500), True)
        patch_cnt = 0
        for line in grid:
            for cell in line:
                patch_cnt += 1
                if file_name in lut:
                    lut[file_name][patch_cnt] = {'w': cell['w'], 'h': cell['h']}
                else:
                    lut[file_name] = {patch_cnt: {'w': cell['w'], 'h': cell['h']}}

    model_names = sorted(os.listdir(pred_dir))
    for model_name in model_names:
        lbl_dir, pre_dir = os.path.join(pred_dir, model_name, 'gt'), os.path.join(pred_dir, model_name, 'pred')
        lbl_files = natsorted(glob(os.path.join(lbl_dir, '*.txt')))
        pre_files =  natsorted(glob(os.path.join(pre_dir, '*.txt')))
        for lbl_file, pre_file in zip(lbl_files, pre_files):
            cnt_file_name = os.path.splitext(os.path.basename(lbl_file))[0]
            cnt, *file_name = cnt_file_name.split('_')
            cnt, file_name = int(cnt), '_'.join(file_name)
            text_file_name = 'h{}_w{}_{}_{}'.format(lut[file_name][cnt]['h'], lut[file_name][cnt]['w'], cnt,
                                                    file_name + '.txt')
            os.system('mv {} {}'.format(lbl_file, os.path.join(lbl_dir, text_file_name)))
            os.system('mv {} {}'.format(pre_file, os.path.join(pre_dir, text_file_name)))

        exit(0)


def rename_back(pred_dir=r'/media/ei-edl01/data/transmission/eccv/pred'):
    model_names = sorted(os.listdir(pred_dir))
    for model_name in model_names:
        lbl_dir, pre_dir = os.path.join(pred_dir, model_name, 'gt'), os.path.join(pred_dir, model_name, 'pred')
        lbl_files = natsorted(glob(os.path.join(lbl_dir, '*.txt')))
        pre_files =  natsorted(glob(os.path.join(pre_dir, '*.txt')))
        for lbl_file, pre_file in zip(lbl_files, pre_files):
            cnt_file_name = os.path.splitext(os.path.basename(lbl_file))[0]
            h, w, *file_name = cnt_file_name.split('_')
            file_name = '_'.join(file_name)
            text_file_name = '{}'.format(file_name + '.txt')
            os.system('mv {} {}'.format(os.path.join(lbl_dir, lbl_file), os.path.join(lbl_dir, text_file_name)))
            os.system('mv {} {}'.format(os.path.join(pre_dir, pre_file), os.path.join(pre_dir, text_file_name)))

        exit(0)


def export_all_models(model_dir=r'/media/ei-edl01/user/bh163/models/eccv/towers', ckpt_id=50000, gpu=0):
    model_names = natsorted(os.listdir(model_dir))
    for model_name in model_names:
        if 'inception' in model_name:
            config_dir = r'/media/ei-edl01/user/bh163/models/towers/faster_rcnn.config'
        elif 'res101' in model_name:
            config_dir = r'/media/ei-edl01/user/bh163/models/towers/faster_rcnn_res101.config'
        elif 'res50' in model_name:
            config_dir = r'/media/ei-edl01/user/bh163/models/towers/faster_rcnn_res50.config'
            ckpt_id = 25000
        else:
            raise NotImplementedError('{} not understood'.format(model_name))
        os.system('bash export_model.sh {} {} {} {}'.format(model_name, ckpt_id, config_dir, gpu))


def vis_towers(tile_name, txt_dir, img_dir=r'/media/ei-edl01/data/transmission/eccv', is_gt=True):
    img_file = os.path.join(img_dir, 'img', '{}.jpg'.format(tile_name))
    img = imageio.imread(img_file)
    txt_files = sorted(glob(os.path.join(txt_dir, '*{}.txt'.format(tile_name))))
    box, conf = get_pred_coords(txt_files, is_gt)

    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.imshow(img)
    for b in box:
        print(b)
        rect = patches.Rectangle((b[0], b[1]), b[2], b[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()


def get_pred_coords(txt_files, is_gt=True):
    box, conf = [], []
    for txt_file in txt_files:
        h, w = os.path.splitext(os.path.basename(txt_file))[0].split('_')[:2]
        h, w = int(''.join([a for a in h if a.isdigit()])), int(''.join([a for a in w if a.isdigit()]))
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if is_gt:
                _, left, top, right, bottom = line.strip().split(' ')
                left, top, right, bottom = int(left), int(top), int(right), int(bottom)
                print(left, top, right, bottom)
                box.append([left+w, top+h, right-left, bottom-top])
                conf.append(1)
            else:
                _, cf, left, top, right, bottom = line.strip().split(' ')
                cf, left, top, right, bottom = float(cf), int(left), int(top), int(right), int(bottom)
                if cf > 0.1:
                    box.append([left + w, top + h, right - left, bottom - top])
                    conf.append(cf)
    return box, conf


if __name__ == '__main__':
    rename_back()
    # vis_towers('NZ_Dunedin_1', r'/media/ei-edl01/data/transmission/eccv/pred/faster_rcnn_inception_'
    #                            r'AZKS_2020-02-13_13-03-05/pred', is_gt=False)
