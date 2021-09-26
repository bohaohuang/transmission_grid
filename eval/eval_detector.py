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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from natsort import natsorted

# Own modules
from utils import misc_utils
from eval import infer_images, eval_utils

# Settings
city = ['NZ_Dunedin_{}',
        'NZ_Gisborne_{}',
        'NZ_Palmerston-North_{}',
        'NZ_Rotorua_{}',
        'NZ_Tauranga_{}',
        'AZ_Tucson_{}',
        'KS_Colwich-Maize_{}'
        ]
idx = [
    list(range(1, 7)),
    list(range(1, 7)),
    list(range(1, 15)),
    list(range(1, 8)),
    list(range(1, 7)),
    list(range(1, 27)),
    list(range(1, 49)),
]
city_lut = {
    'NZ': [0, 1, 2, 3, 4],
    'AZ': [5],
    'KS': [6],
}


def exclude_mask_lbl(img, lbl, mask, img_size=(500, 500), mask_threshold=0.25):
    lbl_exclude = []
    grid = infer_images.make_cell_grid(img, lbl, img_size, True)
    for line in grid:
        for cell in line:
            m = mask[cell['h']:cell['h'] + img_size[0], cell['w']:cell['w'] + img_size[1]]
            ratio = np.sum(m) / (img_size[0] * img_size[1])
            if ratio > mask_threshold:
                continue
            label = cell['label']
            box = cell['box']
            if len(box) > 0:
                for b, l in zip(box, label):
                    lbl_exclude.append({'box': [b[1], b[0], b[3], b[2]],
                                        'center': [int((b[1]+b[3])/2), int((b[0]+b[2])/2)], 'label': 'T'})
    return lbl_exclude


def get_data(city_name, tile_id, root_dir=r'/media/ei-edl01/data/transmission/eccv'):
    img_name = os.path.join(root_dir, 'img', '{}_{}.jpg'.format(city_name, tile_id))
    lbl_name = os.path.join(root_dir, 'gt', '{}_{}.pkl'.format(city_name, tile_id))
    mask_name = os.path.join(root_dir, 'mask', '{}_{}_mask.png'.format(city_name, tile_id))

    img = misc_utils.load_file(img_name)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    mask = misc_utils.load_file(mask_name)
    with open(lbl_name, 'rb') as f:
        data = pickle.load(f)
    node_info, adj_mat, polygons_list = data['node_info'], data['adj_mat'], data['polygons_list']
    lbl = [a for a in node_info if a['label'] != 2]
    lbl = exclude_mask_lbl(img, lbl, mask)
    return img, lbl, mask


def parse_preds(city_name, tile_id, model_name, pred_dir=r'/media/ei-edl01/data/transmission/eccv/pred',
                tower_thresh=0.5):
    pre_dir = os.path.join(pred_dir, model_name, 'pred')
    pre_files = natsorted(glob(os.path.join(pre_dir, '*{}_{}.txt'.format(city_name, tile_id))))
    return get_pred_coords(pre_files, tower_thresh)


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


def vis_towers(img, lbl, pred):
    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.imshow(img)
    for item in lbl:
        box = item['box']
        rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor='g',
                                 facecolor='none')
        ax.add_patch(rect)
    for b in pred:
        rect = patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    # plt.show()
    plt.close()


def save_tower_pred(img, lbl, pred, save_name):
    import cv2
    render = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for item in lbl:
        box = item['box']
        cv2.rectangle(render, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    for box in pred:
        cv2.rectangle(render, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    cv2.imwrite(save_name, render)


def local_maxima_suppression(center_list, conf_list, th=20):
    center_list = np.array(center_list)
    n_samples = center_list.shape[0]
    dist_mat = np.inf * np.ones((n_samples, n_samples))
    merge_list = []
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            dist_mat[i, j] = np.sqrt(np.sum(np.square(center_list[i, :] - center_list[j, :])))
        merge_dist = dist_mat[i, :]
        merge_candidate = np.where(merge_dist < th)
        if merge_candidate[0].shape[0] > 0:
            merge_list.append({i: merge_candidate[0].tolist()})

    remove_idx = []
    for merge_item in merge_list:
        center_points = []
        conf_idx = []
        for k in merge_item.keys():
            center_points.append(center_list[k, :])
            conf_idx.append(k)
            for v in merge_item[k]:
                center_points.append(center_list[v, :])
                conf_idx.append(v)
        center_points = np.mean(center_points, axis=0)

        confs = [conf_list[a] for a in conf_idx]
        keep_idx = int(np.argmax(confs))
        remove_idx.extend([conf_idx[a] for a in range(len(confs)) if a != keep_idx])
        center_list[keep_idx, :] = center_points
        conf_list[keep_idx] = max(confs)

    center_list = [center_list[a] for a in range(n_samples) if a not in remove_idx]
    conf_list = [conf_list[a] for a in range(n_samples) if a not in remove_idx]
    return center_list, conf_list, remove_idx


def non_maxima_suppression(centers, conf, dist_th):
    if len(centers) == 0:
        return [], []

    pick = []
    idx = np.argsort(conf)

    while len(idx) > 0:
        last = len(idx) - 1
        i = idx[last]
        pick.append(i)
        suppress = [last]

        for pos in range(0, last):
            j = idx[pos]

            dist = np.linalg.norm(np.linalg.norm(np.array(centers[i])-np.array(centers[j])))
            if dist < dist_th:
                suppress.append(pos)
        idx = np.delete(idx, suppress)
    return [centers[a] for a in pick], [conf[a] for a in pick]


def get_pred_coords(txt_files, tower_thresh):
    box, conf = [], []
    for txt_file in txt_files:
        h, w = os.path.splitext(os.path.basename(txt_file))[0].split('_')[:2]
        h, w = int(''.join([a for a in h if a.isdigit()])), int(''.join([a for a in w if a.isdigit()]))
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            _, cf, left, top, right, bottom = line.strip().split(' ')
            cf, left, top, right, bottom = float(cf), int(left), int(top), int(right), int(bottom)
            if cf > tower_thresh:
                box.append([left + w, top + h, right + w, bottom + h])
                conf.append(cf)
    box, conf = non_maxima_suppression(box, conf, 10)
    return box, conf


def mean_average_precision(pred_center, lbl_center, link_r=None, eval_range=(5, 10), step=0.5, avg=False):
    if link_r is not None:
        if len(pred_center) != 0:
            link_list = eval_utils.link_pred_gt(pred_center, lbl_center, link_r)
            print(link_list)
            exit(0)
            tp, n_recall, n_precision = eval_utils.grid_score(lbl_center, pred_center, link_list)
            return tp, n_recall, n_precision
        else:
            return 0, 1e-6, 1e-6
    else:
        link_d, precision, recall = [], [], []
        for link_r in np.arange(eval_range[0], eval_range[1], step):
            if len(pred_center) != 0:
                link_list = eval_utils.link_pred_gt(pred_center, lbl_center, link_r)
                tp, n_recall, n_precision = eval_utils.grid_score(lbl_center, pred_center, link_list)
                link_d.append(link_r)
                precision.append(tp/n_precision)
                recall.append(tp/n_recall)
            else:
                precision.append(0)
                recall.append(0)
        if avg:
            map, mrec, mpre = voc_ap(recall, precision)
            return map, mrec, mpre
        else:
            return precision, recall


def mean_average_precision_box(pred_box, lbl_box, link_r=None, eval_range=(0.5, 0.95), step=0.05, avg=False):
    if link_r is not None:
        if len(pred_box) != 0:
            link_list = eval_utils.link_pred_gt_box(pred_box, lbl_box, link_r)
            tp, n_recall, n_precision = eval_utils.grid_score(lbl_box, pred_box, link_list)
            return tp, n_recall, n_precision
        else:
            return 0, 1e-6, 1e-6
    else:
        link_d, precision, recall = [], [], []
        for link_r in np.arange(eval_range[0], eval_range[1], step):
            if len(pred_box) != 0:
                link_list = eval_utils.link_pred_gt_box(pred_box, lbl_box, link_r)
                tp, n_recall, n_precision = eval_utils.grid_score(lbl_box, pred_box, link_list)
                link_d.append(link_r)
                precision.append(tp/n_precision)
                recall.append(tp/n_recall)
            else:
                precision.append(0)
                recall.append(0)
        if avg:
            map, mrec, mpre = voc_ap(recall, precision)
            return map, mrec, mpre
        else:
            return precision, recall


def voc_ap(rec, prec):
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]
    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i)
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def eval_model(model_name, city_name, tile_id, visualize=False, link_r=None, is_box=False):
    img, lbl, _ = get_data(city_name, tile_id)
    bbox, conf = parse_preds(city_name, tile_id, model_name)
    if not is_box:
        lbl_center = [a['center'] for a in lbl]
        pred_center = [[int(a[0]+a[2])/2, int(a[1]+a[3])/2] for a in bbox]
    else:
        lbl_center = [a['box'] for a in lbl]
        pred_center = bbox

    if visualize:
        vis_towers(img, lbl, bbox)

    if not is_box:
        if link_r is None:
            precision, recall = mean_average_precision(pred_center, lbl_center)
            return precision, recall
        else:
            tp, n_recall, n_precision = mean_average_precision(pred_center, lbl_center, link_r)
            return tp, n_recall, n_precision
    else:
        if link_r is None:
            precision, recall = mean_average_precision_box(pred_center, lbl_center)
            return precision, recall
        else:
            tp, n_recall, n_precision = mean_average_precision_box(pred_center, lbl_center, link_r)
            return tp, n_recall, n_precision


def save_imgs(model_name, test_region, save_dir, val_per=0.2):
    result_dir = os.path.join(save_dir, model_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    test_ids = []
    for region in test_region:
        test_ids.extend(city_lut[region])
    for test_id in test_ids:
        city_name = city[test_id][:-3]
        city_ids = idx[test_id]
        val_range = int(len(city_ids) * val_per)
        for tile_id in range(1, val_range + 1):
            print('{}_{}'.format(city_name, tile_id))
            img, lbl, _ = get_data(city_name, tile_id)
            bbox, conf = parse_preds(city_name, tile_id, model_name)
            save_tower_pred(img, lbl, bbox, os.path.join(result_dir, '{}_{}.png'.format(city_name, tile_id)))


def batch_eval(model_name, test_region, val_per=0.2, is_box=False):
    test_ids = []
    mAP_all = []
    for region in test_region:
        test_ids.extend(city_lut[region])
    for test_id in test_ids:
        city_name = city[test_id][:-3]
        city_ids = idx[test_id]
        val_range = int(len(city_ids) * val_per)
        for i in range(1, val_range+1):
            precision, recall = eval_model(model_name, city_name, i, is_box)
            mAP, _, _ = voc_ap(precision, recall)
            mAP_all.append(mAP)
    return np.mean(mAP_all)


def batch_eval_r(model_name, test_region, val_per=0.2, link_r=10, is_box=False):
    rec_all = []
    prec_all = []
    test_ids = []
    for region in test_region:
        test_ids.extend(city_lut[region])
    for test_id in test_ids:
        city_name = city[test_id][:-3]
        city_ids = idx[test_id]
        val_range = int(len(city_ids) * val_per)
        for i in range(1, val_range):
            n_tp, n_recall, n_precision = eval_model(model_name, city_name, i, link_r=link_r, is_box=is_box)
            rec_all.append(n_tp/n_recall)
            prec_all.append(n_tp/n_precision)
    # mAP, _, _ = voc_ap(rec_all, prec_all)
    return np.mean(prec_all)


def tower_results(model_type, pred_dir='/media/ei-edl01/data/transmission/eccv/pred', link_r=10, is_box=False):
    link_dist = int(link_r * 0.3)
    print('{}:'.format(model_type))
    # one-to-one
    print('\tone-to-one--', end='\t')
    for eval_region in ['AZ', 'KS', 'NZ']:
        model_name = glob(os.path.join(pred_dir, 'faster_rcnn_{}_{}_*'.format(model_type, eval_region)))
        assert len(model_name) == 1
        mAP = batch_eval(model_name[0], [eval_region, ], is_box=is_box)
        mAPr = batch_eval_r(model_name[0], [eval_region, ], link_r=link_r, is_box=is_box)
        print('{}: {:.2f}(mAP$_{:d}m$={:.2f})'.format(eval_region, mAP, link_dist, mAPr), end='\t')
    print()

    # all-on-all
    model_name = glob(os.path.join(pred_dir, 'faster_rcnn_{}_All*'.format(model_type)))
    assert len(model_name) == 1
    '''mAP = batch_eval(model_name[0], ['AZ', 'KS', 'NZ'], is_box=is_box)
    mAPr = batch_eval_r(model_name[0], ['AZ', 'KS', 'NZ'], link_r=link_r, is_box=is_box)
    print('\tall-on-all--\t {:.2f}(mAP$_{:d}m$={:.2f})'.format(mAP, link_dist, mAPr))'''
    print('\tall-on-all--', end='\t')
    for eval_region in ['AZ', 'KS', 'NZ']:
        assert len(model_name) == 1
        mAP = batch_eval(model_name[0], [eval_region, ], is_box=is_box)
        mAPr = batch_eval_r(model_name[0], [eval_region, ], link_r=link_r, is_box=is_box)
        print('{}: {:.2f}(mAP$_{:d}m$={:.2f})'.format(eval_region, mAP, link_dist, mAPr), end='\t')
    print()

    # leave-one-out
    print('\tleave-one-out--', end='\t')
    for train_region, eval_region in zip(['NZKS', 'NZAZ', 'AZKS'], ['AZ', 'KS', 'NZ']):
        model_name = glob(os.path.join(pred_dir, 'faster_rcnn_{}_{}_*'.format(model_type, train_region)))
        assert len(model_name) == 1
        mAP = batch_eval(model_name[0], [eval_region, ], is_box=is_box)
        mAPr = batch_eval_r(model_name[0], [eval_region, ], link_r=link_r, is_box=is_box)
        print('{}: {:.2f}(mAP$_{:d}m$={:.2f})'.format(eval_region, mAP, link_dist, mAPr), end='\t')
    print()


if __name__ == '__main__':
    """
    res50 res101 inception
    0.57 0.60; 0.67 0.77; 0.76 0.78
    """
    tower_results('inception', link_r=0.5, is_box=True)
    tower_results('res101', link_r=0.5, is_box=True)
    tower_results('res50', link_r=0.5, is_box=True)

    '''for eval_region in ['All', 'AZ', 'KS', 'NZ', 'NZKS', 'NZAZ', 'AZKS']:
        for model_type in ['inception', 'res101', 'res50']:
            model_name = glob(os.path.join('/media/ei-edl01/data/transmission/eccv/pred',
                                           'faster_rcnn_{}_{}_*'.format(model_type, eval_region)))[0]
            save_imgs(model_name, ['AZ', 'KS', 'NZ'], r'/media/ei-edl01/user/bh163/models/eccv/tower_pred')'''
