"""

"""


# Built-in
import os
from glob import glob

# Libs
import numpy as np
from sklearn.metrics import precision_recall_curve

# Own modules
import eval_detector
from eval import eval_utils

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


def get_y_score(link_list, conf, y_len):
    y_score = [0 for _ in range(y_len)]
    y_true = [1 for _ in range(y_len)]
    for cnt, i in enumerate(link_list):
        if i > 0:
            y_score[i] = conf[cnt]
        else:
            y_score = np.append(y_score, conf[cnt])
            y_true = np.append(y_true, 0)
    return y_true, y_score


def get_conf_pred(model_name, city_name, tile_id, link_r, is_box=False):
    img, lbl, _ = eval_detector.get_data(city_name, tile_id)
    bbox, conf = eval_detector.parse_preds(city_name, tile_id, model_name)
    if is_box:
        lbl_center = [a['box'] for a in lbl]
        pred_center = bbox
        if len(pred_center) == 0:
            return [1 for _ in range(len(lbl_center))], [0 for _ in range(len(lbl_center))]
        link_list = eval_utils.link_pred_gt_box(pred_center, lbl_center, link_r)
        assert len(link_list) == len(conf)
    else:
        lbl_center = [a['center'] for a in lbl]
        pred_center = [[int(a[0] + a[2]) / 2, int(a[1] + a[3]) / 2] for a in bbox]
        if len(pred_center) == 0:
            return [1 for _ in range(len(lbl_center))], [0 for _ in range(len(lbl_center))]
        link_list = eval_utils.link_pred_gt(pred_center, lbl_center, link_r)
        assert len(link_list) == len(conf)
    y_true, y_score = get_y_score(link_list, conf, len(lbl_center))
    return y_true, y_score


def batch_eval(model_name, test_region, val_per=0.2, link_r=10, is_box=False):
    y_true, y_score = [], []
    test_ids = []
    for region in test_region:
        test_ids.extend(city_lut[region])
    for test_id in test_ids:
        city_name = city[test_id][:-3]
        city_ids = idx[test_id]
        val_range = int(len(city_ids) * val_per)
        for tile_id in range(1, val_range+1):
            gt, conf = get_conf_pred(model_name, city_name, tile_id, link_r, is_box)
            y_true.extend(gt)
            y_score.extend(conf)
    prec, rec, _ = precision_recall_curve(y_true, y_score)

    map, mrec, mprec = voc_ap(list(rec)[1:][::-1], list(prec)[1:][::-1])
    return map


def batch_eval_range(model_name, test_region, val_per=0.2, eval_range=(5, 10), step=0.5, is_box=False):
    map = []
    for link_r in np.arange(eval_range[0], eval_range[1], step):
        map.append(batch_eval(model_name, test_region, val_per, link_r, is_box))
    return np.mean(map)


def tower_results(model_type, pred_dir='/media/ei-edl01/data/transmission/eccv/pred', val_per=0.2, link_r=10):
    link_dist = int(link_r * 0.3)
    print('{}:'.format(model_type))

    # one-to-one
    print('\tone-to-one--', end='\t')
    for eval_region in ['AZ', 'KS', 'NZ']:
        model_name = glob(os.path.join(pred_dir, 'faster_rcnn_{}_{}_*'.format(model_type, eval_region)))
        assert len(model_name) == 1
        mAP = batch_eval(model_name[0], [eval_region, ], val_per, link_r)
        mAPr = batch_eval_range(model_name[0], [eval_region, ], val_per, (5, 10), 0.5)
        print('{}: {:.2f}(mAP$_{:d}m$={:.2f})'.format(eval_region, mAPr, link_dist, mAP), end='\t')
    print()

    # all-on-all
    model_name = glob(os.path.join(pred_dir, 'faster_rcnn_{}_All*'.format(model_type)))
    assert len(model_name) == 1
    # mAP = batch_eval(model_name[0], ['AZ', 'KS', 'NZ'], is_box=is_box)
    # mAPr = batch_eval_r(model_name[0], ['AZ', 'KS', 'NZ'], link_r=link_r, is_box=is_box)
    # print('\tall-on-all--\t {:.2f}(mAP$_{:d}m$={:.2f})'.format(mAP, link_dist, mAPr))
    print('\tall-on-all--', end='\t')
    for eval_region in ['AZ', 'KS', 'NZ']:
        assert len(model_name) == 1
        mAP = batch_eval(model_name[0], [eval_region, ], val_per, link_r)
        mAPr = batch_eval_range(model_name[0], [eval_region, ], val_per, (5, 10), 0.5)
        print('{}: {:.2f}(mAP$_{:d}m$={:.2f})'.format(eval_region, mAPr, link_dist, mAP), end='\t')
    print()

    # leave-one-out
    print('\tleave-one-out--', end='\t')
    for train_region, eval_region in zip(['NZKS', 'NZAZ', 'AZKS'], ['AZ', 'KS', 'NZ']):
        model_name = glob(os.path.join(pred_dir, 'faster_rcnn_{}_{}_*'.format(model_type, train_region)))
        assert len(model_name) == 1
        mAP = batch_eval(model_name[0], [eval_region, ], val_per, link_r)
        mAPr = batch_eval_range(model_name[0], [eval_region, ], val_per, (5, 10), 0.5)
        print('{}: {:.2f}(mAP$_{:d}m$={:.2f})'.format(eval_region, mAP, link_dist, mAPr), end='\t')
    print()


def tower_results_iou(model_type, pred_dir='/media/ei-edl01/data/transmission/eccv/pred', val_per=0.2,
                      link_r1=0.75, link_r2=0.5):
    print('{}:'.format(model_type))

    # one-to-one
    print('\tone-to-one--', end='\t')
    for eval_region in ['AZ', 'KS', 'NZ']:
        model_name = glob(os.path.join(pred_dir, 'faster_rcnn_{}_{}_*'.format(model_type, eval_region)))
        assert len(model_name) == 1
        mAP1 = batch_eval(model_name[0], [eval_region, ], val_per, link_r1, is_box=True)
        mAP2 = batch_eval(model_name[0], [eval_region, ], val_per, link_r2, is_box=True)
        mAPr = batch_eval_range(model_name[0], [eval_region, ], val_per, (0.5, 0.95), 0.05, is_box=True)
        print('{}: {:.2f}(mAP$_{}$={:.2f}, mAP$_{}$={:.2f})'.format(eval_region, mAPr, link_r1, mAP1, link_r2, mAP2), end='\t')
    print()

    # all-on-all
    model_name = glob(os.path.join(pred_dir, 'faster_rcnn_{}_All*'.format(model_type)))
    assert len(model_name) == 1
    # mAP = batch_eval(model_name[0], ['AZ', 'KS', 'NZ'], is_box=is_box)
    # mAPr = batch_eval_r(model_name[0], ['AZ', 'KS', 'NZ'], link_r=link_r, is_box=is_box)
    # print('\tall-on-all--\t {:.2f}(mAP$_{:d}m$={:.2f})'.format(mAP, link_dist, mAPr))
    print('\tall-on-all--', end='\t')
    for eval_region in ['AZ', 'KS', 'NZ']:
        assert len(model_name) == 1
        mAP1 = batch_eval(model_name[0], [eval_region, ], val_per, link_r1, is_box=True)
        mAP2 = batch_eval(model_name[0], [eval_region, ], val_per, link_r2, is_box=True)
        mAPr = batch_eval_range(model_name[0], [eval_region, ], val_per, (0.5, 0.95), 0.05, is_box=True)
        print('{}: {:.2f}(mAP$_{}$={:.2f}, mAP$_{}$={:.2f})'.format(eval_region, mAPr, link_r1, mAP1, link_r2, mAP2), end='\t')
    print()

    # leave-one-out
    print('\tleave-one-out--', end='\t')
    for train_region, eval_region in zip(['NZKS', 'NZAZ', 'AZKS'], ['AZ', 'KS', 'NZ']):
        model_name = glob(os.path.join(pred_dir, 'faster_rcnn_{}_{}_*'.format(model_type, train_region)))
        assert len(model_name) == 1
        mAP1 = batch_eval(model_name[0], [eval_region, ], val_per, link_r1, is_box=True)
        mAP2 = batch_eval(model_name[0], [eval_region, ], val_per, link_r2, is_box=True)
        mAPr = batch_eval_range(model_name[0], [eval_region, ], val_per, (0.5, 0.95), 0.05, is_box=True)
        print(
            '{}: {:.2f}(mAP$_{}$={:.2f}, mAP$_{}$={:.2f})'.format(eval_region, mAPr, link_r1, mAP1, link_r2, mAP2),
            end='\t')
    print()


if __name__ == '__main__':
    tower_results_iou('inception')
    tower_results_iou('res101')
    tower_results_iou('res50')
