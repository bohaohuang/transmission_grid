"""

"""


# Built-in

# Libs
import numpy as np
import scipy.spatial

# Own modules


def link_pred_gt(pred, gt, link_r):
    # link predictions
    kdt = scipy.spatial.KDTree(gt)
    d, linked_results = kdt.query(pred)
    link_list = [-1 for _ in range(len(pred))]
    be_linked = {}

    for cnt, item in enumerate(linked_results):
        if d[cnt] > link_r:
            pass
        else:
            be_linked_item = int(item)
            if be_linked_item not in be_linked:
                link_list[cnt] = be_linked_item
                be_linked[be_linked_item] = [d[cnt], cnt]
            else:
                if d[cnt] < be_linked[be_linked_item][0]:
                    link_list[be_linked[be_linked_item][1]] = -1
                    link_list[cnt] = be_linked_item
                    be_linked[be_linked_item] = [d[cnt], cnt]
    return link_list


def coord_iou(coords_a, coords_b):
    """
    This code comes from https://stackoverflow.com/a/42874377
    :param coords_a:
    :param coords_b:
    :return:
    """
    coords_a = np.array(coords_a).reshape((2, 2))
    coords_b = np.array(coords_b).reshape((2, 2))
    y1, x1 = np.min(coords_a, axis=0)
    y2, x2 = np.max(coords_a, axis=0)
    bb1 = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
    y1, x1 = np.min(coords_b, axis=0)
    y2, x2 = np.max(coords_b, axis=0)
    bb2 = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}

    assert bb1['x1'] <= bb1['x2']
    assert bb1['y1'] <= bb1['y2']
    assert bb2['x1'] <= bb2['x2']
    assert bb2['y1'] <= bb2['y2']

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert 0.0 <= iou <= 1.0
    return iou


def link_pred_gt_box(pred, gt, link_r):
    # link predictions
    link_list = [-1 for _ in range(len(pred))]
    be_linked = {}

    for cnt_1, item_1 in enumerate(pred):
        for cnt_2, item_2 in enumerate(gt):
            iou = coord_iou(item_1, item_2)
            if iou < link_r:
                pass
            else:
                be_linked_item = cnt_2
                if be_linked_item not in be_linked:
                    link_list[cnt_1] = be_linked_item
                    be_linked[be_linked_item] = [iou, cnt_1]
                else:
                    if iou < be_linked[be_linked_item][0]:
                        link_list[be_linked[be_linked_item][1]] = -1
                        link_list[cnt_1] = be_linked_item
                        be_linked[be_linked_item] = [iou, cnt_1]
    return link_list


def order_pair(p1, p2):
    if p1 < p2:
        return p1, p2
    else:
        return p2, p1


def grid_score(tower_gt, tower_pred, link_list, line_gt=None, line_pred=None):
    cnt_obj = 0
    for a in link_list:
        if a > -1:
            cnt_obj += 1
    cnt_pred = 0

    if line_gt and line_pred and link_list:
        lp = []
        for cp in line_pred:
            lp.append(order_pair(*cp))
        lp = list(set(lp))

        for cp in lp:
            if (link_list[cp[0]] > -1) and (link_list[cp[1]] > -1):
                if (link_list[cp[0]], link_list[cp[1]]) in line_gt:
                    cnt_pred += 1
    else:
        line_gt = []
        line_pred = []

    tp = cnt_obj + cnt_pred
    n_recall = len(tower_gt) + len(line_gt)
    n_precision = len(tower_pred) + len(line_pred)

    return tp, n_recall, n_precision


if __name__ == '__main__':
    pass
