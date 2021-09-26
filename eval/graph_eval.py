"""

"""


# Built-in
import os
import pickle
from glob import glob

# Libs
import imageio
import numpy as np
from natsort import natsorted

# Own modules
from utils import misc_utils
from pgm_utils import inference_utils, vis_utils, eval_utils
from transmission import ds_parser
# from pgm_utils import inference_utils, vis_utils, eval_utils

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


def parse_preds(city_name, tile_id, model_name, pred_dir=r'/media/ei-edl01/data/transmission/eccv/pred',
                tower_thresh=0.1):
    pre_dir = os.path.join(pred_dir, model_name, 'pred')
    pre_files = natsorted(glob(os.path.join(pre_dir, '*{}_{}.txt'.format(city_name, tile_id))))
    return get_pred_coords(pre_files, tower_thresh)


def get_pred_coords(txt_files, tower_thresh):
    box, center, conf = [], [], []
    for txt_file in txt_files:
        h, w = os.path.splitext(os.path.basename(txt_file))[0].split('_')[:2]
        h, w = int(''.join([a for a in h if a.isdigit()])), int(''.join([a for a in w if a.isdigit()]))
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            _, cf, left, top, right, bottom = line.strip().split(' ')
            cf, left, top, right, bottom = float(cf), int(left), int(top), int(right), int(bottom)
            if cf > tower_thresh:
                box.append([left+w, top+h, right+w, bottom+h])
                center.append([(left+w+right+w)/2, (top+h+bottom+h)/2][::-1])
                conf.append(cf)
    return box, center, conf


def load_data(region_list, backbone, model_type, val_per=0.2, tower_th=0.3,
              data_dir=r'/media/ei-edl01/data/transmission/eccv',
              model_dir=r'/media/ei-edl01/data/transmission/eccv/pred',
              line_dir=r'/hdd/Results/mrs/line_unet_eccv'):
    eval_ids = []
    for region_name in region_list:
        for city_entry in city_lut[region_name]:
            city_ids = idx[city_entry]
            val_range = int(len(city_ids) * val_per)
            for city_id in range(1, val_range + 1):
                eval_ids.append(city[city_entry].format(city_id))

    for city_name in eval_ids:
        rgb_file = os.path.join(data_dir, 'img', '{}.jpg'.format(city_name))
        lbl_file = os.path.join(data_dir, 'gt', '{}.pkl'.format(city_name))
        line_name = os.path.join(data_dir, 'line', '{}_line.png'.format(city_name))
        mask_name = os.path.join(data_dir, 'mask', '{}_mask.png'.format(city_name))
        img = misc_utils.load_file(rgb_file)
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        mask = misc_utils.load_file(mask_name)
        with open(lbl_file, 'rb') as f:
            data = pickle.load(f)
        node_info, adj_mat, polygons_list = data['node_info'], data['adj_mat'], data['polygons_list']

        # node_info = [a for a in node_info if a['label'] == 1]
        # lbl = eval_detector.exclude_mask_lbl(img, lbl, mask)
        line = misc_utils.load_file(line_name)

        *city_name_prefix, city_id = city_name.split('_')
        city_name, city_id = '_'.join(city_name_prefix), int(city_id)
        model_name = glob(os.path.join(model_dir, 'faster_rcnn_{}_{}_*'.format(backbone, model_type)))
        assert len(model_name) == 1
        pred_box, tower_pred, tower_conf = parse_preds(city_name, city_id, model_name[0], tower_thresh=tower_th)

        line_conf = os.path.join(line_dir, model_type.lower(),
                                 # 'ecresnet50_dcunet_dstransmission_lre1e-02_lrd1e-03_ep80_bs7_ds50_dr0p1_crxent1p0',
                                 '{}_{}.npy'.format(city_name, city_id))
        line_conf = misc_utils.load_file(line_conf)
        yield city_name, city_id, img, node_info, adj_mat, line, pred_box, tower_pred, tower_conf, line_conf


def graph_level_eval(eval_region, backbone, model_type, radius=(1500,), width=5, th=0.02,
                     result_dir=r'/media/ei-edl01/data/transmission/eccv/graph', report_file='parameter_record.txt',
                     save_fig=True):
    tp_all, n_recall_all, n_precision_all = 0, 0, 0
    record_file = os.path.join(result_dir, report_file)

    for data in load_data(eval_region, backbone, model_type, line_dir=r'/hdd/Results/mrs/line_mtl_eccv_temp'):
        tile_name, tile_id, rgb_img, node_info, adj_mat, line, pred_box, tower_pred, tower_conf, line_conf = data

        # if tile_name+str(tile_id) not in include_list:
            # print('out:', tile_name, tile_id)
        #     continue

        print('in:', tile_name, tile_id)

        for r in radius:
            try:
                tower_pairs, tower_dists, line_confs = inference_utils.get_edge_info(
                    tower_pred, line_conf, radius=r, width=width, tile_min=(0, 0), tile_max=rgb_img.shape)

                # connect lines
                connected_pairs = inference_utils.connect_lines(tower_pairs, line_confs, th, cut_n=2)
                connected_pairs, unconnected_pairs = inference_utils.prune_lines(connected_pairs, tower_pred)
            except ValueError:
                connected_pairs = []

            # visualize results
            if save_fig:
                img_save_dir = os.path.join(result_dir, backbone, model_type, '_'.join(eval_region))
                if not os.path.exists(img_save_dir):
                    os.makedirs(img_save_dir)
                vis_utils.visualize_results_eccv(
                    rgb_img, node_info, adj_mat, pred_box, tower_pred, connected_pairs,
                    os.path.join(img_save_dir, '{}_{}_cmp.png'.format(tile_name, tile_id)))

            # graph eval
            try:
                tower_gt = [a['center'] for a in node_info if a['label'] == 1]
                # tower_pred_exclude = [tower_pred[a] for a in range(len(tower_pred)) if a not in
                #                     inference_utils.get_unconnected_towers(tower_pred, connected_pairs)]
                link_list = eval_utils.link_pred_gt(tower_pred, tower_gt, 20)

                tp, n_recall, n_precision = eval_utils.grid_score(tower_gt, tower_pred, link_list,
                                                                  eval_utils.adj_to_cp(adj_mat, node_info), connected_pairs)
            except ValueError:
                tp, n_recall, n_precision = 0, 0, 0

            recall = tp / (n_recall + 1e-6)
            precision = tp / (n_precision + 1e-6)
            f1 = 2 * (recall * precision) / (recall + precision + 1e-6)
            print('{}: f1={:.3f}'.format(tile_name, f1))

            tp_all += tp
            n_recall_all += n_recall
            n_precision_all += n_precision

    recall_all = tp_all / (n_recall_all + 1e-6)
    precision_all = tp_all / (n_precision_all + 1e-6)
    f1_all = 2 * (recall_all * precision_all) / (recall_all + precision_all + 1e-6)
    log_str = 'Backbone: {}, eval type: {}, eval region {}\n\tRadius: {}, width: {}, th: {}, Overall f1={:.2f}\n'.\
        format(backbone, model_type, '_'.join(eval_region), radius, width, th, f1_all)
    with open(record_file, 'a+') as f:
        f.writelines(log_str)
    print(log_str)


def main(model_type, eval_region, model_id, prefix, result_dir, radius=(1500,), width=5, th=0.02, step=5, patch_size=(500, 500)):
    tp_all, n_recall_all, n_precision_all = 0, 0, 0
    for data in load_data(eval_region, model_type, 'All', 'AZ'):
        pass

    exit(0)
    for data in load_data(dirs, model_type, model_id, prefix):
        tile_name, rgb_img, node_info, adj_mat, line_map, tower_pred, tower_conf, line_conf = data
        print('Evaluating {}'.format(tile_name))

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        save_name = os.path.join(result_dir, '{}_{}_{}_result.pkl'.format(model_type, model_id, tile_name))
        if True: # not os.path.exists(save_name):
            # get line confidences
            connected_pairs, connected_towers, unconnected_towers = None, None, None
            for r in radius:
                tower_pairs, tower_dists, line_confs = inference_utils.get_edge_info(
                    tower_pred, line_conf, radius=r, width=width, tile_min=(0, 0), tile_max=rgb_img.shape)

                # connect lines
                connected_pairs = inference_utils.connect_lines(tower_pairs, line_confs, th, cut_n=2)
                connected_pairs, unconnected_pairs = inference_utils.prune_lines(connected_pairs, tower_pred)
                # get towers that are not connected
                connected_towers, unconnected_towers = inference_utils.prune_towers(connected_pairs, tower_pred)
                # search line
                try:
                    connected_towers, unconnected_towers, connected_pairs = \
                        inference_utils.towers_online(tower_pred, connected_towers, unconnected_towers, connected_pairs)
                except ValueError:
                    pass
                # update towers
                inference_utils.break_lines(connected_pairs, tower_pred)
            # check the connection length
            line_length_list, attention_pair = inference_utils.linked_length(tower_pred, connected_pairs)
            for ap in attention_pair:
                pred = []
                for sample_patch, top_left in inference_utils.get_samples_between(rgb_img, tower_pred[ap[0]],
                                                                                  tower_pred[ap[1]], step, patch_size):
                    sample_patch = sample_patch[:, :, :3]
                    # Actual detection.
                    if sample_patch.shape[0] != patch_size[0] or sample_patch.shape[1] != patch_size[1]:
                        continue
                    output_dict = inference_utils.run_inference_for_single_image(sample_patch, detection_graph)

                    for db, dc, ds in zip(output_dict['detection_boxes'], output_dict['detection_classes'],
                                          output_dict['detection_scores']):
                        left = int(db[1] * patch_size[1]) + top_left[1]
                        top = int(db[0] * patch_size[0]) + top_left[0]
                        right = int(db[3] * patch_size[1]) + top_left[1]
                        bottom = int(db[2] * patch_size[0]) + top_left[0]
                        confidence = ds
                        class_name = category_index[dc]['name']
                        if confidence > 0.1:
                            pred.append('{} {} {} {} {} {}\n'.format(class_name, confidence, left, top, right, bottom))
                center_list, conf_list, _ = inference_utils.local_maxima_suppression(pred, th=20)
                tower_pred.extend(center_list)
                tower_conf.extend(conf_list)
            tower_pairs, tower_dists, line_confs = inference_utils.get_edge_info(
                tower_pred, line_conf, radius=radius[-1], width=width, tile_min=(0, 0), tile_max=rgb_img.shape)
            # connect lines
            connected_pairs = inference_utils.connect_lines(tower_pairs, line_confs, th, cut_n=2)
            connected_pairs, unconnected_pairs = inference_utils.prune_lines(connected_pairs, tower_pred)
            # get towers that are not connected
            connected_towers, unconnected_towers = inference_utils.prune_towers(connected_pairs, tower_pred)

            # search line
            try:
                connected_towers, unconnected_towers, connected_pairs = \
                    inference_utils.towers_online(tower_pred, connected_towers, unconnected_towers, connected_pairs)
            except ValueError:
                pass

            # visualize results
            pred_img = vis_utils.visualize_results(rgb_img, node_info, adj_mat, tower_pred, connected_pairs)
            imageio.imsave(os.path.join(result_dir, '{}_cmp.png'.format(tile_name)), pred_img)

            # graph eval
            tower_gt = [a['center'] for a in node_info if a['label'] != 2]
            link_list = eval_utils.link_pred_gt(tower_pred, tower_gt, 20)
            tp, n_recall, n_precision = eval_utils.grid_score(tower_gt, tower_pred, link_list,
                                                              eval_utils.adj_to_cp(adj_mat, node_info), connected_pairs)
            result = {'tp': tp, 'n_recall': n_recall, 'n_precission': n_precision}
            with open(save_name, 'wb') as f:
                pickle.dump(result, f)
        else:
            with open(save_name, 'rb') as f:
                result = pickle.load(f)
                tp, n_recall, n_precision = result['tp'], result['n_recall'], result['n_precission']

        recall = tp / (n_recall + 1e-6)
        precision = tp / (n_precision + 1e-6)
        f1 = 2 * (recall * precision) / (recall + precision + 1e-6)
        print('{}: f1={:.3f}'.format(tile_name, f1))

        tp_all += tp
        n_recall_all += (n_recall + 1e-6)
        n_precision_all += (n_precision + 1e-6)
    recall_all = tp_all / (n_recall_all + 1e-6)
    precision_all = tp_all / (n_precision_all + 1e-6)
    f1_all = 2 * (recall_all * precision_all) / (recall_all + precision_all + 1e-6)
    print('Radius: {}, width: {}, th: {}, Overall: f1={:.3f}'.format(radius[0], width, th, f1_all))


if __name__ == '__main__':
    # main('inception', 'AZ_2019-06-30_12-51-03', 'USA_AZ_', r'/media/ei-edl01/user/bh163/tasks/tower_evaluate_graph_fig',
    #      radius=(2500,), width=9, th=0.05, gpu=0)

    '''for radius in [500, 1000, 1500, 2000, 2500, 3000]:
        for width in [3, 5, 7, 9, 11]:
            for th in [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]:
                graph_level_eval(['NZ', 'KS', 'AZ'], 'inception', 'All', radius=(radius,), width=width, th=th)'''

    '''for data in load_data(['NZ', ], 'inception', 'All'):
        city_name, img, node_info, adj_mat, line, tower_pred, tower_conf, line_conf = data
        print(city_name)'''

    report_name = 'report_th.txt'
    radius, width, th = 2000, 9, 0.2
    for th in [0.2, ]:
        for backbone_name in ['inception']:
            graph_level_eval(['NZ', 'KS', 'AZ'], backbone_name, 'All', radius=(radius,), width=width, th=th,
                             report_file=report_name, save_fig=False)
            '''graph_level_eval(['NZ', ], backbone_name, 'All', radius=(radius,), width=width, th=th,
                             report_file=report_name, save_fig=False)
            graph_level_eval(['KS', ], backbone_name, 'All', radius=(radius,), width=width, th=th,
                             report_file=report_name, save_fig=False)
            graph_level_eval(['AZ', ], backbone_name, 'All', radius=(radius,), width=width, th=th,
                             report_file=report_name, save_fig=False)'''
            '''graph_level_eval(['AZ', ], backbone_name, 'NZKS', radius=(radius,), width=width, th=th,
                             report_file='report.txt')
            graph_level_eval(['KS', ], backbone_name, 'NZAZ', radius=(radius,), width=width, th=th,
                             report_file='report.txt')
            graph_level_eval(['NZ', ], backbone_name, 'AZKS', radius=(radius,), width=width, th=th,
                             report_file='report.txt')
            graph_level_eval(['NZ', ], backbone_name, 'NZ', radius=(radius,), width=width, th=th,
                             report_file='report.txt')
            graph_level_eval(['KS', ], backbone_name, 'KS', radius=(radius,), width=width, th=th,
                             report_file='report.txt')
            graph_level_eval(['AZ', ], backbone_name, 'AZ', radius=(radius,), width=width, th=th,
                             report_file='report.txt')'''
