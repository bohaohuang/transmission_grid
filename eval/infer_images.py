# Built-in
import os
import sys
import pickle
sys.path.append(r'/home/jordan/Bohao/code/models/research')
sys.path.append(r'/home/jordan/Bohao/code/models/research/object_detection')

# Libs
import imageio
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from natsort import natsorted

# Own modules

# Settings


# Settings
GPU = 0
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


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def make_dirs_if_not_exist(path_name):
    if not os.path.exists(path_name):
        os.makedirs(path_name)


def get_test_images(data_dir, val_per=0.2):
    img_list, lbl_list, mask_list = [], [], []
    for city_name, city_ids in zip(city, idx):
        for city_id in city_ids:
            file_name = city_name.format(city_id)
            val_range = int(len(city_ids) * val_per)
            img_name = os.path.join(data_dir, 'img', '{}.jpg'.format(file_name))
            lbl_name = os.path.join(data_dir, 'gt', '{}.pkl'.format(file_name))
            mask_name = os.path.join(data_dir, 'mask', '{}_mask.png'.format(file_name))
            if city_id <= val_range:
                img_list.append(img_name)
                lbl_list.append(lbl_name)
                mask_list.append(mask_name)
    return img_list, lbl_list, mask_list


def run_inference_for_single_image(image, graph):
    from object_detection.utils import ops as utils_ops

    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def extract_grids(img, patch_size_h, patch_size_w):
    """
    Get patch grids for given image
    :param img:
    :param patch_size_h:
    :param patch_size_w:
    :return:
    """
    h, w, _ = img.shape
    h_cells = int(np.ceil(h / patch_size_h))
    w_cells = int(np.ceil(w / patch_size_w))
    if h % patch_size_h == 0:
        h_steps = np.arange(0, h, patch_size_h).astype(int)
    else:
        h_steps = np.append(np.arange(0, h-patch_size_h, patch_size_h).astype(int), h-patch_size_h)
    if w % patch_size_w == 0:
        w_steps = np.arange(0, w, patch_size_w).astype(int)
    else:
        w_steps = np.append(np.arange(0, w-patch_size_w, patch_size_w).astype(int), w-patch_size_w)
    grid_cell = [[{} for _ in range(w_cells)] for _ in range(h_cells)]
    for i in range(w_cells):
        for j in range(h_cells):
            grid_cell[j][i]['h'] = h_steps[j]
            grid_cell[j][i]['w'] = w_steps[i]
            grid_cell[j][i]['label'] = []
            grid_cell[j][i]['box'] = []
    return grid_cell, h_steps, w_steps


def get_cell_id(box, h_steps, w_steps, patch_size_h, patch_size_w):
    h_id_0, w_id_0 = None, None
    for cnt, hs in enumerate(h_steps):
        if hs <= box[0] < hs + patch_size_h:
            h_id_0 = cnt
            break
    for cnt, ws in enumerate(w_steps):
        if ws <= box[1] < ws + patch_size_w:
            w_id_0 = cnt
    return h_id_0, w_id_0


def make_cell_grid(img, node_info, patch_size, orig_box=False):
    coords, h_steps, w_steps = extract_grids(img, patch_size[0], patch_size[1])
    for item in node_info:
        box = item['box']
        h_id_0, w_id_0 = get_cell_id(box, h_steps, w_steps, patch_size[0], patch_size[1])
        h_start = coords[h_id_0][w_id_0]['h']
        w_start = coords[h_id_0][w_id_0]['w']
        if not orig_box:
            box = [max(box[0]-h_start, 0), max(box[1]-w_start, 0),
                   min(box[2]-h_start, patch_size[0]), min(box[3]-w_start, patch_size[0])]
        else:
            box = [box[0], box[1], box[2], box[3]]
        coords[h_id_0][w_id_0]['label'].append('T')
        coords[h_id_0][w_id_0]['box'].append(box)
    return coords


# settings
def infer_images(model_name, img_size=(500, 500), mask_threshold=0.25,
                 label_map_dir=r'/media/ei-edl01/user/bh163/models/towers/label_map_t.pbtxt',
                 img_dir=r'/media/ei-edl01/data/transmission/eccv'):
    from utils import label_map_util
    from utils import visualization_utils as vis_util

    frozen_file = r'/media/ei-edl01/user/bh163/models/eccv/export_model/{}/frozen_inference_graph.pb'.format(model_name)
    test_imgs, test_lbls, test_masks = get_test_images(img_dir)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(GPU)
    result_dir = r'/media/ei-edl01/data/transmission/eccv/pred'
    save_dir = os.path.join(result_dir, '{}'.format(model_name))
    make_dirs_if_not_exist(save_dir)
    ground_truth_dir = os.path.join(save_dir, 'gt')
    predicted_dir = os.path.join(save_dir, 'pred')
    make_dirs_if_not_exist(ground_truth_dir)
    make_dirs_if_not_exist(predicted_dir)

    # load frozen tf model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # load label map
    category_index = label_map_util.create_category_index_from_labelmap(label_map_dir, use_display_name=True)

    for image_path, lbl_path, mask_path in zip(test_imgs, test_lbls, test_masks):
        print('Scoring {}...'.format(image_path))
        img = imageio.imread(image_path)
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        with open(lbl_path, 'rb') as f:
            data = pickle.load(f)
        node_info, adj_mat, polygons_list = data['node_info'], data['adj_mat'], data['polygons_list']
        lbl = [a for a in node_info if a['label'] != 2]  # remove edge nodes
        mask = imageio.imread(mask_path)
        grid = make_cell_grid(img, lbl, img_size, True)

        patch_cnt = 0
        for line in grid:
            for cell in line:
                patch_cnt += 1
                patch = img[cell['h']:cell['h'] + img_size[0], cell['w']:cell['w'] + img_size[1], :3]
                m = mask[cell['h']:cell['h'] + img_size[0], cell['w']:cell['w'] + img_size[1]]
                ratio = np.sum(m) / (img_size[0] * img_size[1])
                if ratio > mask_threshold:
                    continue

                label = cell['label']
                box = cell['box']
                if len(box) > 0:
                    for b in box:
                        b[0] = max(0, b[0])
                        b[1] = max(0, b[1])
                        b[2] = min(img_size[0], b[2])
                        b[3] = min(img_size[1], b[3])
                        if (b[2] - b[0]) * (b[3] - b[1]) <= 0:
                            box.remove(b)
                else:
                    continue

                text_file_name = 'h{}_w{}_{}_{}'.format(cell['h'], cell['w'], patch_cnt,
                                                        os.path.basename(image_path)[:-3] + 'txt')
                output_dict = run_inference_for_single_image(patch, detection_graph)

                # write predict
                with open(os.path.join(predicted_dir, text_file_name), 'w+') as f:
                    for db, dc, ds in zip(output_dict['detection_boxes'], output_dict['detection_classes'],
                                          output_dict['detection_scores']):
                        left = int(db[1] * img_size[1])
                        top = int(db[0] * img_size[0])
                        right = int(db[3] * img_size[1])
                        bottom = int(db[2] * img_size[0])
                        confidence = ds
                        class_name = category_index[dc]['name']
                        f.write('{} {} {} {} {} {}\n'.format(class_name, confidence, left, top, right, bottom))

                with open(os.path.join(ground_truth_dir, text_file_name), 'w+') as f:
                    for l, b in zip(label, box):
                        f.write('{} {} {} {} {}\n'.format(l, b[1], b[0], b[3], b[2]))


def infer_all_images():
    model_dir = r'/media/ei-edl01/user/bh163/models/eccv/towers'
    model_names = natsorted(os.listdir(model_dir))
    for model_name in model_names:
        infer_images(model_name)


if __name__ == '__main__':
    infer_all_images()
