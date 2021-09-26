"""

"""


# Built-in
import os
import json
from glob import glob

# Libs
import albumentations as A
import matplotlib.pyplot as plt
from natsort import natsorted
from albumentations.pytorch import ToTensorV2

# Own modules
from data import data_utils
from utils import misc_utils
from network import StackMTLNet, network_utils

# Settings
# MODEL_DIR = r'/hdd6/Models/mrs/line_mtl_eccv/All/20200213_134627'
EPOCH_NUM = 79
GPU = '0'
EVAL_REGION = ['NZ', 'AZ', 'KS']

MODEL_DIRS = [
    r'/hdd6/Models/mrs/line_mtl_eccv/all/20200213_134627',
    r'/hdd6/Models/mrs/line_mtl_eccv/nz/20200216_075139',
    r'/hdd6/Models/mrs/line_mtl_eccv/az/20200217_114718',
    r'/hdd6/Models/mrs/line_mtl_eccv/ks/20200218_173014',
    r'/hdd6/Models/mrs/line_mtl_eccv/azks/20200213_134558',
    r'/hdd6/Models/mrs/line_mtl_eccv/nzaz/20200218_033908',
    r'/hdd6/Models/mrs/line_mtl_eccv/nzks/20200220_194800'
]

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


def get_eval_ids(region_list, val_per=0.2):
    eval_ids = []
    for region_name in region_list:
        for city_entry in city_lut[region_name]:
            city_ids = idx[city_entry]
            val_range = int(len(city_ids) * val_per)
            for city_id in range(1, val_range+1):
                eval_ids.append(city[city_entry].format(city_id))
    return eval_ids


def main(model_dir):
    print(model_dir)
    config_file = glob(os.path.join(model_dir, 'config*.json'))
    assert len(config_file) == 1
    config = json.load(open(config_file[0]))
    mt = model_dir.split('/')[-2].lower()

    # set gpu
    device, parallel = misc_utils.set_gpu(GPU)
    model = StackMTLNet.StackHourglassNetMTL(config['task1_classes'], config['task2_classes'], config['backbone'])
    network_utils.load(model, os.path.join(model_dir, 'epoch-{}.pth.tar'.format(EPOCH_NUM)), disable_parallel=True)
    model.to(device)
    model.eval()

    # eval on dataset
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    tsfm_valid = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    save_dir = os.path.join(r'/hdd/Results/mrs/line_mtl_eccv_temp', mt)
    evaluator = network_utils.Evaluator('transmission', tsfm_valid, device)
    evaluator.evaluate(model, (512, 512), 0, get_eval_ids(EVAL_REGION),
                       pred_dir=save_dir, report_dir=save_dir, save_conf=True)


def compare_results(model_dir):
    results_dir = os.path.join(r'/hdd/Results/line_mtl_cust', os.path.basename(model_dir))
    image_dir = r'~/Documents/bohao/data/transmission_line/raw2'
    pred_files = natsorted(glob(os.path.join(results_dir, '*.png')))
    conf_files = natsorted(glob(os.path.join(results_dir, '*.npy')))
    for pred_file, conf_file in zip(pred_files, conf_files):
        file_name = os.path.splitext(os.path.basename(pred_file))[0]
        rgb_file = os.path.join(image_dir, '{}.tif'.format(file_name))
        rgb = misc_utils.load_file(rgb_file)[..., :3]
        pred = misc_utils.load_file(pred_file)
        # make line map
        lbl_file = os.path.join(r'/media/ei-edl01/data/remote_sensing_data/transmission_line/parsed_annotation',
                                '{}.csv'.format(file_name))
        boxes, lines = data_utils.csv_to_annotation(lbl_file)
        lbl, _ = data_utils.render_line_graph(rgb.shape[:2], boxes, lines)

        plt.figure(figsize=(15, 8))
        ax1 = plt.subplot(131)
        plt.imshow(rgb)
        plt.axis('off')
        plt.subplot(132, sharex=ax1, sharey=ax1)
        plt.title(file_name.replace('_resize', ''))
        plt.imshow(lbl)
        plt.axis('off')
        plt.subplot(133, sharex=ax1, sharey=ax1)
        plt.imshow(pred)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    for md in MODEL_DIRS:
        main(md)
    # compare_results()
