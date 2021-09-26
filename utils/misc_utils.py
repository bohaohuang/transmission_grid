# Built-in
import os
import time
import json
import timeit
import pickle
from functools import wraps

# Libs
import torch
import torchvision
import imageio
import scipy.signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchsummary import summary

# Own modules


def set_gpu(gpu, enable_benchmark=True):
    """
    Set which gpu to use, also return True as indicator for parallel model if multi-gpu selected
    :param gpu: which gpu to use, could a a string with device ids separated by ','
    :param enable_benchmark: if True, will let CUDNN find optimal set of algorithms for input configuration
    :return: device instance
    """
    if len(str(gpu)) > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        parallel = True
        device = torch.device("cuda:{}".format(','.join([str(a) for a in range(len(gpu.split(',')))])))
        print("Devices being used:", device)
    else:
        parallel = False
        device = torch.device("cuda:{}".format(gpu))
        print("Device being used:", device)
    torch.backends.cudnn.benchmark = enable_benchmark
    return device, parallel


def make_dir_if_not_exist(dir_path):
    """
    Make the directory if it does not exists
    :param dir_path: absolute path to the directory
    :return:
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def timer_decorator(func):
    """
    This is a decorator to print out running time of executing func
    :param func:
    :return:
    """
    @wraps(func)
    def timer_wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        duration = time.time() - start_time
        print('duration: {:.3f}s'.format(duration))
    return timer_wrapper


def str2list(s, sep=',', d_type=int):
    """
    Change a {sep} separated string into a list of items with d_type
    :param s: input string
    :param sep: separator for string
    :param d_type: data type of each element
    :return:
    """
    if type(s) is not list:
        s = [d_type(a) for a in s.split(sep)]
    return s


def load_file(file_name, **kwargs):
    """
    Read data file of given path, use numpy.load if it is in .npy format,
    otherwise use pickle or imageio
    :param file_name: absolute path to the file
    :return: file data, or IOError if it cannot be read by either numpy or pickle or imageio
    """
    try:
        if file_name[-3:] == 'npy':
            data = np.load(file_name)
        elif file_name[-3:] == 'pkl' or file_name[-6:] == 'pickle':
            with open(file_name, 'rb') as f:
                data = pickle.load(f)
        elif file_name[-3:] == 'txt':
            with open(file_name, 'r') as f:
                data = f.readlines()
        elif file_name[-3:] == 'csv':
            data = np.genfromtxt(file_name, delimiter=',', dtype=None, encoding='UTF-8')
        elif file_name[-4:] == 'json':
            data = json.load(open(file_name))
        elif 'pil' in kwargs and kwargs['pil']:
            data = Image.open(file_name)
        else:
            data = imageio.imread(file_name)

        return data
    except Exception:  # so many things could go wrong, can't be more specific.
        raise IOError('Problem loading {}'.format(file_name))


def save_file(file_name, data, fmt='%.8e', sort_keys=True, indent=4):
    """
    Save data file of given path, use numpy.load if it is in .npy format,
    otherwise use pickle or imageio
    :param file_name: absolute path to the file
    :param data: data to save
    :return: file data, or IOError if it cannot be saved by either numpy or or pickle imageio
    """
    try:
        if file_name[-3:] == 'npy':
            np.save(file_name, data)
        elif file_name[-3:] == 'pkl':
            with open(file_name, 'wb') as f:
                pickle.dump(data, f)
        elif file_name[-3:] == 'txt':
            with open(file_name, 'w') as f:
                f.writelines(data)
        elif file_name[-3:] == 'csv':
            np.savetxt(file_name, data, delimiter=',', fmt=fmt)
        elif file_name[-4:] == 'json':
            json.dump(data, open(file_name, 'w'), sort_keys=sort_keys, indent=indent)
        else:
            data = Image.fromarray(data.astype(np.uint8))
            data.save(file_name)
    except Exception:  # so many things could go wrong, can't be more specific.
        raise IOError('Problem saving this data')


def get_img_channel_num(file_name):
    """
    Get #channels of the image file
    :param file_name: absolute path to the image file
    :return: #channels or ValueError
    """
    img = load_file(file_name)
    if len(img.shape) == 2:
        channel_num = 1
    elif len(img.shape) == 3:
        channel_num = img.shape[-1]
    else:
        raise ValueError('Image can only have 2 or 3 dimensions')
    return channel_num


def rotate_list(l):
    """
    Rotate a list of lists
    :param l: list of lists to rotate
    :return:
    """
    return list(map(list, zip(*l)))


def make_center_string(char, length, center_str=''):
    """
    Make one line decoration string that has center_str at the center and surrounded by char
    The total length of the string equals to length
    :param char: character to be repeated
    :param length: total length of the string
    :param center_str: string that shown at the center
    :return:
    """
    return center_str.center(length, char)


def float2str(f):
    """
    Return a string for float number and change '.' to character 'p'
    :param f: float number
    :return: changed string
    """
    return '{}'.format(f).replace('.', 'p')


def stem_string(s, lower=True):
    """
    Return a string that with spaces at the begining or end removed and all casted to lower cases
    :param s: input string
    :param lower: if True, the string will be casted to lower cases
    :return: stemmed string
    """
    if lower:
        return s.strip().lower()
    else:
        return s.strip()


def remove_digits(s):
    """
    Remove digits in the given string
    :param s: input string
    :return: digits removed string
    """
    return ''.join([c for c in s if not c.isdigit()])


def get_digits(s):
    """
    Get digits in the given string, cast to int
    :param s: input string
    :return: int from string
    """
    return int(''.join([c for c in s if c.isdigit()]))


def get_model_summary(model, shape, device=None):
    """
    Get model summary with torchsummary
    :param model: the model to visualize summary
    :param shape: shape of the input data
    :return:
    """
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    summary(model.to(device), shape)


def set_random_seed(seed_):
    """
    Set random seed for torch, cudnn and numpy
    :param seed_: random seed to use, could be your lucky number :)
    :return:
    """
    torch.manual_seed(seed_)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_)


def args_getter(inspect_class):
    """
    Inspect parameters inside a class
    :param inspect_class: the class to be inspected
    :return: a dict of key value pairs of all parameters in this class
    """
    params = {}
    for k, v in inspect_class.__dict__.items():
        if not k.startswith('__'):
            params[k] = v
    return params


def args_writer(file_name, inspect_class):
    """
    Save parameters inside a class into json file
    :param file_name: path to the file to be saved
    :param inspect_class: the class which parameters are going to be saved
    :return:
    """
    params = args_getter(inspect_class)
    save_file(file_name, params, sort_keys=True, indent=4)


def read_tensorboard_csv(file, field='Value', smooth=True, window_size=11, order=2):
    """
    Read values from tensorboard csv files, perform savgol smoothing if user specified
    :param file: the csv file downloaded from the tensorboard
    :param field: the name of the column in the csv file to be read
    :param smooth: if True, perform savgol smoothing on the read data
    :param window_size: window size of the savgol filter
    :param order: order of the savgol filter
    :return: data read from the csv file w/o smoothing
    """
    df = pd.read_csv(file, skipinitialspace=True, usecols=['Step', field])
    if smooth:
        value = scipy.signal.savgol_filter(np.array(df[field]), window_size, order)
    else:
        value = np.array(df[field])
    step = np.array(df['Step'])
    return step, value


def get_default_colors():
    """
    Get plt default colors
    :return: a list of rgb colors
    """
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    return colors


def get_color_list():
    """
    Get default color list in plt, convert hex value to rgb tuple
    :return:
    """
    colors = get_default_colors()
    return [tuple(int(a.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) for a in colors]


def decode_label_map(label, label_num=2, label_colors=None):
    """
    #TODO this could be more efficient
    Decode label prediction map into rgb color map
    :param label: label prediction map
    :param label_num: #distinct classes in ground truth
    :param label_colors: list of tuples with RGB value of label colormap
    :return:
    """
    if len(label.shape) == 3:
        label = np.expand_dims(label, -1)
    n, h, w, c = label.shape
    outputs = np.zeros((n, h, w, 3), dtype=np.uint8)
    if not label_colors:
        color_list = get_color_list()
        label_colors = {}
        for i in range(label_num):
            label_colors[i] = color_list[i]
        label_colors[0] = (255, 255, 255)
    for i in range(n):
        pixels = np.zeros((h, w, 3), dtype=np.uint8)
        for j in range(h):
            for k in range(w):
                pixels[j, k] = label_colors[np.int(label[i, j, k, 0])]
        outputs[i] = pixels
    return outputs


def inv_normalize(img, mean, std):
    """
    Do inverse normalize for images
    :param img: the image to be normalized
    :param mean: the original mean
    :param std: the original std
    :return:
    """
    inv_mean = [-a / b for a, b in zip(mean, std)]
    inv_std = [1 / a for a in std]
    if len(img.shape) == 3:
        return (img - inv_mean) / inv_std
    elif len(img.shape) == 4:
        for i in range(img.shape[0]):
            img[i, :, :, :] = (img[i, :, :, :] - inv_mean) / inv_std
        return img


def change_channel_order(data, to_channel_last=True):
    """
    Switch the image type from channel first to channel last
    :param data: the data to switch the channels
    :param to_channel_last: if True, switch the first channel to the last
    :return: the channel switched data
    """
    if to_channel_last:
        if len(data.shape) == 3:
            return np.rollaxis(data, 0, 3)
        else:
            return np.rollaxis(data, 1, 4)
    else:
        if len(data.shape) == 3:
            return np.rollaxis(data, 2, 0)
        else:
            return np.rollaxis(data, 3, 1)


def make_tb_image(img, lbl, pred, n_class, mean, std, chanel_first=True):
    """
    Make validation image for tensorboard
    :param img: the image to display, has shape N * 3 * H * W
    :param lbl: the label to display, has shape N * C * H * W
    :param pred: the pred to display has shape N * C * H * W
    :param n_class: the number of classes
    :param mean: mean used in normalization
    :param std: std used in normalization
    :param chanel_first: if True, the inputs are in channel first format
    :return:
    """
    pred = np.argmax(pred, 1)
    label_image = decode_label_map(change_channel_order(lbl), n_class)
    pred_image = decode_label_map(pred, n_class)
    img_image = inv_normalize(change_channel_order(img), mean, std) * 255
    banner = np.concatenate([img_image, label_image, pred_image], axis=2).astype(np.uint8)
    if chanel_first:
        banner = change_channel_order(banner, False)
    return banner


def write_and_print(writer, phase, current_epoch, total_epoch, loss_dict, s_time):
    """
    Write loss variables to the tensorboard as well as print log message
    :param writer: tensorboardX SummaryWriter
    :param phase: the current phase, will determine where the variables will be written in tensorboard
    :param current_epoch: current epoch number
    :param total_epoch: total number of epochs
    :param loss_dict: a dictionary with loss name and loss value pairs
    :param s_time: the time before this epoch begins, this is used to calculate duration
    :return:
    """
    loss_str = '[{}] Epoch: {}/{} '.format(phase, current_epoch, total_epoch)
    for loss_name, loss_value in loss_dict.items():
        if loss_name == 'image':
            grid = torchvision.utils.make_grid(loss_value)
            writer.add_image('image/{}_epoch'.format(phase), grid, current_epoch)
        else:
            writer.add_scalar('data/{}_{}_epoch'.format(phase, loss_name), loss_value, current_epoch)
            loss_str += '{}: {:.3f} '.format(loss_name, loss_value)
    print(loss_str)
    stop_time = timeit.default_timer()
    print('Execution time: {}\n'.format(str(stop_time - s_time)))


def compare_figures(images, nrows_ncols, show_axis=False, fig_size=(10, 8), show_fig=True,
                    title_list=None):
    """
    Show three figures in a row, link their axes
    :param img_1: image to show on top left
    :param img_2: image to show at top right
    :param img_3: image to show on bottom left
    :param img_4: image to show on bottom right
    :param show_axis: if False, axes will be hide
    :param fig_size: size of the figure
    :param show_fig: show figure or not
    :param color_bar: if True, add color bar to the last plot
    :return:
    """
    from mpl_toolkits.axes_grid1 import Grid
    if title_list:
        assert len(title_list) == len(images)
    fig = plt.figure(figsize=fig_size)
    grid = Grid(fig, rect=111, nrows_ncols=nrows_ncols, axes_pad=0.25, label_mode='L', share_all=True)
    for i, (ax, img) in enumerate(zip(grid, images)):
        ax.imshow(img)
        if not show_axis:
            ax.axis('off')
        if title_list:
            ax.set_title(title_list[i])
    plt.tight_layout()
    if show_fig:
        plt.show()
