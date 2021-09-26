"""
This file defines commonly used functions for using networks
"""

# Built-in
import os
import copy
import timeit

# Libs
import scipy.special
import numpy as np

# PyTorch
import torch
import torchvision
from torch import nn
from torchsummary import summary

# Own modules
from utils import misc_utils, metric_utils
from data import data_utils, patch_extractor


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


def iterate_sublayers(network):
    """
    Iterate through all sublayers
    :param network: the network to iterate through
    :return: a list of layers
    """
    all_layers = []
    for layer in network.children():
        if isinstance(layer, nn.Sequential):
            all_layers.extend(iterate_sublayers(layer))
        if not list(layer.children()):
            all_layers.append(layer)
    return all_layers


def network_summary(network, input_size, **kwargs):
    """
    Make a summary of the network, could be used for debugging purpose
    :param network: network to be summarized
    :param input_size: a tuple of the input size
    :param kwargs: other parameters to initialize the model
    :return:
    """
    net = network(**kwargs)
    summary(net, input_size, device='cpu')


def load_epoch(save_dir, resume_epoch, model, optm):
    """
    Load model from a snapshot, this function can be used to resume training
    :param save_dir: directory that saved the model
    :param resume_epoch: the epoch number to continue training
    :param model: the model created by classes defined in network/
    :param optm: a torch optimizer
    :return:
    """
    checkpoint = torch.load(
        os.path.join(save_dir, 'epoch-' + str(resume_epoch - 1) + '.pth.tar'),
        map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU
    print("Initializing weights from: {}...".format(
        os.path.join(save_dir, 'epoch-' + str(resume_epoch - 1) + '.pth.tar')))
    model.load_state_dict(checkpoint['state_dict'])
    optm.load_state_dict(checkpoint['opt_dict'])


def sequential_load(target, source_state):
    new_dict = {}
    odict_list = list(target.items())
    for k, v in odict_list:
        if 'num_batches_tracked' in k:
            # no need to load this
            odict_list.remove((k, v))
    odict = {k: v for k, v in odict_list}
    for (k1, v1), (k2, v2) in zip(odict.items(), source_state.items()):
        new_dict[k1] = v2
    return new_dict


def flex_load(model_dict, ckpt_dict, relax_load=False, disable_parallel=False):
    # try to load model with relaxed naming restriction
    ckpt_params = [a for a in ckpt_dict.keys()]
    self_params = [a for a in model_dict.keys()]

    # only exists in ckpt
    print('Warning: The following parameters in the pretrained model does not exist in the current model')
    model_params = [a for a in ckpt_params if a not in self_params]
    for mp in model_params:
        print('\t', mp)

    # only exists in self
    print('Warning: The following parameters in the current model does not exist in the pretrained model')
    model_params = [a for a in self_params if a not in ckpt_params]
    for mp in model_params:
        print('\t', mp)

    # size not match
    print('Warning: The size of the following parameters in the current model does not match the ones in the '
          'pretrained model')
    model_params = [a for a in ckpt_params if a in self_params and model_dict[a].size() !=
                    ckpt_dict[a].size()]
    for mp in model_params:
        print('\t', mp)

    if not relax_load and not disable_parallel:
        pretrained_state = {k: v for k, v in ckpt_dict.items() if k in model_dict and
                            v.size() == model_dict[k].size()}
        if len(pretrained_state) == 0:
            raise ValueError('No parameter matches in the current model in pretrained model, please check '
                             'the model definition or enable relax_load')
        print('Try loading without those parameters')
        return pretrained_state
    elif disable_parallel:
        pretrained_state = {k: v for k, v in ckpt_dict.items() if k.replace('module.', '') in model_dict and
                            v.size() == model_dict[k.replace('module.', '')].size()}
        if len(pretrained_state) == 0:
            raise ValueError('No parameter matches in the current model in pretrained model, please check '
                             'the model definition or enable relax_load')
        print('Try loading without those parameters')
        print('{:.2f}% of the model loaded from the pretrained'.format(len(pretrained_state) / len(self_params) * 100))
        return pretrained_state
    else:
        print('Try loading with relaxed naming rule:')
        pretrained_state = {}
        # find one match string
        prefix = ''
        for self_name in self_params:
            if self_name in ckpt_params[0]:
                prefix = copy.deepcopy(ckpt_params[0]).replace(self_name, '')
                print('Prefix in pretrained model {}'.format(prefix))
                break
            elif ckpt_params[0] in self_name:
                prefix = copy.deepcopy(self_name).replace(ckpt_params[0], '')
                print('Prefix in current model {}'.format(prefix))
                break

        for self_name in self_params:
            ckpt_name = '{}{}'.format(prefix, self_name)
            if ckpt_name in ckpt_params:
                print('\tpretrained param: {} -> current param: {}'.format(self_name, ckpt_name))
                if model_dict[self_name].size() == ckpt_dict[ckpt_name].size():
                    pretrained_state[self_name] = ckpt_dict[ckpt_name]
                else:
                    print('\t\tIgnoring: {}->{} (size mismatch)'.format(ckpt_name, self_name))

        print('{:.2f}% of the model loaded from the pretrained'.format(len(pretrained_state) / len(self_params) * 100))
        return pretrained_state


def load(model, model_path, relax_load=False, disable_parallel=False):
    """
    Load the weights in the pretrained model directory, the order of loading method is as follows:
    1. Try load the exact name of tensors in the pretrained model file, if not all names are the same, try 2;
    2. Try only load the tensors with names and sizes match, if still no tensor pair found and relax_load, try 3;
    3. Assume the name of one model has a prefix compared with others, find the prefix first, then try load the model
    :param model: the current model that want to load the data weight
    :param model_path: the path to the pretrained model, should be a .pth or equivalent file
    :param relax_load: if true, the model will be load by assuming there's a prefix in one model's tensor names if
                       necessary
    :return:
    """
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    try:
        model.load_state_dict(checkpoint['state_dict'])
    except RuntimeError:
        pretrained_state = flex_load(model.state_dict(), checkpoint['state_dict'], relax_load, disable_parallel)
        model.load_state_dict(pretrained_state, strict=False)


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


class Evaluator:
    def __init__(self, ds_name, tsfm, device):
        self.tsfm = tsfm
        self.device = device
        if ds_name == 'transmission':
            self.img_dir = r'/media/ei-edl01/data/transmission/eccv/img'
            self.lbl_dir = r'/media/ei-edl01/data/transmission/eccv/line'

    def evaluate(self, model, patch_size, overlap, file_ids, pred_dir=None, report_dir=None, save_conf=False,
                 delta=1e-6):
        iou_a, iou_b = 0, 0
        report = []
        if pred_dir:
            misc_utils.make_dir_if_not_exist(pred_dir)

        for file_id in file_ids:
            img_file = os.path.join(self.img_dir, '{}.jpg'.format(file_id))
            lbl_file = os.path.join(self.lbl_dir, '{}_line.png'.format(file_id))
            rgb = misc_utils.load_file(img_file)
            lbl = misc_utils.load_file(lbl_file)

            # evaluate on tiles
            tile_dim = rgb.shape[:2]
            grid_list = patch_extractor.make_grid(tile_dim, patch_size, overlap)
            tile_preds = []
            for patch in patch_extractor.patch_block(rgb, 0, grid_list, patch_size, False):
                for tsfm in self.tsfm:
                    tsfm_image = tsfm(image=patch)
                    patch = tsfm_image['image']
                patch = torch.unsqueeze(patch, 0).to(self.device)
                pred = model.inference(patch).detach().cpu().numpy()
                tile_preds.append(change_channel_order(pred, True)[0, :, :, :])
            # stitch back to tiles
            tile_preds = patch_extractor.unpatch_block(
                np.array(tile_preds),
                tile_dim,
                patch_size,
                tile_dim,
                patch_size,
                overlap=0
            )

            if save_conf:
                misc_utils.save_file(os.path.join(pred_dir, '{}_line.npy'.format(file_id)),
                                     scipy.special.softmax(tile_preds, axis=-1)[:, :, 1])
            tile_preds = np.argmax(tile_preds, -1)
            a, b = metric_utils.iou_metric(lbl, tile_preds)
            file_name = os.path.splitext(os.path.basename(lbl_file))[0]
            print('{}: IoU={:.2f}'.format(file_name, a/(b+delta)*100))
            report.append('{},{},{},{}\n'.format(file_name, a, b, a/(b+delta)*100))
            iou_a += a
            iou_b += b
            if pred_dir:
                misc_utils.save_file(os.path.join(pred_dir, '{}.png'.format(file_name)), tile_preds)
        print('Overall: IoU={:.2f}'.format(iou_a/iou_b*100))
        report.append('Overall,{},{},{}\n'.format(iou_a, iou_b, iou_a/(iou_b+delta)*100))
        if report_dir:
            misc_utils.make_dir_if_not_exist(report_dir)
            misc_utils.save_file(os.path.join(report_dir, 'result.txt'), report)
