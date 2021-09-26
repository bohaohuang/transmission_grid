"""

"""


# Built-in
import os
import json
import shutil
import timeit
import argparse
import datetime

# Libs
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tensorboardX import SummaryWriter

# PyTorch
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Own modules
from data import loader
from utils import misc_utils, metric_utils
from network import StackMTLNet, network_utils

# Settings
CONFIG_FILE = 'config.json'


def read_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=CONFIG_FILE, type=str, help='config file location')

    flags = parser.parse_args()
    return flags


def train_model(args, device, parallel):
    # TODO more options of network
    model = StackMTLNet.StackHourglassNetMTL(args['task1_classes'], args['task2_classes'], args['backbone'])
    log_dir = os.path.join(args['trainer']['save_dir'], 'log')
    writer = SummaryWriter(log_dir=log_dir)
    try:
        writer.add_graph(model, torch.rand(1, 3, *eval(args['dataset']['input_size'])))
    except (RuntimeError, TypeError):
        print('Warning: could not write graph to tensorboard, this might be a bug in tensorboardX')
    if parallel:
        model.encoder = nn.DataParallel(model.encoder, device_ids=[a for a in range(len(args['gpu'].split(',')))])
        model.decoder = nn.DataParallel(model.decoder, device_ids=[a for a in range(len(args['gpu'].split(',')))])

    start_epoch = 0
    if args['resume_dir'] != 'None':
        print('Resume training from {}'.format(args['resume_dir']))
        ckpt = torch.load(args['resume_dir'])
        start_epoch = ckpt['epoch']
        network_utils.load(model, args['resume_dir'], disable_parallel=True)
    elif args['finetune_dir'] != 'None':
        print('Finetune model from {}'.format(args['finetune_dir']))
        network_utils.load(model, args['finetune_dir'], disable_parallel=True)

    model.to(device)

    # make optimizer
    train_params = [
        {'params': model.encoder.parameters(), 'lr': args['optimizer']['e_lr']},
        {'params': model.decoder.parameters(), 'lr': args['optimizer']['d_lr']}
    ]
    optm = optim.SGD(train_params, lr=args['optimizer']['e_lr'], momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optm, milestones=eval(args['optimizer']['lr_drop_epoch']),
                                               gamma=args['optimizer']['lr_step'])
    angle_weights = torch.ones(args['task2_classes']).to(device)
    road_weights = torch.tensor([1-args['task1_classes'], args['task1_classes']], dtype=torch.float).to(device)
    angle_loss = metric_utils.CrossEntropyLoss2d(weight=angle_weights).to(device)
    road_loss = metric_utils.mIoULoss(weight=road_weights).to(device)
    iou_loss = metric_utils.IoU().to(device)

    # prepare training
    print('Total params: {:.2f}M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    # make data loader
    mean = eval(args['dataset']['mean'])
    std = eval(args['dataset']['std'])
    tsfm_train = A.Compose([
        A.Flip(),
        A.RandomRotate90(),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    tsfm_valid = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    train_loader = DataLoader(loader.TransmissionDataLoader(args['dataset']['data_dir'],
                                                            args['dataset']['train_file'], transforms=tsfm_train),
                              batch_size=args['dataset']['batch_size'], shuffle=True,
                              num_workers=args['dataset']['workers'])
    valid_loader = DataLoader(loader.TransmissionDataLoader(args['dataset']['data_dir'],
                                                            args['dataset']['valid_file'], transforms=tsfm_valid),
                              batch_size=args['dataset']['batch_size'], shuffle=False,
                              num_workers=args['dataset']['workers'])
    print('Start training model')
    train_val_loaders = {'train': train_loader, 'valid': valid_loader}

    # train the model
    for epoch in range(start_epoch, args['trainer']['total_epochs']):
        for phase in ['train', 'valid']:
            start_time = timeit.default_timer()
            if phase == 'train':
                model.train()
                scheduler.step()
            else:
                model.eval()

            loss_dict = model.step(train_val_loaders[phase], device, optm, phase, road_loss, angle_loss, iou_loss,
                                   True, mean, std)
            misc_utils.write_and_print(writer, phase, epoch, args['trainer']['total_epochs'], loss_dict, start_time)

        # save the model
        if epoch % args['trainer']['save_epoch'] == (args['trainer']['save_epoch'] - 1):
            save_name = os.path.join(args['trainer']['save_dir'], 'epoch-{}.pth.tar'.format(epoch))
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optm.state_dict(),
                'loss': loss_dict,
            }, save_name)
            print('Saved model at {}'.format(save_name))
    writer.close()


def main(flags):
    config = json.load(open(flags.config))
    current_time = datetime.datetime.now()
    config['trainer']['save_dir'] = os.path.join(config['trainer']['save_dir'], current_time.strftime('%Y%m%d_%H%M%S'))

    # set gpu
    device, parallel = misc_utils.set_gpu(config['gpu'])
    # set random seed
    misc_utils.set_random_seed(config['seed'])
    # make training directory
    misc_utils.make_dir_if_not_exist(config['trainer']['save_dir'])
    shutil.copy(flags.config, config['trainer']['save_dir'])

    # train the model
    train_model(config, device, parallel)


if __name__ == '__main__':
    main(read_flag())
