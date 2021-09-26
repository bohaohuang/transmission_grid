"""
Train MTL LinkNet34MTL model
"""


# Built-in
import os
import math
import random
from datetime import datetime

# Libs
import numpy as np
import albumentations as A
from model import linknet, stack_module
from utils import util, viz_util
from torch.autograd import Variable
from albumentations.pytorch import ToTensor

# Pytorch
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

# Own modules
import dataset
from loss import CrossEntropyLoss2d, mIoULoss


def weights_init(model, manual_seed=7):
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    random.seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


def summary(model, print_arch=False):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters]) / 1000000.0

    print("*" * 100)
    if print_arch:
        print(model)
    if model.__class__.__name__ == "DataParallel":
        print(
            "Trainable parameters for Model {} : {} M".format(
                model.module.__class__.__name__, params
            )
        )
    else:
        print(
            "Trainable parameters for Model {} : {} M".format(
                model.__class__.__name__, params
            )
        )
    print("*" * 100)


def train(model, optimizer, epoch, task1_classes, task2_classes, train_loader, road_loss, angle_loss,
          train_loss_file, val_loss_file, train_loss_angle_file, val_loss_angle_file):
    train_loss_iou = 0
    train_loss_vec = 0
    model.train()
    optimizer.zero_grad()
    hist = np.zeros((task1_classes, task1_classes))
    hist_angles = np.zeros((task2_classes, task2_classes))
    crop_size = 512
    for i, data in enumerate(train_loader, 0):
        inputsBGR, labels, vecmap_angles = data
        inputsBGR = Variable(inputsBGR.float().cuda())
        outputs, pred_vecmaps = model(inputsBGR)

        loss1 = road_loss(outputs[0], labels[0].long().cuda(), False)
        num_stacks = model.num_stacks
        for idx in range(num_stacks - 1):
            loss1 += road_loss(outputs[idx + 1], labels[0].long().cuda(), False)
        for idx, output in enumerate(outputs[-2:]):
            loss1 += road_loss(output, labels[idx + 1].long().cuda(), False)

        loss2 = angle_loss(pred_vecmaps[0], util.to_variable(vecmap_angles[0]))
        for idx in range(num_stacks - 1):
            loss2 += angle_loss(
                pred_vecmaps[idx + 1], util.to_variable(vecmap_angles[0])
            )
        for idx, pred_vecmap in enumerate(pred_vecmaps[-2:]):
            loss2 += angle_loss(pred_vecmap, util.to_variable(vecmap_angles[idx + 1]))

        outputs = outputs[-1]
        pred_vecmaps = pred_vecmaps[-1]

        train_loss_iou += loss1.detach().item()
        train_loss_vec += loss2.detach().item()

        _, predicted = torch.max(outputs.data, 1)

        correctLabel = labels[-1].view(-1, crop_size, crop_size).long()
        hist += util.fast_hist(
            predicted.view(predicted.size(0), -1).cpu().numpy(),
            correctLabel.view(correctLabel.size(0), -1).cpu().numpy(),
            task1_classes,
        )

        _, predicted_angle = torch.max(pred_vecmaps.data, 1)
        correct_angles = vecmap_angles[-1].view(-1, crop_size, crop_size).long()
        hist_angles += util.fast_hist(
            predicted_angle.view(predicted_angle.size(0), -1).cpu().numpy(),
            correct_angles.view(correct_angles.size(0), -1).cpu().numpy(),
            task2_classes,
        )

        p_accu, miou, road_iou, fwacc = util.performMetrics(
            train_loss_file,
            val_loss_file,
            epoch,
            hist,
            train_loss_iou / (i + 1),
            train_loss_vec / (i + 1),
        )
        p_accu_angle, miou_angle, fwacc_angle = util.performAngleMetrics(
            train_loss_angle_file, val_loss_angle_file, epoch, hist_angles
        )

        viz_util.progress_bar(
            i,
            len(train_loader),
            "Loss: %.6f | VecLoss: %.6f | road miou: %.4f%%(%.4f%%) | angle miou: %.4f%% "
            % (
                train_loss_iou / (i + 1),
                train_loss_vec / (i + 1),
                miou,
                road_iou,
                miou_angle,
            ),
        )

        torch.autograd.backward([loss1, loss2])

        if i % 1 == 0 or i == len(train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()

        del (
            outputs,
            pred_vecmaps,
            predicted,
            correct_angles,
            correctLabel,
            inputsBGR,
            labels,
            vecmap_angles,
        )

    util.performMetrics(
        train_loss_file,
        val_loss_file,
        epoch,
        hist,
        train_loss_iou / len(train_loader),
        train_loss_vec / len(train_loader),
        write=True,
    )
    util.performAngleMetrics(
        train_loss_angle_file, val_loss_angle_file, epoch, hist_angles, write=True
    )


def test(epoch, optimizer, model, task1_classes, task2_classes, val_loader, road_loss, angle_loss,
         train_loss_file, val_loss_file, train_loss_angle_file, val_loss_angle_file, experiment_dir,
         best_accuracy, best_miou):
    model.eval()
    test_loss_iou = 0
    test_loss_vec = 0
    hist = np.zeros((task1_classes, task1_classes))
    hist_angles = np.zeros((task2_classes, task2_classes))
    crop_size = 512
    for i, (inputsBGR, labels, vecmap_angles) in enumerate(val_loader, 0):
        inputsBGR = Variable(
            inputsBGR.float().cuda(), volatile=True, requires_grad=False
        )

        outputs, pred_vecmaps = model(inputsBGR)

        loss1 = road_loss(outputs[0], util.to_variable(labels[0], True), True)
        num_stacks = model.num_stacks
        for idx in range(num_stacks - 1):
            loss1 += road_loss(outputs[idx + 1], util.to_variable(labels[0], True), True)
        for idx, output in enumerate(outputs[-2:]):
            loss1 += road_loss(output, util.to_variable(labels[idx + 1], True), True)

        loss2 = angle_loss(pred_vecmaps[0], util.to_variable(vecmap_angles[0], True))
        for idx in range(num_stacks - 1):
            loss2 += angle_loss(
                pred_vecmaps[idx + 1], util.to_variable(vecmap_angles[0], True)
            )
        for idx, pred_vecmap in enumerate(pred_vecmaps[-2:]):
            loss2 += angle_loss(
                pred_vecmap, util.to_variable(vecmap_angles[idx + 1], True)
            )

        outputs = outputs[-1]
        pred_vecmaps = pred_vecmaps[-1]

        test_loss_iou += loss1.detach().item()
        test_loss_vec += loss2.detach().item()

        _, predicted = torch.max(outputs.data, 1)

        correctLabel = labels[-1].view(-1, crop_size, crop_size).long()
        hist += util.fast_hist(
            predicted.view(predicted.size(0), -1).cpu().numpy(),
            correctLabel.view(correctLabel.size(0), -1).cpu().numpy(),
            task1_classes,
        )

        _, predicted_angle = torch.max(pred_vecmaps.data, 1)
        correct_angles = vecmap_angles[-1].view(-1, crop_size, crop_size).long()
        hist_angles += util.fast_hist(
            predicted_angle.view(predicted_angle.size(0), -1).cpu().numpy(),
            correct_angles.view(correct_angles.size(0), -1).cpu().numpy(),
            task2_classes,
        )

        p_accu, miou, road_iou, fwacc = util.performMetrics(
            train_loss_file,
            val_loss_file,
            epoch,
            hist,
            test_loss_iou / (i + 1),
            test_loss_vec / (i + 1),
            is_train=False,
        )
        p_accu_angle, miou_angle, fwacc_angle = util.performAngleMetrics(
            train_loss_angle_file, val_loss_angle_file, epoch, hist_angles, is_train=False
        )

        viz_util.progress_bar(
            i,
            len(val_loader),
            "Loss: %.6f | VecLoss: %.6f | road miou: %.4f%%(%.4f%%) | angle miou: %.4f%%"
            % (
                test_loss_iou / (i + 1),
                test_loss_vec / (i + 1),
                miou,
                road_iou,
                miou_angle,
            ),
        )
        if i % 100 == 0 or i == len(val_loader) - 1:
            images_path = "{}/images/".format(experiment_dir)
            util.ensure_dir(images_path)
            util.savePredictedProb(
                inputsBGR.data.cpu(),
                labels[-1].cpu(),
                predicted.cpu(),
                F.softmax(outputs, dim=1).data.cpu()[:, 1, :, :],
                predicted_angle.cpu(),
                os.path.join(images_path, "validate_pair_{}_{}.png".format(epoch, i)),
                norm_type='Mean',
            )

        del inputsBGR, labels, predicted, outputs, pred_vecmaps, predicted_angle

    accuracy, miou, road_iou, fwacc = util.performMetrics(
        train_loss_file,
        val_loss_file,
        epoch,
        hist,
        test_loss_iou / len(val_loader),
        test_loss_vec / len(val_loader),
        is_train=False,
        write=True,
    )
    util.performAngleMetrics(
        train_loss_angle_file,
        val_loss_angle_file,
        epoch,
        hist_angles,
        is_train=False,
        write=True,
    )

    if miou > best_miou:
        best_accuracy = accuracy
        best_miou = miou
        util.save_checkpoint(epoch, test_loss_iou / len(val_loader), model, optimizer, best_accuracy, best_miou, experiment_dir)

    return test_loss_iou / len(val_loader)


def main():
    # make model
    task1_classes = 2
    task2_classes = 37
    # model = linknet.LinkNet34MTL(task1_classes, task2_classes)
    model = stack_module.StackHourglassNetMTL(task1_classes, task2_classes)
    model.cuda()

    # make data loader
    data_dir = r'/hdd/pgm/patches_mtl_nz/patches'
    batch_size = 1
    train_file = r'/hdd/pgm/patches_mtl_nz/file_list_train.txt'
    valid_file = r'/hdd/pgm/patches_mtl_nz/file_list_valid.txt'
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    tsfm_train = A.Compose([
        A.Flip(),
        A.RandomRotate90(),
        A.Normalize(mean=mean, std=std),
        ToTensor(sigmoid=False),
    ])
    tsfm_valid = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensor(sigmoid=False),
    ])
    train_loader = DataLoader(dataset.RSDataLoader(data_dir, train_file, transforms=tsfm_train),
                              batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset.RSDataLoader(data_dir, valid_file, transforms=tsfm_valid),
                              batch_size=batch_size, shuffle=False, num_workers=4)

    # prepare training
    experiment_dir = r'/hdd6/Models/line_mtl'
    train_file = "{}/train_loss.txt".format(experiment_dir)
    test_file = "{}/test_loss.txt".format(experiment_dir)
    train_loss_file = open(train_file, "w")
    val_loss_file = open(test_file, "w")
    train_file_angle = "{}/train_angle_loss.txt".format(experiment_dir)
    test_file_angle = "{}/test_angle_loss.txt".format(experiment_dir)
    train_loss_angle_file = open(train_file_angle, "w")
    val_loss_angle_file = open(test_file_angle, "w")

    best_accuracy = 0
    best_miou = 0
    start_epoch = 1
    total_epochs = 120
    lr_drop_epoch = [60, 90, 110]
    lr_step = 0.1
    lr = 1e-3
    seed = 1
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005
    )
    weights_init(model, manual_seed=seed)
    summary(model, print_arch=False)

    scheduler = MultiStepLR(
        optimizer,
        milestones=lr_drop_epoch,
        gamma=lr_step,
    )
    weights = torch.ones(task1_classes).cuda()
    weights_angles = torch.ones(task2_classes).cuda()

    angle_loss = CrossEntropyLoss2d(
        weight=weights_angles, size_average=True, ignore_index=255, reduce=True
    ).cuda()
    road_loss = mIoULoss(
        weight=weights, size_average=True, n_classes=task1_classes
    ).cuda()

    for epoch in range(start_epoch, total_epochs + 1):
        start_time = datetime.now()
        scheduler.step(epoch)
        print("\nTraining Epoch: %d" % epoch)
        train(model, optimizer, epoch, task1_classes, task2_classes, train_loader, road_loss, angle_loss,
              train_loss_file, val_loss_file, train_loss_angle_file, val_loss_angle_file)
        if epoch % 1 == 0:
            print("\nTesting Epoch: %d" % epoch)
            val_loss = test(epoch, optimizer, model, task1_classes, task2_classes, valid_loader, road_loss, angle_loss,
                            train_loss_file, val_loss_file, train_loss_angle_file, val_loss_angle_file, experiment_dir,
                            best_accuracy, best_miou)

        end_time = datetime.now()
        print("Time Elapsed for epoch => {1}".format(epoch, end_time - start_time))


if __name__ == '__main__':
    main()
