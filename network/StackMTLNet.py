"""
The basic structure of the module comes from
https://github.com/anilbatra2185/road_connectivity/blob/master/model/stack_module.py
This is basically a reimplement version of it with some minor modifications
1. Shared encoder could be a well known backbone
"""


# Built-in
import math

# Libs
from tqdm import tqdm

# PyTorch
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

# Own modules
from utils import misc_utils
from network.backbones import encoders


class BasicResnetBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, padding=1, downsample=None):
        super(BasicResnetBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                )
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, group=1):
        super(DecoderBlock, self).__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1, groups=group)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4,
            in_channels // 4,
            3,
            stride=2,
            padding=1,
            output_padding=1,
            groups=group,
        )
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1, groups=group)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class HourglassModuleMTL(nn.Module):
    def __init__(self, block, num_blocks, inplanes, outplanes, depth):
        super(HourglassModuleMTL, self).__init__()
        self.depth = depth
        self.block = block
        self.upsample = nn.Upsample(scale_factor=2)
        self.hg = self._make_hour_glass(block, num_blocks, inplanes, outplanes, depth)

    def _make_residual1(self, block, num_blocks, inplances, outplanes):
        layers = []
        for i in range(0, num_blocks):
            if i == 0:
                layers.append(block(inplances, outplanes))
            else:
                layers.append(block(outplanes, outplanes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, inplanes, outplanes, depth):
        hg = []
        for i in range(depth):
            if i != depth - 1:
                inplanes_temp = outplanes
            else:
                inplanes_temp = inplanes
            res = []
            for j in range(4):
                if j <= 1:
                    res.append(self._make_residual1(block, num_blocks, inplanes_temp, outplanes))
                else:
                    res.append(self._make_residual1(block, num_blocks, outplanes, outplanes))
            if i == 0:
                res.append(self._make_residual1(block, num_blocks, outplanes, outplanes))
                res.append(self._make_residual1(block, num_blocks, outplanes, outplanes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        rows = x.size(2)
        cols = x.size(3)
        up1 = self.hg[n - 1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2, ceil_mode=True)
        low1 = self.hg[n - 1][1](low1)

        if n > 1:
            low2_1, low2_2 = self._hour_glass_forward(n - 1, low1)
        else:
            low2_1 = self.hg[n - 1][4](low1)
            low2_2 = self.hg[n - 1][5](low1)
        low3_1 = self.hg[n - 1][2](low2_1)
        low3_2 = self.hg[n - 1][3](low2_2)
        up2_1 = self.upsample(low3_1)
        up2_2 = self.upsample(low3_2)
        out_1 = up1 + up2_1[:, :, :rows, :cols]
        out_2 = up1 + up2_2[:, :, :rows, :cols]

        return out_1, out_2

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class MTLDecoder(nn.Module):
    def __init__(self, ch, num_stacks, num_feats, inplanes, task1_classes=2, task2_classes=37, block=BasicResnetBlock,
                 hg_num_blocks=3, depth=3):
        super(MTLDecoder, self).__init__()
        self.num_stacks = num_stacks
        self.num_feats = num_feats
        self.inplanes = inplanes
        self.relu = nn.ReLU(inplace=True)
        # build hourglass modules
        hg = []
        res_1, fc_1, score_1, _fc_1, _score_1 = [], [], [], [], []
        res_2, fc_2, score_2, _fc_2, _score_2 = [], [], [], [], []

        for i in range(self.num_stacks):
            hg.append(HourglassModuleMTL(block, hg_num_blocks, ch, self.num_feats, depth))

            res_1.append(self._make_residual(block, self.num_feats, self.num_feats, hg_num_blocks))
            res_2.append(self._make_residual(block, self.num_feats, self.num_feats, hg_num_blocks))

            fc_1.append(self._make_fc(self.num_feats, self.num_feats))
            fc_2.append(self._make_fc(self.num_feats, self.num_feats))

            score_1.append(nn.Conv2d(self.num_feats, task1_classes, kernel_size=1, bias=True))
            score_2.append(nn.Conv2d(self.num_feats, task2_classes, kernel_size=1, bias=True))
            if i < self.num_stacks - 1:
                _fc_1.append(nn.Conv2d(self.num_feats, ch, kernel_size=1, bias=True))
                _fc_2.append(nn.Conv2d(self.num_feats, ch, kernel_size=1, bias=True))
                _score_1.append(nn.Conv2d(task1_classes, ch, kernel_size=1, bias=True))
                _score_2.append(nn.Conv2d(task2_classes, ch, kernel_size=1, bias=True))

        self.hg = nn.ModuleList(hg)
        self.res_1 = nn.ModuleList(res_1)
        self.fc_1 = nn.ModuleList(fc_1)
        self.score_1 = nn.ModuleList(score_1)
        self._fc_1 = nn.ModuleList(_fc_1)
        self._score_1 = nn.ModuleList(_score_1)

        self.res_2 = nn.ModuleList(res_2)
        self.fc_2 = nn.ModuleList(fc_2)
        self.score_2 = nn.ModuleList(score_2)
        self._fc_2 = nn.ModuleList(_fc_2)
        self._score_2 = nn.ModuleList(_score_2)

        # Final Classifier
        self.decoder1 = DecoderBlock(self.num_feats, self.inplanes)
        self.decoder1_score = nn.Conv2d(
            self.inplanes, task1_classes, kernel_size=1, bias=True
        )
        self.finaldeconv1 = nn.ConvTranspose2d(self.inplanes, 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, task1_classes, 2, padding=1)

        # Final Classifier
        self.angle_decoder1 = DecoderBlock(self.num_feats, self.inplanes)
        self.angle_decoder1_score = nn.Conv2d(
            self.inplanes, task2_classes, kernel_size=1, bias=True
        )
        self.angle_finaldeconv1 = nn.ConvTranspose2d(self.inplanes, 32, 3, stride=2)
        self.angle_finalrelu1 = nn.ReLU(inplace=True)
        self.angle_finalconv2 = nn.Conv2d(32, 32, 3)
        self.angle_finalrelu2 = nn.ReLU(inplace=True)
        self.angle_finalconv3 = nn.Conv2d(32, task2_classes, 2, padding=1)

    def forward(self, x, rows, cols):
        out_1 = []
        out_2 = []

        for i in range(self.num_stacks):
            y1, y2 = self.hg[i](x)
            y1, y2 = self.res_1[i](y1), self.res_2[i](y2)
            y1, y2 = self.fc_1[i](y1), self.fc_2[i](y2)

            score1, score2 = self.score_1[i](y1), self.score_2[i](y2)

            out_1.append(
                score1[:, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))]
            )
            out_2.append(
                score2[:, :, : int(math.ceil(rows / 4.0)), : int(math.ceil(cols / 4.0))]
            )
            if i < self.num_stacks - 1:
                _fc_1, _fc_2 = self._fc_1[i](y1), self._fc_2[i](y2)
                _score_1, _score_2 = self._score_1[i](score1), self._score_2[i](score2)
                x = x + _fc_1 + _score_1 + _fc_2 + _score_2

        # Final Classification
        d1 = self.decoder1(y1)[
            :, :, : int(math.ceil(rows / 2.0)), : int(math.ceil(cols / 2.0))
        ]
        d1_score = self.decoder1_score(d1)
        out_1.append(d1_score)
        f1 = self.finaldeconv1(d1)
        f2 = self.finalrelu1(f1)
        f3 = self.finalconv2(f2)
        f4 = self.finalrelu2(f3)
        f5 = self.finalconv3(f4)
        out_1.append(f5)

        # Final Classification
        a_d1 = self.angle_decoder1(y2)[
            :, :, : int(math.ceil(rows / 2.0)), : int(math.ceil(cols / 2.0))
        ]
        a_d1_score = self.angle_decoder1_score(a_d1)
        out_2.append(a_d1_score)
        a_f1 = self.angle_finaldeconv1(a_d1)
        a_f2 = self.angle_finalrelu1(a_f1)
        a_f3 = self.angle_finalconv2(a_f2)
        a_f4 = self.angle_finalrelu2(a_f3)
        a_f5 = self.angle_finalconv3(a_f4)
        out_2.append(a_f5)

        return out_1, out_2

    def _make_residual(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                )
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(outplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(conv, bn, self.relu)


class StackHourglassNetMTL(nn.Module):
    def __init__(
        self,
        task1_classes=2,
        task2_classes=37,
        backbone='resnet34',
        block=BasicResnetBlock,
        num_stacks=2,
        hg_num_blocks=3,
        depth=3,
        pretrained=True
    ):
        super(StackHourglassNetMTL, self).__init__()
        # settings
        self.task1_classes = task1_classes
        self.inplanes = 512
        self.num_stacks = num_stacks
        self.num_feats = 128

        # make encoder
        self.encoder = encoders.models(backbone, pretrained, (2, 2, 1, 1, 1), False)
        ch = self.encoder.chans[0]
        self.decoder = MTLDecoder(ch, self.num_stacks, self.num_feats, self.inplanes, task1_classes, task2_classes,
                                  block, hg_num_blocks, depth)

    def forward(self, x):
        rows = x.size(2)
        cols = x.size(3)

        x = self.encoder(x)
        out_1, out_2 = self.decoder(x, rows, cols)
        return out_1, out_2

    def inference(self, x):
        rows = x.size(2)
        cols = x.size(3)

        x = self.encoder(x)
        out_1, out_2 = self.decoder(x, rows, cols)
        return out_1[-1]

    def step(self, data_loader, device, optm, phase, road_criterion, angle_criterion, iou_criterion, save_image=True,
             mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        loss_dict = {}
        num_stacks = self.num_stacks
        for img_cnt, (image, label, angle) in enumerate(tqdm(data_loader, desc='{}'.format(phase))):
            image = Variable(image, requires_grad=True).to(device)
            label_4 = F.interpolate(label, scale_factor=1/4).long().to(device)
            label_2 = F.interpolate(label, scale_factor=1/2).long().to(device)
            label = Variable(label).long().to(device)
            angle_4 = F.interpolate(angle, scale_factor=1 / 4).long().to(device)
            angle_2 = F.interpolate(angle, scale_factor=1 / 2).long().to(device)
            angle = Variable(angle).long().to(device)
            optm.zero_grad()

            # forward step
            if phase == 'train':
                pred_lbl, pred_ang = self.forward(image)
            else:
                with torch.autograd.no_grad():
                    pred_lbl, pred_ang = self.forward(image)

            # loss after encoder
            loss1 = road_criterion(pred_lbl[0], label_4, False)
            loss2 = angle_criterion(pred_ang[0], angle_4)
            # loss after each stack
            for idx in range(num_stacks - 1):
                loss1 += road_criterion(pred_lbl[idx + 1], label_4, False)
                loss2 += angle_criterion(pred_ang[idx + 1], angle_4)
            # loss after decoders
            loss1 += road_criterion(pred_lbl[-2], label_2, False)
            loss1 += road_criterion(pred_lbl[-1], label, False)
            loss2 += angle_criterion(pred_ang[-2], angle_2)
            loss2 += angle_criterion(pred_ang[-1], angle)
            # IoU loss
            loss3 = iou_criterion(pred_lbl[-1], label)
            iou_criterion.update(loss3, image.size(0))

            pred_lbl, pred_ang = pred_lbl[-1], pred_ang[-1]

            if phase == 'train':
                torch.autograd.backward([loss1, loss2])
                optm.step()
            road_criterion.update(loss1, image.size(0))
            angle_criterion.update(loss2, image.size(0))

            if save_image and img_cnt == 0:
                img_image = image.detach().cpu().numpy()
                lbl_image = label.cpu().numpy()
                pred_image = pred_lbl.detach().cpu().numpy()
                banner = misc_utils.make_tb_image(img_image, lbl_image, pred_image, self.task1_classes, mean, std)
                loss_dict['image'] = torch.from_numpy(banner)
        loss_dict[road_criterion.name] = road_criterion.get_loss()
        loss_dict[angle_criterion.name] = angle_criterion.get_loss()
        loss_dict[iou_criterion.name] = iou_criterion.get_loss()
        road_criterion.reset()
        angle_criterion.reset()
        iou_criterion.reset()

        return loss_dict


if __name__ == '__main__':
    model = StackHourglassNetMTL()
    import torch
    x = torch.randn((1, 3, 512, 512))
    y = model.inference(x)
    print(y.shape)
