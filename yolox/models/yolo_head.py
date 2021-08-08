#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import math
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import bboxes_iou

from .losses import IOUloss
from .network_blocks import BaseConv, DWConv

import ipdb

class YOLOXHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        ##对应paper中图(2)的网络结构
        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        ##type(xin)为'tuple'，len(xin)为3
        ##xin[0].shape为[B, 128, 80, 80]，FPN中的P3层，下采样8倍；以640的输入为例，则为80
        ##xin[1].shape为[B, 256, 40, 40]，FPN中的P4层，下采样16倍；以640的输入为例，则为40
        ##xin[2].shape为[B, 512, 20, 20]，FPN中的P5层，下采样32倍；以640的输入为例，则为20
        ##labels.shape为[B, 120, 5]，[cls, x, y, w, h]
        ##imgs.shape为[2, 3, 640, 640]
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            ##当k=0时，[B, 128, 80, 80]->[B, 128, 80, 80]
            ##当k=1时，[B, 256, 40, 40]->[B, 256, 40, 40]
            ##当k=2时，[B, 512, 20, 20]->[B, 512, 20, 20]
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            ##当k=0时，[B, 128, 80, 80]->[B, 128, 80, 80]
            ##当k=1时，[B, 256, 40, 40]->[B, 256, 40, 40]
            ##当k=2时，[B, 512, 20, 20]->[B, 512, 20, 20]
            cls_feat = cls_conv(cls_x)
            ##当k=0时，[B, 128, 80, 80]->[B, 80, 80, 80]
            ##当k=1时，[B, 256, 40, 40]->[B, 80, 40, 40]
            ##当k=2时，[B, 512, 20, 20]->[B, 80, 20, 20]
            cls_output = self.cls_preds[k](cls_feat)

            ##当k=0时，[B, 128, 80, 80]->[B, 128, 80, 80]
            ##当k=1时，[B, 256, 40, 40]->[B, 256, 40, 40]
            ##当k=2时，[B, 512, 20, 20]->[B, 512, 20, 20]
            reg_feat = reg_conv(reg_x)
            ##当k=0时，[B, 128, 80, 80]->[B, 4, 80, 80]
            ##当k=1时，[B, 256, 40, 40]->[B, 4, 40, 40]
            ##当k=2时，[B, 512, 20, 20]->[B, 4, 20, 20]            
            reg_output = self.reg_preds[k](reg_feat)
            ##当k=0时，[B, 128, 80, 80]->[B, 1, 80, 80]
            ##当k=1时，[B, 256, 40, 40]->[B, 1, 40, 40]
            ##当k=2时，[B, 512, 20, 20]->[B, 1, 20, 20]            
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                ##当k=0时，[[B, 4, 80, 80], [B, 1, 80, 80], [B, 80, 80, 80]]->[2, 85, 80, 80]
                ##当k=1时，[[B, 4, 40, 40], [B, 1, 40, 40], [B, 80, 40, 40]]->[2, 85, 40, 40]
                ##当k=2时，[[B, 4, 20, 20], [B, 1, 20, 20], [B, 80, 20, 20]]->[2, 85, 20, 20]
                output = torch.cat([reg_output, obj_output, cls_output], 1)

                ##当k=0时，output.shape为[B, 6400, 85]，grid.shape为[1, 6400, 2]
                ##当k=1时，output.shape为[B, 1600, 85]，grid.shape为[1, 1600, 2]
                ##当k=2时，output.shape为[B,  400, 85]，grid.shape为[1,  400, 2]
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )

                ##当k=0，x_shifts[k].shape为[1, 6400]，y_shifts[k].shape为[1, 6400]
                ##当k=1，x_shifts[k].shape为[1, 1600]，y_shifts[k].shape为[1, 1600]
                ##当k=2，x_shifts[k].shape为[1,  400]，y_shifts[k].shape为[1,  400]
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                ##expanded_strides代表FPN各层特征的stride值映射到原图上的真实stride值
                ##当k=0，stride_this_level为 8，expanded_strides[k].shape为[1, 6400]
                ##当k=1，stride_this_level为16，expanded_strides[k].shape为[1, 1600]
                ##当k=2，stride_this_level为32，expanded_strides[k].shape为[1,  400]
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())
                ##len(outputs)为3
                ##outputs[0].shape为[B, 6400, 85]，FPN中的P3层，下采样8倍；以640的输入为例，则为80
                ##outputs[1].shape为[B, 1600, 85]，FPN中的P4层，下采样16倍；以640的输入为例，则为40
                ##outputs[2].shape为[B,  400, 85]，FPN中的P5层，下采样32倍；以640的输入为例，则为20

            else:
                ##len(outputs)为3
                ##outputs[0].shape为[B, 85, 80, 80]，FPN中的P3层，下采样8倍；以640的输入为例，则为80
                ##outputs[1].shape为[B, 85, 40, 40]，FPN中的P4层，下采样16倍；以640的输入为例，则为40
                ##outputs[2].shape为[B, 85, 20, 20]，FPN中的P5层，下采样32倍；以640的输入为例，则为20
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
            ##imgs.shape为[B, 3, 640, 640]
            ##type(x_shifts)为'list'，x_shifts[0].shape为[1, 6400]，x_shifts[1].shape为[1, 1600]，x_shifts[2].shape为[1, 400]
            ##type(y_shifts)为'list'，y_shifts[0].shape为[1, 6400]，y_shifts[1].shape为[1, 1600]，y_shifts[2].shape为[1, 400]
            ##type(expanded_strides)为'list'，expanded_strides[0].shape为[1, 6400]，expanded_strides[1].shape为[1, 1600]，expanded_strides[2].shape为[1, 400]
            ##labels.shape为[B, 120, 5]，[cls, x, y, w, h]
            ##type(outputs)为'list'，outputs[0].shape为[B, 6400, 85]，outputs[1].shape为[B, 1600, 85]，outputs[2].shape为[B, 400, 85]
            ##[[B, 6400, 85], [B, 1600, 85], [B, 400, 85]]->[B, 8400, 85]
            ##xin[0].dtype为'torch.float32'
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            ##type(outputs)为'list'，len(outputs)为3
            ##outputs[0].shape为[1, 85, 80, 80]
            ##outputs[1].shape为[1, 85, 40, 40]
            ##outputs[2].shape为[1, 85, 20, 20]
            ##self.hw为[torch.Size([80, 80]), torch.Size([40, 40]), torch.Size([20, 20])]
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]

            ##outputs[0].flatten(start_dim=2)：[1, 85, 80, 80]->[1, 85, 6400]
            ##outputs[1].flatten(start_dim=2)：[1, 85, 40, 40]->[1, 85, 1600]
            ##outputs[2].flatten(start_dim=2)：[1, 85, 20, 20]->[1, 85,  400]
            ##[[1, 85, 6400], [1, 85, 1600], [1, 85, 400]]->[1, 85, 8400]->[1, 8400, 85]
            ##outputs.shape为[1, 8400, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)

            if self.decode_in_inference:
                ##xin[0].shape为[B, 128, 80, 80]，FPN中的P3层，下采样8倍；以640的输入为例，则为80
                ##xin[1].shape为[B, 256, 40, 40]，FPN中的P4层，下采样16倍；以640的输入为例，则为40
                ##xin[2].shape为[B, 512, 20, 20]，FPN中的P5层，下采样32倍；以640的输入为例，则为20
                ##xin[0].type()为'torch.FloatTensor'
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        ##outputs.shape为[1, 8400, 85]
        ##dtype为'torch.FloatTensor'
        grids = []
        strides = []
        ##self.strides为[8, 16, 32]
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            # ipdb.set_trace()
            ##当k=0时，(hsize, wsize) = (80, 80)，stride =  8；yv.shape为[80, 80]，第一行是0最后一行是79；xv.shape为[80, 80]，第一列是0最后一列是79
            ##当k=1时，(hsize, wsize) = (40, 40)，stride = 16；yv.shape为[40, 40]，第一行是0最后一行是39；xv.shape为[40, 40]，第一列是0最后一列是39
            ##当k=2时，(hsize, wsize) = (20, 20)，stride = 32；yv.shape为[20, 20]，第一行是0最后一行是19；xv.shape为[20, 20]，第一列是0最后一列是19
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            ##组合成网格坐标
            ##当k=0时，torch.stack((xv, yv), 2)：([80, 80], [80, 80])->[80, 80, 2]->[1, 6400, 2]
            ##当k=1时，torch.stack((xv, yv), 2)：([40, 40], [40, 40])->[40, 40, 2]->[1, 1600, 2]
            ##当k=2时，torch.stack((xv, yv), 2)：([20, 20], [20, 20])->[20, 20, 2]->[1,  400, 2]
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            ##当k=0时，shape为[1, 6400]
            ##当k=1时，shape为[1, 1600]
            ##当k=2时，shape为[1,  400]
            shape = grid.shape[:2]
            ##当k=0时，torch.full((*shape, 1), stride).shape为[1, 6400, 1]，值为8
            ##当k=1时，torch.full((*shape, 1), stride).shape为[1, 1600, 1]，值为16
            ##当k=2时，torch.full((*shape, 1), stride).shape为[1,  400, 1]，值为32
            strides.append(torch.full((*shape, 1), stride))

        ##grids.shape为[1, 8400, 2]
        grids = torch.cat(grids, dim=1).type(dtype)
        ##strides.shape为[1, 8400, 1]
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        ##outputs.shape为[1, 8400, 85]
        return outputs

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        ##imgs.shape为[B, 3, 640, 640]
        ##type(x_shifts)为'list'，x_shifts[0].shape为[1, 6400]，x_shifts[1].shape为[1, 1600]，x_shifts[2].shape为[1, 400]
        ##type(y_shifts)为'list'，y_shifts[0].shape为[1, 6400]，y_shifts[1].shape为[1, 1600]，y_shifts[2].shape为[1, 400]
        ##type(expanded_strides)为'list'，expanded_strides[0].shape为[1, 6400]，expanded_strides[1].shape为[1, 1600]，expanded_strides[2].shape为[1, 400]
        ##labels.shape为[B, 120, 5]，[cls, x, y, w, h]
        ##outputs.shape为[B, 8400, 85]
        ##xin[0].dtype为'torch.float32'

        ##bbox_preds.shape为[B, 8400, 4]
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        ##obj_preds.shape为[B, 8400, 1]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        ##cls_preds.shape为[B, 8400, 80]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        ##[B, 120, 5]->[B, 120]->[B]；代表这个batch_size中每个imgs的GT目标数量
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects

        ##代表神经网络输出的目标总数量；以640x640的图片为例，(640/8)**2 + (640/16)**2 + (640/32)**2 = 8400
        total_num_anchors = outputs.shape[1]
        ##x_shifts.shape为[1, 8400]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        ##y_shifts.shape为[1, 8400]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        ##expanded_strides.shape为[1, 8400]
        ##expanded_strides代表FPN各层特征的stride值映射到原图上的真实stride值
        ##expanded_strides[0, :6400]都等于8
        ##expanded_strides[0, 6400:8000]都等于16
        ##expanded_strides[0, 8000:8400]都等于32
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            ##num_gt代表imgs[batch_idx]图片中的目标数量
            num_gt = int(nlabel[batch_idx])
            ##num_gts代表这个batch_size的所有图片中的目标数量总数
            num_gts += num_gt
            if num_gt == 0:
                ##cls_target.shape为[0, 80]
                cls_target = outputs.new_zeros((0, self.num_classes))
                ##reg_target.shape为[0, 4]
                reg_target = outputs.new_zeros((0, 4))
                ##l1_target.shape为[0, 4]
                l1_target = outputs.new_zeros((0, 4))
                ##obj_target.shape为[8400, 1]
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                ##fg_mask.shape为[8400]
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                ##labels.shape为[B, 120, 5]，120是设置的每张图片中目标数量的最大值(max_labels)
                ##gt_bboxes_per_image代表imgs[batch_idx]图片中的所有目标的bbox；gt_bboxes_per_image.shape为[num_gt, 4]
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                ##gt_classes代表imgs[batch_idx]图片中的所有目标的cls_idx；gt_classes.shape为[num_gt]
                gt_classes = labels[batch_idx, :num_gt, 0]
                ##bbox_preds.shape为[B, 8400, 4]；bboxes_preds_per_image.shape为[8400, 4]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                ##gt_matched_classes.shape为[11]，代表与gt匹配的预测目标的cls_idx
                ##fg_mask.shape为[8400]，元素为True/False；sum(fg_mask)=11，代表网络输出的预测结果中与GT匹配的为True
                ##pred_ious_this_matching.shape为[11]，代表通过标签匹配得到的预测bbox与GT bbox的iou
                ##matched_gt_inds.shape为[11]，代表与11个预测目标所匹配的GT目标的索引值，例如[3, 5, 1, 6, 5, 1, 6, 3, 0, 2, 0]，案例中GT目标数是7
                ##num_fg_img为11，代表通过标签匹配得到的图像中的前景目标(预测结果)的数量
                torch.cuda.empty_cache()
                ##num_fg代表这个batch_size的所有图片中的通过标签匹配得到的前景目标(预测结果)的总数；结合Dynamic K去理解？
                ##注意：num_gts与num_fg的区别，以及num_gt与num_fg_img的区别
                num_fg += num_fg_img

                ##F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes).shape为[11, 80]，将匹配的前景目标的cls转换成ont-hot向量([11]->[11, 80])
                ##pred_ious_this_matching.unsqueeze(-1)：[11]->[11, 1]
                ##将类别信息与bbox的iou信息相乘
                ##cls_target.shape为[11, 80]
                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                ##[8400]->[8400, 1]
                obj_target = fg_mask.unsqueeze(-1)
                ##gt_bboxes_per_image.shape为[num_gt, 4]；reg_target.shape为[11, 4]
                ##reg_target代表与11个pred bbox匹配的gt bbox
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        ##len(cls_targets)为batch_size；cls_targets[k].shape为[11, 80]
        ##len(reg_targets)为batch_size；reg_targets[k].shape为[11, 4]
        ##len(obj_targets)为batch_size；obj_targets[k].shape为[8400, 1]
        ##len(fg_masks)为batch_size；fg_masks[k].shape为[8400]
        # ipdb.set_trace()

        ##在第0维对batch_size个张量进行拼接
        ##[[11, 80], [11, 80]]->[22, 80]
        cls_targets = torch.cat(cls_targets, 0)
        ##[[11, 4], [11, 4]]->[22, 4]
        reg_targets = torch.cat(reg_targets, 0)
        ##[[8400, 1], [8400, 1]]->[16800, 1]
        obj_targets = torch.cat(obj_targets, 0)
        ##[[8400], [8400]]->[16800]
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        ##num_fg为22.0
        num_fg = max(num_fg, 1)

        ##bbox_preds.shape为[2, 8400, 4]，[2, 8400, 4]->[16800, 4]
        ##fg_masks.shape为[16800]，元素为True/False；sum(fg_masks)=22，代表网络输出的预测结果中与GT匹配的为True
        ##[16800, 4]->[22, 4]
        ##reg_targets.shape为[22, 4]
        loss_iou = (
            self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg
        ##obj_preds.shape为[2, 8400, 1]，[2, 8400, 1]->[16800, 1]
        ##obj_targets.shape为[16800, 1]，元素为1/0；sum(obj_targets)=22，代表网络输出的预测结果中与GT匹配的为1
        loss_obj = (
            self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg        
        ##cls_preds.shape为[2, 8400, 80]，[2, 8400, 80]->[16800, 80]
        ##fg_masks.shape为[16800]，元素为True/False；sum(fg_masks)=22，代表网络输出的预测结果中与GT匹配的为True
        ##[16800, 80]->[22, 80]
        ##cls_targets.shape为[22, 80]
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        imgs,
        mode="gpu",
    ):

        ##batch_idx代表这个batch中的第几张图片
        ##num_gt代表imgs[batch_idx]图片中的目标数量；此案例中为40
        ##total_num_anchors代表神经网络输出的目标总数量；以640x640的图片为例，(640/8)**2 + (640/16)**2 + (640/32)**2 = 8400
        ##gt_bboxes_per_image代表imgs[batch_idx]图片中的所有目标的bbox；gt_bboxes_per_image.shape为[num_gt, 4]
        ##gt_classes代表imgs[batch_idx]图片中的所有目标的cls_idx；gt_classes.shape为[num_gt]
        ##bboxes_preds_per_image代表imgs[batch_idx]图片中的所有预测目标的bbox；bbox_preds.shape为[B, 8400, 4]；bboxes_preds_per_image.shape为[8400, 4]
        ##expanded_strides.shape为[1, 8400]
        ##expanded_strides代表FPN各层特征的stride值映射到原图上的真实stride值
        ##expanded_strides[0, :6400]都等于8
        ##expanded_strides[0, 6400:8000]都等于16
        ##expanded_strides[0, 8000:8400]都等于32
        ##x_shifts.shape为[1, 8400]
        ##y_shifts.shape为[1, 8400]
        ##cls_preds.shape为[B, 8400, 80]
        ##bbox_preds.shape为[B, 8400, 4]
        ##obj_preds.shape为[B, 8400, 1]
        ##labels.shape为[B, 120, 5]，[cls, x, y, w, h]
        ##imgs.shape为[B, 3, 640, 640]

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        ##fg_mask.shape为[8400]，元素为True/False；sum(fg_mask)=5632，代表网络输出的预测结果中与GT匹配的为True
        ##is_in_boxes_and_center.shape为[40, 5632]
        ##此案例中的num_gt为40
        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        ##bboxes_preds_per_image.shape为[5632, 4]
        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        ##cls_preds_.shape为[5632, 80]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        ##obj_preds_.shape为[5632,  1]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        ##num_in_boxes_anchor为5632
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        ##pair_wise_ious.shape为[40, 5632]，表示5632个pred box与gt box的IoU
        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        ##gt_cls_per_image.shape为[40, 5632, 80]
        ##gt_classes.shape为[40]，self.num_classes为80；F.one_hot([40], 80)->[40, 80]
        ##num_in_boxes_anchor为5632；[40, 80]->[40, 5632, 80]
        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        ##pair_wise_ious_loss.shape为[40, 5632]
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        ##cls_preds_.shape为[5632, 80]，[5632, 80]->[1, 5632, 80]->[40, 5632, 80]->[40, 5632, 80]
        ##obj_preds_.shape为[5632,  1]，[5632,  1]->[1, 5632,  1]->[40, 5632,  1]->[40, 5632,  1]
        ##cls_preds_.shape为[40, 5632, 80]，[40, 5632, 80] * [40, 5632,  1] -> [40, 5632, 80]
        cls_preds_ = (
            cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
        )
        ##F.binary_cross_entropy([40, 5632, 80], [40, 5632, 80], reduction="none") -> [40, 5632, 80] -> [40, 5632]
        pair_wise_cls_loss = F.binary_cross_entropy(
            cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
        ).sum(-1)
        del cls_preds_

        # ipdb.set_trace()

        ##cost.shape为[40, 5632]
        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        ##cost.shape为[40, 5632]
        ##pair_wise_ious.shape为[40, 5632]，表示5632个pred box与gt box的IoU
        ##gt_classes.shape为[40]
        ##此案例中num_gt为40
        ##fg_mask.shape为[8400]

        ##此案例中num_fg为39
        ##gt_matched_classes.shape为[39]
        ##pred_ious_this_matching.shape为[39]
        ##matched_gt_inds.shape为[39]
        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    ##3.使用每个GT的预测样本确定它需要分配到的正样本数(dynamic k)
    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------

        ##cost.shape为[40, 5632]
        ##pair_wise_ious.shape为[40, 5632]，表示5632个pred box与gt box的IoU
        ##gt_classes.shape为[40]
        ##此案例中num_gt为40
        ##fg_mask.shape为[8400]
        
        ##matching_matrix.shape为[40, 5632]
        matching_matrix = torch.zeros_like(cost)

        ##ious_in_boxes_matrix.shape为[40, 5632]
        ious_in_boxes_matrix = pair_wise_ious

        ##10这个数字并不敏感，在5-15之间几乎都没有影响
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        ##3.1获取与当前GT的iou前10的样本
        ##topk_ious.shape为[40, 10]，表示与40个gt bbox的IoU排前10的pred bbox的IoU
        ##_.shape为[40, 10]，表示与40个gt bbox对应的10个pred bbox的idx([0, 8400))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        ##dynamic_ks.shape为[40]
        ##3.2将这Top10样本的iou求和取整，就为当前GT的dynamic_k，dynamic_k最小保证为1
        ##为什么这么做？这是人为设定的规则，一头大象和一只蚂蚁的dynamic_k肯定是不一样的
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
        	##4.为每个GT去loss最小的前dynamic_k个样本作为正样本
            ##_应该表示最优匹配的cost值
            ##pos_idx表示与gt_idx最优匹配的预测正样本的idx号
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        ##anchor_matching_gt.shape为torch.Size([5632])，用于表示每个预测目标与多少个GT目标相匹配
        anchor_matching_gt = matching_matrix.sum(0)
        ##如果有的预测目标与多个GT目标相匹配
        if (anchor_matching_gt > 1).sum() > 0:
            ##找出这些与多个GT目标相匹配的预测目标，看看该预测目标与哪个GT的cost更小，并找出这个GT的idx
            ##将该预测目标与其它GT的匹配度值置为0
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            ##5.人工去掉同一个样本被分配到多个GT的正样本的情况（全局信息）
            ##对COCO数据集不会对性能有很大影响，对密集场景数据集会有影响
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            ##将该预测目标与这个GT的匹配度值置为1
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        ##看看哪些预测框有与之匹配的GT框，fg_mask_inboxes.shape为[5632]，其元素为True/False，sum(fg_mask_inboxes)=39
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        ##num_fg代表预测框中的前景框(有与之匹配的GT框)的数量；此案例中是39
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        ##matching_matrix.shape为[40, 5632]，sum(fg_mask_inboxes)=39，matching_matrix[:, fg_mask_inboxes].shape为[40, 39]
        ##matched_gt_inds.shape为[39]，表示依次与这39个前景目标匹配的GT目标的idx
        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        ##gt_matched_classes.shape为[39]，代表这39个前景框的目标类别
        gt_matched_classes = gt_classes[matched_gt_inds]

        ##[40, 5632] * [40, 5632] -> [40, 5632] -> [5632] -> [39]
        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
