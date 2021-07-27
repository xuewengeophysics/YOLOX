#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN

import ipdb

class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            ##num_classes为80
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None):
        ##x.shape为[B, 3, 640, 640]
        ##targets.shape为[B, 120, 5]，[cls, x, y, w, h]
        # fpn output content features of [dark3, dark4, dark5]
        ##type(fpn_outs)为'tuple'，len(fpn_outs)为3
        ##fpn_outs[0].shape为[B, 128, 80, 80]，FPN中的P3层，下采样8倍；以640的输入为例，则为80
        ##fpn_outs[1].shape为[B, 256, 40, 40]，FPN中的P4层，下采样16倍；以640的输入为例，则为40
        ##fpn_outs[2].shape为[B, 512, 20, 20]，FPN中的P5层，下采样32倍；以640的输入为例，则为20
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )

            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs
