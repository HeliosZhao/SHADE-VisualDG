from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN
from model.faster_rcnn.style_hallucination import StyleHallucination

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import ipdb
from model.utils.net_utils import _smooth_l1_loss

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
       'resnet152']


model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

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


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    # it is slightly better whereas slower to set stride = 1
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    self.avgpool = nn.AvgPool2d(7)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)

    return x


def resnet18(pretrained=False):
  """Constructs a ResNet-18 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
  return model


def resnet34(pretrained=False):
  """Constructs a ResNet-34 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
  return model


def resnet50(pretrained=False):
  """Constructs a ResNet-50 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
  return model


def resnet101(pretrained=False):
  """Constructs a ResNet-101 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
  return model


def resnet152(pretrained=False):
  """Constructs a ResNet-152 model.
  Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3])
  if pretrained:
    model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
  return model


class resnet(_fasterRCNN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False, no_freeze=False, eval_bn=False):
    self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.no_freeze = no_freeze
    self.eval_bn = eval_bn

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    resnet = resnet101()

    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

    # Build resnet.
    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)

    self.RCNN_top = nn.Sequential(resnet.layer4)

    self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(2048, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

    if not self.no_freeze:
      # Fix blocks
      for p in self.RCNN_base[0].parameters(): p.requires_grad=False
      for p in self.RCNN_base[1].parameters(): p.requires_grad=False

      assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
      if cfg.RESNET.FIXED_BLOCKS >= 3:
        for p in self.RCNN_base[6].parameters(): p.requires_grad=False
      if cfg.RESNET.FIXED_BLOCKS >= 2:
        for p in self.RCNN_base[5].parameters(): p.requires_grad=False
      if cfg.RESNET.FIXED_BLOCKS >= 1:
        for p in self.RCNN_base[4].parameters(): p.requires_grad=False

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      if self.no_freeze:
        if self.eval_bn:
          # Set fixed blocks to be in eval mode
          self.RCNN_base.eval()
          self.RCNN_base[5].train()
          self.RCNN_base[6].train()
          
        else:
          self.RCNN_base.train()
      else:
        # Set fixed blocks to be in eval mode
        self.RCNN_base.eval()
        self.RCNN_base[5].train()
        self.RCNN_base[6].train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7

class resnetSHADE(_fasterRCNN):
  def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False, add_loss_dict=None, no_freeze=False, detach_feat=False, **kwargs):
    self.model_path = 'data/pretrained_model/resnet101_caffe.pth'
    self.dout_base_model = 1024
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.no_freeze = no_freeze
    self.add_classifier = kwargs.get('add_classifier', False)

    if add_loss_dict is not None:
      self.sc_weight = add_loss_dict.get('sc_weight', 0)
      self.rc_weight = add_loss_dict.get('rc_weight', 0)
      self.ce_weight = add_loss_dict.get('ce_weight', 0)

    _fasterRCNN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    resnet = resnet101()

    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

    # Build resnet.
    self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)

    self.RCNN_top = nn.Sequential(resnet.layer4)

    self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(2048, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

    if not self.no_freeze:
      # Fix blocks
      for p in self.RCNN_base[0].parameters(): p.requires_grad=False
      for p in self.RCNN_base[1].parameters(): p.requires_grad=False

      assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
      if cfg.RESNET.FIXED_BLOCKS >= 3:
        for p in self.RCNN_base[6].parameters(): p.requires_grad=False
      if cfg.RESNET.FIXED_BLOCKS >= 2:
        for p in self.RCNN_base[5].parameters(): p.requires_grad=False
      if cfg.RESNET.FIXED_BLOCKS >= 1:
        for p in self.RCNN_base[4].parameters(): p.requires_grad=False

    self.shm = StyleHallucination(0.0156, 64)
    if self.add_classifier:
      self.add_cls_score = nn.Linear(2048, self.n_classes-1)

    if self.rc_weight:
      ### imnet model
      self.RCNN_base_imnet = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)
      self.RCNN_top_imnet = nn.Sequential(resnet.layer4)

    def set_bn_fix(m):
      classname = m.__class__.__name__
      if classname.find('BatchNorm') != -1:
        for p in m.parameters(): p.requires_grad=False

    self.RCNN_base.apply(set_bn_fix)
    self.RCNN_top.apply(set_bn_fix)

  def train(self, mode=True):
    # Override train so that the training mode is set as we want
    nn.Module.train(self, mode)
    if mode:
      if self.no_freeze:
        self.RCNN_base.train()
      else:
        # Set fixed blocks to be in eval mode
        self.RCNN_base.eval()
        self.RCNN_base[5].train()
        self.RCNN_base[6].train()

      def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
          m.eval()

      self.RCNN_base.apply(set_bn_eval)
      self.RCNN_top.apply(set_bn_eval)

  def _head_to_tail(self, pool5):
    fc7 = self.RCNN_top(pool5).mean(3).mean(2)
    return fc7

  def _head_to_tail_imnet(self, pool5):
    fc7 = self.RCNN_top_imnet(pool5).mean(3).mean(2)
    return fc7


  def style_consistency(self, im_prob, aug_prob):
    
    p_mixture = torch.clamp((aug_prob + im_prob) / 2., 1e-7, 1).log()
    consistency_loss = self.sc_weight * (
                F.kl_div(p_mixture, aug_prob, reduction='batchmean') +
                F.kl_div(p_mixture, im_prob, reduction='batchmean') 
                ) / 2.
    return consistency_loss

  @torch.no_grad()
  def style_feat_obtain(self, im_data):
      x = self.RCNN_base[0](im_data)
      x = self.RCNN_base[1](x)
      x = self.RCNN_base[2](x)
      x = self.RCNN_base[3](x)
      return x


  def forward(self, im_data, im_info, gt_boxes, num_boxes):
      batch_size = im_data.size(0)
      # assert not self.sc_weight ## sc should not be used

      ## gt boxes, bs,N,5: 4 coord and 1 label
      if self.training:
        batch_size *= 2
        im_info = im_info.repeat(2,1)
        gt_boxes = gt_boxes.repeat(2,1,1)
        num_boxes = num_boxes.repeat(2)
      
      im_info = im_info.data # size [bs,3]
      gt_boxes = gt_boxes.data ## size [bs,20,5]
      num_boxes = num_boxes.data # size [bs] the number of each box
        

      # feed image data to base model to obtain base feature map
      # base_feat = self.RCNN_base(im_data)
      x = self.RCNN_base[0](im_data)
      x = self.RCNN_base[1](x)
      x = self.RCNN_base[2](x)
      x = self.RCNN_base[3](x)
      if self.training:
        x_tuple = self.shm(x)
        # ipdb.set_trace()
        x = torch.cat(x_tuple, dim=0)
        
      x = self.RCNN_base[4](x)
      x = self.RCNN_base[5](x)
      base_feat = self.RCNN_base[6](x)
      

      # feed base feature map tp RPN to obtain rois
      rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes) # detect box for the two samples separately

      ## rois B,N,5 : the first item in 5 is the batch index

      if self.training:
          roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
          rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
          # batch_size *= 2
          
          rois_label = rois_label.view(-1).long() 
          rois_target = rois_target.view(-1, rois_target.size(2)) 
          rois_inside_ws = rois_inside_ws.view(-1, rois_inside_ws.size(2)) 
          rois_outside_ws = rois_outside_ws.view(-1, rois_outside_ws.size(2)) 
      else:
          rois_label = None
          rois_target = None
          rois_inside_ws = None
          rois_outside_ws = None
          rpn_loss_cls = 0
          rpn_loss_bbox = 0


      if cfg.POOLING_MODE == 'align': ## use this # N,1024,7,7
          pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
      elif cfg.POOLING_MODE == 'pool':
          pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

      # feed pooled features to top model
      pooled_feat = self._head_to_tail(pooled_feat) # 4*128,2048

      # compute bbox offset
      bbox_pred = self.RCNN_bbox_pred(pooled_feat)
      if self.training and not self.class_agnostic:
          # select the corresponding columns according to roi labels
          bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
          bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
          bbox_pred = bbox_pred_select.squeeze(1)

      # compute object classification probability
      cls_score = self.RCNN_cls_score(pooled_feat) # 512,8
      cls_prob = F.softmax(cls_score, 1)
      loss_rc = loss_sc = torch.tensor(0.)

      RCNN_loss_cls = 0
      RCNN_loss_bbox = 0

      if self.training:
          # classification loss
          RCNN_loss_cls = F.cross_entropy(cls_score, rois_label) 
          
          # bounding box regression L1 loss
          RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

          ## get gt boxes rois
          # ipdb.set_trace()
          gt_rois = []
          gt_box_labels = []
          for bs in range(batch_size):
            fg_mask = gt_boxes[bs, :, -1] > 0
            fg_boxes = gt_boxes[bs, fg_mask] # N,5
            _gt_rois = torch.zeros_like(fg_boxes)
            _gt_rois[:, 0] = bs
            _gt_rois[:, 1:] = fg_boxes[:, :4]
            gt_rois.append(_gt_rois)
            gt_box_labels.append(fg_boxes[:, -1])
            
          gt_rois = torch.cat(gt_rois, dim=0)
          gt_box_labels = torch.cat(gt_box_labels, dim=0).long() # Ngt
          # ipdb.set_trace()
          gt_pooled_feat = self.RCNN_roi_align(base_feat, gt_rois.view(-1, 5)) # Ngt,1024,7,7
          gt_pooled_feat = self._head_to_tail(gt_pooled_feat) # Ngt,2048

          if self.rc_weight:
            with torch.no_grad():
              self.RCNN_base_imnet.eval()
              self.RCNN_top_imnet.eval()

              base_feat_imnet = self.RCNN_base_imnet(im_data) # here the bs is not doubled
              pooled_feat_imnet = self.RCNN_roi_align(base_feat_imnet.repeat(2,1,1,1), gt_rois.view(-1, 5))
              pooled_feat_imnet = self._head_to_tail_imnet(pooled_feat_imnet) # 4*128,2048

            loss_rc = self.rc_weight * F.mse_loss(gt_pooled_feat, pooled_feat_imnet)
          

          if self.sc_weight:
            if self.add_classifier:
              gt_cls_score = self.add_cls_score(gt_pooled_feat) # 512,7 no bg
              gt_box_labels -= 1
            else:
              gt_cls_score = self.RCNN_cls_score(gt_pooled_feat) # 512,8
            gt_cls_prob = F.softmax(gt_cls_score, dim=1)
            im_prob, aug_prob = gt_cls_prob.chunk(2, dim=0)
            loss_sc = self.style_consistency(im_prob, aug_prob)
            if self.ce_weight:
              loss_ce = F.cross_entropy(gt_cls_score, gt_box_labels) * self.ce_weight


      cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
      bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

      return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, loss_sc, loss_rc, loss_ce

