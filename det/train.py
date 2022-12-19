# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from email.policy import strict

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.resnet import resnetSHADE

import ipdb
from tqdm import tqdm
import random

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='dc_fpn', type=str)
  parser.add_argument('--net', dest='net',
                    help='vgg16, res101',
                    default='res101', type=str)
  parser.add_argument('--start_epoch', dest='start_epoch',
                      help='starting epoch',
                      default=1, type=int)
  parser.add_argument('--epochs', dest='max_epochs',
                      help='number of epochs to train',
                      default=20, type=int)
  parser.add_argument('--disp_interval', dest='disp_interval',
                      help='number of iterations to display',
                      default=100, type=int)
  parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                      help='number of iterations to display',
                      default=10000, type=int)

  parser.add_argument('--save_dir', dest='save_dir',
                      help='directory to save models', default="shade_models",
                      type=str)
  parser.add_argument('--output_dir', dest='output_dir',
                      help='output dir', default="shade",
                      type=str)
  parser.add_argument('--nw', dest='num_workers',
                      help='number of worker to load data',
                      default=8, type=int)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')                      
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=2, type=int)
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')

# config optimization
  parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
  parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=0.004, type=float)
  parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=8, type=int)
  parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

# set training session
  parser.add_argument('--s', dest='session',
                      help='training session',
                      default=1, type=int)
  #parser.add_argument('--invariant', type=str, default=None,
   #                   help='invariant (E,W,C,N,H)')

# resume trained model
  parser.add_argument('--r', dest='resume',
                      help='resume checkpoint or not',
                      default=False, type=bool)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load model',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load model',
                      default=0, type=int)
# log and diaplay
  parser.add_argument('--use_tfb', dest='use_tfboard',
                      help='whether use tensorboard',
                      action='store_true')

  ## shade
  parser.add_argument('--sc_weight', 
                    help='weight for style consistency',
                    default=10.0, type=float)
  parser.add_argument('--rc_weight', 
                    help='weight for retrospective consistency',
                    default=1.0, type=float)
  parser.add_argument('--ce_weight', 
                    help='weight for cross entropy of the gt boxes',
                    default=0.5, type=float)

  parser.add_argument('--proto_select_epoch', type=int, default=3,
                      help='epoch to select proto')
  
  ## not freeze backbone
  parser.add_argument('--no_freeze', help='not freeze the first several layers',
                      action='store_true')
  ## detect all
  parser.add_argument('--detect_all', help='detect sparately for augsample and origin sample',
                      action='store_true')

  parser.add_argument('--early_stop', type=int, default=10000000,
                      help='iteration to stop for shm')

  parser.add_argument('--no_flip', help='not use fliped data',
                      action='store_true')

  parser.add_argument('--color_tf', help='use color transformation',
                      action='store_true')

  parser.add_argument('--detach_feat', help='detach style hallucinated feat',
                      action='store_true')

  parser.add_argument('--aug_con_only', help='aug sample only con',
                      action='store_true')
  parser.add_argument('--add_classifier', help='additional classifier without bg',
                      action='store_true')

  args = parser.parse_args()
  return args


class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data


def farthest_point_sample_tensor(point, npoint):
    """
    A sampling algorithm from the `"PointNet++: Deep Hierarchical Feature
    Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper, which iteratively samples the
    most distant point with regard to the rest points.
    
    Input:
        point: point data for sampling, [N, D] 
        npoint: number of samples
    Return:
        centroids: sampled point index, [npoint, D]
    """
    device = point.device
    N, D = point.shape
    xyz = point
    centroids = torch.zeros((npoint,), device=device)
    distance = torch.ones((N,), device=device) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, dim=-1)
    point = point[centroids.long()]
    return point, centroids.long()



@torch.no_grad()
def shm_proto_statis(args, model, dataloader, iters_per_epoch):
  model.eval()
  style_list = torch.empty(size=(0,2,64)).cuda()

  data_iter = iter(dataloader)
  for step in tqdm(range(iters_per_epoch)):
    data = next(data_iter)
    features = model.style_feat_obtain(data[0].cuda()) # B,C,H,W
    img_mean = features.mean(dim=[2,3]) # B,C
    img_var = features.var(dim=[2,3]) # B,C
    img_sig = (img_var+1e-7).sqrt()
    img_statis = torch.stack((img_mean, img_sig), dim=1) # B,2,C
    style_list = torch.cat((style_list, img_statis), dim=0)

    if step > args.early_stop:
      break

  print('final style list size : ', style_list.size())
  style_list = style_list.reshape(style_list.size(0), -1).detach()
  proto_styles, centroids = farthest_point_sample_tensor(style_list, 64) # C,2C


  proto_styles = proto_styles.reshape(64, 2, 64)
  proto_mean, proto_std = proto_styles[:,0], proto_styles[:,1]
  # print('proto style shape ===> ',proto_styles.shape)
  model.shm.proto_mean.copy_(proto_mean)
  model.shm.proto_std.copy_(proto_std)

if __name__ == '__main__':

  args = parse_args()

  args.imdb_name = "voc_2007_train_daytimeclear"
  args.imdbval_name = "voc_2007_train_daytimeclear"
  args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
      
  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)
  random_seed = cfg.RNG_SEED  #304
  torch.manual_seed(random_seed)
  torch.cuda.manual_seed(random_seed)
  torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(random_seed)
  random.seed(random_seed)

  #torch.backends.cudnn.benchmark = True
  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  cfg.TRAIN.USE_FLIPPED = not args.no_flip
  cfg.USE_GPU_NMS = args.cuda
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
  # print(type(roidb))
  train_size = len(roidb)

  print('{:d} roidb entries'.format(len(roidb)))


  output_dir = os.path.join('output', args.output_dir) if args.output_dir else args.save_dir + "/" + args.net + "/" + args.dataset 
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  sampler_batch = sampler(train_size, args.batch_size)

  dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
                           imdb.num_classes, training=True, color_trans=args.color_tf)

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers)

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  # initilize the network here.

  if args.net == 'res101':
    fasterRCNN = resnetSHADE(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic, add_loss_dict={'sc_weight': args.sc_weight, 'rc_weight': args.rc_weight, 'ce_weight': args.ce_weight}, no_freeze=args.no_freeze, add_classifier=args.add_classifier)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  lr = cfg.TRAIN.LEARNING_RATE
  lr = args.lr
  #tr_momentum = cfg.TRAIN.MOMENTUM
  #tr_momentum = args.momentum

  params = []
  for key, value in dict(fasterRCNN.named_parameters()).items():
    if value.requires_grad:
      if 'bias' in key:
        params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
      else:
        params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
  # ipdb.set_trace()
  if args.cuda:
    fasterRCNN.cuda()
      
  if args.optimizer == "adam":
    lr = lr * 0.1
    optimizer = torch.optim.Adam(params)

  elif args.optimizer == "sgd":
    optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

  if args.resume:
    load_name = os.path.join(output_dir,
      'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    args.session = checkpoint['session']
    args.start_epoch = checkpoint['epoch']
    fasterRCNN.load_state_dict(checkpoint['model'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr = optimizer.param_groups[0]['lr']
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))

  if args.mGPUs:
    fasterRCNN = nn.DataParallel(fasterRCNN)

  iters_per_epoch = int(train_size / args.batch_size)

  if args.use_tfboard:
    from tensorboardX import SummaryWriter
    logger = SummaryWriter("logs")

  for epoch in range(0, args.max_epochs):
    # setting to train mode
    if epoch % args.proto_select_epoch == 0:
      shm_proto_statis(args, fasterRCNN, dataloader, iters_per_epoch)

    print(fasterRCNN.shm.proto_mean[0])

    fasterRCNN.train()
    loss_temp = 0
    loss_style_con_temp = 0
    loss_retro_con_temp = 0
    loss_ce_con_temp = 0

    start = time.time()

    if epoch % (args.lr_decay_step + 1) == 0:
        adjust_learning_rate(optimizer, args.lr_decay_gamma)
        lr *= args.lr_decay_gamma

    data_iter = iter(dataloader)
    for step in range(iters_per_epoch):
      data = next(data_iter)
      # print(len(data))
      # print(data)
      with torch.no_grad():
              im_data.resize_(data[0].size()).copy_(data[0])
              im_info.resize_(data[1].size()).copy_(data[1])
              gt_boxes.resize_(data[2].size()).copy_(data[2])
              num_boxes.resize_(data[3].size()).copy_(data[3])

      fasterRCNN.zero_grad()

      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label, loss_sc, loss_rc, loss_ce = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)


      # ipdb.set_trace()
      loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
           + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean() + loss_sc.mean() + loss_rc.mean() + loss_ce.mean()
      loss_temp += loss.item()

      loss_style_con_temp += loss_sc.item()
      loss_retro_con_temp += loss_rc.item()
      loss_ce_con_temp += loss_ce.item()

      # backward
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if step % args.disp_interval == 0:
        end = time.time()
        if step > 0:
          loss_temp /= (args.disp_interval + 1)
          loss_retro_con_temp /= (args.disp_interval + 1)
          loss_style_con_temp /= (args.disp_interval + 1)
          loss_ce_con_temp /= (args.disp_interval + 1)

        if args.mGPUs:
          loss_rpn_cls = rpn_loss_cls.mean().item()
          loss_rpn_box = rpn_loss_box.mean().item()
          loss_rcnn_cls = RCNN_loss_cls.mean().item()
          loss_rcnn_box = RCNN_loss_bbox.mean().item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt
        else:
          loss_rpn_cls = rpn_loss_cls.item()
          loss_rpn_box = rpn_loss_box.item()
          loss_rcnn_cls = RCNN_loss_cls.item()
          loss_rcnn_box = RCNN_loss_bbox.item()
          loss_style_con = loss_sc.item()
          loss_retro_con = loss_rc.item()
          loss_ce_con = loss_ce.item()
          fg_cnt = torch.sum(rois_label.data.ne(0))
          bg_cnt = rois_label.data.numel() - fg_cnt

        print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
        print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
        print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f, sc loss: %.4f, rc loss: %.4f, ce loss: %.4f" \
                      % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box, loss_style_con_temp, loss_retro_con_temp, loss_ce_con_temp))


        loss_temp = 0
        loss_style_con_temp = 0
        loss_retro_con_temp = 0
        loss_ce_con_temp = 0

        start = time.time()

    
    save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))

    save_checkpoint({
            'session': args.session,
            'epoch': epoch + 1,
            'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
            'optimizer': optimizer.state_dict(),
            'pooling_mode': cfg.POOLING_MODE,
            'class_agnostic': args.class_agnostic,
            }, save_name)
    print('save model: {}'.format(save_name))

  if args.use_tfboard:
    logger.close()
