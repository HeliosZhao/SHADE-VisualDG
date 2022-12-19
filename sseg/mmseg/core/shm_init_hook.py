# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from math import inf

import torch
import numpy as np
import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils.data import DataLoader

from mmcv.fileio import FileClient
from mmcv.utils import is_seq_of
from mmcv.runner.hooks.hook import Hook
from mmcv.runner.hooks.logger import LoggerHook
from mmcv.parallel import MMDistributedDataParallel
import copy
import mmcv
import ipdb

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

class SHMHook(Hook):
    """Non-Distributed evaluation hook.

    This hook will regularly perform evaluation in a given interval when
    performing in non-distributed environment.

    Args:
        dataloader (DataLoader): A PyTorch dataloader, whose dataset has
            implemented ``evaluate`` function.
        start (int | None, optional): Evaluation starting epoch. It enables
            evaluation before the training starts if ``start`` <= the resuming
            epoch. If None, whether to evaluate is merely decided by
            ``interval``. Default: None.
        interval (int): Evaluation interval. Default: 1.
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: True.
        save_best (str, optional): If a metric is specified, it would measure
            the best checkpoint during evaluation. The information about best
            checkpoint would be saved in ``runner.meta['hook_msgs']`` to keep
            best score value and best checkpoint path, which will be also
            loaded when resume checkpoint. Options are the evaluation metrics
            on the test dataset. e.g., ``bbox_mAP``, ``segm_mAP`` for bbox
            detection and instance segmentation. ``AR@100`` for proposal
            recall. If ``save_best`` is ``auto``, the first key of the returned
            ``OrderedDict`` result will be used. Default: None.
        rule (str | None, optional): Comparison rule for best score. If set to
            None, it will infer a reasonable rule. Keys such as 'acc', 'top'
            .etc will be inferred by 'greater' rule. Keys contain 'loss' will
            be inferred by 'less' rule. Options are 'greater', 'less', None.
            Default: None.
        test_fn (callable, optional): test a model with samples from a
            dataloader, and return the test results. If ``None``, the default
            test function ``mmcv.engine.single_gpu_test`` will be used.
            (default: ``None``)
        greater_keys (List[str] | None, optional): Metric keys that will be
            inferred by 'greater' comparison rule. If ``None``,
            _default_greater_keys will be used. (default: ``None``)
        less_keys (List[str] | None, optional): Metric keys that will be
            inferred by 'less' comparison rule. If ``None``, _default_less_keys
            will be used. (default: ``None``)
        out_dir (str, optional): The root directory to save checkpoints. If not
            specified, `runner.work_dir` will be used by default. If specified,
            the `out_dir` will be the concatenation of `out_dir` and the last
            level directory of `runner.work_dir`.
            `New in version 1.3.16.`
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details. Default: None.
            `New in version 1.3.16.`
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.

    Note:
        If new arguments are added for EvalHook, tools/test.py,
        tools/eval_metric.py may be affected.
    """

    # Since the key for determine greater or less is related to the downstream
    # tasks, downstream repos may need to overwrite the following inner
    # variable accordingly.


    def __init__(self,
                 dataloader,
                 start=None,
                 interval=1000,
                 num_data=1000,
                 by_epoch=True,
                 out_dir=None,
                 **eval_kwargs):

        if not isinstance(dataloader, DataLoader):
            raise TypeError(f'dataloader must be a pytorch DataLoader, '
                            f'but got {type(dataloader)}')

        if interval <= 0:
            raise ValueError(f'interval must be a positive number, '
                             f'but got {interval}')

        assert isinstance(by_epoch, bool), '``by_epoch`` should be a boolean'

        if start is not None and start < 0:
            raise ValueError(f'The evaluation start epoch {start} is smaller '
                             f'than 0')

        self.dataloader = dataloader
        self.interval = interval
        self.start = start
        self.by_epoch = by_epoch
        self.num_data = num_data

        self.eval_kwargs = eval_kwargs
        self.initial_flag = True

        self.out_dir = out_dir


    def before_train_iter(self, runner):
        """Evaluate the model only at the start of training by iteration."""
        if self.by_epoch or not self.initial_flag:
            return
        if self.start is not None and runner.iter >= self.start:
            self.after_train_iter(runner)
        
        if self.initial_flag: #runner.iter == 0:
            for hook in runner._hooks:
                if isinstance(hook, LoggerHook):
                    hook.after_train_iter(runner)
            runner.log_buffer.clear()

            self._do_evaluate(runner)

        self.initial_flag = False

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        if not (self.by_epoch and self.initial_flag):
            return
        if self.start is not None and runner.epoch >= self.start:
            self.after_train_epoch(runner)
        self.initial_flag = False

    def after_train_iter(self, runner):
        """Called after every training iter to evaluate the results."""
        if not self.by_epoch and self._should_evaluate(runner):
            # Because the priority of EvalHook is higher than LoggerHook, the
            # training log and the evaluating log are mixed. Therefore,
            # we need to dump the training log and clear it before evaluating
            # log is generated. In addition, this problem will only appear in
            # `IterBasedRunner` whose `self.by_epoch` is False, because
            # `EpochBasedRunner` whose `self.by_epoch` is True calls
            # `_do_evaluate` in `after_train_epoch` stage, and at this stage
            # the training log has been printed, so it will not cause any
            # problem. more details at
            # https://github.com/open-mmlab/mmsegmentation/issues/694
            for hook in runner._hooks:
                if isinstance(hook, LoggerHook):
                    hook.after_train_iter(runner)
            runner.log_buffer.clear()

            self._do_evaluate(runner)

    def after_train_epoch(self, runner):
        """Called after every training epoch to evaluate the results."""
        if self.by_epoch and self._should_evaluate(runner):
            self._do_evaluate(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        eval_model = copy.deepcopy(runner.model)
        eval_model.eval()
        prog_bar = mmcv.ProgressBar(min(self.num_data,len(self.dataloader)))
        style_list = []
        # ipdb.set_trace()
        with torch.no_grad():
            for i, data_batch in enumerate(self.dataloader):
                # style_feats = eval_model.module.get_model().extract_style_feat(data_batch['img'].data)
                style_feats = eval_model(**data_batch, return_loss=True, return_style=True)
                img_mean, img_sig = style_feats.mean(dim=[2,3]), style_feats.std(dim=[2,3])
                img_statis = torch.stack((img_mean, img_sig), dim=1) # B,2,C
                style_list.append(img_statis) ## only support one gpu

                prog_bar.update()
                if i >= (self.num_data - 1) :
                    break

        style_list = torch.cat(style_list, dim=0)
        base_style_num = style_list.size(-1)
        style_list = style_list.reshape(style_list.size(0), -1).detach()
        runner.log_buffer.output['style_list_size'] = style_list.size()

        proto_styles, centroids = farthest_point_sample_tensor(style_list, base_style_num) # C,2C
        proto_styles = proto_styles.reshape(base_style_num, 2, base_style_num)
        proto_mean, proto_std = proto_styles[:,0], proto_styles[:,1]
        runner.log_buffer.output['proto_style_size'] = proto_styles.size()

        # ipdb.set_trace()

        # if isinstance(runner.model, MMDistributedDataParallel):
        runner.model.module.model.backbone.shm.proto_mean.copy_(proto_mean)
        runner.model.module.model.backbone.shm.proto_std.copy_(proto_std)
        # else:
        #     runner.model.model.backbone.shm.proto_mean.copy_(proto_mean)
        #     runner.model.model.backbone.shm.proto_std.copy_(proto_std)



    def _should_evaluate(self, runner):
        """Judge whether to perform evaluation.

        Here is the rule to judge whether to perform evaluation:
        1. It will not perform evaluation during the epoch/iteration interval,
           which is determined by ``self.interval``.
        2. It will not perform evaluation if the start time is larger than
           current time.
        3. It will not perform evaluation when current time is larger than
           the start time but during epoch/iteration interval.

        Returns:
            bool: The flag indicating whether to perform evaluation.
        """
        if self.by_epoch:
            current = runner.epoch
            check_time = self.every_n_epochs
        else:
            current = runner.iter
            check_time = self.every_n_iters

        if self.start is None:
            if not check_time(runner, self.interval):
                # No evaluation during the interval.
                return False
        elif (current + 1) < self.start:
            # No evaluation if start is larger than the current time.
            return False
        else:
            # Evaluation only at epochs/iters 3, 5, 7...
            # if start==3 and interval==2
            if (current + 1 - self.start) % self.interval:
                return False
        return True

