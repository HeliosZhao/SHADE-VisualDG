import argparse

import torch
#from IPython.core.debugger import set_trace
from torch import nn
#from torch.nn import functional as F
from data import data_helper
## from IPython.core.debugger import set_trace
from data.data_helper import available_datasets
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler
# from utils.Logger import Logger
import numpy as np
from models.resnet import resnet18
import torchvision.transforms as T
import torch.nn.functional as F
# from utils.Logger import AverageMeter
from utils.fps import farthest_point_sample_tensor
import random
import os

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_root', type=str, default='/data/yyzhao/DG-data/pacs/images',
                        help='path of dataset')
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    # data aug stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscaling a tile")
    #
    parser.add_argument("--limit_source", default=None, type=int,
                        help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int,
                        help="If set, it will limit the number of testing samples")
    parser.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate")
    parser.add_argument("--adv_lr", type=float, default=3.0, help="Learning rate for advstyle")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=7, help="Number of classes")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use", default="resnet18")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default='test', help="Used by the logger to save logs")
    parser.add_argument("--sets", default='a-all', help="settings for DG")
    parser.add_argument("--bias_whole_image", default=0.9, type=float, help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", default=False, type=bool, help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", default=True, type=bool, help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", action='store_true', default=False, help="Use nesterov")
    parser.add_argument('--norsc', action='store_true', default=False,
                        help='Do not use RSC, i.e., use EMR')
    parser.add_argument('--seed', type=int, default=0,
                        help='model seed')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--exp_name', type=str, default='',
                        help='experiment name')

    ## shade configs
    parser.add_argument('--SHM', action='store_true', default=False,
                        help='use SHM')
    parser.add_argument("--concentration_coeff", default=0.0156, type=float, help="")
    parser.add_argument('--base_style_num', type=int, default=64,
                    help='num of base style for style space, it should be same with the style dim, and it can also be larger for over modeling')

    parser.add_argument('--proto_select_epoch', type=int, default=3,
                        help='epoch to select proto')
    parser.add_argument('--set_proto_seed', action='store_true', default=True,
                        help='set seed for prototype dataloader')
    parser.add_argument('--proto_trials', type=int, default=1)

    ## style consistency
    parser.add_argument('--sc_weight', type=float, default=10.0,
                        help='weight for consistency loss, e.g. js loss')
    ## retrospection consistency
    parser.add_argument('--rc_weight', type=float, default=0.1,
                        help='weight for each layer feature of retrospection layer')

    parser.add_argument('--output_dir', type=str, default='',
                    help='output dir')

    parser.add_argument('--no_verbose', action='store_true',
                        help=' not show verbose')

    return parser.parse_args()

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        
        model = resnet18(pretrained=True, classes=args.n_classes, SHM=args.SHM, concentration_coeff=args.concentration_coeff, base_style_num=args.base_style_num)
        teacher_model = resnet18(pretrained=True, classes=args.n_classes)
        self.teacher_model = teacher_model.to(device)


        self.model = model.to(device)
        # print(self.model)
        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args, patches=model.is_patch_based())
        self.target_loader = data_helper.get_val_list_dataloader(args, patches=model.is_patch_based())
        # self.target_loader = data_helper.get_val_dataloader(args, patches=model.is_patch_based())
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        # print("Dataset size: train %d, val %d, test %d" % (
        # len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))
        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, args.train_all,
                                                                 nesterov=args.nesterov)
        self.n_classes = args.n_classes
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None
        

    def style_consistency(self, x):
        outputs_sm = F.softmax(x, dim=1) ##  2B,C first B is x, last B is x_new
        im_prob, aug_prob = outputs_sm.chunk(2, dim=0) 
        
        p_mixture = torch.clamp((aug_prob + im_prob) / 2., 1e-7, 1).log()
        consistency_loss = self.args.sc_weight * (
                    F.kl_div(p_mixture, aug_prob, reduction='batchmean') +
                    F.kl_div(p_mixture, im_prob, reduction='batchmean') 
                    ) / 2.
        return consistency_loss
    
    def retrospective_consistency(self, x, feats):
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outs = self.teacher_model(x)
            teacher_feats = torch.cat((teacher_outs['rc_feats'], teacher_outs['rc_feats']), dim=0)

        loss = self.args.rc_weight * F.mse_loss(teacher_feats, feats)

        return loss



    @torch.no_grad()
    def _SHM_init(self, epoch=None):
        self.model.eval()
        style_list = torch.empty(size=(0,2,self.args.base_style_num)).cuda() # 0,2,C
        for _ in range(self.args.proto_trials):
            for it, ((data, jig_l, class_l), d_idx) in enumerate(self.source_loader):
                data, jig_l, class_l, d_idx = data.to(self.device), jig_l.to(self.device), class_l.to(self.device), d_idx.to(self.device)

                data_flip = torch.flip(data, (3,)).detach().clone()
                data = torch.cat((data, data_flip))
                class_l = torch.cat((class_l, class_l))

                features = self.model(data, class_l, not self.args.norsc, epoch, return_style=True)
                img_mean = features.mean(dim=[2,3]) # B,C
                img_var = features.var(dim=[2,3]) # B,C
                img_sig = (img_var+1e-7).sqrt()
                img_statis = torch.stack((img_mean, img_sig), dim=1) # B,2,C
                style_list = torch.cat((style_list, img_statis), dim=0)

        # print('final style list size : ', style_list.size())
        style_list = style_list.reshape(style_list.size(0), -1).detach()
        proto_styles, centroids = farthest_point_sample_tensor(style_list, self.args.base_style_num) # C,2C


        proto_styles = proto_styles.reshape(self.args.base_style_num, 2, self.args.base_style_num)
        proto_mean, proto_std = proto_styles[:,0], proto_styles[:,1]
        # print('proto style shape ===> ',proto_styles.shape)
        self.model.shm.proto_mean.copy_(proto_mean)
        self.model.shm.proto_std.copy_(proto_std)


    def _do_epoch(self, epoch=None):

        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for it, ((data, jig_l, class_l), d_idx) in enumerate(self.source_loader):
            data, jig_l, class_l, d_idx = data.to(self.device), jig_l.to(self.device), class_l.to(self.device), d_idx.to(self.device)
            self.optimizer.zero_grad()

            data_flip = torch.flip(data, (3,)).detach().clone()
            data = torch.cat((data, data_flip))
            class_l = torch.cat((class_l, class_l))

            ####
            self.optimizer.zero_grad()
            self.model.zero_grad()

            if self.args.SHM:
                class_l = torch.cat((class_l, class_l))

            output = self.model(data, class_l, not self.args.norsc, epoch)
            
            # ipdb.set_trace()
            class_logit = output['logits']
            rc_feats = output['rc_feats']


            class_loss = criterion(class_logit, class_l)
            _, cls_pred = class_logit.max(dim=1)

            
            loss_con_style = self.style_consistency(class_logit)
            loss_con_retro = self.retrospective_consistency(data, rc_feats)

            loss = class_loss + loss_con_style + loss_con_retro

            loss.backward()
            self.optimizer.step()

            # print(epoch, it, len(self.source_loader), class_loss.item(), torch.sum(cls_pred == class_l.data).item(), data.shape[0])

            #self.logger.log(it, len(self.source_loader),
             #               {"class": class_loss.item()},
              #              {"class": torch.sum(cls_pred == class_l.data).item(), }, data.shape[0])
            if (it+1) % self.args.print_freq == 0 and not self.args.no_verbose:
                msg = '[epoch {}], [iter {} / {} ], [loss {:0.6f}], [sc loss {:0.6f}], [rc loss {:0.6f}]'.format(
                    epoch, it + 1, len(self.source_loader), class_loss, loss_con_style, loss_con_retro) #  / args.train_batch_size
                print(msg)

            del loss, class_loss, class_logit

        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)

                class_correct = self.do_test(loader)

                class_acc = float(class_correct) / total
                if not self.args.no_verbose:
                    print('Epoch {} \t {} \t Acc {:.4f}'.format(epoch, phase, class_acc))
                self.results[phase][self.current_epoch] = class_acc

    def do_test(self, loader):
        class_correct = 0
        for it, ((data, nouse, class_l), _) in enumerate(loader):
            data, nouse, class_l = data.to(self.device), nouse.to(self.device), class_l.to(self.device)

            output = self.model(data, class_l, False)
            class_logit = output['logits']
            _, cls_pred = class_logit.max(dim=1)
            

            class_correct += torch.sum(cls_pred == class_l.data)

        return class_correct


    def do_training(self):
        #self.logger = Logger(self.args, update_frequency=30)
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            self.scheduler.step()
            #self.logger.new_epoch(self.scheduler.get_lr())
            if self.current_epoch % self.args.proto_select_epoch == 0:
                self._SHM_init()

            self._do_epoch(self.current_epoch)
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        print("Best val %g, corresponding test %g - best test: %g, best epoch: %g" % (
        val_res.max(), test_res[idx_best], test_res.max(), idx_best))
        #self.logger.save_best(test_res[idx_best], test_res.max())
        if self.args.output_dir:
            out_dir = os.path.join('output', self.args.output_dir)
            os.makedirs(out_dir, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(out_dir, self.args.sets+'.pth'))
        return self.model

def set_seed_all(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def main():
    args = get_args()

    if args.sets == 'a-all':
        args.source = ['art_painting']
        args.target = ['sketch', 'cartoon', 'photo']
    elif args.sets == 'c-all':
        args.source = ['cartoon']
        args.target = ['sketch', 'art_painting', 'photo']
    elif args.sets == 'p-all':
        args.source = ['photo']
        args.target = ['sketch', 'art_painting', 'cartoon']
    elif args.sets == 's-all':
        args.source = ['sketch']
        args.target = ['photo', 'art_painting', 'cartoon']
    elif args.sets == 'all-a':
        args.source = ['photo', 'cartoon', 'sketch']
        args.target = ['art_painting']
    elif args.sets == 'all-c':
        args.source = ['art_painting', 'photo', 'sketch']
        args.target = ['cartoon']
    elif args.sets == 'all-p':
        args.source = ['art_painting', 'cartoon', 'sketch']
        args.target = ['photo']
    elif args.sets == 'all-s':
        args.source = ['art_painting', 'photo', 'cartoon']
        args.target = ['sketch']

    # --------------------------------------------
    print("Exp : {} \t Target domain: {}".format(args.exp_name, args.target))
    set_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
