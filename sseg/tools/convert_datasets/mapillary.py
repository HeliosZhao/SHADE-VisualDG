import argparse
import json
import os.path as osp

import mmcv
import numpy as np
from PIL import Image


id_to_trainid = {}


def gen_id_to_ignore():
    global id_to_trainid
    for i in range(66):
        id_to_trainid[i] = 255

    ### Convert each class to cityscapes one
    ### Road
    # Road
    id_to_trainid[13] = 0
    # Lane Marking - General
    id_to_trainid[24] = 0
    # Manhole
    id_to_trainid[41] = 0

    ### Sidewalk
    # Curb
    id_to_trainid[2] = 1
    # Sidewalk
    id_to_trainid[15] = 1

    ### Building
    # Building
    id_to_trainid[17] = 2

    ### Wall
    # Wall
    id_to_trainid[6] = 3

    ### Fence
    # Fence
    id_to_trainid[3] = 4

    ### Pole
    # Pole
    id_to_trainid[45] = 5
    # Utility Pole
    id_to_trainid[47] = 5

    ### Traffic Light
    # Traffic Light
    id_to_trainid[48] = 6

    ### Traffic Sign
    # Traffic Sign
    id_to_trainid[50] = 7

    ### Vegetation
    # Vegitation
    id_to_trainid[30] = 8

    ### Terrain
    # Terrain
    id_to_trainid[29] = 9

    ### Sky
    # Sky
    id_to_trainid[27] = 10

    ### Person
    # Person
    id_to_trainid[19] = 11

    ### Rider
    # Bicyclist
    id_to_trainid[20] = 12
    # Motorcyclist
    id_to_trainid[21] = 12
    # Other Rider
    id_to_trainid[22] = 12

    ### Car
    # Car
    id_to_trainid[55] = 13

    ### Truck
    # Truck
    id_to_trainid[61] = 14

    ### Bus
    # Bus
    id_to_trainid[54] = 15

    ### Train
    # On Rails
    id_to_trainid[58] = 16

    ### Motorcycle
    # Motorcycle
    id_to_trainid[57] = 17

    ### Bicycle
    # Bicycle
    id_to_trainid[52] = 18

gen_id_to_ignore()

def convert_to_train_id(file):
    # re-assign labels to match the format of Cityscapes
    pil_label = Image.open(file)
    label = np.asarray(pil_label)
    label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
    sample_class_stats = {}
    # import ipdb
    # ipdb.set_trace()
    for k, v in id_to_trainid.items():
        k_mask = label == k
        label_copy[k_mask] = v
        n = int(np.sum(k_mask))
        if n > 0:
            sample_class_stats[v] = n
    new_file = file.replace('.png', '_labelTrainIds.png')
    assert file != new_file
    sample_class_stats['file'] = new_file
    Image.fromarray(label_copy, mode='L').save(new_file)
    return sample_class_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Mapillary annotations to TrainIds')
    parser.add_argument('mapillary_path', help='mapillary data path, need to specify training / testing / validation')
    parser.add_argument('--gt-dir', default='labels', type=str)
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument(
        '--nproc', default=4, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats):
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


def main():
    args = parse_args()
    mapillary_path = args.mapillary_path
    out_dir = args.out_dir if args.out_dir else mapillary_path
    mmcv.mkdir_or_exist(out_dir)

    gt_dir = osp.join(mapillary_path, args.gt_dir)

    poly_files = []
    for poly in mmcv.scandir(
            gt_dir, suffix='.png',
            recursive=True):
        poly_file = osp.join(gt_dir, poly)
        poly_files.append(poly_file)
    poly_files = sorted(poly_files)

    only_postprocessing = False
    if not only_postprocessing:
        if args.nproc > 1:
            sample_class_stats = mmcv.track_parallel_progress(
                convert_to_train_id, poly_files, args.nproc)
        else:
            sample_class_stats = mmcv.track_progress(convert_to_train_id,
                                                     poly_files)
    else:
        with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(out_dir, sample_class_stats)


if __name__ == '__main__':
    main()
