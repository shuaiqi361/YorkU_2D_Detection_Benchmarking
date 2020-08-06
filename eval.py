import pprint
import os
from tqdm import tqdm
from datetime import datetime
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torch.utils.data as data
import time
import yaml
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tensorboardX import SummaryWriter
import argparse
from easydict import EasyDict
import json

from scheduler import adjust_learning_rate
from models import model_entry
from dataset.Datasets import PascalVOCDataset, COCO17Dataset, DetracDataset
from utils import create_logger
from models.utils import detect, detect_focal
from metrics import AverageMeter, calculate_mAP

parser = argparse.ArgumentParser(description='PyTorch 2D object detection training script.')
parser.add_argument('--config', default='', type=str)
parser.add_argument('--load-path', default='', type=str)
parser.add_argument('--save-path', default='', type=str)
parser.add_argument('--recover', action='store_true')  # resume training from checkpoints
parser.add_argument('-e', '--evaluate', action='store_true')
parser.add_argument('--finetune', action='store_true')  # finetune models using partial weights in checkpoints


def main():
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
    config = EasyDict(config)
    config.model_type = args.model_type

    config.save_path = os.path.join(os.path.dirname(args.config), 'snapshots')
    if not os.path.exists(config.save_path):
        os.mkdir(config.save_path)
    config.log_path = os.path.join(os.path.dirname(args.config), 'logs')
    if not os.path.exists(config.log_path):
        os.mkdir(config.log_path)
    config.event_path = os.path.join(os.path.dirname(args.config), 'events')
    if not os.path.exists(config.event_path):
        os.mkdir(config.event_path)

    batch_size = config.batch_size
    config.num_iter_flag = batch_size // config.internal_batchsize
    with open(config.label_path, 'r') as j:
        config.label_map = json.load(j)

    config.rev_coco_label_map = {str(v): k for k, v in config.label_map.items()}

    config.n_classes = len(config.label_map)  # number of different types of objects

    workers = config.workers

    if torch.cuda.device_count() < 2:
        config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        if config.device == 1:  # only for 2 GPUs max
            config.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        else:
            config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    val_data_folder = config.val_data_root

    # Custom dataloaders
    if config.data_name.upper() == 'COCO':
        config.coco = COCO(config.annotation_file)
        test_dataset = COCO17Dataset(val_data_folder, split='val', config=config)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                                  collate_fn=test_dataset.collate_fn, num_workers=workers,
                                                  pin_memory=False)
    elif config.data_name.upper() == 'VOC':
        test_dataset = PascalVOCDataset(val_data_folder, split='val', config=config)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                                  collate_fn=test_dataset.collate_fn, num_workers=workers,
                                                  pin_memory=False)
    elif config.data_name.upper() == 'DETRAC':
        test_dataset = DetracDataset(val_data_folder, split='val', config=config)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                                  collate_fn=test_dataset.collate_fn, num_workers=workers,
                                                  pin_memory=False)
    else:
        raise NotImplementedError

    assert args.load_path is not None
    checkpoint = torch.load(args.load_path)
    model = checkpoint['model']
    saved_epoch = checkpoint['epoch']
    model = model.to(config.device)
    optimizer = checkpoint['optimizer']
    print('Evaluate model from checkpoint %s at epoch %d.\n' % (args.load_path, saved_epoch))

    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_H-%M-%S")
    config.logger = create_logger('global_logger', os.path.join(config.log_path,
                                                                'eval_result_{}_{}.txt'.format(config.model['arch'],
                                                                                               date_time)))
    print('Length of Testing Dataset:', len(test_dataset))
    print('evaluate checkpoint: ', args.load_path, ' at epoch: ', saved_epoch)
    evaluate(test_loader, model, optimizer, config=config)


def evaluate(test_loader, model, optimizer, config):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.train()

    pp = pprint.PrettyPrinter()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()
    detect_speed = list()
    COCO_format_results = list()
    image_all_ids = list()

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, ids, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            images = images.to(config.device)
            boxes = [b.to(config.device) for b in boxes]
            labels = [l.to(config.device) for l in labels]
            difficulties = [d.to(config.device) for d in difficulties]

            # Forward prop.
            time_start = time.time()
            outputs = model(images)

            if len(outputs) == 2:
                predicted_locs, predicted_scores = model(images)
                prior_positives_idx = None
            elif len(outputs) == 7:
                _, _, _, _, predicted_locs, predicted_scores, prior_positives_idx = outputs
            else:
                raise NotImplementedError

            # Detect objects
            if config['focal_type'].lower() == 'sigmoid':
                det_boxes_batch, det_labels_batch, det_scores_batch = \
                    detect_focal(predicted_locs,
                                 predicted_scores,
                                 min_score=config.nms['min_score'],
                                 max_overlap=config.nms['max_overlap'],
                                 top_k=config.nms['top_k'], priors_cxcy=model.priors_cxcy,
                                 config=config, prior_positives_idx=prior_positives_idx)
            elif config['focal_type'].lower() == 'softmax':
                det_boxes_batch, det_labels_batch, det_scores_batch = \
                    detect(predicted_locs,
                           predicted_scores,
                           min_score=config.nms['min_score'],
                           max_overlap=config.nms['max_overlap'],
                           top_k=config.nms['top_k'], priors_cxcy=model.priors_cxcy,
                           config=config, prior_positives_idx=prior_positives_idx)
            else:
                print('focal type should be either softmax or sigmoid.')
                raise NotImplementedError

            time_end = time.time()

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)
            detect_speed.append((time_end - time_start) / len(labels))

            if config.data_name.upper() == 'COCO':
                # store results in COCO formats
                det_boxes_batch = [b.cpu() for b in det_boxes_batch]
                det_labels_batch = [b.cpu() for b in det_labels_batch]
                det_scores_batch = [b.cpu() for b in det_scores_batch]
                for j in range(len(ids)):
                    img = config.coco.loadImgs(ids[j])[0]
                    width = img['width'] * 1.
                    height = img['height'] * 1.
                    bboxes = det_boxes_batch[j]
                    bboxes[:, 2] -= bboxes[:, 0]
                    bboxes[:, 3] -= bboxes[:, 1]
                    bboxes[:, 0] *= width
                    bboxes[:, 2] *= width
                    bboxes[:, 1] *= height
                    bboxes[:, 3] *= height
                    for box_idx in range(det_boxes_batch[j].size(0)):
                        score = float(det_scores_batch[j][box_idx])
                        label_name = config.rev_coco_label_map[str(int(det_labels_batch[j][box_idx]))]
                        label = config.coco.getCatIds(catNms=[label_name])[0]
                        bbox = bboxes[box_idx, :].tolist()
                        image_result = {
                            'image_id': ids[j],
                            'category_id': label,
                            'score': score,
                            'bbox': bbox
                        }
                        COCO_format_results.append(image_result)
                        image_all_ids.append(ids[j])

        # Calculate mAP
        if config.data_name.upper() == 'VOC':
            APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, 0.5,
                                     config.label_map, config.device)

            # Print AP for each class
            pp.pprint(APs)
            details = pprint.pformat(APs)
            config.logger.info(details)

            str_print = 'EVAL: Mean Average Precision {0:.4f}, ' \
                        'avg speed {1:.3f} Hz'.format(mAP, 1. / np.mean(detect_speed))
            config.logger.info(str_print)

        if config.data_name.upper() == 'DETRAC':
            APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties, 0.7,
                                     config.label_map, config.device)

            # Print AP for each class
            pp.pprint(APs)
            details = pprint.pformat(APs)
            config.logger.info(details)

            str_print = 'EVAL: Mean Average Precision {0:.4f}, ' \
                        'avg speed {1:.3f} Hz'.format(mAP, 1. / np.mean(detect_speed))
            config.logger.info(str_print)

        if config.data_name.upper() == 'COCO':
            json.dump(COCO_format_results, open('{}/{}_bbox_results.json'.format(config.save_path,
                                                                                 config.data_name), 'w'), indent=4)
            # run COCO evaluation
            coco_pred = config.coco.loadRes('{}/{}_bbox_results.json'.format(config.save_path,
                                                                             config.data_name))
            coco_eval = COCOeval(config.coco, coco_pred, 'bbox')
            coco_eval.params.imgIds = image_all_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

    return


if __name__ == '__main__':
    main()
