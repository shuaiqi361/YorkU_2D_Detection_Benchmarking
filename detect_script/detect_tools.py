import torch
import torch.nn as nn
from dataset.transforms import *
import torch.nn.functional as F
from torchvision.ops import nms
from operators.iou_utils import diounms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect(predicted_locs, predicted_scores, min_score, max_overlap, top_k, priors_cxcy, config,
           prior_positives_idx=None):
    """
    Decipher the 22536 locations and class scores (output of ths SSD300) to detect objects.

    For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

    :param prior_positives_idx:
    :param priors_cxcy:
    :param predicted_locs: predicted locations/boxes w.r.t the 22536 prior boxes, a tensor of dimensions (N, 22536, 4)
    :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 22536, n_classes)
    :param min_score: minimum threshold for a box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :return: detections (boxes, labels, and scores), lists of length batch_size
    """
    if isinstance(priors_cxcy, list):
        priors_cxcy = torch.cat(priors_cxcy, dim=0)

    box_type = config.model['box_type']
    focal_type = config['focal_type']
    reg_type = config['reg_loss']
    batch_size = predicted_locs.size(0)
    n_classes = predicted_scores.size(2)

    if focal_type.lower() == 'sigmoid':
        predicted_scores = predicted_scores.sigmoid()
    else:
        predicted_scores = predicted_scores.softmax(dim=2)  # softmax activation

    # Lists to store final predicted boxes, labels, and scores for all images
    all_images_boxes = list()
    all_images_labels = list()
    all_images_scores = list()

    for i in range(batch_size):
        # Decode object coordinates from the form we regressed predicted boxes to
        if box_type == 'offset':
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], priors_cxcy)).clamp_(0, 1)
        elif box_type == 'center':
            decoded_locs = cxcy_to_xy(predicted_locs[i]).clamp_(0, 1)
        else:
            decoded_locs = predicted_locs[i].clamp_(0, 1)

        # Lists to store boxes and scores for this image
        image_boxes = list()
        image_labels = list()
        image_scores = list()

        if prior_positives_idx is not None:
            class_scores_all = torch.index_select(predicted_scores[i], dim=0,
                                                  index=prior_positives_idx[i].nonzero().squeeze(-1))
            decoded_locs_all = torch.index_select(decoded_locs, dim=0,
                                                  index=prior_positives_idx[i].nonzero().squeeze(-1))

        else:
            class_scores_all = predicted_scores[i, :, :]
            decoded_locs_all = decoded_locs

        # Check for each class
        for c in range(1, n_classes):
            # Keep only predicted boxes and scores where scores for this class are above the minimum score
            class_scores = class_scores_all[:, c]
            top_k_scores, _ = torch.topk(class_scores, min(5000, len(class_scores) - 1), dim=0)
            min_score = max(min_score, top_k_scores.min())
            score_above_min_score = (class_scores >= min_score).long()  # for indexing

            n_above_min_score = torch.sum(score_above_min_score).item()

            if n_above_min_score == 0:
                continue

            class_scores = torch.index_select(class_scores, dim=0,
                                              index=torch.nonzero(score_above_min_score).squeeze(dim=1))

            class_decoded_locs = torch.index_select(decoded_locs_all, dim=0,
                                                    index=torch.nonzero(score_above_min_score).squeeze(dim=1))

            if reg_type.lower() == 'iou':
                anchor_nms_idx, _ = diounms(class_decoded_locs, class_scores, max_overlap)
            else:
                anchor_nms_idx = nms(class_decoded_locs, class_scores, max_overlap)

            # Store only unsuppressed boxes for this class
            image_boxes.append(class_decoded_locs[anchor_nms_idx, :])
            image_labels.append(torch.LongTensor(anchor_nms_idx.size(0) * [c]).to(device))
            image_scores.append(class_scores[anchor_nms_idx])
        # If no object in any class is found, store a placeholder for 'background'
        if len(image_boxes) == 0:
            image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
            image_labels.append(torch.LongTensor([0]).to(device))
            image_scores.append(torch.FloatTensor([0.]).to(device))

        # Concatenate into single tensors
        image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
        image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
        image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
        n_objects = image_scores.size(0)

        if config.final_nms:
            final_nms_idx = nms(image_boxes, image_scores, 0.7)
            image_scores = image_scores[final_nms_idx]
            image_labels = image_labels[final_nms_idx]
            image_boxes = image_boxes[final_nms_idx, :]

        # Keep only the top k objects
        if n_objects > top_k:
            image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
            image_scores = image_scores[:top_k]
            image_boxes = image_boxes[sort_ind][:top_k]
            image_labels = image_labels[sort_ind][:top_k]

        # Append to lists that store predicted boxes and scores for all images
        all_images_boxes.append(image_boxes)
        all_images_labels.append(image_labels)
        all_images_scores.append(image_scores)

    return all_images_boxes, all_images_labels, all_images_scores


def detect_focal(predicted_locs, predicted_scores, min_score, max_overlap, top_k, priors_cxcy,
                 config, prior_positives_idx=None):
    """
    Decipher the 22536 locations and class scores (output of ths SSD300) to detect objects.

    For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

    :param prior_positives_idx:
    :param config:
    :param priors_cxcy:
    :param predicted_locs: predicted locations/boxes w.r.t the 22536 prior boxes, a tensor of dimensions (N, 22536, 4)
    :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 22536, n_classes)
    :param min_score: minimum threshold for a box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :return: detections (boxes, labels, and scores), lists of length batch_size
    """

    if isinstance(priors_cxcy, list):
        priors_cxcy = torch.cat(priors_cxcy, dim=0)

    box_type = config.model['box_type']
    focal_type = config['focal_type']
    reg_type = config['reg_loss']
    batch_size = predicted_locs.size(0)
    n_classes = predicted_scores.size(2)

    if focal_type.lower() == 'sigmoid':
        predicted_scores = predicted_scores.sigmoid()
    else:
        predicted_scores = predicted_scores.softmax(dim=2)  # softmax activation

    # Lists to store final predicted boxes, labels, and scores for all images
    all_images_boxes = list()
    all_images_labels = list()
    all_images_scores = list()

    for i in range(batch_size):
        # Decode object coordinates from the form we regressed predicted boxes to
        if box_type == 'offset':
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], priors_cxcy)).clamp_(0, 1)
        elif box_type == 'center':
            decoded_locs = cxcy_to_xy(predicted_locs[i]).clamp_(0, 1)
        else:
            decoded_locs = predicted_locs[i].clamp_(0, 1)

        # Lists to store boxes and scores for this image
        image_boxes = list()
        image_labels = list()
        image_scores = list()

        # max_scores, best_label = predicted_scores[i].max(dim=1)  # (22536)
        if prior_positives_idx is not None:
            class_scores_all = torch.index_select(predicted_scores[i], dim=0,
                                                  index=prior_positives_idx[i].nonzero().squeeze(-1))
            decoded_locs_all = torch.index_select(decoded_locs, dim=0,
                                                  index=prior_positives_idx[i].nonzero().squeeze(-1))

        else:
            class_scores_all = predicted_scores[i, :, :]
            decoded_locs_all = decoded_locs

        # Check for each class
        for c in range(n_classes):  # n_classes = 20 for VOC and 80 for COCO
            # Keep only predicted boxes and scores where scores for this class are above the minimum score
            class_scores = class_scores_all[:, c]
            top_k_scores, _ = torch.topk(class_scores, min(3500, len(class_scores) - 1), dim=0)
            min_score = max(min_score, top_k_scores.min())
            score_above_min_score = (class_scores >= min_score).long()  # for indexing

            n_above_min_score = torch.sum(score_above_min_score).item()

            if n_above_min_score == 0:
                continue

            class_scores = torch.index_select(class_scores, dim=0,
                                              index=torch.nonzero(score_above_min_score).squeeze(dim=1))

            class_decoded_locs = torch.index_select(decoded_locs_all, dim=0,
                                                    index=torch.nonzero(score_above_min_score).squeeze(dim=1))

            if reg_type.lower() == 'iou':
                anchor_nms_idx, _ = diounms(class_decoded_locs.data, class_scores, max_overlap)
            else:
                anchor_nms_idx = nms(class_decoded_locs, class_scores, max_overlap)

            image_boxes.append(class_decoded_locs[anchor_nms_idx, :])
            image_labels.append(torch.LongTensor(anchor_nms_idx.size(0) * [c + 1]).to(device))
            image_scores.append(class_scores[anchor_nms_idx])
        # If no object in any class is found, store a placeholder for 'background'
        if len(image_boxes) == 0:
            image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
            image_labels.append(torch.LongTensor([0]).to(device))
            image_scores.append(torch.FloatTensor([0.]).to(device))

        # Concatenate into single tensors
        image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
        image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
        image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
        n_objects = image_scores.size(0)

        if config.final_nms:
            final_nms_idx = nms(image_boxes, image_scores, 0.7)
            image_scores = image_scores[final_nms_idx]
            image_labels = image_labels[final_nms_idx]
            image_boxes = image_boxes[final_nms_idx, :]

        # Keep only the top k objects
        if n_objects > top_k:
            image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
            image_scores = image_scores[:top_k]  # (top_k)
            image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
            image_labels = image_labels[sort_ind][:top_k]  # (top_k)

        # Append to lists that store predicted boxes and scores for all images
        all_images_boxes.append(image_boxes)
        all_images_labels.append(image_labels)
        all_images_scores.append(image_scores)

    return all_images_boxes, all_images_labels, all_images_scores
