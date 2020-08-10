import sys

sys.path.append('../../YorkU_2D_Detection_Benchmarking')
from torchvision import transforms
from PIL import Image
import os
import sys
import cv2
import torch
import json
import time
import numpy as np
import random
import yaml
from easydict import EasyDict
from detect_script.detect_tools import detect, detect_focal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COCO_traffic = {'bus': 6, 'car': 3, 'truck': 8}

to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

"""
Arguments to be modified for testing DETRAC test video
"""
# Transforms
resize = transforms.Resize((540, 960))
# resize = transforms.Resize((512, 512))

root_path = '/home/keyi/Documents/research/code/shape_based_object_detection/experiment/RefineDet_traffic_003'
folder_path = '/home/keyi/Documents/Data/DETRAC/Insight-MVT_Annotation_Test'
model_path = os.path.join(root_path, 'snapshots/refinedetboftraffic_detrac_checkpoint_epoch-30.pth.tar')
config_path = os.path.join(root_path, 'config.yaml')
meta_data_path = '/home/keyi/Documents/research/code/shape_based_object_detection/data/DETRAC/label_map.json'
output_path = os.path.join(root_path, 'detected_results')
output_file_flag = True  # if save detection results flag
output_video_flag = True

# -------------------------------------------------------------------------------------------------------------

if os.path.exists(config_path):
    with open(config_path) as f:
        config = yaml.load(f)
else:
    print('config file path incorrect: {}'.format(config_path))
    exit()

config = EasyDict(config)
config.final_nms = True

if not os.path.exists(output_path):
    os.mkdir(output_path)


def detect_folder(folder_path, model_path, meta_data_path):
    # load model
    checkpoint = torch.load(model_path, map_location=device)
    start_epoch = checkpoint['epoch'] + 1
    print(model_path)
    print('Loading checkpoint from epoch %d.\n' % start_epoch)
    model = checkpoint['model']
    model.device = device
    model = model.to(device)
    model.eval()

    with open(meta_data_path, 'r') as j:
        label_map = json.load(j)
    rev_label_map = {v: k for k, v in label_map.items()}

    # load video
    if not os.path.exists(folder_path):
        print('DETRAC dataset path not found.', folder_path)
        exit()

    width = 960
    height = 540
    fps = 30  # output video configuration

    folder_name = folder_path.split('/')[-1]
    # label_color_map = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (100, 100, 0), (150, 0, 150)]

    if output_video_flag:
        video_out = cv2.VideoWriter(os.path.join(output_path, folder_name + '.mkv'),
                                    cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (width, height))
    if output_file_flag:
        output_file = os.path.join(output_path, folder_name + '_Det.txt')
        f_out = open(output_file, 'w')

    speed_list = list()
    frame_list = os.listdir(folder_path)
    n_frames = len(frame_list)
    for frame_id in range(n_frames):
        # if frame_id >= 250:  # early stop for evaluating specific number of frames
        #     exit()
        frame_name = 'img{:05d}.jpg'.format(frame_id + 1)
        frame_path = os.path.join(folder_path, frame_name)
        # print("Processing frame: ", frame_id, frame_path)
        frame = cv2.imread(frame_path)

        annotated_image, time_pframe, frame_info_list = detect_image(frame, model, 0.45, 0.45, 100,
                                                                     rev_label_map, config)

        speed_list.append(time_pframe)

        if output_video_flag:
            video_out.write(annotated_image)

        if output_file_flag:
            for k in range(len(frame_info_list)):
                f_out.write(str(frame_id + 1) + frame_info_list[k])

        frame_id += 1
        cv2.imshow('DETRAC frames', annotated_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Average speed: {} fps.'.format(1. / np.mean(speed_list)))
    print('Saved to:', output_path)
    print('Video configuration: \nresolution:{}x{}, fps:{}'.format(width, height, fps))


def detect_image(frame, model, min_score, max_overlap, top_k, reverse_label_map, config):
    # Transform
    image_for_detect = frame.copy()
    img = cv2.cvtColor(image_for_detect, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    image = normalize(to_tensor(resize(im_pil)))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    start = time.time()
    outputs = model(image.unsqueeze(0))

    if len(outputs) == 2:
        predicted_locs, predicted_scores = outputs
        prior_positives_idx = None
    elif len(outputs) == 7:
        _, _, _, _, predicted_locs, predicted_scores, prior_positives_idx = outputs
    else:
        raise NotImplementedError

    if config['focal_type'].lower() == 'sigmoid':
        det_boxes, det_labels, det_scores = \
            detect_focal(predicted_locs,
                         predicted_scores,
                         min_score=min_score,
                         max_overlap=max_overlap,
                         top_k=top_k, priors_cxcy=model.priors_cxcy,
                         config=config, prior_positives_idx=prior_positives_idx)
    elif config['focal_type'].lower() == 'softmax':
        det_boxes, det_labels, det_scores = \
            detect(predicted_locs,
                   predicted_scores,
                   min_score=min_score,
                   max_overlap=max_overlap,
                   top_k=top_k, priors_cxcy=model.priors_cxcy,
                   config=config, prior_positives_idx=prior_positives_idx)
    else:
        print('focal type should be either softmax or sigmoid.')
        raise NotImplementedError

    stop = time.time()
    # Move detections to the CPU
    det_boxes_percentage = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [im_pil.width, im_pil.height, im_pil.width, im_pil.height]).unsqueeze(0)
    det_boxes = det_boxes_percentage * original_dims

    # Decode class integer labels
    det_labels_id = [l for l in det_labels[0].to('cpu').tolist()]
    det_labels = [reverse_label_map[l] for l in det_labels[0].to('cpu').tolist()]
    det_labels_scores = [s for s in det_scores[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.']
    # i.e. ['background'] in SSD300.detect_objects() in model.py
    annotated_image = frame.copy()

    if det_labels == ['background']:
        return annotated_image, start - stop, '\n'

    # Annotate
    frame_info_list = []
    for i in range(len(det_labels)):
        # Boxes, this is for evaluate COCO trained model without fintuning on DETRAC data,
        # the min_score should be changed to 0.05
        # if det_labels[i] not in COCO_traffic.keys():
        #     continue

        box_location = det_boxes[i].tolist()
        cv2.rectangle(annotated_image, pt1=(int(box_location[0]), int(box_location[1])),
                      pt2=(int(box_location[2]), int(box_location[3])),
                      color=(0, 255, 0), thickness=2)

        # Text
        # text = det_labels[i].upper()
        text = '{:.3f}'.format(det_labels_scores[i])

        label_score = det_labels_scores[i]
        label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)
        text_location = [box_location[0] + 1, box_location[1] + 1, box_location[0] + 1 + label_size[0][0],
                         box_location[1] + 1 + label_size[0][1]]
        cv2.rectangle(annotated_image, pt1=(int(text_location[0]), int(text_location[1])),
                      pt2=(int(text_location[2]), int(text_location[3])),
                      color=(128, 128, 128), thickness=-1)
        cv2.putText(annotated_image, text, org=(int(text_location[0]), int(text_location[3])),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, thickness=1, fontScale=0.4, color=(255, 255, 255))

        per_object_prediction_info = ',{0},{1:.2f},{2:.2f},{3:.2f},{4:.2f},{5:.4f}\n'.format(i + 1,
                                                                                             box_location[0],
                                                                                             box_location[1],
                                                                                             box_location[2] -
                                                                                             box_location[0],
                                                                                             box_location[3] -
                                                                                             box_location[1],
                                                                                             label_score)
        frame_info_list.append(per_object_prediction_info)

    return annotated_image, - start + stop, frame_info_list


def print_help():
    print('This script is for inference on DETRAC test videos. Specify arguments and try:')
    print('python detect_detrac')

    exit()


if __name__ == '__main__':
    video_list = os.listdir(folder_path)
    for v in video_list:
        sequence_path = os.path.join(folder_path, v)
        detect_folder(sequence_path, model_path, meta_data_path)
