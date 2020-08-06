import sys

sys.path.append('/home/keyi/Documents/research/code/shape_based_object_detection')
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

# Transforms, should adjust accordingly
input_size = (512, 512)
# input_size = (720, 1280)
# resize = transforms.Resize(input_size)  # For traffic data
resize = transforms.Resize(input_size)  # For COCO data

to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def detect_folder(folder_path, model_path, meta_data_path, output_path, config_path,
                  save_image_flag, final_nms):
    """
    :param folder_path: folder that contains all the images to be detected
    :param model_path: path to the checkpoint
    :param meta_data_path: label map file
    :param output_path: path to save detection results
    :param config_path: path to the config.yaml file
    :param save_image_flag:
    :param save_bbox_flag:
    :param final_nms: whether to execute an additional nms for all classes, so that each bbox can only be associated to one class
    :return:
    """
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.load(f)
    else:
        print('config file path incorrect: {}'.format(config_path))
        exit()

    config = EasyDict(config)
    config.final_nms = final_nms

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    output_video_file = os.path.join(output_path,
                                     video_path.split('/')[-1].strip('.mkv').strip('.avi')
                                     + '_detect.avi')
    output_bbox_file = os.path.join(output_path,
                                    video_path.split('/')[-1].strip('.mkv').strip('.jpg').strip('.png').strip('.avi')
                                    + '_bbox.txt')

    # load model
    checkpoint = torch.load(model_path, map_location=device)
    start_epoch = checkpoint['epoch'] + 1
    print(model_path)
    print('Loading checkpoint from epoch %d.\n' % start_epoch)
    model = checkpoint['model']
    model.device = device
    model = model.to(device)
    model.eval()

    if os.path.exists(meta_data_path):
        with open(meta_data_path, 'r') as j:
            label_map = json.load(j)
    else:
        print('meta file path incorrect: {}'.format(meta_data_path))
        exit()

    distinct_colors = []
    config.n_classes = len(label_map)
    for c in range(config.n_classes):
        distinct_colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    rev_label_map = {v: k for k, v in label_map.items()}
    label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

    # load video
    if not os.path.exists(folder_path):
        print('video path incorrect.')
        exit()

    # if save_image_flag:
    #     width = 960  # if this is used, all images should be of same sizes
    #     height = 540
    #     fps = 30
    #     video_out = cv2.VideoWriter(output_video_file,
    #                                 cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (width, height))

    if output_bbox_file is not None and save_bbox_flag:
        f_out = open(output_bbox_file, 'w')

    speed_list = list()
    frame_list = sorted(os.listdir(folder_path))
    n_frames = len(frame_list)
    for frame_id in range(n_frames):
        # modify here if the images has other name patterns for sorting to video frames
        # frame_name = 'img{:04d}.png'.format(frame_id + 1)
        frame_name = frame_list[frame_id]
        frame_path = os.path.join(folder_path, frame_name)
        # print("Processing frame: ", frame_id, frame_path)
        # frame = cv2.resize(cv2.imread(frame_path), dsize=(width, height))
        frame = cv2.imread(frame_path)

        annotated_image, time_pframe, frame_info = detect_image(frame, model, 0.15, 0.65, 200,
                                                                rev_label_map, label_color_map, config)
        speed_list.append(time_pframe)

        if save_image_flag:
            # video_out.write(annotated_image)
            cv2.imwrite(annotated_image, output_path + frame_name.strip('.png').strip('.jpg') + '_bbox.png')
        if save_bbox_flag:
            f_out.write(frame_name + frame_info)

        frame_id += 1
        cv2.imshow('Image detect', annotated_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Average speed: {} fps.'.format(1. / np.mean(speed_list)))
    print('Saved to:', output_path)


def detect_video(video_path, model_path, meta_data_path, output_path, config_path,
                 save_video_flag=False, save_bbox_flag=False, final_nms=True):
    """
    :param video_path: video to be detected
    :param model_path: saved checkpoint
    :param meta_data_path: label map file
    :param output_path: where to save output video and bboxes, if the following flag is True
    :param save_video_flag:
    :param save_bbox_flag:
    :return:
    """
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.load(f)
    else:
        print('config file path incorrect: {}'.format(config_path))
        exit()

    config = EasyDict(config)
    config.final_nms = final_nms

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    output_video_file = os.path.join(output_path,
                                     video_path.split('/')[-1].strip('.mkv').strip('.avi')
                                     + '_detect.avi')
    output_bbox_file = os.path.join(output_path,
                                    video_path.split('/')[-1].strip('.mkv').strip('.jpg').strip('.png').strip('.avi')
                                    + '_bbox.txt')

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
    if not os.path.exists(video_path):
        print('Video path incorrect: {}'.format(video_path))
        exit()

    if save_bbox_flag:
        f_out = open(output_bbox_file, 'w')

    cap_video = cv2.VideoCapture(video_path)
    width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap_video.get(cv2.CAP_PROP_FPS))

    distinct_colors = []
    config.n_classes = len(label_map)
    for c in range(config.n_classes):
        distinct_colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

    if save_video_flag:
        video_out = cv2.VideoWriter(output_video_file,
                                    cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (width, height))

    speed_list = list()
    frame_id = 0
    while True:
        ret, frame = cap_video.read()
        if not ret:
            print('Image processing and detecting done. ')
            break

        annotated_image, time_pframe, frame_info = detect_image(frame, model, 0.05, 0.5, 200,
                                                                rev_label_map, label_color_map, config)
        speed_list.append(time_pframe)

        if save_video_flag:
            video_out.write(annotated_image)

        if save_bbox_flag:
            f_out.write(str(frame_id) + frame_info)

        frame_id += 1
        cv2.imshow('Detecting objects', annotated_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_video.release()
    if save_bbox_flag:
        f_out.close()

    print('Average speed: {} fps.'.format(1. / np.mean(speed_list)))
    print('Saved to:', output_path)
    print('Video configuration: \nresolution:{}x{}, fps:{}'.format(width, height, fps))


def detect_single_image(image_path, model_path, meta_data_path, output_path, config_path,
                        save_image_flag=False, final_nms=True):
    """
    :param image_path:
    :param model_path:
    :param meta_data_path:
    :param output_path:
    :param config_path:
    :param save_image_flag:
    :param final_nms:
    :return:
    """
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = yaml.load(f)
    else:
        print('config file path incorrect: {}'.format(config_path))
        exit()

    config = EasyDict(config)
    config.final_nms = final_nms

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # load model
    checkpoint = torch.load(model_path, map_location=device)
    start_epoch = checkpoint['epoch'] + 1
    print(model_path)
    print('Loading checkpoint from epoch %d.\n' % start_epoch)
    model = checkpoint['model']
    model.device = device
    model = model.to(device)
    model.eval()

    if os.path.exists(meta_data_path):
        with open(meta_data_path, 'r') as j:
            label_map = json.load(j)
    else:
        print('meta file path incorrect: {}'.format(meta_data_path))
        exit()

    distinct_colors = []
    config.n_classes = len(label_map)
    for c in range(config.n_classes):
        distinct_colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

    rev_label_map = {v: k for k, v in label_map.items()}
    label_color_map = {k: distinct_colors[i] for i, k in enumerate(label_map.keys())}

    # load image
    if not os.path.exists(image_path):
        print('Image Not Found.')
        exit()

    frame = cv2.imread(image_path)

    annotated_image, _, _ = detect_image(frame, model, 0.05, 0.5, 200,
                                         rev_label_map, label_color_map, config)

    if save_image_flag:
        cv2.imwrite(annotated_image, os.path.join(output_path, image_path.strip('.png').strip('jpg') + '_bbox.png'))

    cv2.imshow('Image detect', annotated_image)
    cv2.waitKey()

    print('Detected image saved to:', output_path)


def detect_image(frame, model, min_score, max_overlap, top_k, reverse_label_map, label_color_map, config):
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
    frame_info = ''
    for i in range(len(det_labels)):
        # Boxes
        box_location = det_boxes[i].tolist()
        box_coordinates = det_boxes_percentage[i].tolist()
        cv2.rectangle(annotated_image, pt1=(int(box_location[0]), int(box_location[1])),
                      pt2=(int(box_location[2]), int(box_location[3])),
                      color=label_color_map[det_labels[i]], thickness=2)

        # Text
        text = det_labels[i].upper()
        label_id = str(det_labels_id[i])
        label_score = det_labels_scores[i]
        label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.4, 1)
        text_location = [box_location[0] + 1, box_location[1] + 1, box_location[0] + 1 + label_size[0][0],
                         box_location[1] + 1 + label_size[0][1]]
        cv2.rectangle(annotated_image, pt1=(int(text_location[0]), int(text_location[1])),
                      pt2=(int(text_location[2]), int(text_location[3])),
                      color=(50, 50, 50), thickness=-1)
        cv2.putText(annotated_image, text, org=(int(text_location[0]), int(text_location[3])),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, thickness=1, fontScale=0.4, color=(255, 255, 255))

        per_object_prediction_info = ' {0} {1:.4f} {2:.4f} {3:.4f} {4:.4f} {5:.3f}'.format(label_id,
                                                                                           box_coordinates[0],
                                                                                           box_coordinates[1],
                                                                                           box_coordinates[2],
                                                                                           box_coordinates[3],
                                                                                           label_score)
        frame_info += per_object_prediction_info

    return annotated_image, - start + stop, frame_info + '\n'


def print_help():
    print('Modify the arguments in main(), and try one of the following options:')
    print('python detect_bbox --folder(detect for all images under the folder)')
    print('python detect_bbox --video(detect for all frames in the video)')
    print('python detect_bbox --image(single image demo)')
    print('saved images will be put in the same location as input, new folder "detected_results" will be created')
    exit()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print_help()
        exit()
    if sys.argv[1] == '--video':
        root_path = '/media/keyi/Data/Research/traffic/detection/shape_based_object_detection/experiment/SSD512_exp_003'
        video_path = '/media/keyi/Data/Research/traffic/data/Hwy7/20200224_153147_demo4.mkv'
        model_path = os.path.join(root_path, 'snapshots/ssd512_traffic_checkpoint_epoch-3.pth.tar')
        meta_data_path = '/media/keyi/Data/Research/traffic/detection/object_detection_2D/dataset/DETRAC/label_map.json'
        output_path = os.path.join(root_path, 'detected_results')
        config_path = ''
        save_image_flag = True
        save_bbox_flag = True
        final_nms = True
        detect_video(video_path, model_path, meta_data_path, output_path, config_path,
                     save_image_flag, save_bbox_flag, final_nms)

    elif sys.argv[1] == '--folder':  # the folder should contain images from a video, named as "0001.jpg", "0002.jpg"...
        root_path = '/home/keyi/Documents/research/code/shape_based_object_detection/experiment/RefineDet_traffic_003'
        folder_path = '/home/keyi/Documents/Data/Hwy7/20200224_153147_d2_1200'
        model_path = os.path.join(root_path, 'snapshots/refinedetboftraffic_detrac_checkpoint_epoch-10.pth.tar')
        meta_data_path = '/home/keyi/Documents/research/code/shape_based_object_detection/data/DETRAC/label_map.json'
        output_path = os.path.join(root_path, 'detected_results')
        config_path = ''
        save_image_flag = True
        save_bbox_flag = True
        final_nms = True
        detect_folder(folder_path, model_path, meta_data_path, output_path, config_path,
                      save_image_flag, save_bbox_flag, final_nms)

    elif sys.argv[1] == '--image':  # the folder should contain images from a video, named as "0001.jpg", "0002.jpg"...
        root_path = '/home/keyi/Documents/research/code/shape_based_object_detection/experiment/RefineDet_traffic_003'
        image_path = '/home/keyi/Documents/Data/Hwy7/20200224_153147_d2_1200.jpg'
        model_path = os.path.join(root_path, 'snapshots/refinedetboftraffic_detrac_checkpoint_epoch-10.pth.tar')
        meta_data_path = '/home/keyi/Documents/research/code/shape_based_object_detection/data/DETRAC/label_map.json'
        output_path = os.path.join(root_path, 'detected_results')
        config_path = ''
        save_image_flag = True
        final_nms = True
        detect_single_image(image_path, model_path, meta_data_path, output_path, config_path,
                            save_image_flag, final_nms)
    else:
        print_help()
        raise NotImplementedError
