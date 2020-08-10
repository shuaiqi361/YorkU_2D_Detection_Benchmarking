import json
import os
import xml.etree.ElementTree as ET
import random
import cv2
import os
from PIL import Image

traffic_labels_all = ['car', 'van', 'bus', 'others']
traffic_labels = ['vehicle']  # DETRAC dataset has multiple labels for vehicles, but we consider them as one class
traffic_label_map = {k: v + 1 for v, k in enumerate(traffic_labels)}
traffic_label_map['background'] = 0
rev_traffic_label_map = {v: k for k, v in traffic_label_map.items()}  # Inverse mapping

'''
There are 82085 training images containing a total of 594555 objects.
Files have been saved to /media/keyi/Data/Research/course_project/AdvancedCV_2020/AdvanceCV_project/data/DETRAC.
There are 56167 validation images containing a total of 658859 objects.
Files have been saved to /media/keyi/Data/Research/course_project/AdvancedCV_2020/AdvanceCV_project/data/DETRAC.
'''


def parse_annotation(annotation_path, image_folder, down_sample=False):
    tree = ET.parse(annotation_path)

    root = tree.getroot()
    object_list = list()
    image_paths = list()

    # parse ignored regions
    ignore_regions = list()
    region_list = root.find('ignored_region')
    for region in region_list.iter('box'):
        left = float(region.attrib['left'])
        top = float(region.attrib['top'])
        width = float(region.attrib['width'])
        height = float(region.attrib['height'])
        xmin = max(int(left), 0)
        ymin = max(int(top), 0)
        xmax = int(left + width)
        ymax = int(top + height)
        ignore_regions.append([xmin, ymin, xmax, ymax])

    for objects in root.iter('frame'):
        if random.random() > 0.25 and down_sample:
            continue
        boxes = list()
        labels = list()
        difficulties = list()
        occlusions = list()

        frame_id = int(objects.attrib['num'])
        image_name = 'img{:05d}.jpg'.format(frame_id)
        # masked_image_name = 'img{:05d}_masked.jpg'.format(frame_id)

        target_list = objects.find('target_list')
        for target in target_list:
            uni_name = target.find('attribute').attrib['vehicle_type']
            if uni_name not in traffic_labels_all:
                continue
            else:
                # uni_label = traffic_label_map[uni_name]
                uni_label = traffic_label_map['vehicle']

            bbox = target.find('box').attrib
            left = float(bbox['left'])
            top = float(bbox['top'])
            width = float(bbox['width'])
            height = float(bbox['height'])
            xmin = max(int(left), 0)
            ymin = max(int(top), 0)
            xmax = int(left + width)
            ymax = int(top + height)
            if xmin + 2 > xmax or ymin + 2 > ymax:
                print('Invalid bbox: ', os.path.join(image_folder, image_name))
                continue

            difficult = 0  # not really used

            # occlusion info, the quality is not good to use
            occ_status = []
            occ = target.find('occlusion')
            if occ is not None:
                occ_status.append(float(occ.find('region_overlap').attrib['occlusion_status']))

            occlusions.append(occ_status)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(uni_label)
            difficulties.append(difficult)

        if len(boxes) == 0:
            print('Images with no objects: ', os.path.join(image_folder, image_name))
            continue
        object_list.append({'bbox': boxes, 'labels': labels, 'difficulties': difficulties,
                            'image_id': image_name, 'ignore_regions': ignore_regions,
                            'occlusions': occlusions})
        image_paths.append(os.path.join(image_folder, image_name))

        # draw augmented images and bboxes
        # img = cv2.imread(os.path.join(image_folder, image_name))
        #
        # for region in ignore_regions:
        #     cv2.rectangle(img, (region[0], region[1]), (region[2], region[3]), (127, 127, 127), -1)
        #
        # for n in range(len(boxes)):
        #     region = boxes[n]
        #     cv2.rectangle(img, (region[0], region[1]), (region[2], region[3]), (0, 255, 0), 2)
        #
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # temp_image = Image.fromarray(img)
        # temp_image.show()
        #
        # exit()
        # draw = ImageDraw.Draw(temp_image)
        # n_boxes = temp_boxes.size(0)
        # rect_coord = []
        # for i in range(n_boxes):
        #     coord = ((int(temp_boxes[i][0] * resize_dim), int(temp_boxes[i][1] * resize_dim)),
        #              (int(temp_boxes[i][2] * resize_dim), int(temp_boxes[i][3] * resize_dim)))
        #     # rect_coord.append(coord)
        #     draw.rectangle(coord)
        # temp_image.show()
        #
        # exit()

        # show ignored regions
        # img = cv2.imread(os.path.join(image_folder, image_name))
        # img_masked = img.copy()
        # for region in ignore_regions:
        #     cv2.rectangle(img_masked, (region[0], region[1]), (region[2], region[3]), (127, 127, 127), -1)
        #
        # masked_image_path = os.path.join(image_folder, masked_image_name)
        # cv2.imwrite(masked_image_path, img_masked)

        # if os.path.exists(os.path.join(image_folder, masked_image_name)):
        #     print(os.path.join(image_folder, masked_image_name))
        #     os.remove(os.path.join(image_folder, masked_image_name))
        #     exit()

        # for i in range(len(boxes)):
        #     region = boxes[i]
        #     if len(occlusions[i]) > 0 and occlusions[i][0] == 0:
        #         cv2.rectangle(img_masked, (region[0], region[1]), (region[2], region[3]), (0, 0, 255), 2)
        #     else:
        #         cv2.rectangle(img_masked, (region[0], region[1]), (region[2], region[3]), (0, 255, 0), 2)
        #
        # cv2.imshow('Ignored regions', img_masked)
        # cv2.waitKey(1)

    assert len(boxes) == len(labels) == len(difficulties) == len(occlusions)
    assert len(object_list) == len(image_paths)

    return object_list, image_paths


def create_data_lists_detrac(root_path, output_folder):
    # training data
    dataType = 'Train'
    train_images = list()
    train_objects = list()
    n_object = 0

    annotation_folder = 'DETRAC-{}-Annotations-XML'.format(dataType)
    annotation_path = os.path.join(root_path, annotation_folder)
    if not os.path.exists(annotation_path):
        print('annotation_path not exist')
        raise FileNotFoundError

    image_folder = 'Insight-MVT_Annotation_{}'.format(dataType)
    image_root = os.path.join(root_path, image_folder)  # under: sequence_name/image_name
    count = 0
    for video in os.listdir(annotation_path):
        # print('Train data: {}/{}'.format(count + 1, len(os.listdir(annotation_path))))
        if video.endswith('.xml'):
            objects, image_frames_path = parse_annotation(os.path.join(annotation_path, video),
                                                          os.path.join(image_root, video.strip('.xml')))

            if len(objects) == 0:
                continue
            else:
                for obj in objects:
                    n_object += len(obj['bbox'])

            train_objects += objects
            train_images += image_frames_path
            count += 1

    assert len(train_objects) == len(train_images)
    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(traffic_label_map, j)  # save label map too

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_object, os.path.abspath(output_folder)))

    # test data
    dataType = 'Test'
    test_images = list()
    test_objects = list()
    n_object = 0

    annotation_folder = 'DETRAC-{}-Annotations-XML'.format(dataType)
    annotation_path = os.path.join(root_path, annotation_folder)
    if not os.path.exists(annotation_path):
        print('annotation_path not exist')
        raise FileNotFoundError

    image_folder = 'Insight-MVT_Annotation_{}'.format(dataType)
    image_root = os.path.join(root_path, image_folder)  # under: sequence_name/image_name
    count = 0
    for video in os.listdir(annotation_path):
        # print('Test data: {}/{}'.format(count + 1, len(os.listdir(annotation_path))))
        if video.endswith('.xml'):

            objects, image_frames_path = parse_annotation(os.path.join(annotation_path, video),
                                                          os.path.join(image_root, video.strip('.xml')), False)

            if len(objects) == 0:
                continue
            else:
                for obj in objects:
                    n_object += len(obj['bbox'])

            test_objects += objects
            test_images += image_frames_path
            count += 1

    assert len(test_objects) == len(test_images)
    # Save to file
    with open(os.path.join(output_folder, 'VAL_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'VAL_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\nThere are %d validation images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), n_object, os.path.abspath(output_folder)))


if __name__ == '__main__':
    """
    The root_path is the root folder of the downloaded DETRAC dataset, the structure may look like
    DETRAC:
        - DETRAC-Test-Annotations-XML
        - DETRAC-Test-Det
        - DETRAC-Train-Annotations-XML
        - Insight-MVT_Annotation_Test
        - Insight-MVT_Annotation_Train
        ...
    """
    root_path = '/home/keyi/Documents/Data/DETRAC'
    output_folder = '/home/keyi/Documents/research/code/shape_based_object_detection/data/DETRAC'

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    create_data_lists_detrac(root_path, output_folder)
