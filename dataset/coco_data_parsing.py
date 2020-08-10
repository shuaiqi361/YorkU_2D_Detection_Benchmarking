import json
import os
from pycocotools.coco import COCO

'''
There are 117266 training images containing a total of 860001 objects.
Files have been saved to /media/keyi/Data/Research/course_project/AdvancedCV_2020/AdvanceCV_project/data/COCO.
There are 4952 validation images containing a total of 36781 objects.
Files have been saved to /media/keyi/Data/Research/course_project/AdvancedCV_2020/AdvanceCV_project/data/COCO.
'''


def create_data_lists_coco17(coco_root_path, output_folder):
    """
    Create lists of images, the bounding boxes[corner convention] and labels[different from the original]
    of the objects in these images, and save these to file.
    :param coco_root_path: root to coco17 dataset, images and annotations folders should be in this path
    :param output_folder: store the output json files
    """
    image_folder = os.path.join(coco_root_path, 'images')
    annotation_folder = os.path.join(coco_root_path, 'annotations')
    if not os.path.exists(image_folder) or not os.path.exists(annotation_folder):
        print(image_folder)
        print(annotation_folder)
        print('COCO root path does not contain images or annotations.')
        raise NotADirectoryError

    # training data
    train_images = list()
    train_objects = list()

    dataType = 'train2017'
    annFile = '{}/annotations/instances_{}.json'.format(coco_root_path, dataType)
    coco = COCO(annFile)
    cats = coco.loadCats(coco.getCatIds())
    coco_labels = [cat['name'] for cat in cats]  # list of labels

    coco_label_map = {k: v + 1 for v, k in
                      enumerate(coco_labels)}  # this is a customized map, with background class added as label 0
    coco_label_map['background'] = 0
    rev_coco_label_map = {v: k for k, v in coco_label_map.items()}  # Inverse mapping

    catIds = coco.getCatIds(catNms=coco_labels)
    annIds = coco.getAnnIds(catIds=catIds)
    all_anns = coco.loadAnns(ids=annIds)  # get all annotations

    all_images_dict = {}  # dict with image_id(str): {'image_path':'', 'bbox': [[]], 'labels': [], 'cat_names': []}
    count_obj = 0
    for annotation in all_anns:
        count_obj += 1
        if annotation['iscrowd'] == 1:
            continue

        bbox = annotation['bbox']
        if bbox[2] < 1 or bbox[3] < 1:
            print('Eliminate invalid objects for training < 1px.')
            continue

        image_id = annotation['image_id']
        if image_id not in all_images_dict.keys():
            all_images_dict[image_id] = {'image_path': '', 'bbox': [], 'labels': [],
                                         'cat_names': [], 'difficulties': []}

        corner_notation = [int(bbox[0]), int(bbox[1]),
                           int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]
        all_images_dict[image_id]['bbox'].append(corner_notation)

        img = coco.loadImgs(image_id)[0]
        image_path = '%s/%s/%s' % (image_folder, dataType, img['file_name'])
        all_images_dict[image_id]['image_path'] = image_path
        cat_id = annotation['category_id']
        cat_name = coco.loadCats([cat_id])[0]['name']
        all_images_dict[image_id]['cat_names'].append(cat_name)
        all_images_dict[image_id]['difficulties'].append(0)

        if cat_name in coco_label_map.keys():
            label = coco_label_map[cat_name]
            all_images_dict[image_id]['labels'].append(label)
        else:
            del all_images_dict[image_id]  # class must be in these 80 classes

        # print('Loading training objects from COCO17 annotations:', count_obj, '/', len(all_anns))

    for img_id, annots in all_images_dict.items():
        train_images.append(annots['image_path'])
        objects_coco = {'bbox': annots['bbox'], 'labels': annots['labels'], 'image_id': img_id,
                        'difficulties': annots['difficulties']}
        train_objects.append(objects_coco)

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), count_obj, os.path.abspath(output_folder)))
    assert len(train_objects) == len(train_images)

    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(train_objects, j)
    with open(os.path.join(output_folder, 'label_map.json'), 'w') as j:
        json.dump(coco_label_map, j)  # save label map too

    # testing images(val)
    test_images = list()
    test_objects = list()

    dataType = 'val2017'
    annFile = '{}/annotations/instances_{}.json'.format(coco_root_path, dataType)
    coco = COCO(annFile)

    catIds = coco.getCatIds(catNms=coco_labels)
    annIds = coco.getAnnIds(catIds=catIds)
    all_anns = coco.loadAnns(ids=annIds)  # get all annotations

    all_images_dict.clear()  # dict with image_id(str): {'image_path':'', 'bbox': [[]], 'labels': [], 'cat_names': []}
    count_obj = 0
    for annotation in all_anns:
        count_obj += 1
        if annotation['iscrowd'] == 1:
            continue

        bbox = annotation['bbox']
        if bbox[2] < 1 or bbox[3] < 1:
            print('Eliminate invalid objects for testing < 1px.')
            continue

        image_id = annotation['image_id']
        if image_id not in all_images_dict.keys():
            all_images_dict[image_id] = {'image_path': '', 'bbox': [], 'labels': [],
                                         'cat_names': [], 'difficulties': []}

        img = coco.loadImgs(image_id)[0]
        image_path = '%s/%s/%s' % (image_folder, dataType, img['file_name'])
        all_images_dict[image_id]['image_path'] = image_path

        corner_notation = [int(bbox[0]), int(bbox[1]),
                           int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])]
        all_images_dict[image_id]['bbox'].append(corner_notation)

        cat_id = annotation['category_id']
        cat_name = coco.loadCats([cat_id])[0]['name']
        all_images_dict[image_id]['cat_names'].append(cat_name)
        all_images_dict[image_id]['difficulties'].append(0)

        if cat_name in coco_label_map.keys():
            label = coco_label_map[cat_name]
            all_images_dict[image_id]['labels'].append(label)
        else:
            del all_images_dict[image_id]  # class must be in these 80 classes

        # print('Loading validation objects from COCO17 annotations:', count_obj, '/', len(all_anns))

    for img_id, annots in all_images_dict.items():
        test_images.append(annots['image_path'])
        objects_coco = {'bbox': annots['bbox'], 'labels': annots['labels'], 'image_id': img_id,
                        'difficulties': annots['difficulties']}
        test_objects.append(objects_coco)

    # print('Total validation images: ', len(test_images))
    assert len(test_objects) == len(test_images)

    # Save to file
    with open(os.path.join(output_folder, 'VAL_images.json'), 'w') as j:
        json.dump(test_images, j)
    with open(os.path.join(output_folder, 'VAL_objects.json'), 'w') as j:
        json.dump(test_objects, j)

    print('\nThere are %d validation images containing a total of %d objects. Files have been saved to %s.' % (
        len(test_images), count_obj, os.path.abspath(output_folder)))


if __name__ == '__main__':
    """
    The coco_root_path is the root folder of the downloaded dataset, the structure may look like
    COCO_17:
        - annotations
            instances_train2017.json
            instances_val2017.json
            ...
        - images
            train2017
            val2017
            ...
        - ...

    """
    coco_root_path = '/home/lab2/keyi/data/COCO17'
    output_folder = '/home/lab2/keyi/code/detection/YorkU_2D_Detection_Benchmarking/data/COCO'

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    create_data_lists_coco17(coco_root_path, output_folder)
