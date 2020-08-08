import torch.nn as nn
import torch
from math import sqrt, log
import math
import torch.utils.model_zoo as model_zoo
import torchvision
from dataset.transforms import *
from operators.Loss import IouLoss, SigmoidFocalLoss
from metrics import find_jaccard_overlap
from .utils import BasicBlock, Bottleneck
from operators.iou_utils import find_distance

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}


class PyramidFeatures(nn.Module):
    def __init__(self, c3_size, c4_size, c5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(c5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(c4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(c3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(c5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU(inplace=True)
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, c3, c4, c5):
        # print(c3.size(), c4.size(), c5.size())
        P5_x = self.P5_1(c5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(c4)
        # print(P5_upsampled_x.size(), P4_x.size())
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(c3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(c5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return P3_x, P4_x, P5_x, P6_x, P7_x


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(32, feature_size)

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(32, feature_size)

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(32, feature_size)

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.gn4 = nn.GroupNorm(32, feature_size)
        self.act = nn.ReLU(inplace=True)

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.gn1(self.conv1(x))
        out = self.act(out)

        out = self.gn2(self.conv2(out))
        out = self.act(out)

        out = self.gn3(self.conv3(out))
        out = self.act(out)

        out = self.gn4(self.conv4(out))
        out = self.act(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors, num_classes, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(32, feature_size)

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(32, feature_size)

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(32, feature_size)

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.gn4 = nn.GroupNorm(32, feature_size)
        self.act = nn.ReLU(inplace=True)

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.gn1(self.conv1(x))
        out = self.act(out)

        out = self.gn2(self.conv2(out))
        out = self.act(out)

        out = self.gn3(self.conv3(out))
        out = self.act(out)

        out = self.gn4(self.conv4(out))
        out = self.act(out)

        out = self.output(out)

        out1 = out.permute(0, 2, 3, 1)

        return out1.contiguous().view(x.size(0), -1, self.num_classes)


class RetinaATSSNet(nn.Module):
    """
        The RetinaNet network - encapsulates the base ResNet network, Detection, and Classification Head.
        """

    def __init__(self, n_classes, block, layers, prior=0.01, device='cuda:0'):
        super(RetinaATSSNet, self).__init__()
        self.inplanes = 64  # dim of c3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.prior = prior
        self.n_classes = n_classes - 1
        self.device = device

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError("Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], 256)
        self.regressionModel = RegressionModel(256, num_anchors=1)
        self.classificationModel = ClassificationModel(256, num_anchors=1, num_classes=n_classes)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - self.prior) / self.prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        # self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        """
        Freeze BatchNorm layers
        """
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, image):

        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn(x2, x3, x4)

        locs = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        class_scores = torch.cat([self.classificationModel(feature) for feature in features], dim=1)

        return locs, class_scores

    def create_prior_boxes(self):
        """
        Create the 22536 prior (default) boxes for the SSD300, as defined in the paper.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (22536, 4)
        """

        fmap_dims = {'c3': [64, 64],
                     'c4': [32, 32],
                     'c5': [16, 16],
                     'c6': [8, 8],
                     'c7': [4, 4]}

        obj_scales = {'c3': 0.04,
                      'c4': 0.08,
                      'c5': 0.16,
                      'c6': 0.32,
                      'c7': 0.64}

        aspect_ratios = {'c3': [1.],
                         'c4': [1.],
                         'c5': [1.],
                         'c6': [1.],
                         'c7': [1.]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []
        scale_factor = [1.]
        for k, fmap in enumerate(fmaps):
            temp_prior_boxes = []
            for i in range(fmap_dims[fmap][0]):
                for j in range(fmap_dims[fmap][1]):
                    cx = (j + 0.5) / fmap_dims[fmap][1]  # sliding center locations across the feature maps
                    cy = (i + 0.5) / fmap_dims[fmap][0]

                    for ratio in aspect_ratios[fmap]:
                        for fac in scale_factor:
                            temp_prior_boxes.append([cx, cy, obj_scales[fmap] * fac * sqrt(ratio),
                                                     obj_scales[fmap] * fac / sqrt(ratio)])

            temp_prior_boxes = torch.FloatTensor(temp_prior_boxes).to(self.device).contiguous()
            temp_prior_boxes.clamp_(0, 1)
            prior_boxes.append(temp_prior_boxes)

        return prior_boxes


class RetinaATSSNetLoss(nn.Module):
    """
    The RetinaFocalLoss, a loss function for object detection from RetinaNet.
    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, config, n_candidates=9):
        super(RetinaATSSNetLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = [cxcy_to_xy(prior) for prior in self.priors_cxcy]
        self.n_candidates = n_candidates
        self.alpha = config.reg_weights
        self.device = config.device
        self.n_classes = config.n_classes - 1

        fmap_dims = {'c3': [64, 64],
                     'c4': [32, 32],
                     'c5': [16, 16],
                     'c6': [8, 8],
                     'c7': [4, 4]}

        self.prior_split_points = [0]
        fmap_keys = ['c3', 'c4', 'c5', 'c6', 'c7']
        for k in fmap_keys:
            self.prior_split_points.append(self.prior_split_points[-1] + fmap_dims[k][0] * fmap_dims[k][1])

        self.regression_loss = IouLoss(pred_mode='Corner', reduce='mean', losstype='Ciou')
        self.classification_loss = SigmoidFocalLoss(gamma=2.0, alpha=0.25, config=config)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param fmap_dims:
        :param predicted_locs: predicted locations/boxes w.r.t the 22536 prior boxes, a tensor of dimensions (N, 22536, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 22536, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        n_levels = len(self.priors_cxcy)
        batch_size = predicted_locs.size(0)

        decoded_locs = list()  # length is the batch size
        true_locs = list()
        true_classes = list()
        predicted_class_scores = list()

        # For each image
        for i in range(batch_size):
            image_bboxes = boxes[i]

            batch_split_predicted_locs = []
            batch_split_predicted_scores = []
            # split the predictions according to the feature pyramid dimension
            for s in range(len(self.priors_cxcy)):
                batch_split_predicted_locs.append(
                    predicted_locs[i][self.prior_split_points[s]:self.prior_split_points[s + 1], :])
                batch_split_predicted_scores.append(
                    predicted_scores[i][self.prior_split_points[s]:self.prior_split_points[s + 1], :])

            # candidates for positive samples, use to calculate the IOU threshold
            positive_samples_idx = list()
            positive_overlaps = list()
            overlap = list()  # for all
            for level in range(n_levels):
                distance = find_distance(xy_to_cxcy(image_bboxes), self.priors_cxcy[level])  # n_bboxes, n_priors

                _, top_idx_level = torch.topk(-1. * distance, min(self.n_candidates, distance.size(1)), dim=1)
                positive_samples_idx.append(top_idx_level)
                overlap_level = find_jaccard_overlap(image_bboxes, self.priors_xy[level])  # overlap for each level
                positive_overlaps.append(torch.gather(overlap_level, dim=1, index=top_idx_level))

                overlap.append(overlap_level)

            positive_overlaps_cat = torch.cat(positive_overlaps, dim=1)
            overlap_mean = torch.mean(positive_overlaps_cat, dim=1)
            overlap_std = torch.std(positive_overlaps_cat, dim=1)
            # print(overlap_mean, overlap_std)
            iou_threshold = overlap_mean + overlap_std  # n_bboxes, for each object, we have one threshold

            # one prior can only be associated to one gt object
            # For each prior, find the object that has the maximum overlap, return [value, indices]
            true_classes_level = list()
            true_locs_level = list()
            positive_priors = list()  # For all levels
            decoded_locs_level = list()
            for level in range(n_levels):
                positive_priors_per_level = torch.zeros((image_bboxes.size(0), self.priors_cxcy[level].size(0)),
                                                        dtype=torch.uint8).to(self.device)  # indexing, (n,)

                for ob in range(image_bboxes.size(0)):
                    for c in range(len(positive_samples_idx[level][ob])):
                        # print(ob, c, 'Range for c: ', len(positive_samples_idx[level][ob]))
                        current_iou = positive_overlaps[level][ob, c]
                        current_bbox = image_bboxes[ob, :]
                        current_prior = self.priors_cxcy[level][positive_samples_idx[level][ob, c], :]

                        if current_iou > iou_threshold[ob]:
                            if current_bbox[0] < current_prior[0] < current_bbox[2] \
                                    and current_bbox[1] < current_prior[1] < current_bbox[3]:
                                positive_priors_per_level[ob, positive_samples_idx[level][ob, c]] = 1

                positive_priors.append(positive_priors_per_level)

            for level in range(n_levels):  # this is the loop for find the best object for each prior,
                # because one prior could match with more than one objects
                label_for_each_prior_per_level = torch.zeros((self.priors_cxcy[level].size(0)),
                                                             dtype=torch.long).to(self.device)
                true_locs_per_level = list()  # only for positive candidates in the predictions
                decoded_locs_per_level = list()
                total_decoded_locs = cxcy_to_xy(
                    gcxgcy_to_cxcy(batch_split_predicted_locs[level], self.priors_cxcy[level]))

                for c in range(positive_samples_idx[level].size(1)):
                    # for c in range(self.priors_cxcy[level].size(0)):  # loop over each prior in each level
                    current_max_iou = 0.
                    current_max_iou_ob = -1  # index for rows: (n_ob, n_prior)
                    for ob in range(image_bboxes.size(0)):
                        if positive_priors[level][ob, positive_samples_idx[level][ob, c]] == 1:
                            if overlap[level][ob, positive_samples_idx[level][ob, c]] > current_max_iou:
                                current_max_iou_ob = ob
                                current_max_iou = overlap[level][ob, positive_samples_idx[level][ob, c]]

                    if current_max_iou_ob > -1 and current_max_iou > 0.:
                        temp_true_locs = image_bboxes[current_max_iou_ob, :].unsqueeze(0)  # (1, 4)
                        temp_decoded_locs = total_decoded_locs[positive_samples_idx[level][current_max_iou_ob, c],
                                            :].unsqueeze(0)  # (1, 4)
                        label_for_each_prior_per_level[positive_samples_idx[level][current_max_iou_ob, c]] = labels[i][
                            current_max_iou_ob]
                        true_locs_per_level.append(temp_true_locs)
                        decoded_locs_per_level.append(temp_decoded_locs)

                if len(true_locs_per_level) > 0:
                    true_locs_level.append(torch.cat(true_locs_per_level, dim=0).view(-1, 4))  # (1, n_l * 4)
                    decoded_locs_level.append(torch.cat(decoded_locs_per_level, dim=0).view(-1, 4))

                true_classes_level.append(label_for_each_prior_per_level)

            # Store
            true_classes.append(torch.cat(true_classes_level, dim=0))  # batch_size, n_priors
            predicted_class_scores.append(torch.cat(batch_split_predicted_scores, dim=0))
            if len(true_locs_level) > 0:
                true_locs.append(torch.cat(true_locs_level, dim=0))  # batch_size, n_pos, 4
                decoded_locs.append(torch.cat(decoded_locs_level, dim=0))

        # assemble all samples from batches
        true_classes = torch.cat(true_classes, dim=0)
        positive_priors = true_classes > 0
        predicted_scores = torch.cat(predicted_class_scores, dim=0)
        true_locs = torch.cat(true_locs, dim=0)
        decoded_locs = torch.cat(decoded_locs, dim=0)

        # LOCALIZATION LOSS
        loc_loss = self.regression_loss(decoded_locs, true_locs)

        # CONFIDENCE LOSS
        n_positives = positive_priors.sum().float()

        # First, find the loss for all priors
        conf_loss = self.classification_loss(predicted_scores, true_classes) / n_positives
        # conf_loss = self.FocalLoss(predicted_scores, true_classes) / len(true_classes)

        # TOTAL LOSS
        return conf_loss + self.alpha * loc_loss


def RetinaATSS34(num_classes, config, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = RetinaATSSNet(num_classes, BasicBlock, [3, 4, 6, 3], device=config.device)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def RetinaATSS50(num_classes, config, pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param config:
        :param pretrained:
        :param num_classes:
    """
    model = RetinaATSSNet(num_classes, Bottleneck, [3, 4, 6, 3], device=config.device)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def RetinaATSS101(num_classes, config, pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        :param config:
        :param pretrained:
        :param num_classes:
    """
    model = RetinaATSSNet(num_classes, Bottleneck, [3, 4, 23, 3], device=config.device)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model
