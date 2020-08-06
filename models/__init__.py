from .SSD512 import SSD512, MultiBoxLoss512
from .SSD_ATSS import ATSSSSD512, ATSSSSD512Loss
from .NETNet import VGGNETNetDetector, VGGNETNetDetectorLoss
from .RefineDet import RefineDet512, RefineDetLoss
from .RetinaNet import resnet50, resnet101, RetinaFocalLoss
from .RetinaNet_ATSS import RetinaATSSNetLoss, RetinaATSS50, RetinaATSS101


def model_entry(config):
    if config.model['arch'].upper() == 'SSD512':
        print('Loading SSD512 with VGG16 backbone ......')
        return SSD512(config['n_classes'], device=config.device), MultiBoxLoss512
    elif config.model['arch'].upper() == 'ATSSSSD':
        print('Loading ATSS SSD Detector ......')
        return ATSSSSD512(config['n_classes'], config=config), ATSSSSD512Loss
    elif config.model['arch'].upper() == 'RETINA50':
        print('Loading RetinaNet with ResNet-50 backbone ......')
        return resnet50(config['n_classes'], config=config), RetinaFocalLoss
    elif config.model['arch'].upper() == 'RETINA101':
        print('Loading RetinaNet with ResNet-101 backbone ......')
        return resnet101(config['n_classes'], config=config), RetinaFocalLoss
    elif config.model['arch'].upper() == 'RETINAATSS50':
        print('Loading ATSS RetinaNet Detector with ResNet50 ......')
        return RetinaATSS50(config['n_classes'], config=config), RetinaATSSNetLoss
    elif config.model['arch'].upper() == 'RETINAATSS101':
        print('Loading ATSS RetinaNet Detector with ResNet101 ......')
        return RetinaATSS101(config['n_classes'], config=config), RetinaATSSNetLoss
    elif config.model['arch'].upper() == 'REFINEDET':
        print('Loading RefineDet512 with VGG-16 backbone ......')
        return RefineDet512(config['n_classes'], config=config), RefineDetLoss
    elif config.model['arch'].upper() == 'NETNET':
        print('Loading NETNet with VGG16 backbone Detector ......')
        return VGGNETNetDetector(config['n_classes'], config=config), VGGNETNetDetectorLoss
    else:
        print('Models not implemented.')
        raise NotImplementedError
