import torch
import logging


def save_checkpoint(epoch, model, optimizer, name=None):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    :param name: save file path
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    if name is None:
        filename = 'checkpoints/my_checkpoint_ssd300.pth.tar'
    else:
        filename = name
    torch.save(state, filename)


def clip_sigmoid(input):
    out = torch.clamp(torch.sigmoid(input), min=1e-4, max=1 - 1e-4)

    return out


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def create_logger(name, log_file, level=logging.INFO):
    log = logging.getLogger(name)
    formatter = logging.Formatter('[%(asctime)s][%(filename)10s][line:%(lineno)4d][%(levelname)4s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    log.setLevel(level)
    log.addHandler(fh)
    log.addHandler(sh)
    return log
