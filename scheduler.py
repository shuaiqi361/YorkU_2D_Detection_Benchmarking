import json
import os
import torch
import math


def adjust_learning_rate(optimizer, scale):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale

    print("DECAYING learning rate, the new LR is %f" % (optimizer.param_groups[1]['lr'],))


def warm_up_learning_rate(optimizer, rate=5.):
    """
    Scale learning rate by a specified factor.

    :param rate:
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param scale: factor to multiply learning rate with.
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * rate

    print("WARMING up learning rate, the new LR is %f" % (optimizer.param_groups[1]['lr'],))


class WarmUpScheduler(object):
    def __init__(self, target_lr, n_steps, optimizer, types='exp'):
        self.target_lr = target_lr
        self.n_steps = n_steps
        self.optimizer = optimizer
        self.init_scheduler(types)

    def init_scheduler(self, types):
        if types.lower() == 'exp':
            self.rate = 2.
            self.init_lr = self.target_lr / (self.rate ** self.n_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / (self.rate ** self.n_steps)
            print('EXP Warming up lr from {:.6f}'.format(self.init_lr))
        else:
            self.init_lr = self.target_lr * 0.1
            self.rate = (self.target_lr - self.init_lr) / self.n_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.init_lr
            print('Linear Warming up lr from {:.6f}'.format(self.init_lr))

    def update(self, types='exp'):
        if types.lower() == 'exp':
            if self.n_steps > 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * self.rate
                # print(self.n_steps, self.target_lr, self.rate)
                print('New lr {:.6f}'.format(self.target_lr / (self.rate ** (self.n_steps - 1))))

            else:
                return
        else:
            if self.n_steps > 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] + (self.target_lr - self.init_lr) / self.n_steps
                print('New lr {:.6f}'.format(self.target_lr - self.rate * (self.n_steps - 1)))
            else:
                return

        self.n_steps -= 1
