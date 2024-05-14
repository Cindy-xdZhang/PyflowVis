from math import log10
import numpy as np

def PSNR(mse, peak=1.):
    return 10 * log10((peak ** 2) / mse)


class SegMetric:
    def __init__(self, values=0.):
        assert isinstance(values, dict)
        self.miou = values.miou
        self.oa = values.get('oa', None) 
        self.miou = values.miou
        self.miou = values.miou


    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


