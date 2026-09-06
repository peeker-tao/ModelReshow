import os
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import random
import torch
import math
# from kornia.losses import ssim
from torchvision.transforms import ToTensor
from skimage.metrics import structural_similarity as compare_ssim


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            # skip non-string values (e.g. bool flags like strict_load)
            if isinstance(path, str):
                mkdir(path)


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('defusion-plusplus')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = ToTensor()(img1)
    img2 = ToTensor()(img2)
    img1 = img1.squeeze().permute(1, 2, 0).cpu().numpy()
    img2 = img2.squeeze().permute(1, 2, 0).cpu().numpy()
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    # ssim_value = ssim(img1, img2, 11, 'mean')
    # return 1 - ssim_value.item()
    img1 = img1.squeeze().permute(1, 2, 0).cpu().numpy()
    img2 = img2.squeeze().permute(1, 2, 0).cpu().numpy()
    # print("shape", img1.shape, img2.shape)
    ssim_value = compare_ssim(img1, img2, data_range=1, channel_axis = -1, multichannel=True)
    return ssim_value




def calculate_mae(img1, img2):
    mae = torch.mean((img1 - img2).abs(), dim=[2, 3, 1])
    return mae.squeeze().item()


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger (MuM/glog 风格)'''
    lg = logging.getLogger(logger_name)
    # 对齐 MuM 日志前缀：I20260821 10:12:29 3603480 defusion-plusplus utils.py:218] msg
    formatter = logging.Formatter(
        '%(levelname).1s%(asctime)s %(process)s %(name)s %(filename)s:%(lineno)s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S')
    lg.setLevel(level)
    # 幂等：同名 logger 已配置过则不重复添加 handler（否则日志会重复打印）
    if lg.handlers:
        return lg
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)
    return lg


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.random.manual_seed(seed)


def squeeze2d(input, factor):
    if factor == 1:
        return input

    B, C, H, W = input.size()

    assert H % factor == 0 and W % factor == 0, "H or W modulo factor is not 0"

    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)

    return x


class SmoothedValue(object):
    """追踪一组数值，提供窗口滑动统计（median/avg/global_avg）。
    对齐 MuM 的 SmoothedValue，默认格式 '{median:.4f} ({global_avg:.4f})'。"""

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = '{median:.4f} ({global_avg:.4f})'
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, num=1):
        if not math.isfinite(value):
            value = self.median
        self.deque.append(value)
        self.count += num
        self.total += value * num

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    """对齐 MuM 的 MetricLogger，用两个空格分隔各 meter，生成训练日志行。"""

    def __init__(self, delimiter='  '):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append('{}: {}'.format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_message(self, iteration, n_iterations, header, iter_time, data_time):
        """生成 MuM 风格训练日志行：
        Training  [i/n]  eta: ...  loss: ...  grad: ...  batch_size: ...  lr: ...  time: ...  data: ...  max mem: ...
        """
        space_fmt = ':' + str(len(str(n_iterations))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}',
            'max mem: {memory:.0f}',
        ])
        eta_seconds = iter_time.global_avg * (n_iterations - iteration)
        eta_string = str(timedelta(seconds=int(eta_seconds)))
        return log_msg.format(
            iteration, n_iterations,
            eta=eta_string,
            meters=str(self),
            time=str(iter_time),
            data=str(data_time),
            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
        )