import numpy as np
import collections
import torch
import time
import datetime


def strf_time_delta(td):
    td_str = ""
    if td.days > 0:
        td_str += f"{td.days} days, " if td.days > 1 else f"{td.days} day, "
    hours = td.seconds // 3600
    if hours > 0:
        td_str += f"{hours}h "
    minutes = (td.seconds // 60) % 60
    if minutes > 0:
        td_str += f"{minutes}m "
    seconds = td.seconds % 60 + td.microseconds * 1e-6
    td_str += f"{seconds:.1f}s"
    return td_str


class Timer:
    def __init__(self, info=None, log_event=None):
        self.info = info
        self.log_event = log_event

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end.record()
        torch.cuda.synchronize()
        self.duration = self.start.elapsed_time(self.end) / 1000
        if self.info:
            self.info[f"duration/{self.log_event}"] = self.duration


class TimeLog:
    def __init__(self, logger, total_num, event):
        self.logger = logger
        self.total_num = total_num
        self.event = event.upper()
        self.start = time.time()

    def now(self, current_num):
        elapsed = time.time() - self.start
        left = self.total_num * elapsed / (current_num + 1) - elapsed
        elapsed = strf_time_delta(datetime.timedelta(seconds=elapsed))
        left = strf_time_delta(datetime.timedelta(seconds=left))
        self.logger.log_info(
            f"TIME ELAPSED SINCE {self.event} START: {elapsed}"
        )
        self.logger.log_info(f"TIME LEFT UNTIL {self.event} END: {left}")

    def end(self):
        elapsed = time.time() - self.start
        elapsed = strf_time_delta(datetime.timedelta(seconds=elapsed))
        self.logger.log_info(
            f"TIME ELAPSED SINCE {self.event} START: {elapsed}"
        )
        self.logger.log_info(f"{self.event} ENDS")


class MeanTracker(object):
    def __init__(self, name):
        self.values = []
        self.name = name

    def add(self, val):
        self.values.append(float(val))

    def mean(self):
        return np.mean(self.values)

    def flush(self):
        mean = self.mean()
        self.values = []
        return self.name, mean


class _StreamingMean:
    def __init__(self, val=None, counts=None):
        if val is None:
            self.mean = 0.0
            self.counts = 0
        else:
            if isinstance(val, torch.Tensor):
                val = val.data.cpu().numpy()
            self.mean = val
            if counts is not None:
                self.counts = counts
            else:
                self.counts = 1

    def update(self, mean, counts=1):
        if isinstance(mean, torch.Tensor):
            mean = mean.data.cpu().numpy()
        elif isinstance(mean, _StreamingMean):
            mean, counts = mean.mean, mean.counts * counts
        assert counts >= 0
        if counts == 0:
            return
        total = self.counts + counts
        self.mean = self.counts / total * self.mean + counts / total * mean
        self.counts = total

    def __add__(self, other):
        new = self.__class__(self.mean, self.counts)
        if isinstance(other, _StreamingMean):
            if other.counts == 0:
                return new
            else:
                new.update(other.mean, other.counts)
        else:
            new.update(other)
        return new


class StreamingMeans(collections.defaultdict):
    def __init__(self):
        super().__init__(_StreamingMean)

    def __setitem__(self, key, value):
        if isinstance(value, _StreamingMean):
            super().__setitem__(key, value)
        else:
            super().__setitem__(key, _StreamingMean(value))

    def update(self, *args, **kwargs):
        for_update = dict(*args, **kwargs)
        for k, v in for_update.items():
            self[k].update(v)

    def to_dict(self, prefix=""):
        return dict((prefix + k, v.mean) for k, v in self.items())

    def to_str(self):
        return ", ".join([f"{k} = {v:.3f}" for k, v in self.to_dict().items()])