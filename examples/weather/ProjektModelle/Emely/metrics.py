
#-----------------------------------------------------
#---------------------- METRICS ----------------------
#-----------------------------------------------------


class AverageMeter(object):
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

    def to_dict(self):
        return {
            "val": float(self.val),
            "avg": float(self.avg),
            "sum": float(self.sum),
            "count": int(self.count)
        }

