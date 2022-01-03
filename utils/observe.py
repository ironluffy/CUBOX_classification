import torch


# TODO: class-mean Accruacy
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


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


class Observe(object):
    # TODO add class-wise accuracy
    def __init__(self, prefix=''):
        self.prefix = prefix
        self.ce_loss = AverageMeter()
        self.accuracy = AverageMeter()
        self.softmax = torch.nn.Softmax(dim=1)
        self.ce = torch.nn.CrossEntropyLoss(reduction='none')

    def update(self, output, batch):
        with torch.no_grad():
            gt = batch['gt'].cuda()
            num_sample = gt.size(0)

            gt_ce = self.ce(output, gt).mean()
            gt_acc = accuracy(output, gt)[0]

            self.ce_loss.update(gt_ce, num_sample)
            self.accuracy.update(gt_acc, num_sample)

    def get_info_dict(self):
        info_dict = {}
        info_dict["{}/CE".format(self.prefix)] = self.ce_loss.avg
        info_dict["{}/Acc.".format(self.prefix)] = self.accuracy.avg
        return info_dict
