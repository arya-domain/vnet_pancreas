import numpy
import torch
import torch.nn.functional as F


def dice_loss(scores, targets, smooth=1e-5):
    targets = targets.float()
    intersect = torch.sum(scores * targets)
    y_sum = torch.sum(targets * targets)
    z_sum = torch.sum(scores * scores)
    loss = 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)

    return loss


# softmax mse loss
def semi_mse_loss(pred, y_logits):
    y_prob = torch.softmax(y_logits, dim=1)
    pred_prob = torch.softmax(pred, dim=1)

    return F.mse_loss(y_prob, pred_prob, reduction="none")

def mse_loss(inputs, targets ):
    loss = torch.nn.MSELoss()
    output = loss(inputs ,targets)
    return output

def semi_crc_loss(inputs, targets, threshold=0.65, neg_threshold=0.1, conf_mask=True):
    if not conf_mask:
        raise NotImplementedError

    inputs_prob = F.softmax(inputs, dim=1)
    targets_prob = F.softmax(targets, dim=1)

    # for positive
    pos_weight = targets_prob.max(1)[0]
    pos_mask = pos_weight >= threshold

    # for negative
    neg_weight = targets_prob.min(1)[0]
    neg_mask = neg_weight < neg_threshold

    y_tilde = torch.argmax(targets, dim=1)

    ######################## mse mask ################################
    mse_mask = ~pos_mask & ~neg_mask

    if torch.any(mse_mask):
        outputs = mse_loss(inputs_prob[0][mse_mask], targets_prob[0][mse_mask])
    else:
        outputs = torch.tensor([0.0], device=targets.device)

    #######################################################

    # postive
    if not torch.any(pos_mask):
        positive_loss_mat = torch.tensor([0.0], device=targets.device)
    else:
        positive_loss_mat = F.nll_loss(
            torch.log_softmax(inputs, dim=1), y_tilde, reduction="none"
        )
        positive_loss_mat = positive_loss_mat * pos_weight
        positive_loss_mat = positive_loss_mat[pos_mask]

    # negative
    if not torch.any(neg_mask):  # >= 0.1
        negative_loss_mat = torch.tensor([0.0], device=targets.device)
    else:  # < 0.1
        inverse_prob = torch.clamp(
            1 - F.softmax(inputs, dim=1), min=1e-6, max=1.0
        )  # onehot
        negative_loss_mat = F.nll_loss(
            inverse_prob.log(), (1 - y_tilde), reduction="none"
        )
        negative_loss_mat = negative_loss_mat * neg_weight
        negative_loss_mat = negative_loss_mat[neg_mask]

    return positive_loss_mat.mean() + negative_loss_mat.mean() + outputs.mean(), None


class ConsistencyWeight(object):
    """
    ramp_types = ['sigmoid_rampup', 'linear_rampup', 'cosine_rampup', 'log_rampup', 'exp_rampup']
    """

    def __init__(self, config):
        self.final_w = config.unsup_weight
        self.iter_per_epoch = config.iter_per_epoch
        self.start_iter = config.rampup_start * config.iter_per_epoch
        self.rampup_length = config.rampup_length * config.iter_per_epoch
        self.rampup_func = getattr(self, config.rampup_type)
        self.current_rampup = 0

    def __call__(self, current_idx):
        if current_idx <= self.start_iter:
            return 0.0

        self.current_rampup = self.rampup_func(
            current_idx - self.start_iter, self.rampup_length
        )

        return self.final_w * self.current_rampup

    @staticmethod
    def gaussian(start, current, rampup_length):
        assert rampup_length >= 0
        if current == 0:
            return 0.0
        if current < start:
            return 0.0
        if current >= rampup_length:
            return 1.0
        return numpy.exp(-5 * (1 - current / rampup_length) ** 2)

    @staticmethod
    def sigmoid(current, rampup_length):
        if rampup_length == 0:
            return 1.0
        current = numpy.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length

        return float(numpy.exp(-5.0 * phase * phase))

    @staticmethod
    def linear(current, rampup_length):
        assert current >= 0 and rampup_length >= 0
        if current >= rampup_length:
            return 1.0
        return current / rampup_length
