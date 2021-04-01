import torch

def bpr_loss(pos, neg, b=0.0, collapse=True):
    """
    Usual BPR loss that penalizes when negative score is greater than positive one.

    Parameters
    ----------
    pos: torch.tensor
        positive score

    neg: torch.tensor
        negative scores. Can be a batch of negatives.

    b: float
        desired boundary between positive and negative examples

    collapse: bool
        If True collapse batch with mean.

    Returns
    -------
    torch.tensor
    """
    res = torch.sigmoid(neg - pos + b)
    if collapse:
        res = res.mean()
    return res


def warp_loss(pos, neg, b=1, collapse=True):
    """
    Batch version of WARP loss.

    Regular version samples one negative example until violation is met.
    This number of samples become the estimation of rank and weight is produced as some function of this rank.

    This version uses a batch of negatives and estimates rank as a number of violated examples.

    If you use the number of first violation as rank this will degenerate to usual WARP with a limit on draws.

    Parameters
    ----------
    pos: torch.tensor
        positive score

    neg: torch.tensor
        negative scores. Can be a batch of negatives.

    b: float
        desired boundary between positive and negative examples

    collapse: bool
        If True collapse batch with mean.

    Returns
    -------
    torch.tensor
    """
    loss = bpr_loss(pos, neg, b, collapse=False)
    m = (loss > 0.5).float()
    m *= torch.log(m.sum() + 1) + 1
    res = m * loss
    if collapse:
        res = res.mean()
    return res