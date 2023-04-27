import numpy as np
from data.data_utils import CircleParams
import torch


def iou(a: CircleParams, b: CircleParams) -> float:
    """Calculate the intersection over union of two circles"""
    r1, r2 = a.radius, b.radius
    d = np.linalg.norm(np.array([a.row, a.col]) - np.array([b.row, b.col]))
    if d > r1 + r2:
        return 0
    if d <= abs(r1 - r2):
        return 1
    r1_sq, r2_sq = r1 ** 2, r2 ** 2
    d1 = (r1_sq - r2_sq + d ** 2) / (2 * d)
    d2 = d - d1
    h1 = r1_sq * np.arccos(d1 / r1)
    h2 = d1 * np.sqrt(r1_sq - d1 ** 2)
    h3 = r2_sq * np.arccos(d2 / r2)
    h4 = d2 * np.sqrt(r2_sq - d2 ** 2)
    intersection = h1 + h2 + h3 + h4
    union = np.pi * (r1_sq + r2_sq) - intersection
    return intersection / union


def iou_tensor(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Tensor a and tensor b are of shape (batch_size, 3) where each element is of the form (center_x, center_y, radius)
    This function returns a tensor of shape (batch_size, 1) where each element is the IOU between the corresponding circles in a and b
    """
    assert a.shape == b.shape, "Tensors supplied to iou_tensor must have the same shape"
    assert (
        a.shape[1] == 3
    ), "Tensors supplied to iou_tensor must have shape (batch_size, 3)"

    r1 = a[:, 2]
    r2 = b[:, 2]

    d = torch.linalg.norm(a[:, :2] - b[:, :2], dim=1)

    r1_sq, r2_sq = r1 ** 2, r2 ** 2
    d1 = (r1_sq - r2_sq + d ** 2) / (2 * d)
    d2 = d - d1
    h1 = r1_sq * torch.arccos(d1 / r1)
    h2 = d1 * torch.sqrt(r1_sq - d1 ** 2)
    h3 = r2_sq * torch.arccos(d2 / r2)
    h4 = d2 * torch.sqrt(r2_sq - d2 ** 2)
    intersection = h1 + h2 + h3 + h4
    union = torch.pi * (r1_sq + r2_sq) - intersection

    ans = intersection / union
    ans[r1 + r2 < d] = 0
    ans[torch.abs(r1 - r2) > d] = 1

    return ans


def circle_prediction_accuracy(
    threshold: int, circlesA: torch.tensor, circlesB: torch.tensor
) -> float:
    """
    This function returns the accuracy of the model on the dataset
    circlesA and circlesB are tensors of shape (batch_size, 3) where each element is of the form (center_x, center_y, radius)
    """
    iou = iou_tensor(circlesA, circlesB)
    accuracy = (
        iou_tensor(circlesA, circlesB) >= threshold
    ).sum().item() / circlesA.shape[0]

    return accuracy
