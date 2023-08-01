import torch.nn as nn
import torch
from typing import Optional
import warnings

Tensor = torch.Tensor


class IntervalScore(nn.Module):
    """Computes the interval score for observations and their corresponding predicted interval.
    `loss = (q_upper - q_lower) + max((q_lower - target), 0) + max((target - q_upper), 0)`

    Args:
        target (Tensor): Ground truth values. shape = `[batch_size, d0, .. dn]`
        q_lower (Tensor): The predicted left/lower quantile. shape = `[batch_size, d0, .. dn]`
        q_upper (Tensor): The predicted right/upper quantile. shape = `[batch_size, d0, .. dn]`
        alpha (Float): Alpha level for (1-alpha) interval
        reduce (bool, optional): Boolean value indicating whether reducing the loss to one value or to
            a Tensor with shape = `[batch_size]`.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``.

    Raises:
        ValueError: One of the input shapes does not match

    Returns:
        interval_score: 1-D float `Tensor` with shape [batch_size] or Float if reduction = True

    References:
        Gneiting, T. and A. E. Raftery (2007). Strictly proper scoring rules, prediction,
        and estimation. Journal of the American Statistical Association 102(477), 359-378.
    """

    def __init__(
        self,
        alpha: Tensor,
        reduce: Optional[bool] = True,
        reduction: Optional[str] = "mean",
    ):
        super(IntervalScore, self).__init__()
        self.alpha = alpha
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, target: Tensor, q_lower: Tensor, q_upper: Tensor) -> Tensor:
        if not (target.size() == q_lower.size() == q_upper.size()):
            raise ValueError("Mismatching target and prediction shapes")
        if torch.any(q_lower > q_upper):
            warnings.warn(
                "Left quantile should be smaller than right quantile, check consistency!"
            )
        sharpness = q_upper - q_lower
        calibration = (
            (
                torch.clamp_min(q_lower - target, min=0)
                + torch.clamp_min(target - q_upper, min=0)
            )
            * 2
            / self.alpha
        )
        interval_score = sharpness + calibration
        if not self.reduce:
            return interval_score
        else:
            if self.reduction == "sum":
                return torch.sum(interval_score)
            else:
                return torch.mean(interval_score)


class QuantileScore(nn.Module):
    """Computes the qauntile score (pinball loss) between `y_true` and `y_pred`.
    `loss = maximum(alpha * (y_true - y_pred), (alpha - 1) * (y_true - y_pred))`

    Args:
        target (Tensor): Ground truth values. shape = `[batch_size, d0, .. dn]`
        prediction (Tensor): The predicted values. shape = `[batch_size, d0, .. dn]`
        alpha: Float in [0, 1] or a tensor taking values in [0, 1] and shape = `[d0,..., dn]`.
        reduce (bool, optional): Boolean value indicating whether reducing the loss to one value or to
            a Tensor with shape = `[batch_size]`.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``.

    Raises:
        ValueError: If input and target size do not match.

    Returns:
        quantile_score: 1-D float `Tensor` with shape [batch_size] or Float if reduction = True

    References:
      - https://en.wikipedia.org/wiki/Quantile_regression
    """

    def __init__(
        self,
        alpha: Tensor,
        reduce: Optional[bool] = True,
        reduction: Optional[str] = "mean",
    ):
        super(QuantileScore, self).__init__()
        self.alpha = alpha
        self.reduce = reduce
        self.reduction = reduction

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        if not (target.size() == prediction.size()):
            raise ValueError(
                "Using a target size ({}) that is different to the input size ({}). "
                "".format(target.size(), prediction.size())
            )
        errors = target - prediction
        quantile_score = torch.max((self.alpha - 1) * errors, self.alpha * errors)
        if not self.reduce:
            return quantile_score
        else:
            if self.reduction == "sum":
                return torch.sum(quantile_score)
            else:
                return torch.mean(quantile_score)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    q_upper = torch.rand(size=(3, 2)) + 1
    q_lower = torch.rand(size=(3, 2))
    target = torch.rand(size=(3, 2))
    interval_score = IntervalScore(alpha=0.1)
    res = interval_score(target, q_lower, q_upper)
    print(res.shape)
    print(res)
    # plt.plot(prediction, res)
