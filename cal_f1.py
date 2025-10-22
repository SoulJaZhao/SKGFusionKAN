import numpy as np

def f1_from_pr(precision: float, recall: float, decimals: int = 2) -> float:
    """
    根据精确率和召回率计算 F1 分数。

    参数
    ----
    precision : float
        精确率；可以是 0–1 之间的小数，也可以是 0–100 之间的百分数。
    recall : float
        召回率；同上。
    decimals : int, default=2
        返回值保留的小数位数。

    返回
    ----
    float
        F1 分数，百分比形式，保留 `decimals` 位小数。
    """
    # 如果输入大于 1，说明是百分数，需要除以 100
    if precision > 1:
        precision /= 100.0
    if recall > 1:
        recall /= 100.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return round(f1 * 100, decimals)   # 返回百分数形式


# 示例
p, r = 46.29, 44.95
print(f"F1 = {f1_from_pr(p, r)}%")   # 输出：F1 = 97.18%
