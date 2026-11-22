import numpy as np

from .transforms import MODELS, compute_error


def ransac(matches, iters=1000, threshold=3.0, mode="homography", seed=None):
    # 根據指定變換取得對應的函數及所需最少配對點數量
    fit, sample_size = MODELS[mode]
    rng = np.random.default_rng(seed)
    n = len(matches)
    best_mask, best_transform, best_err = np.zeros(n, dtype=bool), None, np.inf

    # 若總配對數連最少需求配對點都不到，直接回傳
    if n < sample_size:
        return best_transform, best_mask


    for i in range(iters):
        # 1. 隨機抽取計算變換矩陣的配對樣本，不要重複
        index = rng.choice(n, size=sample_size, replace=False)
        T_ran = fit(matches[index])

        # 2. 根據轉換矩陣計算每個點各自的重投影誤差，小於 threshold 視為 inlier(內點)
        errors = compute_error(T_ran, matches)
        cur_inliers = errors < threshold

        # 3. 更新目前最高內點數量的轉換矩陣
        if np.sum(cur_inliers) > np.sum(best_mask) or (np.sum(cur_inliers) == np.sum(best_mask) and best_err > np.sum(errors[cur_inliers])):
            best_mask = cur_inliers
            best_transform = T_ran
            best_err = np.sum(errors[cur_inliers])

    # 最後，用歷史最高的變換矩陣內全部的內點再計算一次變換矩陣，降低誤差
    if np.sum(best_mask) >= sample_size:
        best_transform = fit(matches[best_mask])
    else:
        best_transform = None

    return best_transform, best_mask
