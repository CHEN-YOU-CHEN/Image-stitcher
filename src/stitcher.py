import cv2
import numpy as np

from .features import match_images
from .ransac import ransac
from .transforms import apply_transform



def compute_canvas(shape_l, shape_r, T):
    h_l, w_l = shape_l[:2]
    h_r, w_r = shape_r[:2]

    # 左圖，右圖的四個角落
    corners_l = np.array([[0, 0], [w_l, 0], [0, h_l], [w_l, h_l]])
    corners_r = np.array([[0, 0], [w_r, 0], [0, h_r], [w_r, h_r]])

    # 左圖進行變換
    corners_l_warped = apply_transform(T, corners_l)

    # 合併右圖及變換後的左圖共 8 個座標點，找出容納得下的兩張圖的畫布矩形
    all_corners = np.vstack([corners_l_warped, corners_r])

    # 左上:(min_x, min_y)  右下:(max_x, max_y)
    min_x = np.floor(all_corners[:, 0].min())
    min_y = np.floor(all_corners[:, 1].min())
    max_x = np.ceil(all_corners[:, 0].max())
    max_y = np.ceil(all_corners[:, 1].max())

    size =  (int(max_x - min_x), int(max_y - min_y))

    # 所有座標點進行平移，使得 (min_x, min_y) -> (0, 0)
    M = np.array([
      [1, 0, -min_x],
      [0, 1, -min_y],
      [0, 0, 1]
    ])

    return size, M


# 把影像透過變換矩陣 M 投影到大畫布，並產生對應的有效區域布林遮罩
def warp_to_canvas(img, M, size):

    # 1. 將 img 以矩陣 M 變換並貼到尺寸為 size = (寬, 高) 的畫布上 (預設使用雙線性內插，空白處補 0)
    #    先轉成 float32，避免內插結果被截成整數而損失精度
    warped = cv2.warpPerspective(img.astype(np.float32), M, size)

    # 2. 建立與原圖同尺寸的全 1 遮罩，套用相同的變換 (用最近鄰內插，讓遮罩維持 0 / 1)
    ones_mask = np.ones(img.shape[:2], dtype = np.uint8)
    warped_mask = cv2.warpPerspective(ones_mask, M, size, flags=cv2.INTER_NEAREST)

    # 3. 將有效遮罩往內侵蝕 (Erode) 1 px
    kernel = np.ones((3, 3), dtype = np.uint8)
    eroded_mask = cv2.erode(warped_mask, kernel, iterations=1)

    # 4. 轉成布林遮罩 (True 代表有效像素，False 代表背景或已剔除的暗邊)
    mask = eroded_mask.astype(bool)

    return warped, mask


# 把平面影像投影到半徑為 f 的圓柱上，用 cv2.remap 做反向映射
# 圓柱上的像素 (xc, yc)：theta = (xc - cx) / f，來源 x = f*tan(theta) + cx，來源 y = (yc - cy) / cos(theta) + cy
# 回傳 (投影後影像, bool 有效遮罩)，遮罩外的像素為黑色
def cylindrical_projection(img, f):
    h, w = img.shape[:2]
    # 以影像中心為圓柱的中心軸
    cx, cy = (w - 1) / 2, (h - 1) / 2
    xc, yc = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))

    # 對圓柱上每個像素，反推它在原始平面影像上的位置
    theta = (xc - cx) / f
    map_x = f * np.tan(theta) + cx
    map_y = (yc - cy) / np.cos(theta) + cy

    # 只有角度在 ±90 度內、且來源座標落在原圖範圍內的像素才有效
    mask = ((np.abs(theta) < np.pi / 2)
            & (map_x >= 0) & (map_x <= w - 1)
            & (map_y >= 0) & (map_y <= h - 1))
    projected = cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32),
                          interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    projected[~mask] = 0
    return projected, mask




# 避免分母為 0 的極小值
EPS = 1e-7

# 平均融合：重疊區取兩張影像的平均，其他地方取有效的那張，都沒覆蓋到的地方為黑色
def average_blend(warped_l, warped_r, mask_l, mask_r):
    m_l = mask_l[:, :, None].astype(np.float32)
    m_r = mask_r[:, :, None].astype(np.float32)

    blended = (warped_l * m_l + warped_r * m_r) / (m_l + m_r + EPS)

    return blended
    return blended


# feather_blend：alpha = d_l / (d_l + d_r)，I = alpha * I_l + (1 - alpha) * I_r
def feather_blend(warped_l, warped_r, mask_l, mask_r):

    # 計算每個有效像素到自己影像邊界的距離（遮罩外為 0），越靠近影像中心權重越大
    d_l = cv2.distanceTransform(mask_l.astype(np.uint8), cv2.DIST_L2, 3)
    d_r = cv2.distanceTransform(mask_r.astype(np.uint8), cv2.DIST_L2, 3)

    d_l = d_l[:, :, None]
    d_r = d_r[:, :, None]

    blended = (warped_l * d_l + warped_r * d_r) / (d_l + d_r + EPS)

    return blended
    return blended


# blend 方法名稱 -> blend 函式
BLENDERS = {"average": average_blend, "feather": feather_blend}


# 拼接兩張影像，T 把左圖對應到右圖
def stitch_pair(left, right, T, blend="feather"):
    size, M = compute_canvas(left.shape, right.shape, T)
    wl, ml = warp_to_canvas(left, M @ T, size)
    wr, mr = warp_to_canvas(right, M, size)
    return BLENDERS[blend](wl, wr, ml, mr), M @ T, M



def stitch_multiple(images, masks, transforms):
    # masks 為 None 時，視為每張影像的像素全部有效
    if masks is None:
        masks = [None] * len(images)

    all_corners = []

    for img, T in zip(images, transforms):
        h, w = img.shape[:2]
        corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=float)

        warped_corners = apply_transform(T, corners)
        all_corners.append(warped_corners)

    all_corners = np.vstack(all_corners)
    min_x = np.floor(all_corners[:, 0].min())
    min_y = np.floor(all_corners[:, 1].min())
    max_x = np.ceil(all_corners[:, 0].max())
    max_y = np.ceil(all_corners[:, 1].max())

    size = (int(max_x - min_x), int(max_y - min_y))
    M = np.array([
        [1.0, 0.0, -min_x],
        [0.0, 1.0, -min_y],
        [0.0, 0.0,  1.0 ]
    ])

    sum_img = np.zeros((size[1], size[0], images[0].shape[2]), dtype=np.float32)
    sum_dist = np.zeros((size[1], size[0], 1), dtype=np.float32)

    for img, mask, T in zip(images, masks, transforms):
        M_i = M @ T

        warped_img, valid_mask = warp_to_canvas(img, M_i, size)

        # 若有額外遮罩 (例如圓柱投影的有效區)，也一起變形後取交集
        final_mask = valid_mask
        if mask is not None:
            warped_input_mask = cv2.warpPerspective(mask.astype(np.uint8), M_i, size, flags=cv2.INTER_NEAREST)
            final_mask = valid_mask & warped_input_mask.astype(bool)

        dist = cv2.distanceTransform(final_mask.astype(np.uint8), cv2.DIST_L2, 3)
        dist = dist[..., None]

        sum_img += warped_img * dist
        sum_dist += dist

    panorama = sum_img / (sum_dist + EPS)

    return panorama, M

# 串接相鄰影像間的轉換矩陣：pairwise[k] 把第 k+1 張對應到第 k 張
# 回傳 A_i，把第 i 張對應到第 0 張（A_0 = I，A_i = A_{i-1} @ pairwise[i-1]）
def chain_transforms(pairwise):
    acc = [np.eye(3)]
    for M in pairwise:
        acc.append(acc[-1] @ M)
    return acc

# 建立圓柱全景圖：把每張影像投影到圓柱上，對齊相鄰的影像，串接轉換矩陣後繪製全景
# 回傳 (全景圖, 累積的轉換矩陣)
def build_cylindrical_panorama(images, focal, mode="affine", ratio=0.75, threshold=3.0, iters=1000,
                               border_erode=15, seed=0):
    # 1. 圓柱投影，取得投影後的影像與有效遮罩
    cyl = [cylindrical_projection(im, focal) for im in images]
    cyl_imgs = [p for p, _ in cyl]
    cyl_masks = [m for _, m in cyl]

    # 把遮罩往內侵蝕，避免在投影邊緣偵測到不穩定的特徵點
    kernel = np.ones((border_erode, border_erode), np.uint8)
    det_masks = [cv2.erode(m.astype(np.uint8), kernel).astype(bool) for m in cyl_masks]

    # 2. 估計每一對相鄰影像的轉換矩陣 (第 i 張 -> 第 i-1 張)：SIFT + 比值檢定 -> RANSAC
    pairwise = []
    for i in range(1, len(cyl_imgs)):
        matches = match_images(cyl_imgs[i], cyl_imgs[i - 1], ratio, det_masks[i], det_masks[i - 1])
        T, _ = ransac(matches, iters, threshold, mode, seed)
        if T is None:
            raise RuntimeError(f"alignment of frame {i} -> {i - 1} failed "
                               f"({len(matches)} putative matches)")
        pairwise.append(T)

    # 3. 串接成每張影像到第 0 張的轉換矩陣，再一次融合所有影像
    accumulated = chain_transforms(pairwise)
    pano, _ = stitch_multiple(cyl_imgs, cyl_masks, accumulated)
    return pano, accumulated

# 裁掉黑邊：只保留大部分都有影像覆蓋的列 / 行
# row_cover、col_cover 為該列 / 行有影像覆蓋的比例門檻
def crop_panorama(pano, row_cover=0.9, col_cover=0.5):
    covered = pano.sum(axis=2) > 0
    rows = np.where(covered.mean(axis=1) > row_cover)[0]
    cols = np.where(covered.mean(axis=0) > col_cover)[0]
    # 找不到符合條件的列或行就不裁切
    if len(rows) == 0 or len(cols) == 0:
        return pano
    return pano[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
