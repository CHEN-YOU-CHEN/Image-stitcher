import argparse
from pathlib import Path

import cv2
import numpy as np

from src import FOCAL, ensure_frames
from src import match_images
from src import ransac
from src import build_cylindrical_panorama, chain_transforms, crop_panorama, stitch_multiple, stitch_pair
from src.transforms import apply_transform

# 整體流程：
#   0. 準備資料：
#      - 沒有給 --input：使用測試資料，data/frames 缺少影像時自動下載 skybox 並渲染 32 張相機繞垂直軸旋轉拍攝的影像
#      - 有給 --input：讀取自己的影像 (資料夾或檔案清單，依檔名排序，需由左到右依序拍攝)，過大的影像先縮小
#   1. 兩張影像拼接：SIFT + 比值檢定找配對 -> RANSAC 估計 homography -> 變形到同一畫布 -> 融合
#   2. 多張影像拼接 (平面)：相鄰影像兩兩估計 homography -> 串接成對應到中間那張的轉換 -> 一次融合
#   3. 圓柱全景：先把每張影像投影到圓柱上，相鄰影像就近似只差平移，可以拼接很大的視角

# 自己的影像可接受的副檔名
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# 平面多張拼接的畫布寬度上限 (單張寬度的倍數)，超過代表視角太大、homography 兩側被拉得太長
MAX_CANVAS_RATIO = 6


# 讀取第 idx 張測試影像（<data_dir>/00.jpg ... 31.jpg），格式為 OpenCV 原生的 uint8 BGR
def load_frame(data_dir, idx):
    # 檔名固定兩位數，例如 idx = 3 -> 03.jpg
    return load_image(Path(data_dir) / f"{idx:02d}.jpg")


# 讀取單張影像，max_size > 0 時把長邊縮小到 max_size 以內
def load_image(path, max_size=0):
    # cv2.imread 不支援中文路徑，所以先用 numpy 讀成位元組再解碼
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    # 讀不到或格式錯誤時不會報錯，只會默默回傳 None，所以要自己檢查
    if img is None:
        raise FileNotFoundError(f"cannot read image: {path}")

    # 手機拍的照片動輒 4000 px，SIFT 與配對的計算量會大幅增加，先等比例縮小
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    if 0 < scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img


# 讀取 --input 指定的自己的影像：可以是一個資料夾 (取裡面所有影像) 或多個影像檔
# 依檔名排序，所以檔名要照拍攝順序 (由左到右) 命名，例如 01.jpg、02.jpg ...
def load_inputs(inputs, max_size):
    paths = []
    for p in map(Path, inputs):
        if p.is_dir():
            paths += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        else:
            paths.append(p)

    if len(paths) < 2:
        raise SystemExit("need at least two input images")
    print(f"loaded {len(paths)} images: {[p.name for p in paths]}")
    return [load_image(p, max_size) for p in paths]


# 把拼接結果（BGR）存檔，資料夾不存在時自動建立
def save_image(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    # 融合後的結果是 float32（數值 0～255），但圖檔只能存 uint8，所以要先 clip、四捨五入再轉回
    # （直接把 float 交給 cv2.imwrite，存 PNG 會失敗或存成幾乎全黑的圖）
    img = np.clip(img, 0, 255).round().astype(np.uint8)
    cv2.imwrite(str(path), img)


# 估計把 src 對應到 dst 的轉換矩陣：SIFT + 比值檢定找候選配對，再用 RANSAC 剔除錯誤配對
def align(src, dst, mode, args):
    matches = match_images(src, dst, args.ratio)
    T, inliers = ransac(matches, args.iters, args.threshold, mode, seed=0)
    # 配對點太少或 RANSAC 找不到模型時對齊失敗 (例如兩張重疊太少，或重疊區都是天空、牆面等沒有紋理的地方)
    if T is None:
        raise RuntimeError(f"alignment failed ({len(matches)} putative matches)")
    print(f"  {inliers.sum()} inliers / {len(matches)} putative matches")
    return T


# 1. 兩張影像拼接：左圖變形到右圖的座標系，分別用 average、feather 兩種方式融合
def run_pair(left, right, tag, args, out):
    print(f"[1] pair stitching: {tag}")

    # 相機只做旋轉時，兩個視角之間剛好是一個 homography (H = K R K^-1)
    T = align(left, right, "homography", args)
    print("  T =\n", np.array2string(T, precision=5, suppress_small=True))

    # 比較兩種融合方式：average 在重疊區邊界會有明顯接縫，feather 會平滑過渡
    for blend in ("average", "feather"):
        mosaic, _, _ = stitch_pair(left, right, T, blend)
        save_image(out / f"pair_{tag}_{blend}.png", mosaic)


# 2. 多張影像拼接 (平面)：所有影像都變形到中間那張的平面上
def run_multiple(imgs, args, out):
    print(f"[2] multi-image stitching: {len(imgs)} images")

    # 2-1. 相鄰影像兩兩估計 homography：pairwise[k] 把第 k+1 張對應到第 k 張
    pairwise = [align(imgs[k + 1], imgs[k], "homography", args) for k in range(len(imgs) - 1)]

    # 2-2. 串接成每張影像到第 0 張的轉換，再左乘第 0 張到中間那張的轉換 (inv(A_mid))
    #      以中間的影像為基準，左右兩側的變形才會對稱，不會一路越拉越大
    acc = chain_transforms(pairwise)
    mid = len(imgs) // 2
    to_mid = np.linalg.inv(acc[mid])
    transforms = [to_mid @ A for A in acc]

    # 2-3. 先檢查畫布大小：總視角接近 180 度時平面投影兩側會被無限拉長，畫布會大到無法配置記憶體
    corners = np.vstack([apply_transform(T, np.array([[0, 0], [w, 0], [0, h], [w, h]], float))
                         for T, (h, w) in zip(transforms, (im.shape[:2] for im in imgs))])
    width = np.ptp(corners[:, 0])
    if not np.isfinite(width) or width > MAX_CANVAS_RATIO * imgs[mid].shape[1]:
        print("  skipped: field of view too wide for a planar mosaic, see the cylindrical panorama instead")
        return

    # 2-4. 一次把所有影像以 feather blend 畫到畫布上 (平面影像的像素全部有效，所以 masks 傳 None)
    mosaic, _ = stitch_multiple(imgs, None, transforms)
    save_image(out / "multiple.png", mosaic)


# 3. 圓柱全景：平面投影在視角變大時兩側會被無限拉長，改投影到圓柱上就能拼接更大的視角
def run_panorama(imgs, focal, args, out):
    print(f"[3] cylindrical panorama: {len(imgs)} images, focal = {focal:.1f} px")

    # 圓柱投影 -> 相鄰影像對齊 (近似只差平移加上微小旋轉，用 affine 就夠) -> 串接轉換矩陣 -> 融合
    pano, _ = build_cylindrical_panorama(imgs, focal, mode="affine", ratio=args.ratio,
                                         threshold=args.threshold, iters=args.iters)

    # 存檔：完整全景圖，以及裁掉上下左右黑邊的版本
    save_image(out / "panorama.png", pano)
    save_image(out / "panorama_cropped.png", crop_panorama(pano))
    print(f"  panorama {pano.shape[1]}x{pano.shape[0]}")


# 解析參數、準備資料，依序執行三種拼接並把結果存到 --out
def main():
    p = argparse.ArgumentParser(description="Feature-based image stitching")
    p.add_argument("--input", nargs="+", help="your own images: a folder or image files, ordered left to right")
    p.add_argument("--max-size", type=int, default=1000)                                # 自己的影像長邊縮到多少 px (0 = 不縮)
    p.add_argument("--data", default="data/frames", help="folder with 00.jpg ... 31.jpg")
    p.add_argument("--out", default="results")
    p.add_argument("--pair-ids", type=int, nargs=2, default=[1, 3])                     # 兩張拼接用的影像
    p.add_argument("--multi-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])         # 多張拼接用的影像
    p.add_argument("--pano-ids", type=int, nargs="+", default=[0, 2, 4, 6, 8, 10])      # 圓柱全景用的影像
    p.add_argument("--ratio", type=float, default=0.75)                                 # 比值檢定門檻
    p.add_argument("--threshold", type=float, default=3.0)                              # RANSAC inlier 的誤差門檻 (px)
    p.add_argument("--iters", type=int, default=1000)                                   # RANSAC 迭代次數
    p.add_argument("--focal", type=float, default=None)                                 # 圓柱投影用的焦距 (px)
    p.add_argument("--skybox", default="data/skybox", help="skybox folder (auto-downloaded)")
    args = p.parse_args()
    out = Path(args.out)

    # 要執行的步驟：(函式, 參數)
    # 自己的影像：2 張做兩張拼接，3 張以上做多張拼接，最後都做圓柱全景
    if args.input:
        imgs = load_inputs(args.input, args.max_size)
        # 不知道相機焦距時，以影像寬度當作焦距 (約等於 53 度的水平視角，一般相機大致落在這附近)
        focal = args.focal or imgs[0].shape[1]
        if len(imgs) == 2:
            steps = [(run_pair, (imgs[0], imgs[1], "input", args, out))]
        else:
            steps = [(run_multiple, (imgs, args, out))]
        steps.append((run_panorama, (imgs, focal, args, out)))

    # 測試資料：缺少影像時自動下載 skybox 並渲染，再依序執行三種拼接
    else:
        if len(args.multi_ids) < 2 or len(args.pano_ids) < 2:
            p.error("--multi-ids and --pano-ids need at least two frames")
        ensure_frames(args.data, args.skybox)
        focal = args.focal or FOCAL

        tag = f"{args.pair_ids[0]:02d}_{args.pair_ids[1]:02d}"
        steps = [
            (run_pair, (*(load_frame(args.data, i) for i in args.pair_ids), tag, args, out)),
            (run_multiple, ([load_frame(args.data, i) for i in args.multi_ids], args, out)),
            (run_panorama, ([load_frame(args.data, i) for i in args.pano_ids], focal, args, out)),
        ]

    # 依序執行，某一步對齊失敗時只跳過那一步，其他步驟照樣執行
    for fn, fn_args in steps:
        try:
            fn(*fn_args)
        except RuntimeError as e:
            print(f"  skipped: {e}")

    print(f"results written to {out}/")


if __name__ == "__main__":
    main()
