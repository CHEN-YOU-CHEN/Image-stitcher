import cv2
import numpy as np


def sift(img):
    # 固定參數
    n_oct = 4        # octave 數量，不同 octvate 表示不同大小圖片，每次縮小成原圖一半
    s = 3            # 每個 octave 中實際拿來找偵測特徵點的有效層數
    sigma = 1.6      # 第一層的高斯模糊量，會等比上升，第 s 層高斯模糊量為 2*sigma
    contrast = 0.04  # 低對比門檻：|DoG| < contrast / s 的點剔除
    edge_r = 10.0    # 邊緣門檻：主曲率比值 > edge_r 的點剔除

    # img 轉為灰階 (H, W)，數值正規化 [0.0, 1.0]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    base = gray.astype(np.float32) / 255.0

    # 假設相機拍攝原圖已有 0.5 的模糊
    base = cv2.GaussianBlur(base, (0, 0), np.sqrt(sigma ** 2 - 0.5 ** 2))   

    # 每一層相對上一層要再補的模糊量
    k = 2 ** (1 / s)

    # 計算模糊程度增量: (依變異數可加性 σ_目標² = σ_上層² + σ_增量²)
    inc = [np.sqrt((sigma * k ** i) ** 2 - (sigma * k ** (i - 1)) ** 2) for i in range(1, s + 3)]
    gauss, DoG = [], []

    for _ in range(n_oct):
        if min(base.shape) < 32:                        
            break

        octave = [base]
        for sg in inc:
            octave.append(cv2.GaussianBlur(octave[-1], (0, 0), sg))

        octave = np.stack(octave)                        # list 轉為 numpy 形狀 (s+3, H, W)
        gauss.append(octave)
        DoG.append(octave[1::1] - octave[0:-1:1])             # (s+2, H, W)
        base = octave[s][::2, ::2]                       # 模糊量 2σ 的那層，downsample 成下一個 octave

    # ptd : 存放特徵點 (x, y),  descs : 128 維特徵描述子向量
    pts, descs = [], []

    for o, (G, D) in enumerate(zip(gauss, DoG)):
        # L (Levels)：DoG 的總層數
        L, H, W = D.shape

        # 每一層的梯度大小與方向
        gx = np.zeros_like(G)
        gy = np.zeros_like(G)

        gx[:, :, 1:-1] = G[:, :, 2:] - G[:, :, :-2]
        gy[:, 1:-1, :] = G[:, 2:, :] - G[:, :-2, :]

        # 計算梯度強度 = sqrt((gx)^2 + (gx)^2), 角度 = arctan(gy/gx)
        MAG = np.hypot(gx, gy)
        ANG = np.degrees(np.arctan2(gy, gx)) % 360

        # 與上層 9 個, 下層 9 個, 周圍 8 個，共 26 個鄰居比較是否為極值
        c = D[1:-1, 1:-1, 1:-1]         # 不看第一層及最後一層，以及圖片周圍一圈 1 px
        mx = np.full_like(c, -np.inf)   # 記錄每個位置周圍 26 個鄰居的最大值
        mn = np.full_like(c, np.inf)    # 記錄每個位置周圍 26 個鄰居的最小值

        # 看過周圍鄰居，共 26 個方向
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == dy == dx == 0:
                        continue
                    n = D[1 + dz:L - 1 + dz, 1 + dy:H - 1 + dy, 1 + dx:W - 1 + dx]
                    np.maximum(mx, n, out=mx)
                    np.minimum(mn, n, out=mn)

        # 篩選候選極值點：必須嚴格大於或小於 26 個鄰居，且先以正式對比門檻的一半 (0.5 * contrast / s) 粗篩掉平坦雜訊       
        cand = ((c > mx) | (c < mn)) & (np.abs(c) > 0.5 * contrast / s)


        for z, y, x in zip(*np.nonzero(cand)):
            # 找到特徵候選點並還原座標
            z, y, x = int(z) + 1, int(y) + 1, int(x) + 1
        
            converged = False
            # 利用 3D 二階泰勒展開式逼近 DoG 函數，透過牛頓法迭代尋找真正的連續極值點

            for _ in range(5):  # 最多迭代 5 次

                # 確保點沒有跑到影像邊界外 (留 5 像素安全距離)
                if not (1 <= z < L - 1 and 5 <= y < H - 5 and 5 <= x < W - 5):
                    break

                v = D[z, y, x]

                # 計算一階偏導數向量 g = [dD/dx, dD/dy, dD/ds]
                g = 0.5 * np.array([D[z, y, x + 1] - D[z, y, x - 1],
                                    D[z, y + 1, x] - D[z, y - 1, x],
                                    D[z + 1, y, x] - D[z - 1, y, x]])

                # 計算二階偏導數，組成 3x3 Hessian 矩陣
                dxx = D[z, y, x + 1] - 2 * v + D[z, y, x - 1]
                dyy = D[z, y + 1, x] - 2 * v + D[z, y - 1, x]
                dss = D[z + 1, y, x] - 2 * v + D[z - 1, y, x]
                dxy = 0.25 * (D[z, y + 1, x + 1] - D[z, y + 1, x - 1] - D[z, y - 1, x + 1] + D[z, y - 1, x - 1])
                dxs = 0.25 * (D[z + 1, y, x + 1] - D[z + 1, y, x - 1] - D[z - 1, y, x + 1] + D[z - 1, y, x - 1])
                dys = 0.25 * (D[z + 1, y + 1, x] - D[z + 1, y - 1, x] - D[z - 1, y + 1, x] + D[z - 1, y - 1, x])
                hess = np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
                
                try:
                    # 令一階導數為 0，解線性方程組 H * off = -g，求得次像素偏移量 off = [dx, dy, ds]
                    off = -np.linalg.solve(hess, g)      # 泰勒展開求極值位置的偏移量
                except np.linalg.LinAlgError:
                    break

                # 若三個維度的偏移量都小於 0.5，代表極值點就落在目前像素內，收斂完成
                if np.all(np.abs(off) < 0.5):
                    converged = True
                    break

                # 否則將整數座標移動到更接近極值點的相鄰像素，繼續迭代
                x, y, z = x + int(round(off[0])), y + int(round(off[1])), z + int(round(off[2]))
            if not converged:
                continue

            # 剔除低對比點
            if abs(v + 0.5 * g @ off) < contrast / s:    # 對比太低，容易受雜訊影響
                continue

            # 剔除掉邊緣上的點
            tr, det = dxx + dyy, dxx * dyy - dxy ** 2
            if det <= 0 or tr ** 2 / det >= (edge_r + 1) ** 2 / edge_r:   # 在邊緣上，沿邊緣滑動看起來都一樣
                continue

            # 將次像素座標乘回 2^o，還原成原始輸入影像的座標系
            ox, oy = (x + off[0]) * 2 ** o, (y + off[1]) * 2 ** o           # 換回原圖座標
            sig = sigma * 2 ** ((z + off[2]) / s)        # 此特徵點在 octave 內的尺度
            mag, ang = MAG[z], ANG[z]

            # 主方向：36 格梯度方向直方圖，# 採樣半徑設為 3 * (1.5 * sig) 的高斯視窗
            r = int(round(3 * 1.5 * sig))
            y0, y1, x0, x1 = max(y - r, 0), min(y + r + 1, H), max(x - r, 0), min(x + r + 1, W)
            yy, xx = np.mgrid[y0:y1, x0:x1]

            # 計算高斯距離權重：越靠近特徵點中心的像素，對主方向投票權重越高
            w = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * (1.5 * sig) ** 2))

            # 將 0~360 度切成 36 個 bin (每格 10 度)，以「梯度大小 * 高斯權重」累計投票
            hist = np.bincount((ang[y0:y1, x0:x1] // 10).astype(int).ravel() % 36,
                               (mag[y0:y1, x0:x1] * w).ravel(), minlength=36)
            
            hist = np.convolve(np.r_[hist[-2:], hist, hist[:2]], [1, 4, 6, 4, 1], "valid") / 16   # 平滑
            thetas = []
            for i in np.nonzero(hist >= 0.8 * hist.max())[0]:   # 達最高峰 80% 的成為一個特徵點
                lft, mid, rgt = hist[i - 1], hist[i], hist[(i + 1) % 36]
                if mid > lft and mid > rgt:                      # 局部峰值，用拋物線內插求精確角度
                    thetas.append(((i + 0.5 * (lft - rgt) / (lft - 2 * mid + rgt)) * 10 + 5) % 360)

            #描述子：4x4 cell x 8 方向 = 128 維
            hw = 3 * sig                                 # 每個 cell 的寬度
            r = int(round(hw * np.sqrt(2) * 5 / 2))
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            for theta in thetas:
                cos, sin = np.cos(np.radians(theta)), np.sin(np.radians(theta))
                # 旋轉到主方向的座標系，達成旋轉不變
                rx = (cos * xx + sin * yy) / hw + 1.5
                ry = (-sin * xx + cos * yy) / hw + 1.5
                px, py = xx + x, yy + y
                ok = ((rx > -0.5) & (rx < 3.5) & (ry > -0.5) & (ry < 3.5)
                      & (px > 0) & (px < W - 1) & (py > 0) & (py < H - 1))
                rx_, ry_, px_, py_ = rx[ok], ry[ok], px[ok], py[ok]
                m = mag[py_, px_] * np.exp(-((rx_ - 1.5) ** 2 + (ry_ - 1.5) ** 2) / 8)
                ob = ((ang[py_, px_] - theta) % 360) / 45    # 相對主方向的角度，分成 8 個 bin
               
                # 三線性內插：把每個像素的權重分到相鄰的 (row, col, 方向) bin，避免描述子跳變
                desc = np.zeros((6, 6, 8))
                r0, c0, o0 = np.floor(ry_).astype(int), np.floor(rx_).astype(int), np.floor(ob).astype(int)
                fr, fc, fo = ry_ - r0, rx_ - c0, ob - o0
                for dr in (0, 1):
                    for dc in (0, 1):
                        for do in (0, 1):
                            wt = m * (fr if dr else 1 - fr) * (fc if dc else 1 - fc) * (fo if do else 1 - fo)
                            np.add.at(desc, (r0 + dr + 1, c0 + dc + 1, (o0 + do) % 8), wt)
                vec = desc[1:5, 1:5].ravel()
                vec /= np.linalg.norm(vec) + 1e-7
                vec = np.minimum(vec, 0.2)               # 壓掉過大的值，對光照變化更穩
                vec /= np.linalg.norm(vec) + 1e-7
                descs.append(np.minimum(vec * 512, 255))  # 跟 OpenCV 一樣縮放到 0~255
                pts.append((ox, oy))

    return (np.array(pts, np.float64).reshape(-1, 2),
            np.array(descs, np.float32).reshape(-1, 128))


def match_images(img1, img2, ratio=0.75, mask1=None, mask2=None):
    p1, d1 = sift(img1)
    p2, d2 = sift(img2)

    # 用 mask 剔除落在有效區域外的特徵點
    if mask1 is not None:
        keep1 = mask1.astype(bool)[np.clip(np.round(p1[:, 1]).astype(int), 0, mask1.shape[0] - 1),
                                   np.clip(np.round(p1[:, 0]).astype(int), 0, mask1.shape[1] - 1)]
        p1, d1 = p1[keep1], d1[keep1]
    if mask2 is not None:
        keep2 = mask2.astype(bool)[np.clip(np.round(p2[:, 1]).astype(int), 0, mask2.shape[0] - 1),
                                   np.clip(np.round(p2[:, 0]).astype(int), 0, mask2.shape[1] - 1)]
        p2, d2 = p2[keep2], d2[keep2]

     # 少於 2 個點無法做比值檢定
    if len(d1) == 0 or len(d2) < 2:                     
        return np.empty((0, 4), np.float64)

    # 相似程度：描述子之間的歐氏距離 (N1, N2)，用 |a-b|^2 = |a|^2 + |b|^2 - 2a·b 一次算完
    d1, d2 = d1.astype(np.float64), d2.astype(np.float64)
    sq = (d1 ** 2).sum(1)[:, None] + (d2 ** 2).sum(1)[None, :] - 2 * d1 @ d2.T
    dist = np.sqrt(np.maximum(sq, 0))                    # 浮點誤差可能出現極小負數

    # 最相似的兩個特徵點要比第二相似的配對比值小於 ratio，才當作一組配對
    sorted_idx = np.argsort(dist, axis=1)

    idx_pairs = []
    for i in range(dist.shape[0]):
        if dist[i][sorted_idx[i][0]] / dist[i][sorted_idx[i][1]] < ratio:
            idx_pairs.append([i, sorted_idx[i][0]])

    if len(idx_pairs) > 0:
        idx_pairs = np.array(idx_pairs)
    else:
        idx_pairs = np.empty((0, 2), dtype=int)

    # 索引配對 -> 座標 [x1, y1, x2, y2]
    return np.hstack([p1[idx_pairs[:, 0]], p2[idx_pairs[:, 1]]])
