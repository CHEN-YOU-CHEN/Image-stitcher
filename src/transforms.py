import numpy as np

# 估計平移矩陣 (2 個自由度)：為 x2 = x1 + tx, y2 = y1 + ty
# matches 形狀為 (N, 4)，每列為 [x1, y1, x2, y2]；最小平方法的解剛好就是所有位移量的平均值
def fit_translation(matches: np.ndarray) -> np.ndarray:
    dx = matches[:, 2] - matches[:, 0]
    dy = matches[:, 3] - matches[:, 1]
    
    tx = np.mean(dx)
    ty = np.mean(dy)
    # 組成 3x3 齊次座標平移矩陣
    translation_matrix = np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0]
    ])

    return translation_matrix

# 估計 affine 變換矩陣 (6 個自由度：旋轉、縮放、錯切、平移)
# 模型：x2 = a*x1 + b*y1 + tx, y2 = c*x1 + d*y1 + ty，改寫成線性方程組 A @ W = b 用最小平方法求解
def fit_affine(matches: np.ndarray) -> np.ndarray:
   
    N = matches.shape[0]
    # 每組配對點提供 x、y 兩條方程式，共 2N 條方程式解 6 個未知數 W = [a, b, c, d, tx, ty]
    A = np.zeros((2*N, 6))
    b = np.zeros((2*N))

    x1 = matches[:, 0]
    y1 = matches[:, 1]
    x2 = matches[:, 2]
    y2 = matches[:, 3]
    # 偶數列 (0, 2, 4...) 放 x 方向方程式：a*x1 + b*y1 + 1*tx = x2
    A[0::2, 0] = x1
    A[0::2, 1] = y1
    A[0::2, 4] = 1.0
    b[0::2] = x2

    # 奇數列 (1, 3, 5...) 放 y 方向方程式：c*x1 + d*y1 + 1*ty = y2
    A[1::2, 2] = x1
    A[1::2, 3] = y1
    A[1::2, 5] = 1.0
    b[1::2] = y2
    # 求解最小平方解 W (使 ||A @ W - b||^2 最小)
    W, residuals, rank, s = np.linalg.lstsq(A, b, rcond = None)

    affine_matrix = np.array([
       [W[0], W[1], W[4]],
       [W[2], W[3], W[5]],
       [0, 0, 1.0]
    ])

    return affine_matrix


def normalization(pts: np.ndarray) -> np.ndarray:
    # Hartley 正規化：把重心移到原點，並讓點到原點的平均距離為 sqrt(2)
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2) / d if d > 0 else 1.0
    return np.array([[s, 0, -s * c[0]],
                     [0, s, -s * c[1]],
                     [0, 0, 1.0]])

# 估計 homography 變換矩陣 (8 個自由度，至少需要 4 組不共線的配對點)
# 使用 DLT (Direct Linear Transform) 演算法，將透視投影公式改寫為齊次方程組 A @ h = 0 並以 SVD 求解
def fit_homography(matches: np.ndarray, normalize: bool = True) -> np.ndarray:
    N = matches.shape[0]

    p1 = matches[:, 0:2]
    p2 = matches[:, 2:4]
    if normalize:
        T1 = normalization(p1)
        T2 = normalization(p2)
        p1 = apply_transform(T1, p1)
        p2 = apply_transform(T2, p2)

    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]

    # 每組配對點提供 x、y 兩條方程式，共 2N 條方程式解 9 個矩陣元素 h = [h1, h2, ..., h9]^T
    A = np.zeros((2*N, 9))

    # 偶數列 (0, 2, 4...) 放 x 方向方程式：h1*x1 + h2*y1 + h3 - h7*x1*x2 - h8*y1*x2 - h9*x2 = 0
    A[0::2, 0] = x1
    A[0::2, 1] = y1
    A[0::2, 2] = 1.0
    A[0::2, 6] = (-1)*x1*x2
    A[0::2, 7] = (-1)*y1*x2
    A[0::2, 8] = -x2

    # 奇數列 (1, 3, 5...) 放 y 方向方程式：h4*x1 + h5*y1 + h6 - h7*x1*y2 - h8*y1*y2 - h9*y2 = 0
    A[1::2, 3] = x1
    A[1::2, 4] = y1
    A[1::2, 5] = 1.0
    A[1::2, 6] = (-1)*x1*y2
    A[1::2, 7] = (-1)*y1*y2
    A[1::2, 8] = -y2

    # 對 A 做奇異值分解 (SVD)：在 ||h|| = 1 的限制下，最小平方解為最小奇異值對應的右奇異向量 (Vh 最後一列)
    U, S, Vh = np.linalg.svd(A)
    H_vector = Vh[-1, :]

    homography_matrix = np.array([
        [H_vector[0], H_vector[1], H_vector[2]],
        [H_vector[3], H_vector[4], H_vector[5]],
        [H_vector[6], H_vector[7], H_vector[8]]
    ])

    # 反正規化：H 是在正規化座標下求得的 (p2' = H p1')，換回原始座標 H = T2^-1 @ H @ T1
    if normalize:
        homography_matrix = np.linalg.inv(T2) @ homography_matrix @ T1

    # 正規化:除以右下角元素讓它變成 1
    if homography_matrix[2, 2] != 0:
        homography_matrix = homography_matrix / homography_matrix[2, 2]

    return homography_matrix


def apply_transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    # 把 3x3 轉換矩陣套用到 (N, 2) 的點上，包含除以最後一個齊次座標
    ph = np.hstack([pts, np.ones((len(pts), 1))]) @ T.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return ph[:, :2] / ph[:, 2:3]

# 以歐式距離計算重投影誤差，
def compute_error(T: np.ndarray, matches: np.ndarray) -> np.ndarray:
    N = matches.shape[0]

    pts1 = matches[:, 0:2]
    pts2_proj = apply_transform(T, pts1)

    proj_x = pts2_proj[:, 0]
    proj_y = pts2_proj[:, 1]

    true_x = matches[:, 2]
    true_y = matches[:, 3]

    errors = np.sqrt((proj_x - true_x)**2 + (proj_y - true_y)**2)
    # 如果 Homography 投影時分母為 0 產生 NaN，改設為 inf
    errors[np.isnan(errors)] = np.inf

    return errors


# 模型名稱 -> (擬合函式, 最少抽樣數)
MODELS = {
    "translation": (fit_translation, 1),
    "affine": (fit_affine, 3),
    "homography": (fit_homography, 4),
}
