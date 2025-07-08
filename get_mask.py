import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder


def get_mask(view_num, data_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data.

        Args:
          view_num: view number
          data_len: number of samples
          missing_rate: Defined in section 4.1 of the paper
        Returns:
          mask

    """
    missing_rate = missing_rate / view_num
    one_rate = 1.0 - missing_rate
    if one_rate <= (1 / view_num):
        # 毒热编码
        enc = OneHotEncoder()
        # 分类编码变量，将每一个类可能取值的特征变换为二进制特征向量，每一类的特征向量只有一个地方是1，其余位置都是0
        view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(data_len, view_num))
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder()
        # shape: (data_len * view_num) 该函数就是分类
        view_preserve = enc.fit_transform(randint(0, view_num, size=(data_len, 1))).toarray()
        # 完整的样本个数
        one_num = view_num * data_len * one_rate - data_len
        ratio = one_num / (view_num * data_len)
        matrix_iter = (randint(0, 100, size=(data_len, view_num)) < int(ratio * 100)).astype(np.int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * data_len)
        matrix_iter = (randint(0, 100, size=(data_len, view_num)) < int(ratio * 100)).astype(np.int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
        # 完整样本的占比
        ratio = np.sum(matrix) / (view_num * data_len)
        error = abs(one_rate - ratio)

    return matrix


def get_mask_fixed_pattern(data_len, view_num=2):
    """
    Generate mask with 30% View A missing, 30% View B missing, and 40% complete samples.

    Returns:
        mask: numpy array of shape [data_len, 2]
    """
    assert view_num == 2, "This fixed mask pattern only supports 2 views."

    # 计算样本数量
    num_A_only = int(data_len * 0.3)
    num_B_only = int(data_len * 0.3)
    num_full = data_len - num_A_only - num_B_only  # 剩下的都完整

    # 构造对应的 mask 样本
    mask_A_only = np.array([[1, 0]] * num_A_only)
    mask_B_only = np.array([[0, 1]] * num_B_only)
    mask_full = np.array([[1, 1]] * num_full)

    # 合并并打乱顺序
    mask = np.vstack([mask_A_only, mask_B_only, mask_full])
    np.random.shuffle(mask)

    return mask

