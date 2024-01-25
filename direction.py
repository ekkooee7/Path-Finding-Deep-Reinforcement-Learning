import numpy as np

# 定义方向单位向量
direction_vectors = {
    'E': np.array([1, 0]),
    'NE': np.array([np.sqrt(2)/2, np.sqrt(2)/2]),
    'N': np.array([0, 1]),
    'NW': np.array([-np.sqrt(2)/2, np.sqrt(2)/2]),
    'W': np.array([-1, 0]),
    'SW': np.array([-np.sqrt(2)/2, -np.sqrt(2)/2]),
    'S': np.array([0, -1]),
    'SE': np.array([np.sqrt(2)/2, -np.sqrt(2)/2])
}


def get_direction(vector):
    # 计算与各方向向量的点积
    dot_products = {name: np.dot(vec, vector) for name, vec in direction_vectors.items()}

    # 找出点积最大的方向
    max_direction = max(dot_products, key=dot_products.get)

    # 创建八维方向数组，所属方位为1，其它为0
    direction_array = np.zeros(8)
    direction_names = list(direction_vectors.keys())
    direction_array[direction_names.index(max_direction)] = 1

    return direction_array

direction_array = get_direction([1,6])
# 输出方向数组
print(direction_array)