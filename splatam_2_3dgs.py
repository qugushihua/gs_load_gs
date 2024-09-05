import numpy as np
from plyfile import PlyData, PlyElement

data = np.load('params.npz')

# 查看文件中包含的数组名
print(data.files)

# # 读取指定数组
means3D = data['means3D']
rgb_colors = data['rgb_colors']
unnorm_rotations = data['unnorm_rotations']
logit_opacities = data['logit_opacities']
log_scales = data['log_scales']
cam_unnorm_rots = data['cam_unnorm_rots']
cam_trans = data['cam_trans']
timestep = data['timestep']
intrinsics = data['intrinsics']
w2c = data['w2c']
org_width = data['org_width']
org_height = data['org_height']
gt_w2c_all_frames = data['gt_w2c_all_frames']
keyframe_time_indices = data['keyframe_time_indices']

import numpy as np

print("rgb_colors",rgb_colors)

from plyfile import PlyData, PlyElement

def generate_random_point_cloud(num_points):
    colors = rgb_colors
    xyz = means3D
    opacities = logit_opacities
    scales = log_scales
    rots = unnorm_rotations
    return xyz, opacities, colors, scales, rots

def save_random_point_cloud(path, num_points):
    xyz, opacities, colors, scales, rots = generate_random_point_cloud(num_points)

    # 创建点云数据
    dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')] + \
                 [('red', 'f4'), ('green', 'f4'), ('blue', 'f4')] + \
                 [('opacity', 'f4')] + \
                 [(f'scale_{i}', 'f4') for i in range(scales.shape[1])] + \
                 [(f'rot_{i}', 'f4') for i in range(rots.shape[1])]

    elements = np.empty(num_points, dtype=dtype_full)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['red'] = colors[:, 0]
    elements['green'] = colors[:, 1]
    elements['blue'] = colors[:, 2]
    elements['opacity'] = opacities[:, 0]

    for idx in range(scales.shape[1]):
        elements[f'scale_{idx}'] = scales[:, idx]

    for idx in range(rots.shape[1]):
        elements[f'rot_{idx}'] = rots[:, idx]

    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

# 示例：生成一个包含 100 个点的点云文件
save_random_point_cloud('points3D.ply', num_points=means3D.shape[0])