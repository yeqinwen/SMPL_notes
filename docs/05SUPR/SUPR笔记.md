
 **SUPR: A Sparse Unified Part-Based Human Representation**


我使用的运行环境是dgp_env

安装chumpy
```python
# 安装chumpy
# 下载这个github版本
# https://github.com/mattloper/chumpy
# c:
# cd C:\Users\yeqin\Downloads\chumpy-master\chumpy-master
pip install . --no-build-isolation

```


Foot3D数据集
[OllieBoyne/Foot3D: Dataset of scanned 3D feet](https://github.com/OllieBoyne/Foot3D)


[Scan&Fit – Base Protection World](https://www.baseprotection.com/scanfit/)
使用手机扫描重建脚部模型



## 1 supr生成人体
### 1.1 supr生成人体
```python
from supr.pytorch.supr import SUPR
import numpy as np
import torch
import polyscope as ps

betas = np.zeros((1, 300))
num_betas = 300
batch_size = 1

path_model = 'D:\yeqinwen\PycharmProjects\pytorch3d_code\SUPR-master\model_data\supr_female.npy'

supr = SUPR(path_model, num_betas=num_betas)
supr.display_info()

poses = torch.cuda.FloatTensor(np.zeros((batch_size, supr.num_pose)))
betas = torch.cuda.FloatTensor(np.zeros((batch_size, supr.num_betas)))
trans = torch.cuda.FloatTensor(np.zeros((batch_size, 3)))

model = supr.forward(poses, betas, trans)
v = model[0].cpu().numpy()
f = model.f.astype("int32")

print(v.shape, f.shape)

ps.init()
ps.register_surface_mesh("mesh", v, f,
                         color=np.array([0, 91, 255]) / 255,
                         edge_width=0.003,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )

# ps.set_build_gui(False)
# ps.set_ui_scale(1.5)
ps.set_window_size(1200,800)
ps.set_view_projection_mode("perspective")  # orthographic 正交投影   perspective 透视投影
# ps.set_navigation_style("planar")  # ['turntable','free','planar','none','first_person']
ps.set_ground_plane_mode("shadow_only")  # ['none','tile','tile_reflection','shadow_only']
# ps.set_ground_plane_height(-0.001)  # 设置地平面高度
ps.set_shadow_blur_iters(3)  # 设置地平面阴影模糊程度
ps.set_shadow_darkness(0.5)  # 设置地平面阴影明暗程度
ps.set_up_dir("y_up")  # 设置y轴正方向向上（这个和设置视角会冲突，因此在添加视角参数时，这一行要注释掉）
ps.set_front_dir('z_front')  # 设置z轴正方向向前
ps.set_SSAA_factor(4)  # 在截图时，设置为4时，截图会更清晰
ps.show()

```

打印：
```python
Kinematic Tree: Un-Constrained
Number of Pose Parameters: 225
Number of Joints: 75
Number of Vertices:10475
(10475, 3) (20908, 3)
```

![](assets/Pasted%20image%2020261104094633.png)


### 1.2 打印模型数据参数

```python
import numpy as np

path_model = 'D:\yeqinwen\PycharmProjects\pytorch3d_code\SUPR-master\model_data\supr_female.npy'

# 根据动力学树数据，获得关节连接关系
model_data = np.load(path_model, allow_pickle=True, encoding='latin1')[()]


for key, value in model_data.items():

    try:
        print(key, np.shape(value))
    except:
        print(key, type(value))
```

打印：
```python
f (20908, 3)
J_regressor (225, 31426)
kintree_table (2, 75)
J (75, 3)
bs_style ()
weights (10475, 75)
posedirs (10475, 3, 300)
v_template (10475, 3)
shapedirs (10475, 3, 400)
bs_type ()
```

### 1.3 可视化关节点和骨架 
```python
from supr.pytorch.supr import SUPR
import numpy as np
import torch
import polyscope as ps

betas = np.zeros((1, 300))
num_betas = 300
batch_size = 1

# Set the Path of the model.
path_model = 'D:\yeqinwen\PycharmProjects\pytorch3d_code\SUPR-master\model_data\supr_female.npy'

supr = SUPR(path_model, num_betas=num_betas)
supr.display_info()

# Passing Pose and Shape Parameters
poses = torch.cuda.FloatTensor(np.zeros((batch_size, supr.num_pose)))
betas = torch.cuda.FloatTensor(np.zeros((batch_size, supr.num_betas)))
trans = torch.cuda.FloatTensor(np.zeros((batch_size, 3)))

model = supr.forward(poses, betas, trans)
v = model[0].cpu().numpy()
f = model.f.astype("int32")
Joints = model.J_transformed[0].detach().cpu().numpy()

# 根据动力学树数据，获得关节连接关系
model_data = np.load(path_model, allow_pickle=True, encoding='latin1')[()]
kintree_table = model_data['kintree_table']

parents = kintree_table[0].astype(np.int64)
joint_ids = kintree_table[1].astype(np.int64)

id_to_index = {
    int(joint_id): i
    for i, joint_id in enumerate(joint_ids)
}

edges = []

for i in range(1, len(joint_ids)):
    parent_id = int(parents[i])

    parent_index = id_to_index[parent_id]

    edges.append([parent_index, i])

edges = np.asarray(edges, dtype=np.int32)

print("Skeleton edges:", edges.shape)

print(v.shape, f.shape)

ps.init()
ps.register_surface_mesh("mesh", v, f,
                         color=np.array([0, 91, 255]) / 255,
                         edge_width=0.0003,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         transparency=0.6,
                         # material="flat"
                         )


# 可视化关节点和连接关系
ps.register_point_cloud("SUPR Joints", Joints,
                        color=[1.0, 0.0, 0.0],
                        radius=0.008,
                        enabled=True
                        )

ps.register_curve_network("SUPR Skeleton",
                          Joints,
                          edges,
                          color=[0.0, 1.0, 0.0],
                          radius=0.003,
                          enabled=True
                          )

# ps.set_build_gui(False)
# ps.set_ui_scale(1.5)
ps.set_view_projection_mode("perspective")  # orthographic 正交投影   perspective 透视投影
# ps.set_navigation_style("planar")  # ['turntable','free','planar','none','first_person']
ps.set_ground_plane_mode("shadow_only")  # ['none','tile','tile_reflection','shadow_only']
# ps.set_ground_plane_height(-0.001)  # 设置地平面高度
ps.set_shadow_blur_iters(3)  # 设置地平面阴影模糊程度
ps.set_shadow_darkness(0.5)  # 设置地平面阴影明暗程度
ps.set_up_dir("y_up")  # 设置y轴正方向向上（这个和设置视角会冲突，因此在添加视角参数时，这一行要注释掉）
ps.set_front_dir('z_front')  # 设置z轴正方向向前
ps.set_SSAA_factor(4)  # 在截图时，设置为4时，截图会更清晰
ps.show()

```


![](assets/Pasted%20image%2020261104095605.png)


### 1.4 生成不同姿态人体
```python
from supr.pytorch.supr import SUPR
import numpy as np
import torch
import polyscope as ps

betas = np.zeros((1, 300))
num_betas = 300
batch_size = 1

# Set the Path of the model.
path_model = 'D:\yeqinwen\PycharmProjects\pytorch3d_code\SUPR-master\model_data\supr_female.npy'

supr = SUPR(path_model, num_betas=num_betas)
supr.display_info()

# 初始化A姿态（双臂张开）
A_pose_np = np.zeros((1, 225))
A_pose_np[0, 3 * 13 + 2] = -0.8  # 左手
A_pose_np[0, 3 * 14 + 2] = 0.8  # 右手

poses = torch.cuda.FloatTensor(A_pose_np)  # 姿态参数 [1,72]
betas = torch.cuda.FloatTensor(np.zeros((batch_size, supr.num_betas)))
trans = torch.cuda.FloatTensor(np.zeros((batch_size, 3)))

model = supr.forward(poses, betas, trans)
v = model[0].cpu().numpy()
f = model.f.astype("int32")

ps.init()
ps.register_surface_mesh("mesh", v, f,
                         color=np.array([0, 91, 255]) / 255,
                         edge_width=0.0003,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # transparency=0.6,
                         # material="flat"
                         )

# ps.set_build_gui(False)
# ps.set_ui_scale(1.5)
ps.set_view_projection_mode("perspective")  # orthographic 正交投影   perspective 透视投影
# ps.set_navigation_style("planar")  # ['turntable','free','planar','none','first_person']
ps.set_ground_plane_mode("shadow_only")  # ['none','tile','tile_reflection','shadow_only']
# ps.set_ground_plane_height(-0.001)  # 设置地平面高度
ps.set_shadow_blur_iters(3)  # 设置地平面阴影模糊程度
ps.set_shadow_darkness(0.5)  # 设置地平面阴影明暗程度
ps.set_up_dir("y_up")  # 设置y轴正方向向上（这个和设置视角会冲突，因此在添加视角参数时，这一行要注释掉）
ps.set_front_dir('z_front')  # 设置z轴正方向向前
ps.set_SSAA_factor(4)  # 在截图时，设置为4时，截图会更清晰
ps.show()

```

![](Pasted%20image%2020261104100316.png)


## 2 supr拟合扫描人体
### 2.1 将扫描人体与supr人体对齐
```python
import numpy as np
import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from supr.pytorch.supr import SUPR
import polyscope as ps


# ==================== 1. 设备设置 ====================
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

# ==================== 2. 参数配置 ====================
BATCH_SIZE = 1
NUM_BETAS = 300  # 300,  # 使用300个形状参数以更好拟合头部等细节
MAX_ITER = 1000

# 损失函数权重：控制各项损失在总优化目标中的重要性
loss_weights_dict = {
    'chamfer': 10000000,  # 主要数据项：强制supr顶点贴近扫描点云
    'shape': 10,  # 形状正则化：防止体形参数过度变形
    'pose': 1000,  # 姿态正则化：把人体向A姿态拉
}

# ==================== 3. 初始化supr人体 ====================
# 初始化A姿态（双臂双脚略微张开）
A_pose_np = np.zeros((1, 225))
A_pose_np[0, 3 * 13 + 2] = -0.8  # 左手
A_pose_np[0, 3 * 14 + 2] = 0.8  # 右手


path_model = 'D:\yeqinwen\PycharmProjects\pytorch3d_code\SUPR-master\model_data\supr_female.npy'
supr = SUPR(path_model, num_betas=NUM_BETAS).to(device)

poses = torch.cuda.FloatTensor(A_pose_np)  # (1,225)
betas = torch.cuda.FloatTensor(np.zeros((BATCH_SIZE, NUM_BETAS)))  # 形状参数 [1,300]
trans = torch.cuda.FloatTensor(np.zeros((BATCH_SIZE, 3)))  # 平移参数 [1,3]

model = supr.forward(poses, betas, trans)  # 前向生成初始人体

supr_v_np = model[0].cpu().numpy()  # 顶点坐标 [6890,3]
supr_f_np = model.f.astype("int32")  # 面片索引

# ==================== 4. 加载扫描人体 ====================
path = "dhu_scan_1_sc.obj"
verts, faces, aux = load_obj(path)
scan_v_np = verts.cpu().numpy()
scan_f_np = faces.verts_idx.cpu().numpy()

# 粗略对齐：重心和底部对齐到supr位置
scan_v_np = scan_v_np - scan_v_np.mean(axis=0) + supr_v_np.mean(axis=0)
scan_v_np[:, 1] = scan_v_np[:, 1] - np.min(scan_v_np[:, 1]) + np.min(supr_v_np[:, 1])

scan_v_tensor = torch.tensor(scan_v_np, dtype=torch.float32, device=device)
scan_f_tensor = torch.tensor(scan_f_np, dtype=torch.int64, device=device)

scan_mesh = Meshes(verts=[scan_v_tensor], faces=[scan_f_tensor])

print(f"扫描人体顶点数: {scan_v_tensor.shape[1]}")
print(f"扫描人体面数: {scan_f_tensor.shape[0]}")

# ==================== 5. 初始化可视化 ====================

ps.set_program_name("Fitting 3D body scan")
ps.init()
ps_supr_mesh = ps.register_surface_mesh("supr", supr_v_np, supr_f_np,
                                        color=np.array([0, 91, 255]) / 255,
                                        edge_color=[1, 1, 1],
                                        smooth_shade=True
                                        )
ps_scan_mesh = ps.register_surface_mesh("scan", scan_v_np, scan_f_np,
                                        color=np.array([255, 164, 0]) / 255,
                                        edge_color=[1, 1, 1],
                                        smooth_shade=True
                                        )


ps.set_build_gui(False)
# ps.set_ui_scale(1.5)
ps.set_window_size(600,800)
ps.set_view_projection_mode("perspective")  # orthographic 正交投影   perspective 透视投影
# ps.set_navigation_style("planar")  # ['turntable','free','planar','none','first_person']
ps.set_ground_plane_mode("shadow_only")  # ['none','tile','tile_reflection','shadow_only']
# ps.set_ground_plane_height(-0.001)  # 设置地平面高度
ps.set_shadow_blur_iters(3)  # 设置地平面阴影模糊程度
ps.set_shadow_darkness(0.5)  # 设置地平面阴影明暗程度
ps.set_up_dir("y_up")  # 设置y轴正方向向上（这个和设置视角会冲突，因此在添加视角参数时，这一行要注释掉）
ps.set_front_dir('z_front')  # 设置z轴正方向向前
ps.set_SSAA_factor(4)  # 在截图时，设置为4时，截图会更清晰
ps.show()

```


### 2.2 使用supr拟合扫描人体
```python
import numpy as np
import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from supr.pytorch.supr import SUPR
import polyscope as ps

from loss import Chamfer_loss, Shape_reg_loss, Pose_reg_loss

# ==================== 1. 设备设置 ====================
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

# ==================== 2. 参数配置 ====================
BATCH_SIZE = 1
NUM_BETAS = 300  # 300,  # 使用300个形状参数以更好拟合头部等细节
MAX_ITER = 1000

# 损失函数权重：控制各项损失在总优化目标中的重要性
loss_weights_dict = {
    'chamfer': 10000000,  # 主要数据项：强制supr顶点贴近扫描点云
    'shape': 10,  # 形状正则化：防止体形参数过度变形
    'pose': 1000,  # 姿态正则化：把人体向A姿态拉
}

# ==================== 3. 初始化supr人体 ====================
# 初始化A姿态（双臂双脚略微张开）
A_pose_np = np.zeros((1, 225))
A_pose_np[0, 3 * 13 + 2] = -0.8  # 左手
A_pose_np[0, 3 * 14 + 2] = 0.8  # 右手

path_model = 'D:\yeqinwen\PycharmProjects\pytorch3d_code\SUPR-master\model_data\supr_female.npy'
supr = SUPR(path_model, num_betas=NUM_BETAS).to(device)

poses = torch.cuda.FloatTensor(A_pose_np)  # (1,225)
betas = torch.cuda.FloatTensor(np.zeros((BATCH_SIZE, NUM_BETAS)))  # 形状参数 [1,300]
trans = torch.cuda.FloatTensor(np.zeros((BATCH_SIZE, 3)))  # 平移参数 [1,3]

model = supr.forward(poses, betas, trans)  # 前向生成初始人体

supr_v_np = model[0].cpu().numpy()  # 顶点坐标 [6890,3]
supr_f_np = model.f.astype("int32")  # 面片索引

# ==================== 4. 加载扫描人体 ====================
path = "dhu_scan_1_sc.obj"
verts, faces, aux = load_obj(path)
scan_v_np = verts.cpu().numpy()
scan_f_np = faces.verts_idx.cpu().numpy()

# 粗略对齐：重心和底部对齐到supr位置
scan_v_np = scan_v_np - scan_v_np.mean(axis=0) + supr_v_np.mean(axis=0)
scan_v_np[:, 1] = scan_v_np[:, 1] - np.min(scan_v_np[:, 1]) + np.min(supr_v_np[:, 1])

scan_v_tensor = torch.tensor(scan_v_np, dtype=torch.float32, device=device)
scan_f_tensor = torch.tensor(scan_f_np, dtype=torch.int64, device=device)

scan_mesh = Meshes(verts=[scan_v_tensor], faces=[scan_f_tensor])

print(f"扫描人体顶点数: {scan_v_tensor.shape[1]}")
print(f"扫描人体面数: {scan_f_tensor.shape[0]}")

# ==================== 5. 初始化可视化 ====================
ps.set_build_gui(False)
ps.set_program_name("Fitting 3D body scan")
ps.init()
ps_supr_mesh = ps.register_surface_mesh("supr", supr_v_np, supr_f_np,
                                        color=np.array([0, 91, 255]) / 255,
                                        edge_color=[1, 1, 1],
                                        smooth_shade=True
                                        )
ps_scan_mesh = ps.register_surface_mesh("scan", scan_v_np, scan_f_np,
                                        color=np.array([255, 164, 0]) / 255,
                                        edge_color=[1, 1, 1],
                                        smooth_shade=True
                                        )
ps.set_ground_plane_mode("shadow_only")
ps.set_shadow_blur_iters(3)
ps.set_shadow_darkness(0.5)
ps.set_up_dir("y_up")
ps.set_SSAA_factor(4)
ps.show()

# ==================== 6. 优化器设置 ====================
# 优化姿态、形状、平移三个参数组（约75个参数），而非直接优化每个顶点
poses.requires_grad_(True)
betas.requires_grad_(True)
trans.requires_grad_(True)
optim_params = [poses, betas, trans]

optimizer = torch.optim.Adam(optim_params, lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

supr_f_tensor = torch.tensor(supr_f_np, dtype=torch.int64, device=device)

# ==================== 7. 训练循环 ====================
for i in range(MAX_ITER):
    optimizer.zero_grad()

    # 前向：根据当前参数生成supr网格
    model = supr.forward(poses, betas, trans)  # [1, 6890, 3]
    supr_mesh = Meshes(
        verts=[model[0]],
        faces=[supr_f_tensor]
    )

    # ==================== 8. 损失函数定义（从外部导入） ====================
    loss_dict = {}
    chamfer_loss = Chamfer_loss(supr_mesh, scan_mesh, num_samples=30000)  # 采样5000，速度更快，采样30000，拟合更精确
    shape_reg_loss = Shape_reg_loss(betas)
    pose_reg_loss = Pose_reg_loss(poses)

    loss_dict["chamfer"] = loss_weights_dict["chamfer"] * chamfer_loss
    loss_dict["shape"] = loss_weights_dict["shape"] * shape_reg_loss
    loss_dict["pose"] = loss_weights_dict["pose"] * pose_reg_loss

    loss = sum(loss_dict.values())

    # 反向传播：计算loss对poses, betas, trans的梯度
    loss.backward()
    torch.nn.utils.clip_grad_norm_(optim_params, max_norm=1.0)  # 梯度裁剪，防止梯度爆炸
    optimizer.step()  # 更新参数
    scheduler.step()  # 更新学习率

    # ==================== 9. 实时更新可视化 ====================
    with torch.no_grad():
        updated_vertices = model[0].cpu().numpy()
        if 'ps_supr_mesh' in globals():
            ps_supr_mesh.update_vertex_positions(updated_vertices)
            ps.frame_tick()

# ==================== 10. 保存结果 ====================
with torch.no_grad():
    model = supr.forward(poses, betas, trans)
    final_supr_v_np = model[0].detach().cpu().numpy()

save_obj("final_supr.obj", torch.Tensor(final_supr_v_np), torch.Tensor(supr_f_np))
save_obj("final_scan.obj", scan_v_tensor, scan_f_tensor)
print("结果已保存！")

ps.remove_all_structures()
ps.init()
ps.register_surface_mesh("supr_fitted", final_supr_v_np, supr_f_np,
                         color=np.array([0, 91, 255]) / 255,
                         edge_color=[1, 1, 1],
                         smooth_shade=True
                         )
ps.register_surface_mesh("scan", scan_v_np, scan_f_np,
                         color=np.array([255, 164, 0]) / 255,
                         edge_color=[1, 1, 1],
                         smooth_shade=True
                         )
ps.set_up_dir("y_up")
ps.set_ground_plane_mode("shadow_only")
ps.set_shadow_blur_iters(3)
ps.set_shadow_darkness(0.5)
ps.set_SSAA_factor(4)
ps.show()

```

loss.py
```python
import torch
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
import numpy as np

# ==================== 1. 设备设置 ====================
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")


def Chamfer_loss(star_mesh, scan_mesh, num_samples=30000):
    """计算Chamfer距离"""
    # 在mesh上采样点云
    star_mesh_sample_points = sample_points_from_meshes(star_mesh, num_samples)
    scan_mesh_sample_points = sample_points_from_meshes(scan_mesh, num_samples)

    # 计算Chamfer距离
    chamfer_dist, _ = chamfer_distance(star_mesh_sample_points, scan_mesh_sample_points)
    return chamfer_dist


def Shape_reg_loss(betas):
    """体型正则化损失"""
    return torch.mean(betas * betas)


A_pose_np = np.zeros((1, 225))
A_pose_np[0, 3 * 13 + 2] = -0.8  # 左手
A_pose_np[0, 3 * 14 + 2] = 0.8  # 右手


def Pose_reg_loss(poses: torch.Tensor):
    # 将姿态约束在你设定的A_pose附近
    # 也就是说：允许微调，但不允许偏离太远
    A_pose_tensor = torch.tensor(A_pose_np, device=device)
    loss = torch.mean((poses - A_pose_tensor) ** 2)
    return loss

```

### 2.3 可视化对齐结果
```python
import numpy as np
import pymeshlab
import polyscope as ps

ms = pymeshlab.MeshSet()

ms.load_new_mesh('final_supr.obj')
m = ms.current_mesh()
supr_v, supr_f = m.vertex_matrix(), m.face_matrix()

ms.load_new_mesh('final_scan.obj')
m = ms.current_mesh()
scan_v, scan_f = m.vertex_matrix(), m.face_matrix()

print(scan_v.shape, scan_f.shape)
print(supr_v.shape, supr_f.shape)

ps.init()
ps.register_surface_mesh("supr", supr_v, supr_f,
                         color=np.array([0, 91, 255]) / 255,
                         edge_width=0.001,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )
ps.register_surface_mesh("scan", scan_v, scan_f,
                         color=np.array([255, 164, 0]) / 255,
                         # color=np.array([255, 255, 255]) / 255,
                         edge_width=0.001,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # transparency= 0.5
                         # material="flat"
                         )

ps.set_build_gui(False)
# ps.set_ui_scale(1.5)
ps.set_window_size(600,800)
ps.set_view_projection_mode("perspective")  # orthographic 正交投影   perspective 透视投影
# ps.set_navigation_style("planar")  # ['turntable','free','planar','none','first_person']
ps.set_ground_plane_mode("shadow_only")  # ['none','tile','tile_reflection','shadow_only']
# ps.set_ground_plane_height(-0.001)  # 设置地平面高度
ps.set_shadow_blur_iters(3)  # 设置地平面阴影模糊程度
ps.set_shadow_darkness(0.5)  # 设置地平面阴影明暗程度
ps.set_up_dir("y_up")  # 设置y轴正方向向上（这个和设置视角会冲突，因此在添加视角参数时，这一行要注释掉）
ps.set_front_dir('z_front')  # 设置z轴正方向向前
ps.set_SSAA_factor(4)  # 在截图时，设置为4时，截图会更清晰
ps.show()

```

### 2.4 可视化误差
```python
import numpy as np
import pymeshlab
import polyscope as ps

ms = pymeshlab.MeshSet()
ms.load_new_mesh('final_supr.obj')
ms.load_new_mesh('final_scan.obj')

m = ms.current_mesh()
scan_v, scan_f = m.vertex_matrix(), m.face_matrix()

# hausdorff_distance
ms.get_hausdorff_distance(targetmesh=0, sampledmesh=1, samplevert=True)
dis = ms.mesh(1).vertex_scalar_array()

vmin = 0
vmax = np.percentile(dis, 95)  # p95 or p99

print(np.min(dis), np.max(dis))
print(vmin, vmax)

ps.init()
ps_scan = ps.register_surface_mesh("scan", scan_v, scan_f, edge_width=0.001, smooth_shade=True)
ps_scan.add_scalar_quantity("distance", dis,
                            defined_on="vertices",
                            # vminmax=[vmin, vmax],
                            vminmax=[0, 0.008],
                            cmap="jet",  # colormap
                            onscreen_colorbar_enabled=True,  # show the colorbar
                            enabled=True,
                            )

ps.set_build_gui(False)
# ps.set_ui_scale(1.5)
ps.set_window_size(600,800)
ps.set_view_projection_mode("perspective")  # orthographic 正交投影   perspective 透视投影
# ps.set_navigation_style("planar")  # ['turntable','free','planar','none','first_person']
ps.set_ground_plane_mode("shadow_only")  # ['none','tile','tile_reflection','shadow_only']
# ps.set_ground_plane_height(-0.001)  # 设置地平面高度
ps.set_shadow_blur_iters(3)  # 设置地平面阴影模糊程度
ps.set_shadow_darkness(0.5)  # 设置地平面阴影明暗程度
ps.set_up_dir("y_up")  # 设置y轴正方向向上（这个和设置视角会冲突，因此在添加视角参数时，这一行要注释掉）
ps.set_front_dir('z_front')  # 设置z轴正方向向前
ps.set_SSAA_factor(4)  # 在截图时，设置为4时，截图会更清晰
ps.show()

```

|                       对齐                        |                      拟合结果                       |                      可视化误差                      |
| :---------------------------------------------: | :---------------------------------------------: | :---------------------------------------------: |
| ![](assets/Pasted%20image%2020261104102804.png) | ![](assets/Pasted%20image%2020261104102947.png) | ![](assets/Pasted%20image%2020261104103015.png) |
