

| ![](assets/Pasted%20image%2020260920111546.png) | ![](assets/Pasted%20image%2020260920111737.png) | ![](assets/Pasted%20image%2020260920111836.png) |     |
| :---------------------------------------------: | :---------------------------------------------: | :---------------------------------------------: | --- |
|                      SMPL                       |                     SMPL-H                      |                     SMPL-X                      |     |
| ![](assets/Pasted%20image%2020260921182047.png) | ![](assets/Pasted%20image%2020260920112051.png) | ![](assets/Pasted%20image%2020260920112148.png) |     |
|                   ANNY(四边形网格)                   |                       MHR                       |                     SOMA-X                      |     |
| ![](assets/Pasted%20image%2020260920112321.png) | ![](assets/Pasted%20image%2020260920112344.png) |                                                 |     |
|                      FLAME                      |                       GNM                       |                                                 |     |
| ![](assets/Pasted%20image%2020260920112408.png) | ![](assets/Pasted%20image%2020260920112439.png) | ![](assets/Pasted%20image%2020260920112900.png) |     |
|                      MANO                       |                      SKEL                       |                       GM                        |     |

**body-models项目网站**

[abcamiletto/body-models: Unified Python interface for parametric and articulated human, anatomical, and robot models across NumPy, PyTorch, and JAX, with optional Warp acceleration.](https://github.com/abcamiletto/body-models)
```
https://github.com/abcamiletto/body-models
```


## 1 配置运行环境

```python
# 创建运行环境
conda create -n body_models python=3.11

# 安装pytorch
pip install torch --index-url https://download.pytorch.org/whl/cu121

Installing collected packages: mpmath, typing-extensions, sympy, networkx, MarkupSafe, fsspec, filelock, jinja2, torch
Successfully installed MarkupSafe-3.0.3 filelock-3.32.3 fsspec-2026.7.0 jinja2-3.1.6 mpmath-1.3.0 networkx-3.6.1 sympy-1.13.1 torch-2.5.1+cu121 typing-extensions-4.16.0


pip install "body-models[torch]"


python -m pip install polyscope

pip3 install pymeshlab

pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.5.1cu121

pip install pillow
```

安装其他库
```python
pip install viser


body-models-viser
# https://github.com/abcamiletto/body-models-viser

body-models-viser需要'>=3.12'
所以在安装body_models运行环境时，安装python3.12

```



```
# 自动下载的模型保存的位置

C:\Users\yeqin\AppData\Local\body-models\body-models\Cache

```



```
# 设置SMPL模型文件路径
body-models set smpl-neutral D:\body_models\SMPL\SMPL_NEUTRAL.pkl 
body-models set smpl-male D:\body_models\SMPL\SMPL_MALE.pkl
body-models set smpl-female D:\body_models\SMPL\SMPL_FEMALE.pkl

# 设置SMPLX
body-models set smplx-neutral D:\body_models\SMPLX\SMPLX_NEUTRAL.pkl 
body-models set smplx-male D:\body_models\SMPLX\SMPLX_MALE.pkl
body-models set smplx-female D:\body_models\SMPLX\SMPLX_FEMALE.pkl

# 设置SMPLH
body-models set smplh-neutral D:\body_models\SMPLH\neutral\model.npz
body-models set smplh-male D:\body_models\SMPLH\male\model.npz
body-models set smplh-female D:\body_models\SMPLH\female\model.npz

# 设置FLAME
body-models set flame D:\body_models\FLAME\FLAME_female.pkl 

# 设置MONO
body-models set mano-left D:\body_models\MANO\MANO_LEFT.pkl
body-models set mano-right D:\body_models\MANO\MANO_RIGHT.pkl

# 设置SKEL
body-models set skel-female D:\body_models\SKEL\skel_female.pkl
body-models set skel-male D:\body_models\SKEL\skel_male.pkl



# 数据集的命名
Expected one of: 
smpl-male, smpl-female, smpl-neutral,       
smplx-male, smplx-female, smplx-neutral, 
smplh-male, smplh-female, smplh-neutral,
 mano-right, mano-left, 
 skel-male,  skel-female, 
anny,
 mhr, 
flame, 
gnm, 
soma, 
garment-measurements


```


这几个模型是一起的
1. **SMPL**：身体（body）
2. **MANO**：手（hands）
3. **FLAME**：头脸（head + face）
4. **SMPL-H**：SMPL + MANO
5. **SMPL-X**：SMPL + MANO + FLAME




## 2 SMPL
### 2.1 生成人体numpy

```python
from body_models.smpl.numpy import SMPL
import polyscope as ps

model = SMPL(gender="female")
params = model.get_rest_pose(batch_dims=(1,))

vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
faces = model.faces

print(vertices.shape)  # [1, 6890, 3]
print(skeleton.shape)  # ([1, 24, 4, 4]
print(faces.shape)  # [13776, 3]

print(type(vertices))
print(type(faces))
print(type(skeleton))

ps.set_build_gui(False)
ps.init()
ps.register_surface_mesh("mesh", vertices[0], faces,
                         color=[0, 91 / 255, 255 / 255],
                         edge_width=0.003,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )

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
(1, 6890, 3)
(1, 24, 4, 4)
(13776, 3)
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
```

![](assets/Pasted%20image%2020260920141403.png)

### 2.2 生成人体torch_cpu

```python
# from body_models.smpl.numpy import SMPL
from body_models.smpl.torch import SMPL
import torch
import polyscope as ps

model = SMPL(gender="female")
params = model.get_rest_pose(batch_dims=(1,))

vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
faces = model.faces

print(vertices.shape)  # [1, 6890, 3]
print(skeleton.shape)  # ([1, 24, 4, 4]
print(faces.shape)  # [13776, 3]

# 查看每个参数在cpu上还是gpu上
for k, v in params.items():
    if torch.is_tensor(v):
        print(k, v.device)

ps.set_build_gui(False)
ps.init()
ps.register_surface_mesh("mesh", vertices[0], faces,
                         color=[0, 91 / 255, 255 / 255],
                         edge_width=0.003,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )

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
torch.Size([1, 6890, 3])
torch.Size([1, 24, 4, 4])
torch.Size([13776, 3])
shape cpu
body_pose cpu
pelvis_rotation cpu
global_rotation cpu
global_translation cpu
```

### 2.3 生成人体torch_gpu
```python
# from body_models.smpl.numpy import SMPL
from body_models.smpl.torch import SMPL
import torch
import polyscope as ps

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

model = SMPL(gender="female").to(device)  # 将SMPL模型放到GPU上
params = model.get_rest_pose(batch_dims=(1,))

vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
faces = model.faces

print(vertices.shape)  # [1, 6890, 3]
print(skeleton.shape)  # ([1, 24, 4, 4]
print(faces.shape)  # [13776, 3]

for k, v in params.items():
    if torch.is_tensor(v):
        print(k, v.device)

# 生成的顶点、面、骨骼数据都是gpu tensor，需要使用.cpu().numpy()进行转化，再可视化
print(type(vertices))
print(type(faces))
print(type(skeleton))

vertices = vertices[0].cpu().numpy()
faces = faces.cpu().numpy()

ps.set_build_gui(False)
ps.init()
ps.register_surface_mesh("mesh", vertices, faces,
                         color=[0, 91 / 255, 255 / 255],
                         edge_width=0.003,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )

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
torch.Size([1, 6890, 3])
torch.Size([1, 24, 4, 4])
torch.Size([13776, 3])
shape cuda:0
body_pose cuda:0
pelvis_rotation cuda:0
global_rotation cuda:0
global_translation cuda:0
<class 'torch.Tensor'>
<class 'torch.Tensor'>
<class 'torch.Tensor'>
```

### 2.4 人体参数
```python
# from body_models.smpl.numpy import SMPL
from body_models.smpl.torch import SMPL
import torch
import polyscope as ps

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

model = SMPL(gender="female").to(device)  # 将SMPL模型放到GPU上
params = model.get_rest_pose(batch_dims=(1,))

# 打印参数
print(params.keys())
for k, v in params.items():
    print(k, v.shape)

vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
faces = model.faces

vertices = vertices[0].cpu().numpy()
faces = faces.cpu().numpy()

ps.init()
ps.register_surface_mesh("mesh", vertices, faces,
                         color=[0, 91 / 255, 255 / 255],
                         edge_width=0.003,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )

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
dict_keys(['shape', 'body_pose', 'pelvis_rotation', 'global_rotation', 'global_translation'])
shape torch.Size([1, 10])
body_pose torch.Size([1, 23, 3])
pelvis_rotation torch.Size([1, 3])
global_rotation torch.Size([1, 3])
global_translation torch.Size([1, 3])
```

1. **`body_pose`** — `torch.Size([1, 23, 3])`  这是 23 个关节（不含骨盆根节点）相对于其父关节的**局部旋转**（通常是轴角表示）。它决定了四肢和躯干各部位的姿势，是姿态的主要控制参数。
2. **`pelvis_rotation`** — `torch.Size([1, 3])`  这是骨盆（根节点）的旋转，控制整个人体在空间中的**朝向/转身**。它和 `body_pose` 一起构成完整的关节旋转（即 SMPL 中的 `global_orient` + `body_pose`）。
3. **`global_rotation`** — `[1, 3]`：通常是整个人体相对于世界坐标系的全局旋转（有时与 `pelvis_rotation` 含义重叠，取决于具体模型实现）。
4. **`global_translation`** — `[1, 3]`：控制人体在空间中的**位置平移**，不属于姿态，属于位置。
 5. **`shape`** — `[1, 10]`：控制人体的**体型/身材**（高矮胖瘦），不是姿态。



### 2.5 可视化pose
```python
# from body_models.smpl.numpy import SMPL
from body_models.smpl.torch import SMPL
import torch
import polyscope as ps

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

model = SMPL(gender="female").to(device)  # 将SMPL模型放到GPU上
params = model.get_rest_pose(batch_dims=(1,))

skeleton = model.forward_skeleton(**params)
vertices = model.forward_vertices(**params)
faces = model.faces

vertices = vertices[0].cpu().numpy()
faces = faces.cpu().numpy()

J = skeleton[0].cpu().numpy()  # [24, 4, 4]
joints = J[:, :3, 3]  # [24, 3]  每个关节的 (x, y, z)

# SMPL 关节父子连接表 (parent, child)
bones = [
    (0, 1), (0, 2), (0, 3),  # 骨盆 -> 左右髋、脊柱
    (1, 4), (4, 7), (7, 10),  # 左腿: 髋->膝->踝->脚
    (2, 5), (5, 8), (8, 11),  # 右腿
    (3, 6), (6, 9), (9, 12),  # 脊柱: spine1->2->3->neck
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),  # 左臂
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),  # 右臂
    (12, 15),  # 脖子 -> 头
]


ps.init()
ps.set_build_gui(False)
ps.register_surface_mesh("mesh", vertices, faces,
                         color=[0, 91 / 255, 255 / 255],
                         edge_width=0.003,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         transparency=0.5
                         # material="flat"
                         )

ps.register_point_cloud("j", joints, radius=0.005, color=[1, 0, 0])
ps.register_curve_network("bone", joints, bones, radius=0.0015,
                          color=[255 / 255, 0, 218 / 255])

# ps.set_view_projection_mode("perspective")  # orthographic 正交投影   perspective 透视投影
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


![](assets/Pasted%20image%2020260920141339.png)

![](assets/Pasted%20image%2020260929143913.png)


### 2.6 设置shape

```python
# from body_models.smpl.numpy import SMPL
from body_models.smpl.torch import SMPL
import torch
import numpy as np
import polyscope as ps

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

model = SMPL(gender="female").to(device)  # 将SMPL模型放到GPU上
params = model.get_rest_pose(batch_dims=(1,))

# 设置shape参数
params["shape"] = torch.tensor([[1.0, -2, 0, 0, 0, 0, 0, 0, 0, 0]],
                               dtype=torch.float32, device=device)

vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
faces = model.faces

vertices = vertices[0].cpu().numpy()
faces = faces.cpu().numpy()

ps.init()
ps.set_build_gui(False)
ps.register_surface_mesh("mesh", vertices, faces,
                         color=[0, 91 / 255, 255 / 255],
                         edge_width=0.003,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )

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


![](assets/Pasted%20image%2020260920141539.png)
### 2.7 生成不同shape

```python
# from body_models.smpl.numpy import SMPL
from body_models.smpl.torch import SMPL
import torch
import numpy as np
import polyscope as ps

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

model = SMPL(gender="female").to(device)  # 将SMPL模型放到GPU上
params = model.get_rest_pose(batch_dims=(1,))

ps.init()
ps.set_build_gui(False)

# params["shape"] 为 torch.zeros((1,10),dtype=torch.int64, device=device)
for i in range(7):
    params["shape"][0, 1] = i - 3

    vertices = model.forward_vertices(**params)
    skeleton = model.forward_skeleton(**params)
    faces = model.faces

    vertices = vertices[0].cpu().numpy()
    faces = faces.cpu().numpy()

    vertices = vertices + np.array([0, 0, i * 0.5])

    ps.register_surface_mesh("mesh_{}".format(i - 3), vertices, faces,
                             color=[0, 91 / 255, 255 / 255],
                             edge_width=0.003,
                             edge_color=[1, 1, 1],
                             smooth_shade=True,
                             # material="flat"
                             )

ps.set_view_projection_mode("orthographic")  # orthographic 正交投影   perspective 透视投影
# ps.set_navigation_style("planar")  # ['turntable','free','planar','none','first_person']
ps.set_ground_plane_mode("shadow_only")  # ['none','tile','tile_reflection','shadow_only']
# ps.set_ground_plane_height(-0.001)  # 设置地平面高度
ps.set_shadow_blur_iters(3)  # 设置地平面阴影模糊程度
ps.set_shadow_darkness(0.5)  # 设置地平面阴影明暗程度
ps.set_up_dir("y_up")  # 设置y轴正方向向上（这个和设置视角会冲突，因此在添加视角参数时，这一行要注释掉）
ps.set_front_dir('neg_x_front')  # 设置z轴正方向向前
ps.set_SSAA_factor(4)  # 在截图时，设置为4时，截图会更清晰
ps.show()

```


![](assets/Pasted%20image%2020260920142013.png)



## 3 SMPLH
### 3.1 生成人体numpy

```python
from body_models.smplh.numpy import SMPLH
import polyscope as ps

model = SMPLH(gender="female")
params = model.get_rest_pose(batch_dims=(1,))

vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
faces = model.faces

print(vertices.shape)  # [1, 6890, 3]
print(skeleton.shape)  # ([1, 24, 4, 4]
print(faces.shape)  # [13776, 3]

print(type(vertices))
print(type(faces))
print(type(skeleton))

ps.set_build_gui(False)
ps.init()
ps.register_surface_mesh("mesh", vertices[0], faces,
                         color=[0, 91 / 255, 255 / 255],
                         edge_width=0.003,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )

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
(1, 6890, 3)
(1, 52, 4, 4)
(13776, 3)
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
```

![](assets/Pasted%20image%2020260929142021.png)

### 3.2 生成人体torch_cpu
```python
from body_models.smplh.torch import SMPLH
import torch
import polyscope as ps

model = SMPLH(gender="female")
params = model.get_rest_pose(batch_dims=(1,))

vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
faces = model.faces

print(vertices.shape)  # [1, 6890, 3]
print(skeleton.shape)  # ([1, 24, 4, 4]
print(faces.shape)  # [13776, 3]

# 查看每个参数在cpu上还是gpu上
for k, v in params.items():
    if torch.is_tensor(v):
        print(k, v.device)

ps.set_build_gui(False)
ps.init()
ps.register_surface_mesh("mesh", vertices[0], faces,
                         color=[0, 91 / 255, 255 / 255],
                         edge_width=0.003,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )

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
torch.Size([1, 6890, 3])
torch.Size([1, 52, 4, 4])
torch.Size([13776, 3])
shape cpu
body_pose cpu
hand_pose cpu
pelvis_rotation cpu
global_rotation cpu
global_translation cpu
```

### 3.3 生成人体torch_gpu
```python
from body_models.smplh.torch import SMPLH
import torch
import polyscope as ps

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

model = SMPLH(gender="female").to(device)  # 将SMPL模型放到GPU上
params = model.get_rest_pose(batch_dims=(1,))

vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
faces = model.faces

print(vertices.shape)  # [1, 6890, 3]
print(skeleton.shape)  # ([1, 24, 4, 4]
print(faces.shape)  # [13776, 3]

for k, v in params.items():
    if torch.is_tensor(v):
        print(k, v.device)

# 生成的顶点、面、骨骼数据都是gpu tensor，需要使用.cpu().numpy()进行转化，再可视化
print(type(vertices))
print(type(faces))
print(type(skeleton))

vertices = vertices[0].cpu().numpy()
faces = faces.cpu().numpy()

ps.set_build_gui(False)
ps.init()
ps.register_surface_mesh("mesh", vertices, faces,
                         color=[0, 91 / 255, 255 / 255],
                         edge_width=0.003,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )

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
torch.Size([1, 6890, 3])
torch.Size([1, 52, 4, 4])
torch.Size([13776, 3])
shape cuda:0
body_pose cuda:0
hand_pose cuda:0
pelvis_rotation cuda:0
global_rotation cuda:0
global_translation cuda:0
<class 'torch.Tensor'>
<class 'torch.Tensor'>
<class 'torch.Tensor'>
```


### 3.4 人体参数
```python
from body_models.smplh.torch import SMPLH
import torch
import polyscope as ps

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

model = SMPLH(gender="female").to(device)  # 将SMPL模型放到GPU上
params = model.get_rest_pose(batch_dims=(1,))

# 打印参数
print(params.keys())
for k, v in params.items():
    print(k, v.shape)

vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
faces = model.faces

vertices = vertices[0].cpu().numpy()
faces = faces.cpu().numpy()

ps.init()
ps.register_surface_mesh("mesh", vertices, faces,
                         color=[0, 91 / 255, 255 / 255],
                         edge_width=0.003,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )

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
dict_keys(['shape', 'body_pose', 'hand_pose', 'pelvis_rotation', 'global_rotation', 'global_translation'])
shape torch.Size([1, 16])
body_pose torch.Size([1, 21, 3])
hand_pose torch.Size([1, 30, 3])
pelvis_rotation torch.Size([1, 3])
global_rotation torch.Size([1, 3])
global_translation torch.Size([1, 3])
```



### 3.5 可视化pose

```python
from body_models.smplh.torch import SMPLH
import torch
import polyscope as ps

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

model = SMPLH(gender="female").to(device)  # 将SMPL模型放到GPU上
params = model.get_rest_pose(batch_dims=(1,))

# 这里的参数是姿态参数，指的是旋转轴角
print(params["body_pose"].shape)
print(params["hand_pose"].shape)
print(params["pelvis_rotation"].shape)

skeleton = model.forward_skeleton(**params)
vertices = model.forward_vertices(**params)
faces = model.faces

vertices = vertices[0].cpu().numpy()
faces = faces.cpu().numpy()

J = skeleton[0].cpu().numpy()  # [24, 4, 4]
joints = J[:, :3, 3]  # [24, 3]  每个关节的 (x, y, z)

print(vertices.shape)
print(faces.shape)
print(skeleton.shape)
print(joints.shape)


# SMPL 关节父子连接表 (parent, child)
bones = [
    # pelvis -> legs / spine
    (0, 1), (0, 2), (0, 3),

    # left leg
    (1, 4), (4, 7), (7, 10),

    # right leg
    (2, 5), (5, 8), (8, 11),

    # spine
    (3, 6), (6, 9), (9, 12), (12, 15),

    # left arm
    (9, 13),
    (13, 16),
    (16, 18),
    (18, 20),

    # right arm
    (9, 14),
    (14, 17),
    (17, 19),
    (19, 21),


    # ------------------
    # left hand
    # ------------------

    # index finger
    (20,22), (22,23), (23,24),

    # middle finger
    (20,25), (25,26), (26,27),

    # pinky
    (20,28), (28,29), (29,30),

    # ring
    (20,31), (31,32), (32,33),

    # thumb
    (20,34), (34,35), (35,36),


    # ------------------
    # right hand
    # ------------------

    # index finger
    (21,37), (37,38), (38,39),

    # middle finger
    (21,40), (40,41), (41,42),

    # pinky
    (21,43), (43,44), (44,45),

    # ring
    (21,46), (46,47), (47,48),

    # thumb
    (21,49), (49,50), (50,51),
]



ps.init()
ps.set_build_gui(False)
ps.register_surface_mesh("mesh", vertices, faces,
                         color=[0, 91 / 255, 255 / 255],
                         edge_width=0.003,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         transparency=0.5
                         # material="flat"
                         )

ps.register_point_cloud("j", joints, radius=0.005, color=[1, 0, 0])
ps.register_curve_network("bone", joints, bones, radius=0.0015,
                          color=[255 / 255, 0, 218 / 255])

# ps.set_view_projection_mode("perspective")  # orthographic 正交投影   perspective 透视投影
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
torch.Size([1, 21, 3])
torch.Size([1, 30, 3])
torch.Size([1, 3])
(6890, 3)
(13776, 3)
torch.Size([1, 52, 4, 4])
(52, 3)
```

![](assets/Pasted%20image%2020260929152558.png)


### 3.6 设置shape

```python
from body_models.smplh.torch import SMPLH
import torch
import numpy as np
import polyscope as ps

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

model = SMPLH(gender="female").to(device)  # 将SMPL模型放到GPU上
params = model.get_rest_pose(batch_dims=(1,))

# 设置shape参数
params["shape"] = torch.tensor([[1.0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                               dtype=torch.float32, device=device)

vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
faces = model.faces

vertices = vertices[0].cpu().numpy()
faces = faces.cpu().numpy()

ps.init()
ps.set_build_gui(False)
ps.register_surface_mesh("mesh", vertices, faces,
                         color=[0, 91 / 255, 255 / 255],
                         edge_width=0.003,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )

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

![](assets/Pasted%20image%2020260929152955.png)


### 3.7 生成不同shape

```python
from body_models.smplh.torch import SMPLH
import torch
import numpy as np
import polyscope as ps

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

model = SMPLH(gender="female").to(device)  # 将SMPL模型放到GPU上
params = model.get_rest_pose(batch_dims=(1,))

ps.init()
ps.set_build_gui(False)

# params["shape"] 为 torch.zeros((1,10),dtype=torch.int64, device=device)
for i in range(7):
    params["shape"][0, 1] = i - 3

    vertices = model.forward_vertices(**params)
    skeleton = model.forward_skeleton(**params)
    faces = model.faces

    vertices = vertices[0].cpu().numpy()
    faces = faces.cpu().numpy()

    vertices = vertices + np.array([0, 0, i * 0.5])

    ps.register_surface_mesh("mesh_{}".format(i - 3), vertices, faces,
                             color=[0, 91 / 255, 255 / 255],
                             edge_width=0.003,
                             edge_color=[1, 1, 1],
                             smooth_shade=True,
                             # material="flat"
                             )

ps.set_view_projection_mode("orthographic")  # orthographic 正交投影   perspective 透视投影
# ps.set_navigation_style("planar")  # ['turntable','free','planar','none','first_person']
ps.set_ground_plane_mode("shadow_only")  # ['none','tile','tile_reflection','shadow_only']
# ps.set_ground_plane_height(-0.001)  # 设置地平面高度
ps.set_shadow_blur_iters(3)  # 设置地平面阴影模糊程度
ps.set_shadow_darkness(0.5)  # 设置地平面阴影明暗程度
ps.set_up_dir("y_up")  # 设置y轴正方向向上（这个和设置视角会冲突，因此在添加视角参数时，这一行要注释掉）
ps.set_front_dir('neg_x_front')  # 设置z轴正方向向前
ps.set_SSAA_factor(4)  # 在截图时，设置为4时，截图会更清晰
ps.show()

```

![](assets/Pasted%20image%2020260929153159.png)

## 4 SMPLX
### 4.1 生成人体numpy
```python
from body_models.smplx.numpy import SMPLX
import polyscope as ps

model = SMPLX(gender="female")
params = model.get_rest_pose(batch_dims=(1,))

vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
faces = model.faces

print(vertices.shape)  # [1, 6890, 3]
print(skeleton.shape)  # ([1, 24, 4, 4]
print(faces.shape)  # [13776, 3]

print(type(vertices))
print(type(faces))
print(type(skeleton))

ps.set_build_gui(False)
ps.init()
ps.register_surface_mesh("mesh", vertices[0], faces,
                         color=[0, 91 / 255, 255 / 255],
                         edge_width=0.003,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )

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
(1, 10475, 3)
(1, 55, 4, 4)
(20908, 3)
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
```

![](assets/Pasted%20image%2020260929153353.png)

### 4.2 生成人体torch_cpu
```python
from body_models.smplx.torch import SMPLX
import torch
import polyscope as ps

model = SMPLX(gender="female")
params = model.get_rest_pose(batch_dims=(1,))

vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
faces = model.faces

print(vertices.shape)  # [1, 6890, 3]
print(skeleton.shape)  # ([1, 24, 4, 4]
print(faces.shape)  # [13776, 3]

# 查看每个参数在cpu上还是gpu上
for k, v in params.items():
    if torch.is_tensor(v):
        print(k, v.device)

ps.set_build_gui(False)
ps.init()
ps.register_surface_mesh("mesh", vertices[0], faces,
                         color=[0, 91 / 255, 255 / 255],
                         edge_width=0.003,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )

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
torch.Size([1, 10475, 3])
torch.Size([1, 55, 4, 4])
torch.Size([20908, 3])
shape cpu
expression cpu
body_pose cpu
head_pose cpu
hand_pose cpu
pelvis_rotation cpu
global_rotation cpu
global_translation cpu
```


### 4.3 生成人体torch_gpu
```python
from body_models.smplx.torch import SMPLX
import torch
import polyscope as ps

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

model = SMPLX(gender="female").to(device)  # 将SMPL模型放到GPU上
params = model.get_rest_pose(batch_dims=(1,))

vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
faces = model.faces

print(vertices.shape)  # [1, 6890, 3]
print(skeleton.shape)  # ([1, 24, 4, 4]
print(faces.shape)  # [13776, 3]

for k, v in params.items():
    if torch.is_tensor(v):
        print(k, v.device)

# 生成的顶点、面、骨骼数据都是gpu tensor，需要使用.cpu().numpy()进行转化，再可视化
print(type(vertices))
print(type(faces))
print(type(skeleton))

vertices = vertices[0].cpu().numpy()
faces = faces.cpu().numpy()

ps.set_build_gui(False)
ps.init()
ps.register_surface_mesh("mesh", vertices, faces,
                         color=[0, 91 / 255, 255 / 255],
                         edge_width=0.003,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )

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
torch.Size([1, 10475, 3])
torch.Size([1, 55, 4, 4])
torch.Size([20908, 3])
shape cuda:0
expression cuda:0
body_pose cuda:0
head_pose cuda:0
hand_pose cuda:0
pelvis_rotation cuda:0
global_rotation cuda:0
global_translation cuda:0
<class 'torch.Tensor'>
<class 'torch.Tensor'>
<class 'torch.Tensor'>
```

### 4.4 人体参数
```python
from body_models.smplx.torch import SMPLX
import torch
import polyscope as ps

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

model = SMPLX(gender="female").to(device)  # 将SMPL模型放到GPU上
params = model.get_rest_pose(batch_dims=(1,))

# 打印参数
print(params.keys())
for k, v in params.items():
    print(k, v.shape)

vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
faces = model.faces

vertices = vertices[0].cpu().numpy()
faces = faces.cpu().numpy()

ps.init()
ps.register_surface_mesh("mesh", vertices, faces,
                         color=[0, 91 / 255, 255 / 255],
                         edge_width=0.003,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )

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
dict_keys(['shape', 'expression', 'body_pose', 'head_pose', 'hand_pose', 'pelvis_rotation', 'global_rotation', 'global_translation'])
shape torch.Size([1, 300])
expression torch.Size([1, 10])
body_pose torch.Size([1, 21, 3])
head_pose torch.Size([1, 3, 3])
hand_pose torch.Size([1, 30, 3])
pelvis_rotation torch.Size([1, 3])
global_rotation torch.Size([1, 3])
global_translation torch.Size([1, 3])
```


### 4.5 可视化pose
**但要特别注意：不同 SMPL-X 代码/数据集的 joint 顺序可能不一样。**

```python
# from body_models.smpl.numpy import SMPL
from body_models.smplx.torch import SMPLX
import torch
import polyscope as ps

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

model = SMPLX(gender="female").to(device)  # 将SMPL模型放到GPU上
params = model.get_rest_pose(batch_dims=(1,))

skeleton = model.forward_skeleton(**params)
vertices = model.forward_vertices(**params)
faces = model.faces

vertices = vertices[0].cpu().numpy()
faces = faces.cpu().numpy()

J = skeleton[0].cpu().numpy()  # [24, 4, 4]
joints = J[:, :3, 3]  # [24, 3]  每个关节的 (x, y, z)

# SMPL 关节父子连接表 (parent, child)
bones = [
    # =========================
    # 身体
    # =========================

    # pelvis -> hips / spine
    (0, 1), (0, 2), (0, 3),

    # left leg
    (1, 4), (4, 7), (7, 10),

    # right leg
    (2, 5), (5, 8), (8, 11),

    # spine
    (3, 6), (6, 9), (9, 12),

    # left arm
    (9, 13), (13, 16), (16, 18), (18, 20),

    # right arm
    (9, 14), (14, 17), (17, 19), (19, 21),

    # neck -> head
    (12, 15),

    # head
    (15, 22),  # jaw
    (15, 23),  # left_eye
    (15, 24),  # right_eye


    # =========================
    # 左手
    # =========================

    # wrist -> index
    (20, 25),
    (25, 26),
    (26, 27),

    # wrist -> middle
    (20, 28),
    (28, 29),
    (29, 30),

    # wrist -> pinky
    (20, 31),
    (31, 32),
    (32, 33),

    # wrist -> ring
    (20, 34),
    (34, 35),
    (35, 36),

    # wrist -> thumb
    (20, 37),
    (37, 38),
    (38, 39),


    # =========================
    # 右手
    # =========================

    # wrist -> index
    (21, 40),
    (40, 41),
    (41, 42),

    # wrist -> middle
    (21, 43),
    (43, 44),
    (44, 45),

    # wrist -> pinky
    (21, 46),
    (46, 47),
    (47, 48),

    # wrist -> ring
    (21, 49),
    (49, 50),
    (50, 51),

    # wrist -> thumb
    (21, 52),
    (52, 53),
    (53, 54),
]


ps.init()
ps.set_build_gui(False)
ps.register_surface_mesh("mesh", vertices, faces,
                         color=[0, 91 / 255, 255 / 255],
                         edge_width=0.003,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         transparency=0.5
                         # material="flat"
                         )

ps.register_point_cloud("j", joints, radius=0.005, color=[1, 0, 0])
ps.register_curve_network("bone", joints, bones, radius=0.0015,
                          color=[255 / 255, 0, 218 / 255])

# ps.set_view_projection_mode("perspective")  # orthographic 正交投影   perspective 透视投影
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

![](assets/Pasted%20image%2020260929154512.png)


### 4.6 设置shape
```python
from body_models.smplx.torch import SMPLX
import torch
import numpy as np
import polyscope as ps

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

model = SMPLX(gender="female").to(device)  # 将SMPL模型放到GPU上
params = model.get_rest_pose(batch_dims=(1,))

# 设置shape参数
print(params["shape"])
print(params["shape"].shape)

params["shape"][0,0] = 1
params["shape"][0,1] = -2

print(params["shape"])

vertices = model.forward_vertices(**params)
skeleton = model.forward_skeleton(**params)
faces = model.faces

vertices = vertices[0].cpu().numpy()
faces = faces.cpu().numpy()

ps.init()
ps.set_build_gui(False)
ps.register_surface_mesh("mesh", vertices, faces,
                         color=[0, 91 / 255, 255 / 255],
                         edge_width=0.003,
                         edge_color=[1, 1, 1],
                         smooth_shade=True,
                         # material="flat"
                         )

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
tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], device='cuda:0')
torch.Size([1, 300])
tensor([[ 1., -2.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.]], device='cuda:0')
```

![](assets/Pasted%20image%2020260929154940.png)


### 4.7 生成不同shape

```python
from body_models.smplx.torch import SMPLX
import torch
import numpy as np
import polyscope as ps

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

model = SMPLX(gender="female").to(device)  # 将SMPL模型放到GPU上
params = model.get_rest_pose(batch_dims=(1,))

ps.init()
ps.set_build_gui(False)

# params["shape"] 为 torch.zeros((1,300),dtype=torch.int64, device=device)
for i in range(7):
    params["shape"][0, 1] = i - 3

    vertices = model.forward_vertices(**params)
    skeleton = model.forward_skeleton(**params)
    faces = model.faces

    vertices = vertices[0].cpu().numpy()
    faces = faces.cpu().numpy()

    vertices = vertices + np.array([0, 0, i * 0.5])

    ps.register_surface_mesh("mesh_{}".format(i - 3), vertices, faces,
                             color=[0, 91 / 255, 255 / 255],
                             edge_width=0.003,
                             edge_color=[1, 1, 1],
                             smooth_shade=True,
                             # material="flat"
                             )

ps.set_view_projection_mode("orthographic")  # orthographic 正交投影   perspective 透视投影
# ps.set_navigation_style("planar")  # ['turntable','free','planar','none','first_person']
ps.set_ground_plane_mode("shadow_only")  # ['none','tile','tile_reflection','shadow_only']
# ps.set_ground_plane_height(-0.001)  # 设置地平面高度
ps.set_shadow_blur_iters(3)  # 设置地平面阴影模糊程度
ps.set_shadow_darkness(0.5)  # 设置地平面阴影明暗程度
ps.set_up_dir("y_up")  # 设置y轴正方向向上（这个和设置视角会冲突，因此在添加视角参数时，这一行要注释掉）
ps.set_front_dir('neg_x_front')  # 设置z轴正方向向前
ps.set_SSAA_factor(4)  # 在截图时，设置为4时，截图会更清晰
ps.show()

```



![](assets/Pasted%20image%2020260929155245.png)


## 5 SMPL家族人体说明

> **SMPL = 身体基础版**  
> **SMPL-H = SMPL + 高质量双手**  
> **SMPL-X = 身体 + 双手 + 脸 + 表情的统一模型**  
> **STAR = SMPL 的改进版，重点不是增加身体部位，而是改善身体形变模型、降低参数量和提高局部形变合理性。**

### 5.1 总体对比

|模型|身体|手|脸/下颌|表情|主要特点|
|---|---|---|---|---|---|
|**SMPL**|✅|❌|❌|❌|最经典、最简单、生态最大|
|**SMPL-H / SMPL+H**|✅|✅|❌|❌|SMPL + MANO 双手|
|**SMPL-X**|✅|✅|✅|✅|完整表达人体|
|**STAR**|✅|❌|❌|❌|SMPL 的形变改进版|

其中 SMPL-X 有 **10,475 个顶点、54 个关节**，包括手指、下颌和眼球等；STAR 则定位为 SMPL 的 drop-in replacement。 GitHub+1


### 5.2 SMPL、SMPLH、SMPLX、STAR

### 5.3 基本结构

SMPL 可以理解成：

```
                    SMPL
                     │
          ┌──────────┴──────────┐
          │                     │
        Shape                  Pose
        β                      θ
          │                     │
          └──────────┬──────────┘
                     ↓
                  3D Body
```



SMPL-H 可以理解为：

```
             SMPL-H
                │
       ┌────────┴────────┐
       │                 │
     SMPL              MANO
    身体                双手
```

它的核心思想非常简单：

> **SMPL 的身体 + MANO 的手。**

MANO 本身就是专门为手建模的参数化模型，SMPL-H 将 MANO 手连接到 SMPL 身体上。



SMPL-X 是这几个模型里面**表达能力最强的一个**。
可以理解成：

```
                    SMPL-X
                       │
       ┌───────────────┼───────────────┐
       │               │               │
     Body             Hands           Face
       │               │               │
     SMPL             MANO           FLAME
```

SMPL-X 的目标就是统一：

```
身体
+
双手
+
脸
+
表情
```


SMPL-X大概可以理解成：

```
                SMPL-X
                   │
       ┌───────────┼────────────┐
       │           │            │
     Shape        Pose       Expression
       │           │            │
     β            θ            ψ
       │           │            │
       └───────────┼────────────┘
                   ↓
              Full Human
```




STAR 和 SMPL-H、SMPL-X 有一个非常重要的区别。

**STAR 并不是“SMPL + 某个身体部位”。**

而是：

> **重新设计 SMPL 的 deformation model。**

所以：

```
SMPL
  │
  │ 改进形变方式
  ↓
STAR
```

STAR 全称：

> Sparse Trained Articulated Human Body Regressor

它的核心问题是 SMPL 的 **pose corrective blend shapes**。

SMPL 使用比较密集的 pose corrective：

```
一个关节动作
        ↓
可能影响大量甚至远距离的 vertices
```

这可能产生一些不自然的远距离形变。

STAR 改成了：

```
Joint
  ↓
局部 sparse corrective
  ↓
附近 vertices
```

也就是：

> **每个关节主要影响它附近的网格区域。**

STAR 官方介绍中还特别强调了它使用稀疏、空间局部的 corrective blend shapes，并将参数量降低到 SMPL 的约 20%。

STAR 的另一个重要改进
SMPL 大致是：

```
Shape deformation
+
Pose deformation
```

两者相对独立。

STAR 则进一步考虑：

```
Shape
   +
Pose
   ↓
Pose-dependent deformation
```

也就是说：
> **不同体型的人，在同一个姿势下，肌肉/软组织的形变可以不同。**

例如：

```
瘦人弯手肘
      ↓
肘部形变 A

胖人弯手肘
      ↓
肘部形变 B
```

STAR 引入了 shape-dependent pose corrective，并使用更丰富的扫描数据训练 shape space。 



 Mesh 数量也有明显区别

| 模型     | Mesh规模    | 重点         |
| ------ | --------- | ---------- |
| SMPL   | 6,890     | 身体         |
| SMPL-H | 6890      | 身体 + 手     |
| SMPL-X | **10475** | 身体 + 手 + 脸 |
| STAR   | 6890      | 更紧凑的身体模型   |

SMPL-X 的 10,475 vertices 和 54 joints 是官方明确给出的。 GitHub



我建议你不要把它们理解成四个完全独立的模型，而是理解成：

```
                       Parametric Human Models
                                │
                ┌───────────────┴───────────────┐
                │                               │
           Body Modeling                  Expressiveness
                │                               │
        ┌───────┴───────┐                 SMPL-X
        │               │
      SMPL             STAR
        │
        │
      + MANO
        │
        ↓
      SMPL-H
```


### 5.4 总结
1. **SMPL解决如何用少量参数描述人体身体？**
2. **SMPLH解决如何在 SMPL 身体上加入真实的手指运动？**
3. **SMPL-X解决如何统一描述身体 + 手 + 脸 + 表情？**
4. **STAR解决如何让 SMPL 的身体形变更加局部、紧凑、真实？**

