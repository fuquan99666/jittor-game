# new/ — 点云去噪重构版 + DSM 改进

原 `milestone/` 下 PyTorch 点云去噪代码的重构版本。目标：**结构清晰、命名直观、隐式约定显式化**。

包含两个模型：
- **VelocityModule**（原版重构）：监督位移向量，Langevin 4 步去噪
- **ScoreModule**（新实现）：完整 Denoising Score Matching，退火 Langevin 多级去噪

所有命令在 `new/` 目录下执行。数据集放在 `milestone/`（路径用 `../milestone/dataset_train`、`../milestone/datalist`）。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 调试: 跑一轮 dataloader 打印 Asset 形状(不训练)
python run.py --task configs/task/debug.yaml --device cpu

# --- VelocityModule (原版重构) ---
# 训练(每轮存 experiments/vm/checkpoint_<epoch>.pt)
python run.py --task configs/task/train.yaml --device cuda:0
# 推理(输出 results/<...>/denoised.npy)
python run.py --task configs/task/predict.yaml --device cuda:0

# --- ScoreModule (DSM 改进, 推荐) ---
# 训练(每轮存 experiments/score/checkpoint_<epoch>.pt)
python run.py --task configs/task/train_score.yaml --device cuda:0
# 推理
python run.py --task configs/task/predict_score.yaml --device cuda:0

# 评测
python evaluate.py --pred_dir results --gt_dir <gt> --noisy_dir <noisy> [--mesh_dir <mesh>]
```

---

## 文件说明

### 入口

| 文件 | 作用 |
|------|------|
| `run.py` | 统一入口。按 task YAML 的 `mode`(train/predict/debug) 组装 data→transform→model→system 并执行。开头设置线程环境变量(必须在 import torch 前)。predict 模式会断言 `predict_transform` 为空。 |
| `evaluate.py` | 评测脚本(独立单文件)。算 Chamfer Distance 和 Point-to-Surface，评分 `100*(1-pred/noisy)`。无 mesh 时只算 CD。多进程并行。 |
| `requirements.txt` | 依赖：torch, numpy, trimesh, scipy, omegaconf, tqdm, point-cloud-utils(可选, P2S 精确计算需要)。 |

### src/data/ — 数据层

| 文件 | 作用 |
|------|------|
| `asset.py` | `Asset` 数据载体(贯穿 dataloader→augment→model) + `Exporter`(写 obj)。`Asset.transform()` 对全部坐标字段施加仿射变换。 |
| `spec.py` | `ConfigSpec` 基类：可从 YAML 解析的 dataclass 基类，`check_keys()` 拒绝未知字段。 |
| `datapath.py` | 数据路径管理 + 惰性加载。`ObjLazyAsset`/`NpyLazyAsset` 只存路径，访问时才读文件。`Datapath` 支持顺序遍历(`use_prob=False`)和按类别权重概率采样(`use_prob=True`)。 |
| `dataset.py` | `PCDataset`(torch Dataset) + `PCDatasetModule`(管理 train/validate/predict 三套 dataloader)。`prepare_batch()` 把 Asset 列表 collate 成 tensor batch(stack/cat/non 三类字段)。 |
| `transform.py` | `Transform`：有序 augment 容器，`apply()` 依次执行。 |
| `augment.py` | 各类 Augment 实现 + 工厂 `get_augments()`。基类统一提供 `parse()`，子类只需声明字段 + 实现 `apply()`。支持 `register_augment()` 扩展。 |
| `sampling.py` | 纯 numpy 的几何运算：网格表面面积加权采样、重心坐标插值、随机欧拉旋转、单位球归一化。集中在此方便日后替换成高性能库。 |

### src/model/ — 模型层

| 文件 | 作用 |
|------|------|
| `spec.py` | `ModelSpec` 基类(nn.Module)：管理 config、checkpoint、train/predict 切换。`process_fn_wrapper()` 在非训练模式把原始 Asset 挂到 `non["asset"]`(writer 需要它恢复输出路径)。 |
| `parse.py` | 模型工厂：`__target__` 查表实例化。`register_model()` 扩展。 |
| `layers.py` | 网络层：`pairwise_squared_distance`、`EdgeConv`/`DynamicEdgeConv`、`FeatureExtraction`(DGCNN 风格三层 EdgeConv)、`Decoder`(特征→3维位移)。 |
| `ops.py` | 点云几何运算的 torch 实现：`farthest_point_sampling`、`knn_query`、`patch_based_denoise`(FPS切patch→去噪→加权拼回)。推理用。 |
| `vm.py` | `VelocityModule`：原版模型。训练时在 noisy/clean 间随机插值构造混合点云，监督预测位移方向(MSE)。推理时 Langevin 动力学迭代 4 步。 |
| `score.py` | `ScoreModule`：**DSM 改进版**。训练时学得分函数 s(x,σ)≈-ε/σ，推理时退火 Langevin 多级 σ 迭代去噪。复用 FeatureExtraction/Decoder，新增 sigma_encoder 注入噪声级别。 |

### src/system/ — 系统层(训练/推理循环)

| 文件 | 作用 |
|------|------|
| `spec.py` | `BaseSystem`(训练/推理循环主体) + `BaseWriter`(输出基类) + `get_optimizer()` 工厂。一个 epoch = 训练→(可选)验证→存 checkpoint。 |
| `parse.py` | system/writer 工厂。`register` 扩展。 |
| `vm.py` | `VMSystem`(目前是 BaseSystem 别名，预留扩展) + `VMWriter`(把去噪结果按测试集目录布局写成 npy/obj)。 |

### configs/ — 配置(OmegaConf)

| 路径 | 作用 |
|------|------|
| `task/*.yaml` | 任务入口：设 `mode` + `components` + optimizer/loss/trainer/writer/load_ckpt。`train_score.yaml`/`predict_score.yaml` 是 DSM 版。 |
| `data/*.yaml` | 数据集配置：路径、batch_size、num_workers、采样方式。`train.yaml` 用概率采样(每 epoch 10000 batch)。 |
| `transform/vm.yaml` | VelocityModule 的 augment 链(sample→normalize→noise→patch)。**`predict_transform` 必须为空**。 |
| `transform/score.yaml` | ScoreModule 的 augment 链(sample→normalize→score_perturb)。用高斯噪声 + 得分目标替代 patch 插值。 |
| `model/vm.yaml` | VelocityModule 超参。 |
| `model/score.yaml` | ScoreModule 超参。含退火 Langevin 的 σ 序列和步数。 |
| `system/vm.yaml` | checkpoint 存 `experiments/vm/`。 |
| `system/score.yaml` | checkpoint 存 `experiments/score/`。 |

### scripts/

| 文件 | 作用 |
|------|------|
| `build_datalists.py` | 生成 `../milestone/datalist/{train,validate,test}.txt`。仅换数据集或重新划分时运行。默认 seed=123, val_ratio=0.05。 |

---

## 与原版的区别(重构要点)

1. **命名**：`DummySystem/DummyWriter` → `BaseSystem/BaseWriter`(原名误导，以为是"假的")。
2. **去样板**：Augment 子类的 `parse()` 统一提到基类，子类只声明字段 + `apply()`。
3. **显式校验**：predict 模式下断言 `predict_transform` 为空(原版无校验，写错会静默产出垃圾)。
4. **去硬编码**：`predict_step` 里的 `patch_size/seed_k/seed_k_alpha/langevin_steps` 从代码提到 `model/vm.yaml`。
5. **性能**：`num_workers` 从 0 提到 4(train)/2(val/predict)，缓解 IO 瓶颈。
6. **文件拆分**：原 `utils.py` 拆成 `sampling.py`(几何运算)；原 `feature.py` 拆成 `layers.py`(网络层) + `ops.py`(点云操作)。
7. **工厂可扩展**：`register_model/register_augment/register_system` 对外暴露，加新组件无需改源码映射表。
8. **注释**：关键算法(velocity field 思想、Langevin 迭代、patch 拼接权重、坐标轴翻转)都加了说明。

---

## 架构一览

```
run.py
  │  加载 task.yaml -> 按 components 加载各层 config
  │  __target__ 工厂查表实例化
  ▼
PCDatasetModule ──train_dataloader──> Asset 列表
  │                                       │ transform.apply() (sample→normalize→noise→patch)
  │                                       ▼
  └─prepare_batch ──> model.process_fn_wrapper(Asset→dict) ──> tensor batch
                          │
                          ▼
                   model.training_step (encoder→decoder→MSE loss)
                   model.predict_step  (FPS切patch→Langevin 4步→拼回)
                          │
                          ▼
                   VMWriter.write ──> results/<...>/denoised.npy
```

**核心算法**：把去噪建模成 velocity field。模型学习从噪声点指向干净点的位移向量。训练时在 noisy/clean 之间随机插值，让模型预测 `pc_clean - pc_noisy`。推理时迭代 `pc = pc + v(pc)/steps` 跑 4 步(Langevin 动力学的离散化)。本质是 score-based denoising 的简化形式。

---

## 后续可优化的降噪方法(比当前 velocity field 更好)

当前方法是把"位移向量"当监督目标，是 score-based 方法的简化。以下是公认更好且代码量适中的方向，按推荐度排序：

### 1. 完整的 Denoising Score Matching (DSM) ★★★ 已实现

**状态**：已实现为 `ScoreModule`（`src/model/score.py`），用 `train_score.yaml` / `predict_score.yaml` 训练推理。

**原理**：不直接学位移，而是学噪声扰动后点云的"得分函数" ∇log p(x)。推理时用退火 Langevin 动力学从大 σ 到小 σ 多轮迭代去噪。当前代码的 `velocity ≈ score * σ` 已是雏形，升级成本低。

**为什么更好**：
- 学的是概率密度梯度，理论上更通用，能处理多模态分布(同一噪声点可能对应多个干净点)。
- 退火(Langevin on decreasing σ)能处理不同噪声级别，鲁棒性更强。

**实现细节**（本仓库）：
- 新增 `AugmentScorePerturb`（`augment.py`）：在 clean 点上加高斯噪声 σε，计算得分目标 -ε/σ
- 新增 `ScoreModule`（`score.py`）：复用 FeatureExtraction + Decoder，加 `sigma_encoder` 把 σ 编码后注入点特征
- 训练目标：`||s_θ(x+σε, σ) - (-ε/σ)||²`
- 推理：退火 Langevin，σ 序列 [0.02, 0.01, 0.005, 0.0025]，每级 5 步
- 复用现有 `patch_based_denoise` 做 patch 切分

**用法**：
```bash
python run.py --task configs/task/train_score.yaml --device cuda:0
python run.py --task configs/task/predict_score.yaml --device cuda:0
```

**调参建议**：
- `sigma_min/sigma_max`（transform）：应覆盖测试集的实际噪声范围
- `predict_sigmas`（model）：从大到小，最后一级应接近最小噪声
- `predict_steps_per_sigma`：越多越精细但越慢
- `langevin_step_scale`：0.5 默认，收敛慢则调大，不稳定则调小

### 2. PointCleanNet (两阶段) ★★★ 推荐

**原理**：ECCV 2020，分两步：(1) 估计每个点的噪声大小(标量)；(2) 估计每个点的位移向量。用剔除法剔除离群点后再回归位置。

**为什么更好**：
- 显式区分离群点(outlier)和正常噪声，对离群点鲁棒(当前方法不处理离群)。
- 两阶段解耦，每个子网络更简单、更易训练。

**实现要点**：
- 第一阶段：Decoder 输出 1 维(噪声分数)，用 max-pool 聚合(当前 `Decoder` 已预留 `out_dim==1` 分支)。
- 第二阶段：Decoder 输出 3 维位移(就是当前实现)。
- 开源参考：https://github.com/ryanlsen/PointCleanNet

**代码量**：新增一个子网络 + 改 predict_step 串两阶段，约 100 行。

### 3. Deep Prior Optimization (DPO) ★★ 无需训练

**原理**：不训练任何模型。对每个输入点云，随机初始化一个网络，让它"过拟合"输出该点云，但加正则约束输出平滑。过拟合过程中网络会先学到结构、再学到噪声，提前停止就得到去噪结果。

**为什么好**：
- **零训练成本**，拿到新数据立刻能用。
- 不需要配对训练数据。
- 适合作为 baseline 或快速验证。

**实现要点**：
- 复用现有 `FeatureExtraction` + `Decoder`，去掉训练循环，改成对单个输入做梯度下降。
- 损失 = 重建误差 + 平滑正则(如 Laplacian)。
- 参考：Deep Prior of Point Clouds (CVPR 2021)。

**代码量**：新增一个 `deep_prior_denoise.py`，约 80 行，不动现有训练代码。

### 4. 经典非学习方法(快速 baseline) ★★ 一行调用

用 `point-cloud-utils` 直接调，无需训练，适合做对比基线或快速预处理：

| 方法 | 接口 | 特点 |
|------|------|------|
| Bilateral Filter | `pcu.bilateral_filter_point_cloud` | 保边去噪，类似图像双边滤波 |
| Jet Smoothing | `pcu.smooth_point_cloud_pca` / jet fit | 局部多项式拟合投影，几何保真 |
| WLOP | `pcu.wlop` | 局部最优投影，均匀化+去噪 |

**建议**：先用这些跑出 baseline 分数，再训练学习类方法对比。

### 5. IterativePFN (迭代去噪网络) ★

**原理**：训练一个轻量网络，迭代应用多次(类似 UNet 的深度可交换)。每次细化一点。

**为什么好**：参数少、推理快、效果稳定。但需要重新设计训练流程，代码改动较大。

---

### 优化路线建议

1. **先跑经典方法**(Bilateral/Jet)拿到 baseline，10 分钟出结果。
2. **升级当前 velocity → 完整 DSM**(改动最小，理论提升最明显)。
3. **加 PointCleanNet 两阶段**处理离群点(如果测试集有离群点的话)。
4. **DPO 作为补充**，对单个难样本做无训练精修。

---

## 常见坑

- **predict_transform 不能加 augment**：测试集已是 `noisy.npy`，再采样/归一化/加噪会破坏数据。`run.py` 有断言。
- **checkpoint 文件名 0-indexed**：`epochs: 100` → 最终 `checkpoint_99.pt`。改 epochs 要同步改 `predict.yaml` 的 `load_ckpt`。
- **从 `new/` 目录运行**：config 里路径是 `../milestone/dataset_train`、`../milestone/datalist`，从别的目录跑会 `FileNotFoundError`。
- **`OMP_NUM_THREADS=1` 等**：在 `run.py` 最前面设置，必须在 `import torch` 前，否则 numpy/torch 线程冲突。
