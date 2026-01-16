# 已知问题：CUDA Engine Error (RuntimeError: GET was unable to find an engine...)

## 状态：待解决 (To Be Fixed)

## 问题描述
在使用 `EconomicGrasp` 进行推理时，对于某些特定物体（通常是细长物体如"线"、"香蕉"），Pipeline 可能会崩溃并报错：
```
RuntimeError: GET was unable to find an engine to execute this computation
```
报错通常发生在 `modules_economicgrasp.py` 的卷积层 (`Conv1d`) 中。

## 根本原因 (Root Cause)
该错误是由于 **有效点数过少或为零** 导致的底层 CuDNN 算子输入形状异常。

`EconomicGrasp` 模型内部有两个筛选步骤：
1. **Objectness Mask**: 筛选属于物体的点。
2. **Graspness Mask**: 筛选抓取置信度高于 `graspness_threshold` 的点 (默认 0.1)。

对于某些难以检测或几何特征不明显的物体，所有点可能都会被这两个 Mask 过滤掉。当有效点数为 0 时，后续的 `furthest_point_sample` (最远点采样) 会产生无效索引或空张量，导致传递给后续卷积层的 Input Shape 非法，从而触发 CUDA 引擎错误。

## 临时方案 (Workaround)
代码中（`economic_grasp/models/economicgrasp.py`）已加入防崩溃逻辑：
- 当检测到有效点数为 0 时，强制回退使用原始点云的前 N 个点或随机点进行填充。
- **注意**：这只是为了防止程序崩溃。此时生成的抓取结果是基于随机/未筛选点的，**质量可能极不可靠**。

## 建议优化方向
1. **自适应阈值**：如果 `graspness_threshold` 过滤后点数过少，尝试动态降低阈值，而不是使用硬阈值。
2. **Top-K 策略**：总是保留得分最高的 $K$ 个点，而不是只保留大于阈值的点。
3. **模型微调**：针对细长或低置信度物体进行针对性训练。
