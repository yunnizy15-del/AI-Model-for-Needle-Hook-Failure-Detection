# 针钩失效判断模型说明

本项目基于摩擦系数 CSV 数据训练二分类模型：

- `valid/*.csv` 对应标签 `valid`（正常）
- `invalid/*.csv` 对应标签 `invalid`（失效）

每个 CSV 文件视为一个样本。程序会从 `mu_true` 序列中提取时序特征，并使用随机森林进行训练。

## 1) 训练模型

```powershell
python train_model.py
```

输出文件：

- `model/needle_hook_model.joblib`
- `model/metrics.json`

可选参数示例：

```powershell
python train_model.py --valid-dir valid --invalid-dir invalid --test-size 0.2 --n-estimators 500
```

## 2) 预测单个 CSV

```powershell
python predict_model.py --input valid\000001.csv
```

## 3) 预测整个文件夹

```powershell
python predict_model.py --input invalid --output-csv model\invalid_predictions.csv
```

## 4) GUI 界面（论文图表导出）

```powershell
python gui_app.py
```

如果 GUI 启动时报缺少 `Tcl/Tk`，请安装带 `tkinter` 组件的 Python 版本。

GUI 支持：

- 可配置训练参数
- 单文件或目录批量预测
- 自动导出论文常用图表
- 在界面内预览导出的图表

训练阶段图表：

- 类别分布图
- 混淆矩阵图
- ROC 曲线
- PR 曲线
- 特征重要性 Top 图
- 测试集失效概率分布图
- OOB 训练过程误差曲线（可选）

预测阶段图表：

- 预测类别计数图
- 失效概率直方图
- 高风险样本 Top20 柱状图
- `mu_mean` 与 `mu_std` 散点图
- 风险排序曲线
- 高风险样本信号曲线（`t_s` vs `mu_true`）

## 5) 数据格式要求

- CSV 必须包含列：`t_s`、`mu_true`
- 即使个别文件长度略短，程序仍可完成特征提取与预测
