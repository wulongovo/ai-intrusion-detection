# 🛡️ AI入侵检测系统 (AI Intrusion Detection System)

基于机器学习的网络流量异常检测系统，使用 **NSL-KDD 真实数据集** 训练，支持 5 类流量识别和 **实时网络抓包检测**。

## 📋 功能特性

| 模块 | 说明 |
|------|------|
| 📊 数据集概览 | 可视化查看 NSL-KDD 数据集分布和统计信息 |
| 🔍 实时检测 | 输入流量特征参数，手动测试模型判断结果 |
| 🔴 实时抓包 | 本机网络流量实时抓取 + AI 模型分析，攻击自动告警 |
| 📁 批量检测 | 上传 CSV 文件批量分析，结果导出下载 |
| 📈 模型评估 | 三种模型性能对比可视化 |

## 🎯 检测能力

| 类型 | 说明 | 典型攻击 | 风险等级 |
|------|------|----------|----------|
| 正常流量 | 合法网络访问 | Web浏览、邮件等 | ✅ 安全 |
| DOS攻击 | 拒绝服务攻击 | Neptune, Smurf, Back | 🔴 高危 |
| Probe探测 | 端口扫描、网络探测 | Nmap, Portsweep, Satan | 🟡 中危 |
| R2L远程攻击 | 远程权限获取 | FTP写入, 密码猜测 | 🔴 高危 |
| U2R提权攻击 | 本地用户提权为root | Buffer Overflow, Rootkit | ⚫ 严重 |

## 🧠 模型

基于 NSL-KDD 数据集训练了 3 个模型:

| 模型 | 准确率 | 训练耗时 | 说明 |
|------|--------|----------|------|
| 随机森林 (Random Forest) | 75.47% | 0.5s | 集成学习，快速稳定 |
| 梯度提升 (HistGradientBoosting) | 75.23% | 2.8s | 梯度提升树 |
| MLP神经网络 | 74.82% | 3.6s | 多层感知机 |

> 注：R2L 和 U2R 类别在 NSL-KDD 训练集中样本极少（分别为 115 和 5 条），检测难度大，这是该数据集的已知挑战。

## 🏗️ 技术栈

- **机器学习**: scikit-learn (RandomForest, HistGradientBoosting, MLP)
- **实时抓包**: scapy (BPF 过滤器, 特征提取)
- **数据处理**: pandas, numpy
- **可视化**: matplotlib, seaborn
- **前端**: Streamlit
- **数据集**: NSL-KDD (Canadian Institute for Cybersecurity)

## 📁 项目结构

```
ai-intrusion-detection/
├── data/
│   ├── KDDTrain+.txt              # NSL-KDD 训练集 (13,688条)
│   ├── KDDTest+.txt               # NSL-KDD 测试集 (5,334条)
│   ├── train.csv                  # 处理后训练数据
│   └── test.csv                   # 处理后测试数据
├── models/
│   ├── random_forest.pkl           # 随机森林模型
│   ├── gradient_boosting.pkl       # 梯度提升模型
│   ├── mlp.pkl                     # MLP神经网络
│   ├── scaler.pkl                  # 标准化器
│   ├── encoders.pkl                # 类别编码器
│   ├── feature_names.pkl           # 特征名列表
│   ├── metrics.json                # 模型评估指标
│   └── dataset_info.json           # 数据集信息
├── src/
│   ├── generate_data.py            # 模拟数据生成器
│   ├── training/
│   │   ├── train_models.py         # 模拟数据训练
│   │   └── train_nsl_kdd.py        # NSL-KDD 真实数据训练
│   └── utils/
│       ├── predictor.py            # 模型预测器
│       └── realtime_capture.py     # 实时抓包特征提取
├── frontend/
│   └── app.py                      # Streamlit 前端
├── requirements.txt
└── README.md
```

## 🚀 快速开始

### 环境要求
- Python 3.10+
- Windows/Linux/macOS

### 安装依赖
```bash
pip install scikit-learn pandas numpy matplotlib seaborn streamlit scapy joblib
```

### 运行步骤

```bash
# 1. 使用 NSL-KDD 真实数据集训练模型
python src/training/train_nsl_kdd.py

# 2. 启动前端 (自动打开浏览器)
streamlit run frontend/app.py
```

或者使用模拟数据:
```bash
python src/generate_data.py
python src/training/train_models.py
streamlit run frontend/app.py
```

### 实时抓包检测
1. 在前端页面选择「实时抓包检测」
2. 选择模型和分析窗口
3. 可选填 BPF 过滤器 (如 `tcp port 80`)
4. 点击开始抓包 (需要管理员/root权限)

## 📊 数据集说明

**NSL-KDD** 是网络安全领域最经典的基准数据集之一:
- 来源: Canadian Institute for Cybersecurity (CIC)
- 训练集: 13,688 条流量记录
- 测试集: 5,334 条流量记录
- 特征: 36 维数值特征 + 3 维类别特征
- 标签: 正常流量 + 4 大类攻击 (含 39 种攻击子类型)

## 📝 参考资料

- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html)
- [KDD Cup 1999](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- [A Detailed Analysis of the KDD CUP 99 Data Set](https://scholar.google.com/scholar?q=nsl+kdd+intrusion+detection)

## 📄 License

MIT License
