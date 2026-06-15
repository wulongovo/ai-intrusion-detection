# 🛡️ AI入侵检测系统 (AI Intrusion Detection System)

基于机器学习的网络流量异常检测系统，使用 **NSL-KDD 真实数据集** 训练，支持 **服务器日志入侵检测**。

## 📋 功能特性

| 模块 | 说明 |
|------|------|
| 📊 数据集概览 | NSL-KDD数据分布可视化 |
| 🔍 实时检测 | 手动输入特征值，AI判断是否为攻击 |
| 🖥️ 服务器日志检测 | **上传SSH/Nginx/防火墙日志，自动检测入侵** |
| 📁 批量检测 | CSV批量分析+结果导出 |
| 📈 模型评估 | 三种模型性能对比 |

## 🖥️ 服务器日志检测能力

| 攻击类型 | 检测方法 | 风险等级 |
|----------|----------|----------|
| SSH暴力破解 | 同一IP失败≥5次 | 🔴 高危 |
| SQL注入 | union select等关键词 | 🔴 高危 |
| XSS攻击 | script标签等 | 🔴 高危 |
| 目录遍历 | ../等路径穿越 | 🔴 高危 |
| CC攻击 | 单IP请求≥1000次 | 🔴 高危 |
| 端口扫描 | 多端口探测 | 🔴 高危 |
| Root直接登录 | auth.log检测 | 🟡 中危 |
| 提权尝试 | sudo危险命令 | ⚫ 严重 |
| WebShell | eval/base64等 | 🔴 高危 |
| 扫描器识别 | nikto/sqlmap等UA | 🔴 高危 |

## 🧠 模型 (NSL-KDD)

| 模型 | 准确率 | 训练耗时 |
|------|--------|----------|
| 随机森林 | 75.47% | 0.5s |
| 梯度提升 | 75.23% | 2.8s |
| MLP神经网络 | 74.82% | 3.6s |

## 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 下载NSL-KDD数据集 (可从GitHub下载)
# 放入 data/ 目录

# 训练模型
python src/training/train_nsl_kdd.py

# 启动前端
streamlit run frontend/app.py
```

### 服务器日志检测使用
```bash
# 从服务器下载日志
scp root@your-server:/var/log/auth.log ./auth.log

# 打开前端 → 选择「服务器日志检测」→ 上传日志 → 自动分析
streamlit run frontend/app.py
```

## 📁 项目结构

```
ai-intrusion-detection/
├── data/samples/              # 测试日志样本
│   ├── sample_auth.log        # SSH攻击测试样本
│   └── sample_nginx.log       # Web攻击测试样本
├── frontend/
│   └── app.py                 # Streamlit前端 (7个功能页)
├── src/
│   ├── generate_data.py       # 模拟数据生成
│   ├── training/
│   │   └── train_nsl_kdd.py   # NSL-KDD训练脚本
│   └── utils/
│       ├── predictor.py       # 模型预测器
│       ├── realtime_capture.py # 实时抓包模块
│       └── server_log_analyzer.py # 服务器日志分析
├── requirements.txt
└── README.md
```

## 🛠️ 技术栈

Python / scikit-learn / Streamlit / scapy / pandas / matplotlib

## 📄 License

MIT License
