# -*- coding: utf-8 -*-
"""
AI入侵检测系统 - Streamlit前端 (含实时抓包)
运行: streamlit run frontend/app.py
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.utils.predictor import IntrusionDetector, LABEL_MAP, LABEL_CN
from src.utils.realtime_capture import RealtimeDetector, SCAPY_AVAILABLE

st.set_page_config(page_title="AI入侵检测系统", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.risk-safe { color: green; font-weight: bold; font-size: 20px; }
.risk-warning { color: orange; font-weight: bold; font-size: 20px; }
.risk-danger { color: red; font-weight: bold; font-size: 20px; }
.risk-critical { color: darkred; font-weight: bold; font-size: 20px; }
.stMetric .stMetricValue { font-size: 28px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    return IntrusionDetector(model_dir='models')

def get_dataset_info():
    try:
        with open('models/dataset_info.json', 'r') as f:
            return json.load(f)
    except:
        return None

PRESETS = {
    "正常Web浏览流量": {
        'duration': 50, 'protocol_type': 'tcp', 'service': 'http', 'flag': 'S1',
        'src_bytes': 800, 'dst_bytes': 2000, 'land': 0, 'wrong_fragment': 0,
        'urgent': 0, 'hot': 0, 'num_failed_logins': 0, 'logged_in': 1,
        'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0,
        'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0,
        'count': 30, 'srv_count': 25, 'serror_rate': 0.01, 'srv_serror_rate': 0.01,
        'rerror_rate': 0.02, 'srv_rerror_rate': 0.02, 'same_srv_rate': 0.95,
        'diff_srv_rate': 0.03, 'srv_diff_host_rate': 0.1,
        'dst_host_count': 100, 'dst_host_srv_count': 60,
        'dst_host_same_srv_rate': 0.85, 'dst_host_diff_srv_rate': 0.05,
        'dst_host_serror_rate': 0.01, 'dst_host_srv_serror_rate': 0.01,
        'dst_host_rerror_rate': 0.02, 'dst_host_srv_rerror_rate': 0.02
    },
    "Neptune DOS攻击": {
        'duration': 0, 'protocol_type': 'tcp', 'service': 'private', 'flag': 'S0',
        'src_bytes': 0, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0,
        'urgent': 0, 'hot': 0, 'num_failed_logins': 0, 'logged_in': 0,
        'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0,
        'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0,
        'count': 511, 'srv_count': 511, 'serror_rate': 1.0, 'srv_serror_rate': 1.0,
        'rerror_rate': 0.0, 'srv_rerror_rate': 0.0, 'same_srv_rate': 1.0,
        'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0,
        'dst_host_count': 255, 'dst_host_srv_count': 255,
        'dst_host_same_srv_rate': 1.0, 'dst_host_diff_srv_rate': 0.0,
        'dst_host_serror_rate': 1.0, 'dst_host_srv_serror_rate': 1.0,
        'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0
    },
    "Smurf DOS攻击": {
        'duration': 0, 'protocol_type': 'icmp', 'service': 'ecr_i', 'flag': 'SF',
        'src_bytes': 1032, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0,
        'urgent': 0, 'hot': 0, 'num_failed_logins': 0, 'logged_in': 0,
        'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0,
        'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0,
        'count': 511, 'srv_count': 511, 'serror_rate': 0.0, 'srv_serror_rate': 0.0,
        'rerror_rate': 0.0, 'srv_rerror_rate': 0.0, 'same_srv_rate': 1.0,
        'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0,
        'dst_host_count': 255, 'dst_host_srv_count': 255,
        'dst_host_same_srv_rate': 1.0, 'dst_host_diff_srv_rate': 0.0,
        'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0,
        'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0
    },
    "端口扫描探测": {
        'duration': 0, 'protocol_type': 'tcp', 'service': 'private', 'flag': 'REJ',
        'src_bytes': 0, 'dst_bytes': 0, 'land': 0, 'wrong_fragment': 0,
        'urgent': 0, 'hot': 0, 'num_failed_logins': 0, 'logged_in': 0,
        'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0,
        'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0,
        'count': 229, 'srv_count': 10, 'serror_rate': 0.0, 'srv_serror_rate': 0.0,
        'rerror_rate': 1.0, 'srv_rerror_rate': 1.0, 'same_srv_rate': 0.04,
        'diff_srv_rate': 0.06, 'srv_diff_host_rate': 0.0,
        'dst_host_count': 255, 'dst_host_srv_count': 1,
        'dst_host_same_srv_rate': 0.0, 'dst_host_diff_srv_rate': 0.06,
        'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0,
        'dst_host_rerror_rate': 1.0, 'dst_host_srv_rerror_rate': 1.0
    },
    "暴力破解": {
        'duration': 120, 'protocol_type': 'tcp', 'service': 'ftp', 'flag': 'SF',
        'src_bytes': 200, 'dst_bytes': 800, 'land': 0, 'wrong_fragment': 0,
        'urgent': 0, 'hot': 3, 'num_failed_logins': 4, 'logged_in': 0,
        'num_compromised': 0, 'root_shell': 0, 'su_attempted': 0, 'num_root': 0,
        'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 1,
        'count': 50, 'srv_count': 40, 'serror_rate': 0.0, 'srv_serror_rate': 0.0,
        'rerror_rate': 0.0, 'srv_rerror_rate': 0.0, 'same_srv_rate': 0.9,
        'diff_srv_rate': 0.05, 'srv_diff_host_rate': 0.05,
        'dst_host_count': 100, 'dst_host_srv_count': 50,
        'dst_host_same_srv_rate': 0.8, 'dst_host_diff_srv_rate': 0.1,
        'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0,
        'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0
    },
    "Root提权攻击": {
        'duration': 300, 'protocol_type': 'tcp', 'service': 'telnet', 'flag': 'SF',
        'src_bytes': 5000, 'dst_bytes': 10000, 'land': 0, 'wrong_fragment': 0,
        'urgent': 1, 'hot': 8, 'num_failed_logins': 0, 'logged_in': 1,
        'num_compromised': 3, 'root_shell': 1, 'su_attempted': 1, 'num_root': 8,
        'num_file_creations': 5, 'num_shells': 1, 'num_access_files': 3,
        'count': 10, 'srv_count': 8, 'serror_rate': 0.0, 'srv_serror_rate': 0.0,
        'rerror_rate': 0.0, 'srv_rerror_rate': 0.0, 'same_srv_rate': 0.95,
        'diff_srv_rate': 0.02, 'srv_diff_host_rate': 0.02,
        'dst_host_count': 50, 'dst_host_srv_count': 25,
        'dst_host_same_srv_rate': 0.9, 'dst_host_diff_srv_rate': 0.02,
        'dst_host_serror_rate': 0.0, 'dst_host_srv_serror_rate': 0.0,
        'dst_host_rerror_rate': 0.0, 'dst_host_srv_rerror_rate': 0.0
    }
}


def show_dataset_overview():
    st.header("📊 数据集概览")
    info = get_dataset_info()
    if info:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("数据集", info.get('name', 'Unknown'))
        col2.metric("训练样本", f"{info.get('train_samples', 0):,}")
        col3.metric("测试样本", f"{info.get('test_samples', 0):,}")
        col4.metric("特征数", info.get('features', 0))
        st.subheader("类别分布")
        cats = info.get('categories', {})
        if cats:
            st.bar_chart(pd.Series(cats))
    else:
        st.info("请先运行训练脚本生成数据集信息")


def show_realtime_detection():
    st.header("🔍 实时流量检测")
    st.markdown("输入网络流量特征值，选择模型实时判断是否为攻击流量")
    try:
        detector = load_detector()
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return
    model_name = st.selectbox("选择检测模型",
        ['random_forest', 'gradient_boosting', 'mlp'],
        format_func=lambda x: {'random_forest': '🌲 随机森林',
                               'gradient_boosting': '🚀 梯度提升',
                               'mlp': '🧠 神经网络(MLP)'}[x])
    st.subheader("快速测试 (选择预设场景)")
    scenario = st.selectbox("选择场景", ["自定义输入"] + list(PRESETS.keys()))
    feature_names = detector.feature_names
    defaults = PRESETS.get(scenario, {})
    with st.form("detection_form"):
        st.subheader("流量特征参数")
        cols = st.columns(4)
        user_input = {}
        for i, feat in enumerate(feature_names):
            with cols[i % 4]:
                default_val = defaults.get(feat, 0.0)
                if isinstance(default_val, str):
                    user_input[feat] = st.text_input(feat, value=default_val, key=feat)
                else:
                    user_input[feat] = st.number_input(
                        feat, value=float(default_val),
                        step=0.01, format="%.4f", key=feat)
        submitted = st.form_submit_button("🔍 开始检测", use_container_width=True)
    if submitted:
        with st.spinner("AI模型分析中..."):
            result = detector.predict(user_input, model_name)[0]
        st.markdown("---")
        st.subheader("检测结果")
        col1, col2, col3 = st.columns(3)
        with col1:
            risk_map = {'green': 'risk-safe', 'orange': 'risk-warning',
                       'red': 'risk-danger', 'darkred': 'risk-critical'}
            rc = risk_map.get(result.get('risk_color', 'green'), 'risk-safe')
            st.markdown(f'<p class="{rc}">检测结果: {result["label_cn"]}</p>',
                       unsafe_allow_html=True)
        with col2:
            st.metric("风险等级", result['risk_level'])
        with col3:
            if 'confidence' in result:
                st.metric("置信度", f"{result['confidence']:.1%}")
        if 'probabilities' in result:
            st.subheader("各类别概率")
            prob_df = pd.DataFrame({
                '类别': [LABEL_CN.get(k, k) for k in result['probabilities']],
                '概率': list(result['probabilities'].values())
            })
            st.bar_chart(prob_df.set_index('类别'))


def show_realtime_capture():
    st.header("🔴 实时网络抓包检测")
    st.markdown("抓取本机网络流量，AI实时分析是否存在攻击行为")

    if not SCAPY_AVAILABLE:
        st.error("scapy未安装，请运行: pip install scapy")
        return

    try:
        detector = load_detector()
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return

    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("控制面板")
        model_name = st.selectbox("模型",
            ['random_forest', 'gradient_boosting', 'mlp'],
            format_func=lambda x: {'random_forest': '🌲 RF',
                                   'gradient_boosting': '🚀 GB',
                                   'mlp': '🧠 MLP'}[x],
            key='rt_model')

        window = st.slider("分析窗口(秒)", 1, 10, 2)
        bpf_filter = st.text_input("BPF过滤器", value="",
                                    placeholder="如: tcp port 80")

        if 'rt_detector' not in st.session_state:
            st.session_state.rt_detector = None
            st.session_state.rt_running = False

        start_btn = st.button("▶ 开始抓包", use_container_width=True, disabled=st.session_state.get('rt_running', False))
        stop_btn = st.button("⏹ 停止", use_container_width=True, disabled=not st.session_state.get('rt_running', False))

        if start_btn:
            rt = RealtimeDetector(detector, window_size=window, capture_filter=bpf_filter or None)
            try:
                rt.start()
                st.session_state.rt_detector = rt
                st.session_state.rt_running = True
                st.success("抓包已启动!")
                st.rerun()
            except Exception as e:
                st.error(f"启动失败: {e}")

        if stop_btn and st.session_state.rt_detector:
            st.session_state.rt_detector.stop()
            st.session_state.rt_running = False
            st.success("已停止")
            st.rerun()

    with col2:
        if st.session_state.get('rt_running') and st.session_state.rt_detector:
            rt = st.session_state.rt_detector
            stats = rt.get_stats()
            st.metric("总包数", stats['total_packets'])
            st.metric("速率", f"{stats['pps']} pps")

            # Auto refresh
            time.sleep(window + 0.5)
            result = rt.analyze(model_name)

            if result:
                risk_map = {'green': 'safe', 'orange': 'warning', 'red': 'danger', 'darkred': 'critical'}
                st.subheader(f"最新检测: {result['timestamp']}")
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("包数", result.get('packet_count', 0))
                col_b.metric("源IP数", result.get('unique_src_ips', 0))
                col_c.metric("检测结果", result.get('label_cn', '-'))
                col_d.metric("置信度", f"{result.get('confidence', 0):.1%}")

            # 历史记录
            history = rt.get_history()
            if history:
                st.subheader("检测历史")
                hist_df = pd.DataFrame(history)[['timestamp', 'label_cn', 'risk_level', 'packet_count',
                                                  'top_src_ip', 'top_dst_ip', 'top_service', 'confidence']]
                st.dataframe(hist_df.tail(20), use_container_width=True)
            else:
                st.info("等待流量数据...")
        elif not st.session_state.get('rt_running'):
            st.info("点击左侧「开始抓包」启动实时检测")
            st.markdown("""
            **使用说明:**
            1. 选择检测模型和分析窗口
            2. 可选填BPF过滤器 (如 `tcp port 80` 只抓HTTP)
            3. 点击开始抓包，系统自动分析网络流量
            4. 检测结果实时显示，攻击流量会高亮告警

            **注意:** 需要管理员权限运行抓包
            """)


def show_batch_detection():
    st.header("📁 批量流量检测")
    st.markdown("上传CSV文件，批量检测网络流量")
    try:
        detector = load_detector()
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        return
    model_name = st.selectbox("选择模型",
        ['random_forest', 'gradient_boosting', 'mlp'],
        format_func=lambda x: {'random_forest': '🌲 随机森林',
                               'gradient_boosting': '🚀 梯度提升',
                               'mlp': '🧠 MLP'}[x], key='batch_model')
    uploaded = st.file_uploader("上传CSV文件", type=['csv'])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(), use_container_width=True)
        if st.button("开始批量检测", use_container_width=True):
            with st.spinner("分析中..."):
                results = detector.predict(df, model_name)
            result_df = pd.DataFrame(results)
            col1, col2, col3, col4 = st.columns(4)
            total = len(results)
            attacks = sum(1 for r in results if r['prediction'] != 'normal')
            col1.metric("总流量", total)
            col2.metric("攻击流量", attacks)
            col3.metric("正常流量", total - attacks)
            col4.metric("攻击比例", f"{attacks/total:.1%}")
            st.subheader("检测结果分布")
            st.bar_chart(result_df['label_cn'].value_counts())
            st.subheader("详细结果")
            display_df = df.copy()
            display_df['检测结果'] = result_df['label_cn']
            display_df['风险等级'] = result_df['risk_level']
            st.dataframe(display_df, use_container_width=True)
            csv = display_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("下载检测结果", csv, "detection_results.csv", "text/csv")


def show_model_evaluation():
    st.header("📈 模型评估报告")
    if not os.path.exists('models/metrics.json'):
        st.warning("模型未训练，请先运行训练脚本")
        return
    with open('models/metrics.json') as f:
        metrics = json.load(f)
    model_names = {'random_forest': '🌲 随机森林', 'gradient_boosting': '🚀 梯度提升', 'mlp': '🧠 MLP'}
    cols = st.columns(3)
    for i, (name, m) in enumerate(metrics.items()):
        with cols[i]:
            st.markdown(f"### {model_names.get(name, name)}")
            st.metric("准确率", f"{m['accuracy']:.2%}")
            st.metric("训练耗时", f"{m['time']:.1f}s")
    st.subheader("模型准确率对比")
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [model_names.get(n, n) for n in metrics.keys()]
    accs = [m['accuracy'] for m in metrics.values()]
    bars = ax.barh(names, accs, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax.set_xlim(0.8, 1.0)
    ax.set_xlabel("Accuracy")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() - 0.05, bar.get_y() + bar.get_height()/2,
                f'{acc:.2%}', ha='center', va='center', color='white', fontweight='bold')
    st.pyplot(fig)


def show_about():
    st.header("ℹ️ 系统说明")
    info = get_dataset_info()
    ds_name = info.get('name', 'NSL-KDD') if info else 'NSL-KDD'
    st.markdown(f"""
    ### AI入侵检测系统

    **数据集:** {ds_name} (真实网络流量数据)

    **技术栈:**
    - 机器学习: scikit-learn (随机森林、梯度提升、MLP)
    - 实时抓包: scapy (BPF过滤、特征提取)
    - 数据处理: pandas, numpy
    - 前端: Streamlit

    **功能模块:**
    | 模块 | 说明 |
    |------|------|
    | 📊 数据集概览 | 查看训练数据分布和统计 |
    | 🔍 实时检测 | 输入特征值，手动测试 |
    | 🔴 实时抓包 | 本机网络流量实时抓取+AI检测 |
    | 📁 批量检测 | 上传CSV批量分析 |
    | 📈 模型评估 | 多模型性能对比 |

    **检测能力:**
    | 类型 | 说明 | 风险等级 |
    |------|------|----------|
    | 正常流量 | 合法网络访问 | 安全 |
    | DOS攻击 | 拒绝服务攻击 (Neptune/Smurf等) | 高危 |
    | Probe探测 | 端口扫描 (Nmap/Portsweep等) | 中危 |
    | R2L远程攻击 | 远程权限获取 (FTP写入/密码猜测等) | 高危 |
    | U2R提权攻击 | 本地用户提权为root | 严重 |

    **使用方法:**
    ```bash
    python src/training/train_nsl_kdd.py   # 训练真实数据模型
    streamlit run frontend/app.py           # 启动前端
    ```

    **项目地址:** [GitHub](https://github.com/wulongovo/ai-intrusion-detection)
    """)


def main():
    st.title("🛡️ AI入侵检测系统")
    info = get_dataset_info()
    ds = info.get('name', '') if info else ''
    st.markdown(f"基于{'真实' if 'KDD' in ds or 'CIC' in ds else '模拟'}网络流量的机器学习入侵检测 | 5类流量识别 | 实时抓包")
    st.sidebar.title("功能导航")
    page = st.sidebar.radio("选择功能", [
        "📊 数据集概览", "🔍 实时检测", "🔴 实时抓包检测",
        "📁 批量检测", "📈 模型评估", "ℹ️ 系统说明"
    ])
    if page == "📊 数据集概览":
        show_dataset_overview()
    elif page == "🔍 实时检测":
        show_realtime_detection()
    elif page == "🔴 实时抓包检测":
        show_realtime_capture()
    elif page == "📁 批量检测":
        show_batch_detection()
    elif page == "📈 模型评估":
        show_model_evaluation()
    elif page == "ℹ️ 系统说明":
        show_about()

if __name__ == '__main__':
    main()
