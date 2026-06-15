# -*- coding: utf-8 -*-
"""
实时网络抓包特征提取模块
从实时网络流量中提取 NSL-KDD 兼容特征，送入模型预测
"""
import time
import threading
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from datetime import datetime

try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False


class PacketFeatureExtractor:
    """从抓包数据中提取NSL-KDD兼容的41维特征"""

    # 常见端口到service的映射
    PORT_SERVICE_MAP = {
        20: 'ftp_data', 21: 'ftp', 22: 'ssh', 23: 'telnet', 25: 'smtp',
        53: 'domain_u', 67: 'dhcp', 68: 'dhcp', 69: 'tftp_u', 80: 'http',
        110: 'pop_3', 111: 'auth', 119: 'nntp', 123: 'ntp_u', 135: 'microsoft_ds',
        137: 'netbios_ns', 138: 'netbios_dgm', 139: 'netbios_ssn', 143: 'imap4',
        161: 'snmp', 162: 'snmp', 443: 'http_443', 445: 'microsoft_ds',
        514: 'shell', 515: 'printer', 993: 'imaps', 995: 'pop3s',
        1080: 'socks', 1433: 'sql_net', 1521: 'oracle', 2049: 'efs',
        3306: 'mysql', 3389: 'remote_job', 5432: 'postgresql', 5900: 'vmnet',
        6000: 'X11', 6667: 'IRC', 8080: 'http', 8443: 'http_443',
    }

    def __init__(self, window_size=2):
        self.window_size = window_size
        self.packets = deque()
        self.lock = threading.Lock()
        self.connection_stats = defaultdict(lambda: {
            'count': 0, 'srv_count': 0, 'src_bytes': 0, 'dst_bytes': 0,
            'serror': 0, 'rerror': 0, 'same_srv': 0, 'diff_srv': 0,
            'dst_host_count': 0, 'dst_host_srv_count': 0,
            'services': set(), 'flags': [],
        })
        self.current_window = []
        self.total_packets = 0
        self.start_time = time.time()

    def _get_service(self, pkt):
        """根据端口识别服务"""
        if TCP in pkt:
            sport = pkt[TCP].sport
            dport = pkt[TCP].dport
        elif UDP in pkt:
            sport = pkt[UDP].sport
            dport = pkt[UDP].sport
        else:
            return 'other'
        port = dport if dport < 65535 else sport
        return self.PORT_SERVICE_MAP.get(port, 'other')

    def _get_protocol(self, pkt):
        """获取协议类型"""
        if TCP in pkt:
            return 'tcp'
        elif UDP in pkt:
            return 'udp'
        elif ICMP in pkt:
            return 'icmp'
        return 'other'

    def _get_flag(self, pkt):
        """获取TCP标志"""
        if TCP not in pkt:
            return 'OTH'
        flags = pkt[TCP].flags
        if flags == 0x02: return 'S0'      # SYN only
        elif flags == 0x12: return 'S1'    # SYN+ACK
        elif flags == 0x10: return 'ACK'   # ACK only
        elif flags == 0x04: return 'RSTO'  # RST
        elif flags == 0x14: return 'RSTR'  # RST+ACK
        elif flags == 0x01: return 'REJ'   # FIN
        elif flags == 0x11: return 'SH'    # FIN+ACK
        elif flags == 0x00: return 'S2'    # NULL
        elif flags == 0x29: return 'S3'    # FIN+PSH+URG
        else: return 'OTH'

    def packet_callback(self, pkt):
        """抓包回调，每个包调用一次"""
        if not (IP in pkt):
            return

        with self.lock:
            self.total_packets += 1
            now = time.time()

            pkt_info = {
                'time': now,
                'src': pkt[IP].src,
                'dst': pkt[IP].dst,
                'protocol': self._get_protocol(pkt),
                'service': self._get_service(pkt),
                'flag': self._get_flag(pkt),
                'src_bytes': len(pkt),
                'dst_bytes': 0,
                'is_tcp': TCP in pkt,
                'is_udp': UDP in pkt,
                'is_icmp': ICMP in pkt,
            }
            self.current_window.append(pkt_info)

    def extract_features(self):
        """从当前时间窗口提取NSL-KDD格式的41维特征"""
        with self.lock:
            if not self.current_window:
                return None

            pkts = list(self.current_window)
            self.current_window.clear()

        n = len(pkts)
        now = time.time()

        # 基础统计
        protocols = [p['protocol'] for p in pkts]
        services = [p['service'] for p in pkts]
        flags = [p['flag'] for p in pkts]
        src_bytes_list = [p['src_bytes'] for p in pkts]
        dst_bytes_list = [p['dst_bytes'] for p in pkts]

        # 连接级别统计
        src_ips = [p['src'] for p in pkts]
        dst_ips = [p['dst'] for p in pkts]
        conns = list(zip(src_ips, dst_ips))

        unique_src_dst = set(conns)
        count = n
        srv_count = sum(1 for s in services if s == services[0] if services else 0)

        # 错误率
        serror_count = sum(1 for f in flags if f in ['S0', 'REJ', 'RSTO', 'RSTR'])
        rerror_count = sum(1 for f in flags if f in ['REJ', 'RSTO'])

        same_srv = sum(1 for s in services if s == (services[0] if services else '')) / max(n, 1)
        diff_srv = len(set(services)) / max(n, 1)

        # 目标主机统计
        dst_host = dst_ips[0] if dst_ips else ''
        dst_host_count = len(set(src_ips))
        dst_host_srv = sum(1 for p in pkts if p['dst'] == dst_host and p['service'] == (services[0] if services else ''))
        dst_host_same = dst_host_srv / max(dst_host_count, 1)

        # 持续时间
        duration = pkts[-1]['time'] - pkts[0]['time'] if n > 1 else 0

        # TCP flags 统计
        syn_count = sum(1 for f in flags if f in ['S0', 'S1'])
        fin_count = sum(1 for f in flags if f in ['SH', 'REJ'])
        rst_count = sum(1 for f in flags if f in ['RSTO', 'RSTR'])

        features = {
            'duration': duration,
            'protocol_type': protocols[0] if protocols else 'tcp',
            'service': services[0] if services else 'other',
            'flag': flags[0] if flags else 'OTH',
            'src_bytes': sum(src_bytes_list),
            'dst_bytes': sum(dst_bytes_list),
            'land': 1 if src_ips and dst_ips and src_ips[0] == dst_ips[0] else 0,
            'wrong_fragment': 0,
            'urgent': 0,
            'hot': 0,
            'num_failed_logins': 0,
            'logged_in': 1 if syn_count > 0 and rst_count == 0 else 0,
            'num_compromised': 0,
            'root_shell': 0,
            'su_attempted': 0,
            'num_root': 0,
            'num_file_creations': 0,
            'num_shells': 0,
            'num_access_files': 0,
            'count': count,
            'srv_count': srv_count,
            'serror_rate': serror_count / max(n, 1),
            'srv_serror_rate': serror_count / max(n, 1),
            'rerror_rate': rerror_count / max(n, 1),
            'srv_rerror_rate': rerror_count / max(n, 1),
            'same_srv_rate': same_srv,
            'diff_srv_rate': diff_srv,
            'srv_diff_host_rate': len(set(dst_ips)) / max(n, 1),
            'dst_host_count': min(dst_host_count, 255),
            'dst_host_srv_count': min(dst_host_srv, 255),
            'dst_host_same_srv_rate': dst_host_same,
            'dst_host_diff_srv_rate': diff_srv,
            'dst_host_serror_rate': serror_count / max(n, 1),
            'dst_host_srv_serror_rate': serror_count / max(n, 1),
            'dst_host_rerror_rate': rerror_count / max(n, 1),
            'dst_host_srv_rerror_rate': rerror_count / max(n, 1),
        }

        # 附加元数据（不参与模型，用于前端展示）
        meta = {
            'packet_count': n,
            'unique_src_ips': len(set(src_ips)),
            'unique_dst_ips': len(set(dst_ips)),
            'top_src_ip': max(set(src_ips), key=src_ips.count) if src_ips else '-',
            'top_dst_ip': max(set(dst_ips), key=dst_ips.count) if dst_ips else '-',
            'top_service': max(set(services), key=services.count) if services else '-',
            'top_protocol': max(set(protocols), key=protocols.count) if protocols else '-',
            'syn_count': syn_count,
            'rst_count': rst_count,
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'total_bytes': sum(src_bytes_list) + sum(dst_bytes_list),
        }

        return features, meta

    def get_stats(self):
        """获取总统计"""
        with self.lock:
            elapsed = time.time() - self.start_time
            return {
                'total_packets': self.total_packets,
                'uptime': f"{elapsed:.0f}s",
                'pps': f"{self.total_packets / max(elapsed, 1):.0f}",
                'window_packets': len(self.current_window),
            }


class RealtimeDetector:
    """实时抓包检测引擎"""

    def __init__(self, predictor, window_size=2, capture_filter=None):
        """
        predictor: IntrusionDetector实例
        window_size: 分析窗口(秒)
        capture_filter: BPF过滤器 (如 'tcp port 80')
        """
        self.predictor = predictor
        self.extractor = PacketFeatureExtractor(window_size=window_size)
        self.window_size = window_size
        self.capture_filter = capture_filter
        self.running = False
        self.history = deque(maxlen=100)
        self.thread = None

    def start(self, interface=None):
        """启动抓包线程"""
        if not SCAPY_AVAILABLE:
            raise RuntimeError("scapy未安装，无法抓包")

        self.running = True

        def _sniff():
            try:
                sniff(
                    iface=interface,
                    prn=self.extractor.packet_callback,
                    filter=self.capture_filter,
                    store=False,
                    stop_filter=lambda x: not self.running,
                )
            except Exception as e:
                print(f"抓包错误: {e}")

        self.thread = threading.Thread(target=_sniff, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def analyze(self, model_name='random_forest'):
        """分析当前窗口流量"""
        result = self.extractor.extract_features()
        if result is None:
            return None

        features, meta = result
        predictions = self.predictor.predict(features, model_name)

        if predictions:
            pred = predictions[0]
            record = {**pred, **meta}
            self.history.append(record)
            return record
        return None

    def get_history(self):
        return list(self.history)

    def get_stats(self):
        return self.extractor.get_stats()
