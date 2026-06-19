# core/ad_checker.py
import numpy as np
import pandas as pd
import os
import onnxruntime as ort

class ApplicabilityDomainChecker:
    """TransIA 适用域（AD）判别器（ONNX 运行时推理版本）"""
    
    def __init__(self, model_path='data_processed/ad_checker.onnx'):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到 AD 判别器 ONNX 模型: {model_path}")
        
        # 限制线程数以避免 gunicorn 多 worker 下线程过度订阅
        intra = int(os.getenv("ORT_INTRA_OP_NUM_THREADS", "1"))
        inter = int(os.getenv("ORT_INTER_OP_NUM_THREADS", "1"))
        
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(intra, 1)
        so.inter_op_num_threads = max(inter, 1)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.sess = ort.InferenceSession(
            model_path,
            sess_options=so,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name
        
        # 从模型元数据加载常量
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.h_star = float(meta["h_star"])
        self.mz_mean = float(meta["mz_mean"])
        self.mz_std = float(meta["mz_std"])
        self.max_intensity_mz_mean = float(meta["max_intensity_mz_mean"])
        self.max_intensity_mz_std = float(meta["max_intensity_mz_std"])
        
        self.nonafi_characteristic_peaks = [
            58.0651, 72.0808, 84.0808, 99.0917, 113.1073,
            135.0441, 147.0077, 151.0866, 166.0975, 169.076,
            197.0709, 250.0863, 256.0955, 262.0862, 283.1195,
            297.1346, 299.1139, 302.0812, 312.1581, 315.091,
            327.1274, 341.1608, 354.2, 377.1, 396.203
        ]
        
        self.peak_groups = {
            'low_mass': [58.0651, 72.0808, 84.0808, 99.0917, 113.1073],
            'middle_mass': [135.0441, 147.0077, 151.0866, 166.0975, 169.076, 197.0709],
            'high_mass': [250.0863, 256.0955, 262.0862, 283.1195, 297.1346, 299.1139, 302.0812],
            'very_high_mass': [312.1581, 315.091, 327.1274, 341.1608, 354.2, 377.1, 396.203]
        }
        
        self.key_peaks = [58.0651, 72.0808, 135.0441, 166.0975, 250.0863]
        self.max_nodes = 10
        self.node_dim = 10

    def read_excel_to_ms_string(self, file_path):
        """读取 Excel 文件并转换为质谱字符串格式"""
        if not os.path.exists(file_path):
            if not file_path.endswith(('.xlsx', '.xls')):
                test_path = file_path + '.xlsx'
                if os.path.exists(test_path):
                    file_path = test_path
                else:
                    test_path = file_path + '.xls'
                    if os.path.exists(test_path):
                        file_path = test_path
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")
        
        df = pd.read_excel(file_path, header=None)
        
        if df.shape[1] < 2:
            raise ValueError(f"需要至少2列数据，当前只有 {df.shape[1]} 列")
        
        if df.shape[0] == 0:
            raise ValueError("文件为空")
        
        peaks = []
        for _, row in df.iterrows():
            try:
                mz = float(row[0])
                intensity = float(row[1])
                if intensity > 0:
                    peaks.append(f"{mz:.4f}:{intensity:.2f}")
            except (ValueError, TypeError):
                continue
        
        if not peaks:
            raise ValueError("没有有效的峰数据")
        
        return ",".join(peaks)
    
    def parse_ms_string(self, ms_str):
        """解析质谱字符串"""
        if not ms_str or ms_str == '':
            return np.zeros((self.max_nodes, self.node_dim)), np.eye(self.max_nodes)
        
        peaks_data = []
        for peak in ms_str.split(','):
            try:
                parts = peak.split(':')
                if len(parts) >= 2:
                    mz = float(parts[0].strip())
                    intensity = float(parts[1].strip())
                    peaks_data.append((mz, intensity))
            except:
                continue
        
        if not peaks_data:
            return np.zeros((self.max_nodes, self.node_dim)), np.eye(self.max_nodes)
        
        peaks_data.sort(key=lambda x: x[1], reverse=True)
        peaks_data = peaks_data[:self.max_nodes]
        
        if len(peaks_data) < self.max_nodes and len(peaks_data) > 0:
            last_peak = peaks_data[-1]
            while len(peaks_data) < self.max_nodes:
                peaks_data.append(last_peak)
        
        mz_values = [p[0] for p in peaks_data]
        max_intensity_mz = peaks_data[0][0] if peaks_data else 0
        
        node_features = np.zeros((self.max_nodes, self.node_dim))
        for j in range(self.max_nodes):
            if j < len(peaks_data):
                mz = peaks_data[j][0]
            else:
                mz = 0
            features = self._compute_node_features(mz, j, len(peaks_data), mz_values, max_intensity_mz)
            node_features[j, :] = features
        
        adjacency_matrix = self._build_adjacency_matrix(mz_values)
        
        return node_features, adjacency_matrix
    
    def _compute_node_features(self, mz, position, total_peaks, all_mz_values, max_intensity_mz):
        """计算10维节点特征"""
        mz_norm = (mz - self.mz_mean) / self.mz_std if self.mz_std and self.mz_std > 0 else 0.0
        
        position_ratio = position / max(total_peaks, 1)
        is_first_peak = 1.0 if position == 0 else 0.0
        is_last_peak = 1.0 if position == total_peaks - 1 else 0.0
        
        rounded_mz = round(mz, 1)
        rounded_characteristic = [round(p, 1) for p in self.nonafi_characteristic_peaks]
        is_characteristic = 1.0 if rounded_mz in rounded_characteristic else 0.0
        
        if self.nonafi_characteristic_peaks:
            min_diff = min([abs(mz - cp) for cp in self.nonafi_characteristic_peaks])
            characteristic_mz_diff = min_diff / 100.0
        else:
            characteristic_mz_diff = 1.0
        
        mass_region_feature = 0.0
        for group_name, group_peaks in self.peak_groups.items():
            group_rounded = [round(p, 1) for p in group_peaks]
            if rounded_mz in group_rounded:
                mass_region_feature = {
                    'low_mass': 0.25,
                    'middle_mass': 0.5,
                    'high_mass': 0.75,
                    'very_high_mass': 1.0
                }[group_name]
                break
        
        rounded_key_peaks = [round(p, 1) for p in self.key_peaks]
        is_key_peak = 1.0 if rounded_mz in rounded_key_peaks else 0.0
        
        if self.max_intensity_mz_std is not None and self.max_intensity_mz_std > 0:
            max_intensity_mz_norm = (max_intensity_mz - self.max_intensity_mz_mean) / self.max_intensity_mz_std
        else:
            max_intensity_mz_norm = 0.0
        mz_relative_to_max = mz / max_intensity_mz if max_intensity_mz > 0 else 1.0
        
        features = [
            mz_norm,
            position_ratio,
            is_first_peak,
            is_last_peak,
            is_characteristic,
            characteristic_mz_diff,
            mass_region_feature,
            is_key_peak,
            max_intensity_mz_norm,
            mz_relative_to_max
        ]
        
        return features
    
    def _build_adjacency_matrix(self, mz_values):
        """构建邻接矩阵"""
        adj_matrix = np.eye(self.max_nodes)
        n_nodes = len(mz_values)
        
        for i in range(min(n_nodes, self.max_nodes)):
            for j in range(min(n_nodes, self.max_nodes)):
                if i != j:
                    mz_diff = abs(mz_values[i] - mz_values[j])
                    similarity = np.exp(-mz_diff ** 2 / (2 * 50.0 ** 2))
                    adj_matrix[i, j] = similarity
        
        return adj_matrix
    
    def extract_feature_vector(self, ms_str):
        """提取完整特征向量"""
        node_features, adj_matrix = self.parse_ms_string(ms_str)
        return np.concatenate([node_features.flatten(), adj_matrix.flatten()])
    
    def check_ad_from_file(self, file_path, verbose=True):
        """从 Excel 文件读取质谱并使用 ONNX 模型检查是否在适用域内"""
        ms_str = self.read_excel_to_ms_string(file_path)
        peaks = ms_str.split(',')
        
        feature_vector = self.extract_feature_vector(ms_str).astype(np.float32)
        
        # ONNX 推理计算 leverage
        feed = {self.input_name: feature_vector.reshape(1, -1)}
        out = self.sess.run([self.output_name], feed)
        leverage = float(np.asarray(out[0]).reshape(-1)[0])
        
        within_ad = leverage <= self.h_star
        
        if within_ad:
            confidence_level = 'High'
            message = '该谱图在模型适用域内，预测结果可信度高。'
        elif leverage <= 1.5 * self.h_star:
            confidence_level = 'Medium'
            message = '该谱图处于适用域边缘，预测结果仅供参考，建议结合MS2碎片手动核实。'
        else:
            confidence_level = 'Out of AD'
            message = '警告：该谱图超出模型适用域，预测结果不确定性高，不建议作为监管依据。'
        
        result = {
            'within_ad': bool(within_ad),
            'leverage': round(float(leverage), 6),
            'h_star': float(self.h_star),
            'confidence_level': confidence_level,
            'message': message,
            'num_peaks': len(peaks)
        }
        
        return result


def load_or_train_ad_checker(
    model_path='data_processed/ad_checker.onnx',
    train_data_path=None,  # 未使用，保留参数以兼容旧调用
    h_star=0.11
) -> ApplicabilityDomainChecker:
    """加载 TransIA 适用域（AD）判别器 ONNX 模型"""
    # 兼容处理：如果传入 pkl 路径，自动映射为 onnx 路径
    if model_path.endswith('.pkl'):
        model_path = model_path.replace('.pkl', '.onnx')
    return ApplicabilityDomainChecker(model_path=model_path)
