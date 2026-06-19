# core/ad_checker.py
import numpy as np
import typing
import pandas as pd
import pickle
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class ADDataPreprocessor:
    """与训练代码完全一致的数据预处理器"""
    
    def __init__(self, max_nodes=10, node_dim=10):
        self.max_nodes = max_nodes
        self.node_dim = node_dim
        
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
        
        self.mz_mean = None
        self.mz_std = None
        self.max_intensity_mz_mean = None
        self.max_intensity_mz_std = None
        
        self.graph_data = []
        # 初始化样本标签和训练/验证/测试集索引，使用 typing.Any 绕过 sklearn 返回类型不确定的 assignment issue
        self.labels: np.ndarray = np.array([], dtype=int)
        self.train_indices: typing.Any = None
        self.val_indices: typing.Any = None
        self.test_indices: typing.Any = None
    
    def load_and_preprocess_data(self, file_path):
        """加载并预处理数据"""
        print(f"加载训练数据: {file_path}")
        df = pd.read_excel(file_path)
        
        labels_cleaned = []
        for val in df['label']:
            if pd.isna(val):
                labels_cleaned.append(0)
            else:
                str_val = str(val).strip().lower()
                if str_val in ['0', '0.0', '非那非', '否', 'negative', 'n', 'false', 'f', '非']:
                    labels_cleaned.append(0)
                elif str_val in ['1', '1.0', '那非', '是', 'positive', 'y', 'true', 't', '是']:
                    labels_cleaned.append(1)
                else:
                    try:
                        num_val = float(str_val)
                        labels_cleaned.append(int(num_val) if num_val in [0, 1] else 0)
                    except:
                        labels_cleaned.append(0)
        
        self.labels = np.array(labels_cleaned, dtype=int)
        print(f"标签分布: 那非={sum(self.labels)}, 非那非={len(self.labels)-sum(self.labels)}")
        
        all_mz = []
        all_max_intensity_mz = []
        
        for i, row in df.iterrows():
            ms_str = str(row['MS'])
            if pd.isna(ms_str) or ms_str == 'nan' or ms_str.strip() == '':
                continue
            
            peaks = ms_str.split(',')
            max_intensity = -1
            max_intensity_mz = 0
            
            for peak in peaks:
                try:
                    parts = peak.split(':')
                    if len(parts) >= 2:
                        mz = float(parts[0].strip())
                        intensity = float(parts[1].strip())
                        all_mz.append(mz)
                        if intensity > max_intensity:
                            max_intensity = intensity
                            max_intensity_mz = mz
                except:
                    continue
            
            if max_intensity > 0:
                all_max_intensity_mz.append(max_intensity_mz)
        
        if all_mz:
            self.mz_mean = np.mean(all_mz)
            self.mz_std = np.std(all_mz) if np.std(all_mz) > 0 else 1
        else:
            self.mz_mean = 0
            self.mz_std = 1
        
        if all_max_intensity_mz:
            self.max_intensity_mz_mean = np.mean(all_max_intensity_mz)
            self.max_intensity_mz_std = np.std(all_max_intensity_mz) if np.std(all_max_intensity_mz) > 0 else 1
        else:
            self.max_intensity_mz_mean = 0
            self.max_intensity_mz_std = 1
        
        print(f"m/z 均值: {self.mz_mean:.2f}, 标准差: {self.mz_std:.2f}")
        print(f"最大强度 m/z 均值: {self.max_intensity_mz_mean:.2f}, 标准差: {self.max_intensity_mz_std:.2f}")
        
        print("构建图表示...")
        # 使用 enumerate 以确保索引 i 具有明确的 int 类型
        for i, (_, row) in enumerate(df.iterrows()):
            if i % 1000 == 0 and i > 0:
                print(f"  已处理 {i}/{len(df)} 个样本")
            
            ms_str = str(row['MS'])
            if pd.isna(ms_str) or ms_str == 'nan' or ms_str.strip() == '':
                node_features = np.zeros((self.max_nodes, self.node_dim))
                adjacency_matrix = np.eye(self.max_nodes)
            else:
                node_features, adjacency_matrix = self._parse_ms_string(ms_str)
            
            graph_info = {
                'node_features': node_features,
                'adjacency_matrix': adjacency_matrix
            }
            self.graph_data.append(graph_info)
        
        print(f"数据处理完成! 构建了 {len(self.graph_data)} 个图")
        return df
    
    def _parse_ms_string(self, ms_str):
        """解析质谱字符串"""
        peaks = ms_str.split(',')
        peak_data = []
        max_intensity = -1
        max_intensity_mz = 0
        
        for peak in peaks:
            try:
                peak = peak.strip()
                if not peak:
                    continue
                parts = peak.split(':')
                if len(parts) >= 2:
                    mz = float(parts[0].strip())
                    intensity = float(parts[1].strip())
                    if intensity > max_intensity:
                        max_intensity = intensity
                        max_intensity_mz = mz
                    peak_data.append((mz, intensity))
                elif len(parts) == 1:
                    mz = float(parts[0].strip())
                    intensity = 1.0
                    if intensity > max_intensity:
                        max_intensity = intensity
                        max_intensity_mz = mz
                    peak_data.append((mz, intensity))
            except:
                continue
        
        peak_data.sort(key=lambda x: x[1], reverse=True)
        peak_data = peak_data[:self.max_nodes]
        
        if len(peak_data) < self.max_nodes and len(peak_data) > 0:
            last_peak = peak_data[-1]
            while len(peak_data) < self.max_nodes:
                peak_data.append(last_peak)
        
        mz_values = [p[0] for p in peak_data]
        
        node_features = np.zeros((self.max_nodes, self.node_dim))
        for j in range(self.max_nodes):
            if j < len(peak_data):
                mz = peak_data[j][0]
            else:
                mz = 0
            features = self._compute_node_features(mz, j, len(peak_data), mz_values, max_intensity_mz)
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
    
    def get_train_features(self, train_indices):
        """获取训练集特征矩阵"""
        X_train_raw = []
        for idx in train_indices:
            graph = self.graph_data[idx]
            node_flat = graph['node_features'].flatten()
            adj_flat = graph['adjacency_matrix'].flatten()
            feature_vec = np.concatenate([node_flat, adj_flat])
            X_train_raw.append(feature_vec)
        return np.array(X_train_raw)
    
    def split_data(self, test_size=0.2, val_size=0.2):
        """分割数据集"""
        n_samples = len(self.graph_data)
        indices = np.arange(n_samples)
        labels_for_split = self.labels.astype(int)
        
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            stratify=labels_for_split,
            random_state=42
        )
        
        val_ratio = val_size / (1 - test_size)
        train_val_labels = labels_for_split[train_val_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio,
            stratify=train_val_labels,
            random_state=42
        )
        
        self.train_indices = train_idx
        self.val_indices = val_idx
        self.test_indices = test_idx
        
        return self.train_indices, self.val_indices, self.test_indices


class ApplicabilityDomainChecker:
    """TransIA 适用域（AD）判别器"""
    
    def __init__(self, h_star=0.11):
        self.h_star = h_star
        self.pca = None
        self.X_train_pca = None
        self.inv_cov = None
        self.mz_mean = None
        self.mz_std = None
        self.max_intensity_mz_mean = None
        self.max_intensity_mz_std = None
        
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
    
    def fit_from_preprocessor(self, preprocessor):
        """从预处理器获取训练数据并拟合 PCA 模型"""
        X_train_raw = preprocessor.get_train_features(preprocessor.train_indices)
        
        self.mz_mean = preprocessor.mz_mean
        self.mz_std = preprocessor.mz_std
        self.max_intensity_mz_mean = preprocessor.max_intensity_mz_mean
        self.max_intensity_mz_std = preprocessor.max_intensity_mz_std
        
        self.pca = PCA(n_components=0.95)
        self.X_train_pca = self.pca.fit_transform(X_train_raw)
        
        XTX = self.X_train_pca.T @ self.X_train_pca
        self.inv_cov = np.linalg.pinv(XTX)
        
        return self
    
    def read_excel_to_ms_string(self, file_path):
        """
        读取 Excel 文件并转换为质谱字符串格式
        """
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
        
        # 提取 m/z 和 intensity，做 try-except 跳过非数值表头
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
        """
        从 Excel 文件读取质谱并检查是否在适用域内
        """
        if self.pca is None:
            raise ValueError("请先调用 fit_from_preprocessor() 方法")
        
        ms_str = self.read_excel_to_ms_string(file_path)
        peaks = ms_str.split(',')
        
        feature_vector = self.extract_feature_vector(ms_str)
        query_pca = self.pca.transform(feature_vector.reshape(1, -1)).flatten()
        leverage = query_pca @ self.inv_cov @ query_pca.T
        
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
    
    def save_model(self, filepath='ad_checker_model.pkl'):
        """保存模型"""
        model_data = {
            'pca': self.pca,
            'X_train_pca': self.X_train_pca,
            'inv_cov': self.inv_cov,
            'h_star': self.h_star,
            'mz_mean': self.mz_mean,
            'mz_std': self.mz_std,
            'max_intensity_mz_mean': self.max_intensity_mz_mean,
            'max_intensity_mz_std': self.max_intensity_mz_std
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)


def load_or_train_ad_checker(
    model_path='data_processed/ad_checker_model.pkl',
    train_data_path='data/化合物-7-1.xlsx',
    h_star=0.11
) -> ApplicabilityDomainChecker:
    """
    加载或训练 TransIA 适用域（AD）判别器
    """
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            checker = ApplicabilityDomainChecker(h_star=model_data.get('h_star', h_star))
            checker.pca = model_data['pca']
            checker.X_train_pca = model_data['X_train_pca']
            checker.inv_cov = model_data['inv_cov']
            checker.mz_mean = model_data['mz_mean']
            checker.mz_std = model_data['mz_std']
            checker.max_intensity_mz_mean = model_data['max_intensity_mz_mean']
            checker.max_intensity_mz_std = model_data['max_intensity_mz_std']
            return checker
        except Exception as e:
            print(f"警告: 加载 AD 模型失败 ({e})，将重新训练。")
    
    # 训练模型
    preprocessor = ADDataPreprocessor(max_nodes=10, node_dim=10)
    preprocessor.load_and_preprocess_data(train_data_path)
    preprocessor.split_data(test_size=0.2, val_size=0.2)
    
    checker = ApplicabilityDomainChecker(h_star=h_star)
    checker.fit_from_preprocessor(preprocessor)
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    checker.save_model(model_path)
    return checker
