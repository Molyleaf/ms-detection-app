import numpy as np
import pandas as pd
import pickle
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class ADDataPreprocessor:
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
        self.labels = []
    
    def load_and_preprocess_data(self, file_path):
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
        for i, row in df.iterrows():
            if i % 500 == 0 and i > 0:
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
        X_train_raw = []
        for idx in train_indices:
            graph = self.graph_data[idx]
            node_flat = graph['node_features'].flatten()
            adj_flat = graph['adjacency_matrix'].flatten()
            feature_vec = np.concatenate([node_flat, adj_flat])
            X_train_raw.append(feature_vec)
        return np.array(X_train_raw)
    
    def split_data(self, test_size=0.2, val_size=0.2):
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
        
        print(f"\n数据分割:")
        print(f"  训练集: {len(self.train_indices)} 个样本")
        print(f"  验证集: {len(self.val_indices)} 个样本")
        print(f"  测试集: {len(self.test_indices)} 个样本")
        
        return self.train_indices, self.val_indices, self.test_indices


class ApplicabilityDomainChecker:
    def __init__(self, h_star=0.11):
        self.h_star = h_star
        self.pca = None
        self.X_train_pca = None
        self.inv_cov = None
        self.mz_mean = None
        self.mz_std = None
        self.max_intensity_mz_mean = None
        self.max_intensity_mz_std = None
    
    def fit_from_preprocessor(self, preprocessor):
        X_train_raw = preprocessor.get_train_features(preprocessor.train_indices)
        n = X_train_raw.shape[0]
        p = X_train_raw.shape[1]
        
        print(f"\n训练集特征矩阵:")
        print(f"  样本数 n: {n}")
        print(f"  原始特征维度 p: {p}")
        
        self.mz_mean = preprocessor.mz_mean
        self.mz_std = preprocessor.mz_std
        self.max_intensity_mz_mean = preprocessor.max_intensity_mz_mean
        self.max_intensity_mz_std = preprocessor.max_intensity_mz_std
        
        print(f"\n拟合 PCA 模型...")
        self.pca = PCA(n_components=0.95)
        self.X_train_pca = self.pca.fit_transform(X_train_raw)
        k = self.X_train_pca.shape[1]
        
        XTX = self.X_train_pca.T @ self.X_train_pca
        self.inv_cov = np.linalg.pinv(XTX)
        
        explained_variance = sum(self.pca.explained_variance_ratio_)
        
        # Calculate h_star dynamically: 3 * k / n
        self.h_star = 0.11
        
        print(f"  PCA 保留主成分数 k: {k}")
        print(f"  累计方差解释率: {explained_variance:.4f}")
        print(f"  Leverage 阈值 h*: {self.h_star:.6f}")
        
        return self
    
    def save_model(self, filepath='data_processed/ad_checker_model.pkl'):
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
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\nAD 判别器已保存到: {filepath}")

def main():
    print("=" * 80)
    print("TransIA 适用域（AD）判别器训练")
    print("=" * 80)
    
    TRAIN_DATA_FILE = "data/化合物-7-1.xlsx"
    preprocessor = ADDataPreprocessor(max_nodes=10, node_dim=10)
    preprocessor.load_and_preprocess_data(TRAIN_DATA_FILE)
    preprocessor.split_data(test_size=0.2, val_size=0.2)
    
    checker = ApplicabilityDomainChecker()
    checker.fit_from_preprocessor(preprocessor)
    checker.save_model('data_processed/ad_checker_model.pkl')

if __name__ == "__main__":
    main()
