import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# 导入质谱匹配需要的库（简化版）
import os
from tqdm import tqdm

# 简化导入，只导入必要的模块
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    # 不导入图形相关模块
    RDKIT_AVAILABLE = True
except ImportError as e:
    print(f"RDKit导入错误: {e}")
    print("部分功能可能受限")
    RDKIT_AVAILABLE = False

try:
    from matchms import Spectrum
    from matchms.similarity import CosineGreedy
    MATCHMS_AVAILABLE = True
except ImportError as e:
    print(f"matchms导入错误: {e}")
    print("部分功能可能受限")
    MATCHMS_AVAILABLE = False

class SimplifiedAttentionClassifier:
    def __init__(self, max_nodes=10, node_dim=10):
        """
        简化版注意力分类器 - 专用于加载预训练模型进行推理
        """
        self.max_nodes = max_nodes
        self.node_dim = node_dim
        self.graph_data = []
        self.labels = []
        self.model = None
        
        # 特征峰列表
        self.nonafi_characteristic_peaks = [
            58.0651, 72.0808, 84.0808, 99.0917, 113.1073,
            135.0441, 147.0077, 151.0866, 166.0975, 169.076,
            197.0709, 250.0863, 256.0955, 262.0862, 283.1195,
            297.1346, 299.1139, 302.0812, 312.1581, 315.091,
            327.1274, 341.1608, 354.2, 377.1, 396.203
        ]
        
        # 峰组分类
        self.peak_groups = {
            'low_mass': [58.0651, 72.0808, 84.0808, 99.0917, 113.1073],
            'middle_mass': [135.0441, 147.0077, 151.0866, 166.0975, 169.076, 197.0709],
            'high_mass': [250.0863, 256.0955, 262.0862, 283.1195, 297.1346, 299.1139, 302.0812],
            'very_high_mass': [312.1581, 315.091, 327.1274, 341.1608, 354.2, 377.1, 396.203]
        }
        
        # 关键特征峰
        self.key_peaks = [58.0651, 72.0808, 135.0441, 166.0975, 250.0863]
    
    def load_data_for_prediction(self, file_path, ms_column='peaks'):
        """
        仅加载数据用于预测
        """
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("仅支持.csv和.xlsx格式的文件")
        
        if ms_column not in df.columns:
            raise ValueError(f"列 '{ms_column}' 不存在")
        
        # 预处理标签（如果存在）
        if 'label' in df.columns:
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
        else:
            self.labels = np.zeros(len(df), dtype=int)
        
        # 计算统计信息
        all_mz = []
        all_max_intensity_mz = []
        
        for i, row in df.iterrows():
            ms_str = str(row[ms_column]) if not pd.isna(row[ms_column]) else ""
            
            if pd.isna(ms_str) or ms_str == 'nan' or ms_str.strip() == '':
                continue
                
            peaks = ms_str.replace(';', ',').split(',')
            max_intensity = -1
            max_intensity_mz = 0
            
            for peak in peaks[:self.max_nodes]:
                try:
                    peak = peak.strip()
                    if not peak:
                        continue
                    
                    parts = peak.split(':')
                    if len(parts) >= 2:
                        mz = float(parts[0].strip())
                        intensity = float(parts[1].strip())
                        all_mz.append(mz)
                        
                        if intensity > max_intensity:
                            max_intensity = intensity
                            max_intensity_mz = mz
                    elif len(parts) == 1:
                        mz = float(parts[0].strip())
                        intensity = 1.0
                        all_mz.append(mz)
                        
                        if intensity > max_intensity:
                            max_intensity = intensity
                            max_intensity_mz = mz
                except:
                    continue
            
            if max_intensity > 0:
                all_max_intensity_mz.append(max_intensity_mz)
        
        # 计算统计量
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
        
        # 构建图数据
        self.graph_data = []
        
        for i, row in df.iterrows():
            ms_str = str(row[ms_column]) if not pd.isna(row[ms_column]) else ""
            
            if ms_str is None or pd.isna(ms_str) or ms_str == 'nan' or ms_str.strip() == '':
                node_features = np.zeros((self.max_nodes, self.node_dim))
                adjacency_matrix = np.eye(self.max_nodes)
            else:
                node_features, adjacency_matrix = self._parse_ms_string(ms_str)
            
            self.graph_data.append({
                'node_features': node_features,
                'adjacency_matrix': adjacency_matrix
            })
        
        return df
    
    def _parse_ms_string(self, ms_str):
        """解析质谱字符串为图数据"""
        peaks = ms_str.replace(';', ',').split(',')
        peak_data = []
        
        # 首先找到最大强度的m/z
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
        
        # 按强度降序排序
        peak_data.sort(key=lambda x: x[1], reverse=True)
        peak_data = peak_data[:self.max_nodes]
        
        # 提取m/z值
        mz_values = []
        for j, (mz, intensity) in enumerate(peak_data):
            mz_values.append(mz)
        
        # 计算节点特征
        node_features = np.zeros((self.max_nodes, self.node_dim))
        for j in range(self.max_nodes):
            if j < len(peak_data):
                mz = peak_data[j][0]
                intensity = peak_data[j][1]
            elif len(peak_data) > 0:
                mz = peak_data[-1][0]
                intensity = 1.0
            else:
                mz = 0
                intensity = 0
            
            features = self._compute_node_features(mz, j, len(peak_data), mz_values, max_intensity_mz)
            for k in range(min(len(features), self.node_dim)):
                node_features[j, k] = features[k]
        
        # 构建邻接矩阵
        adjacency_matrix = self._build_adjacency_matrix(mz_values)
        
        return node_features, adjacency_matrix
    
    def _compute_node_features(self, mz, position, total_peaks, all_mz_values, max_intensity_mz):
        """计算节点特征（10维）"""
        # 1. 标准化m/z
        mz_norm = (mz - self.mz_mean) / self.mz_std
        
        # 2. 位置特征
        position_ratio = position / max(total_peaks, 1)
        is_first_peak = 1.0 if position == 0 else 0.0
        is_last_peak = 1.0 if position == total_peaks - 1 else 0.0
        
        # 3. 那非特征峰匹配
        rounded_mz = round(mz, 1)
        rounded_characteristic = [round(p, 1) for p in self.nonafi_characteristic_peaks]
        is_characteristic = 1.0 if rounded_mz in rounded_characteristic else 0.0
        
        # 4. 与最近特征峰的m/z差异
        if self.nonafi_characteristic_peaks:
            min_diff = min([abs(mz - cp) for cp in self.nonafi_characteristic_peaks])
            characteristic_mz_diff = min_diff / 100.0
        else:
            characteristic_mz_diff = 1.0
        
        # 5. 质量区域特征
        mass_region_feature = 0.0
        for group_name, group_peaks in self.peak_groups.items():
            group_rounded = [round(p, 1) for p in group_peaks]
            if rounded_mz in group_rounded:
                mass_region_feature = {'low_mass': 0.25, 'middle_mass': 0.5, 
                                     'high_mass': 0.75, 'very_high_mass': 1.0}[group_name]
                break
        
        # 6. 是否关键峰
        rounded_key_peaks = [round(p, 1) for p in self.key_peaks]
        is_key_peak = 1.0 if rounded_mz in rounded_key_peaks else 0.0
        
        # 7. 最大强度m/z特征
        if max_intensity_mz > 0:
            max_intensity_mz_norm = (max_intensity_mz - self.max_intensity_mz_mean) / self.max_intensity_mz_std
            mz_relative_to_max = mz / max_intensity_mz
        else:
            max_intensity_mz_norm = 0.0
            mz_relative_to_max = 1.0
        
        # 构建特征向量
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
        n_nodes = len(mz_values)
        adjacency_matrix = np.eye(self.max_nodes)
        
        if n_nodes > 0:
            for i in range(min(n_nodes, self.max_nodes)):
                for j in range(min(n_nodes, self.max_nodes)):
                    if i != j:
                        mz_diff = abs(mz_values[i] - mz_values[j])
                        similarity = np.exp(-mz_diff**2 / (2 * 50.0**2))
                        adjacency_matrix[i, j] = similarity
        
        return adjacency_matrix
    
    def prepare_batch_data(self):
        """准备批次数据"""
        batch_size = len(self.graph_data)
        nodes_batch = np.zeros((batch_size, self.max_nodes, self.node_dim))
        adj_batch = np.zeros((batch_size, self.max_nodes, self.max_nodes))
        
        for i, graph in enumerate(self.graph_data):
            nodes_batch[i] = graph['node_features']
            adj_batch[i] = graph['adjacency_matrix']
        
        return [nodes_batch, adj_batch]
    
    def load_best_model(self, model_path='251229.h5'):
        """加载最佳模型"""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ 模型加载成功: {model_path}")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            self.model = None
        
        return self.model
    
    def predict_and_evaluate(self):
        """进行预测和评估"""
        if self.model is None:
            print("❌ 模型未加载")
            return None, None, False
        
        # 准备数据
        X = self.prepare_batch_data()
        
        # 预测
        y_pred_prob = self.model.predict(X, verbose=0).flatten()
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # 计算平均概率
        avg_prob = float(y_pred_prob.mean())
        positive_count = int(y_pred.sum())
        
        # 根据要求输出结果
        if positive_count > 0:
            # 如果是Positive，直接输出avg_prob
            print(f"✅ Positive, probability={avg_prob:.4f}")
            return y_pred, y_pred_prob, True
        else:
            # 如果是Negative，输出1-avg_prob
            negative_prob = 1.0 - avg_prob
            print(f"✅ Negative, probability={negative_prob:.4f}")
            return y_pred, y_pred_prob, False


class MS_SMILES_Matcher:
    def __init__(self, tolerance=0.2):
        """
        质谱相似度匹配器
        """
        self.tolerance = tolerance
        self.test_spectrum = None
        self.compounds_data = []
        self.similarity_results = []
        
        # 检查依赖
        if not RDKIT_AVAILABLE:
            print("⚠️ 警告: RDKit不可用，分子结构分析功能将受限")
        
        if not MATCHMS_AVAILABLE:
            print("⚠️ 警告: matchms不可用，谱图匹配功能将受限")
        
    def parse_ms_string(self, ms_string):
        """解析质谱字符串"""
        try:
            ms_string = str(ms_string).strip()
            ms_string = ms_string.replace('"', '').replace("'", "")
            
            if ';' in ms_string:
                peaks = ms_string.split(';')
            elif ',' in ms_string:
                peaks = ms_string.split(',')
            else:
                peaks = ms_string.split()
            
            mz_list = []
            intensity_list = []
            
            for peak in peaks:
                peak = peak.strip()
                if not peak:
                    continue
                
                if ':' in peak:
                    parts = peak.split(':')
                elif ' ' in peak:
                    parts = peak.split()[:2]
                else:
                    continue
                
                if len(parts) >= 2:
                    try:
                        mz = float(parts[0].strip())
                        intensity = float(parts[1].strip())
                        mz_list.append(mz)
                        intensity_list.append(intensity)
                    except:
                        continue
            
            if not mz_list:
                return None, None
            
            paired = list(zip(mz_list, intensity_list))
            paired.sort(key=lambda x: x[0])
            
            sorted_mz = [p[0] for p in paired]
            sorted_intensity = [p[1] for p in paired]
            
            if max(sorted_intensity) > 0:
                max_int = max(sorted_intensity)
                sorted_intensity = [i/max_int*100 for i in sorted_intensity]
            
            return sorted_mz, sorted_intensity
            
        except Exception as e:
            print(f"解析MS数据错误: {e}")
            return None, None
    
    def load_compounds_data(self, file_path="ku.txt"):
        """加载化合物数据库"""
        print("=" * 60)
        print(f"📂 加载化合物数据库: {file_path}")
        print("=" * 60)
        
        try:
            if not os.path.exists(file_path):
                print(f"❌ 错误: 文件 {file_path} 不存在")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            self.compounds_data = []
            
            for i, line in enumerate(tqdm(lines, desc="加载化合物")):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 2:
                    smiles = parts[0].strip()
                    ms_data = parts[1].strip()
                    
                    mz_list, intensity_list = self.parse_ms_string(ms_data)
                    
                    if mz_list and intensity_list:
                        if MATCHMS_AVAILABLE:
                            spectrum = Spectrum(
                                mz=np.array(mz_list),
                                intensities=np.array(intensity_list),
                                metadata={
                                    "compound_id": i + 1,
                                    "smiles": smiles,
                                    "peaks_count": len(mz_list),
                                    "mz_range": f"{min(mz_list):.2f}-{max(mz_list):.2f}"
                                }
                            )
                        else:
                            # 如果没有matchms，创建一个简单的字典
                            spectrum = {
                                "mz": np.array(mz_list),
                                "intensities": np.array(intensity_list),
                                "metadata": {
                                    "compound_id": i + 1,
                                    "smiles": smiles,
                                    "peaks_count": len(mz_list),
                                    "mz_range": f"{min(mz_list):.2f}-{max(mz_list):.2f}"
                                }
                            }
                        
                        mol = None
                        if RDKIT_AVAILABLE and smiles != "N/A":
                            try:
                                mol = Chem.MolFromSmiles(smiles)
                            except:
                                mol = None
                        
                        self.compounds_data.append({
                            'id': i + 1,
                            'smiles': smiles,
                            'spectrum': spectrum,
                            'mol': mol
                        })
                else:
                    ms_data = line
                    smiles = f"Compound_{i+1}"
                    
                    mz_list, intensity_list = self.parse_ms_string(ms_data)
                    
                    if mz_list and intensity_list:
                        if MATCHMS_AVAILABLE:
                            spectrum = Spectrum(
                                mz=np.array(mz_list),
                                intensities=np.array(intensity_list),
                                metadata={
                                    "compound_id": i + 1,
                                    "smiles": smiles,
                                    "peaks_count": len(mz_list),
                                    "mz_range": f"{min(mz_list):.2f}-{max(mz_list):.2f}"
                                }
                            )
                        else:
                            spectrum = {
                                "mz": np.array(mz_list),
                                "intensities": np.array(intensity_list),
                                "metadata": {
                                    "compound_id": i + 1,
                                    "smiles": smiles,
                                    "peaks_count": len(mz_list),
                                    "mz_range": f"{min(mz_list):.2f}-{max(mz_list):.2f}"
                                }
                            }
                        
                        self.compounds_data.append({
                            'id': i + 1,
                            'smiles': smiles,
                            'spectrum': spectrum,
                            'mol': None
                        })
            
            print(f"\n✅ 数据库加载完成!")
            print(f"  成功加载: {len(self.compounds_data)} 个化合物")
            
            if self.compounds_data:
                smiles_list = [c['smiles'] for c in self.compounds_data]
                valid_smiles = [s for s in smiles_list if s != "N/A"]
                
                if RDKIT_AVAILABLE:
                    valid_mols = [c['mol'] for c in self.compounds_data if c['mol'] is not None]
                    print(f"  可解析分子: {len(valid_mols)}")
            
            return len(self.compounds_data) > 0
            
        except Exception as e:
            print(f"❌ 加载化合物数据库错误: {e}")
            return False
    
    def load_test_spectrum_from_excel(self, file_path, ms_column='peaks'):
        """从Excel文件加载测试谱"""
        print("\n" + "=" * 60)
        print(f"📂 从Excel文件加载测试谱: {file_path}")
        print("=" * 60)
        
        try:
            if not os.path.exists(file_path):
                print(f"❌ 错误: 文件 {file_path} 不存在")
                return False
            
            df = pd.read_excel(file_path)
            
            if ms_column not in df.columns:
                print(f"❌ 错误: 列 '{ms_column}' 不存在")
                return False
            
            ms_data = str(df[ms_column].iloc[0]) if len(df) > 0 else ""
            
            mz_list, intensity_list = self.parse_ms_string(ms_data)
            
            if not mz_list:
                print("❌ 解析测试谱失败")
                return False
            
            if MATCHMS_AVAILABLE:
                self.test_spectrum = Spectrum(
                    mz=np.array(mz_list),
                    intensities=np.array(intensity_list),
                    metadata={
                        "title": "测试谱",
                        "peaks_count": len(mz_list),
                        "mz_range": f"{min(mz_list):.2f}-{max(mz_list):.2f}"
                    }
                )
            else:
                self.test_spectrum = {
                    "mz": np.array(mz_list),
                    "intensities": np.array(intensity_list),
                    "metadata": {
                        "title": "测试谱",
                        "peaks_count": len(mz_list),
                        "mz_range": f"{min(mz_list):.2f}-{max(mz_list):.2f}"
                    }
                }
            
            print("✅ 测试谱加载成功!")
            print(f"  峰数量: {len(mz_list)}")
            print(f"  m/z范围: {min(mz_list):.2f}-{max(mz_list):.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载测试谱错误: {e}")
            return False
    
    def calculate_similarities(self):
        """计算相似度"""
        print("\n" + "=" * 60)
        print("🔍 计算相似度（匹配峰数量）")
        print("=" * 60)
        
        if not self.test_spectrum:
            print("❌ 错误: 请先加载测试谱")
            return False
        
        if not self.compounds_data:
            print("❌ 错误: 化合物数据库为空")
            return False
        
        self.similarity_results = []
        
        # 获取测试谱的m/z
        if MATCHMS_AVAILABLE:
            test_mz = self.test_spectrum.peaks.mz
        else:
            test_mz = self.test_spectrum["mz"]
        
        test_peaks_count = len(test_mz)
        
        print(f"测试谱 vs {len(self.compounds_data)} 个化合物")
        print(f"测试谱有 {test_peaks_count} 个峰")
        
        for compound in tqdm(self.compounds_data, desc="计算相似度"):
            try:
                if MATCHMS_AVAILABLE:
                    compound_mz = compound['spectrum'].peaks.mz
                    compound_metadata = compound['spectrum'].metadata
                else:
                    compound_mz = compound['spectrum']["mz"]
                    compound_metadata = compound['spectrum']["metadata"]
                
                # 计算匹配峰数量
                matched_count = 0
                matched_peaks = []
                
                for mz_val1 in test_mz:
                    match_found = False
                    match_details = None
                    
                    for mz_val2 in compound_mz:
                        if abs(mz_val1 - mz_val2) <= self.tolerance:
                            match_found = True
                            match_details = {
                                'mz1': mz_val1,
                                'mz2': mz_val2,
                                'mz_diff': abs(mz_val1 - mz_val2)
                            }
                            break
                    
                    if match_found:
                        matched_count += 1
                        if match_details:
                            matched_peaks.append(match_details)
                
                similarity_score = matched_count / test_peaks_count if test_peaks_count > 0 else 0.0
                
                # 计算分子信息
                mol_info = {}
                if compound['mol'] is not None and RDKIT_AVAILABLE:
                    try:
                        mol = compound['mol']
                        mol_info['mol_weight'] = Descriptors.ExactMolWt(mol)
                        mol_info['formula'] = Chem.rdMolDescriptors.CalcMolFormula(mol)
                        mol_info['num_atoms'] = mol.GetNumAtoms()
                        mol_info['num_bonds'] = mol.GetNumBonds()
                    except:
                        mol_info = {}
                
                result = {
                    'compound_id': compound['id'],
                    'smiles': compound['smiles'],
                    'similarity': similarity_score,
                    'matches': matched_count,
                    'peaks_count': compound_metadata['peaks_count'],
                    'mz_range': compound_metadata['mz_range'],
                    'mol_info': mol_info,
                    'matched_peaks': matched_peaks
                }
                
                self.similarity_results.append(result)
                
            except Exception as e:
                print(f"处理化合物 {compound.get('id', '未知')} 时出错: {e}")
                self.similarity_results.append({
                    'compound_id': compound.get('id', 0),
                    'smiles': compound.get('smiles', 'N/A'),
                    'similarity': 0.0,
                    'matches': 0,
                    'error': str(e)
                })
        
        self.similarity_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        if self.similarity_results:
            print(f"\n✅ 相似度计算完成!")
            print(f"  最高相似度: {self.similarity_results[0]['similarity']:.4f} ({self.similarity_results[0]['matches']} 个匹配峰)")
            print(f"  最低相似度: {self.similarity_results[-1]['similarity']:.4f} ({self.similarity_results[-1]['matches']} 个匹配峰)")
        else:
            print(f"\n❌ 相似度计算失败，无结果")
        
        return len(self.similarity_results) > 0
    
    def display_top_results(self, top_n=10):
        """显示Top N结果"""
        if not self.similarity_results:
            print("❌ 错误: 无结果可显示")
            return
        
        print("\n" + "=" * 80)
        print(f"🏆 TOP {top_n} 最相似化合物（基于匹配峰数量）")
        print("=" * 80)
        
        top_results = self.similarity_results[:top_n]
        
        for i, result in enumerate(top_results, 1):
            print(f"\n{'='*60}")
            print(f"排名 #{i}: 化合物 ID {result['compound_id']}")
            print(f"{'='*60}")
            
            print(f"相似度: {result['similarity']:.4f}")
            print(f"匹配峰数量: {result['matches']}")
            print(f"测试谱总峰数: {len(self.test_spectrum['mz'] if isinstance(self.test_spectrum, dict) else self.test_spectrum.peaks.mz)}")
            print(f"化合物总峰数: {result['peaks_count']}")
            print(f"m/z范围: {result['mz_range']}")
            print(f"SMILES: {result['smiles']}")
            
            if result['mol_info']:
                if 'formula' in result['mol_info']:
                    print(f"分子式: {result['mol_info']['formula']}")
                if 'mol_weight' in result['mol_info']:
                    print(f"分子量: {result['mol_info']['mol_weight']:.2f}")
            
            print(f"{'='*60}")
    
    def save_results(self, filename="similarity_results.csv"):
        """保存结果到CSV文件"""
        if not self.similarity_results:
            print("❌ 错误: 无结果可保存")
            return
        
        data = []
        for result in self.similarity_results:
            row = {
                'compound_id': result['compound_id'],
                'smiles': result['smiles'],
                'similarity': result['similarity'],
                'matches': result['matches'],
                'peaks_count': result['peaks_count'],
                'mz_range': result['mz_range']
            }
            
            if 'mol_info' in result and result['mol_info']:
                for key, value in result['mol_info'].items():
                    if key in ['mol_weight', 'num_atoms', 'num_bonds']:
                        row[key] = value
            
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        print(f"\n💾 结果保存到: {filename}")
        print(f"总记录数: {len(df)}")


def run_similarity_matching(test_file="L2-448-processed.xlsx"):
    """运行质谱相似度匹配"""
    # 创建匹配器
    matcher = MS_SMILES_Matcher(tolerance=0.2)
    
    # 加载化合物数据库
    print("\n1. 加载化合物数据库...")
    if not matcher.load_compounds_data("ku.txt"):
        print("❌ 加载化合物数据库失败")
        return
    
    # 从Excel加载测试谱
    print("\n2. 加载测试谱...")
    if not matcher.load_test_spectrum_from_excel(test_file, ms_column='peaks'):
        print("❌ 加载测试谱失败")
        return
    
    # 计算相似度
    print("\n3. 计算相似度...")
    if not matcher.calculate_similarities():
        print("❌ 计算相似度失败")
        return
    
    # 显示Top 10结果
    print("\n4. 显示Top 10结果...")
    matcher.display_top_results(top_n=10)
    
    # 保存结果
    print("\n5. 保存结果...")
    matcher.save_results("similarity_results.csv")
    
    print("\n" + "=" * 80)
    print("✅ 质谱相似度匹配完成!")
    print("=" * 80)


def main():
    """
    主函数：整个流程控制
    """
    print("=" * 80)
    print("🔬 那非类药物检测系统")
    print("=" * 80)
    
    # 1. 初始化分类器
    classifier = SimplifiedAttentionClassifier(max_nodes=10, node_dim=10)
    
    # 2. 加载预训练模型
    best_model_path = "251229.h5"
    classifier.load_best_model(best_model_path)
    
    if classifier.model is None:
        print("❌ 模型加载失败，程序终止")
        return
    
    # 3. 加载数据并进行预测
    external_file_path = "L2-475-processed.xlsx"
    
    try:
        # 使用peaks列作为质谱数据
        external_df = classifier.load_data_for_prediction(external_file_path, ms_column='peaks')
        if external_df is None:
            print("❌ 数据加载失败，程序终止")
            return
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return
    
    # 4. 进行预测
    print("\n" + "=" * 60)
    print("🔍 进行那非类检测...")
    print("=" * 60)
    
    y_pred, y_pred_prob, is_positive = classifier.predict_and_evaluate()
    
    # 5. 根据预测结果决定下一步
    print("\n" + "=" * 60)
    print("📋 检测结果分析")
    print("=" * 60)
    
    if not is_positive:
        print("检测结论: 阴性")
        print("仅对阳性样本进行质谱相似度匹配")
    else:
        print("检测结论: 阳性")
        print("开始质谱相似度匹配...")
        print("\n" + "=" * 60)
        run_similarity_matching("L2-448-processed.xlsx")


if __name__ == "__main__":
    main()