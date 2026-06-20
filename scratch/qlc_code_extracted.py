
# ==================== CELL 2 ====================
import pandas as pd
import numpy as np
import os

def remove_isotope_peaks(df, mass_tolerance=2.0):
    """
    去除同位素峰（在指定Da范围内只保留最强的峰）
    
    参数:
    df: 包含Mass和Intensity列的DataFrame
    mass_tolerance: 同位素峰的质量容差（默认2.0 Da）
    
    返回:
    过滤后的DataFrame
    """
    # 按Mass排序
    df = df.sort_values('Mass').reset_index(drop=True)
    
    # 获取Mass和Intensity数组
    masses = df['Mass'].values
    intensities = df['Intensity'].values
    
    # 标记要保留的峰
    keep = np.ones(len(masses), dtype=bool)
    
    i = 0
    while i < len(masses):
        # 查找与当前峰相差在mass_tolerance Da以内的峰
        j = i + 1
        while j < len(masses) and masses[j] - masses[i] <= mass_tolerance:
            j += 1
        
        # 如果在这个范围内有多个峰，只保留Intensity最高的
        if j - i > 1:
            # 找到这个范围内的最大Intensity的索引
            max_idx_in_range = i + np.argmax(intensities[i:j])
            
            print(f"  同位素峰组 [{i}:{j-1}], Mass范围: {masses[i]:.6f} - {masses[j-1]:.6f}")
            print(f"    保留索引 {max_idx_in_range}: Mass={masses[max_idx_in_range]:.6f}, Intensity={intensities[max_idx_in_range]:.2f}")
            
            # 标记其他峰为不保留
            for k in range(i, j):
                if k != max_idx_in_range:
                    keep[k] = False
                    print(f"    移除索引 {k}: Mass={masses[k]:.6f}, Intensity={intensities[k]:.2f}")
            
            # 跳到下一组
            i = j
        else:
            i += 1
    
    # 只保留标记为True的行
    df_filtered = df[keep].copy()
    
    # 再次按Mass排序
    df_filtered = df_filtered.sort_values('Mass').reset_index(drop=True)
    
    return df_filtered

def normalize_intensity(df):
    """
    将Intensity归一化到0-100范围
    
    参数:
    df: 包含Intensity列的DataFrame
    
    返回:
    包含归一化Intensity的DataFrame
    """
    if 'Intensity' not in df.columns:
        print("错误: DataFrame中没有Intensity列")
        return df
    
    intensities = df['Intensity'].values
    
    # 找到最小值和最大值
    min_val = intensities.min()
    max_val = intensities.max()
    
    print(f"  原始Intensity范围: {min_val:.4f} - {max_val:.4f}")
    
    if max_val == min_val:
        print("  警告: 所有Intensity值相同，归一化后所有值将为0")
        df['Normalized_Intensity'] = 0.0
    else:
        # 归一化到0-100范围
        df['Normalized_Intensity'] = 100 * (intensities - min_val) / (max_val - min_val)
        
        # 显示归一化后的统计信息
        norm_min = df['Normalized_Intensity'].min()
        norm_max = df['Normalized_Intensity'].max()
        norm_mean = df['Normalized_Intensity'].mean()
        print(f"  归一化后范围: {norm_min:.2f} - {norm_max:.2f}")
        print(f"  归一化后平均值: {norm_mean:.2f}")
    
    return df

def check_isotope_removal(df, mass_tolerance=2.0):
    """
    检查去除同位素峰后的结果，确保没有2Da之内的峰
    
    参数:
    df: 处理后的DataFrame
    mass_tolerance: 质量容差
    
    返回:
    检查结果
    """
    if len(df) == 0:
        return "数据为空，无法检查"
    
    masses = df['Mass'].values
    
    # 检查是否有2Da之内的峰
    violations = []
    for i in range(len(masses) - 1):
        mass_diff = masses[i+1] - masses[i]
        if mass_diff <= mass_tolerance:
            violations.append({
                'index1': i,
                'index2': i+1,
                'mass1': masses[i],
                'mass2': masses[i+1],
                'diff': mass_diff,
                'intensity1': df.iloc[i]['Intensity'],
                'intensity2': df.iloc[i+1]['Intensity']
            })
    
    if violations:
        print(f"\n⚠️ 检查发现 {len(violations)} 个违规（2Da范围内仍有多个峰）:")
        for v in violations[:10]:  # 只显示前10个
            print(f"  索引 {v['index1']} & {v['index2']}: {v['mass1']:.6f} - {v['mass2']:.6f} = {v['diff']:.6f} Da")
            print(f"    强度: {v['intensity1']:.2f} vs {v['intensity2']:.2f}")
        if len(violations) > 10:
            print(f"  ... 还有 {len(violations)-10} 个违规未显示")
        return False
    else:
        print(f"\n✅ 检查通过：所有峰之间至少间隔 {mass_tolerance} Da")
        return True

def process_excel_file(input_file="L1-475.xlsx", output_file="L1-475-1.xlsx", mass_tolerance=2.0, min_intensity=1.0):
    """
    处理Excel文件，先归一化Intensity，再清理同位素峰，然后删除低强度峰
    
    参数:
    input_file: 输入Excel文件路径
    output_file: 输出Excel文件路径
    mass_tolerance: 同位素峰质量容差
    min_intensity: 归一化后最小强度阈值
    """
    print(f"正在处理Excel文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"质量容差: {mass_tolerance} Da")
    print(f"最小强度阈值: {min_intensity}")
    print("-" * 50)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        print("请确保文件在当前目录")
        
        # 尝试查找可能的文件
        possible_files = [f for f in os.listdir('.') if f.endswith(('.xlsx', '.xls'))]
        if possible_files:
            print(f"\n当前目录找到以下Excel文件:")
            for i, file in enumerate(possible_files, 1):
                print(f"  {i}. {file}")
            print("请将文件重命名为 'test-L1.xlsx' 或修改代码中的文件名")
        return None
    
    try:
        # 读取Excel文件
        print(f"读取Excel文件...")
        xls = pd.ExcelFile(input_file)
        print(f"工作表: {xls.sheet_names}")
        
        # 读取第一个工作表
        df = pd.read_excel(input_file, sheet_name=0)
        print(f"原始数据形状: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        
        # 检查列名
        print(f"\n检查列名...")
        
        # 尝试找到Mass和Intensity列
        mass_col = None
        intensity_col = None
        
        # 可能的列名变体
        mass_possible_names = ['Mass', 'mass', 'm/z', 'M/Z', 'mz', 'MASS']
        intensity_possible_names = ['Intensity', 'intensity', 'Int', 'int', 'INTENSITY']
        
        # 查找Mass列
        for col in df.columns:
            col_str = str(col)
            for name in mass_possible_names:
                if name in col_str:
                    mass_col = col
                    print(f"找到Mass列: {mass_col}")
                    break
            if mass_col:
                break
        
        # 查找Intensity列
        for col in df.columns:
            col_str = str(col)
            for name in intensity_possible_names:
                if name in col_str:
                    intensity_col = col
                    print(f"找到Intensity列: {intensity_col}")
                    break
            if intensity_col:
                break
        
        # 如果没有找到，使用默认的列索引
        if not mass_col or not intensity_col:
            print("未找到明确的Mass/Intensity列名，使用默认列索引")
            if len(df.columns) >= 2:
                mass_col = df.columns[0]
                intensity_col = df.columns[1]
                print(f"使用第1列作为Mass: {mass_col}")
                print(f"使用第2列作为Intensity: {intensity_col}")
            else:
                print("错误: 数据列不足")
                return None
        
        # 重命名列以便处理
        original_cols = {}
        if mass_col != 'Mass':
            original_cols['Mass'] = mass_col
        if intensity_col != 'Intensity':
            original_cols['Intensity'] = intensity_col
            
        df = df.rename(columns={mass_col: 'Mass', intensity_col: 'Intensity'})
        
        # 只保留需要的列
        df = df[['Mass', 'Intensity']].copy()
        
        # 删除包含NaN的行
        df_clean = df.dropna(subset=['Mass', 'Intensity']).copy()
        print(f"删除NaN后数据形状: {df_clean.shape}")
        
        # 确保数据类型
        df_clean['Mass'] = pd.to_numeric(df_clean['Mass'], errors='coerce')
        df_clean['Intensity'] = pd.to_numeric(df_clean['Intensity'], errors='coerce')
        
        # 再次删除NaN
        df_clean = df_clean.dropna().copy()
        
        original_count = len(df_clean)
        print(f"\n原始总行数: {original_count}")
        
        # 显示一些统计数据
        print(f"\n原始数据统计:")
        print(f"  Mass范围: {df_clean['Mass'].min():.4f} - {df_clean['Mass'].max():.4f}")
        print(f"  Intensity范围: {df_clean['Intensity'].min():.2f} - {df_clean['Intensity'].max():.2f}")
        print(f"  平均Intensity: {df_clean['Intensity'].mean():.2f}")
        
        # 步骤1: 先归一化Intensity到0-100范围
        print(f"\n{'='*60}")
        print("步骤1: Intensity归一化到0-100范围...")
        df_normalized = normalize_intensity(df_clean.copy())
        
        # 替换原始Intensity列为归一化后的值
        df_normalized['Intensity'] = df_normalized['Normalized_Intensity']
        df_normalized = df_normalized[['Mass', 'Intensity']].copy()
        
        # 步骤2: 删除归一化后Intensity为0的行
        print(f"\n{'='*60}")
        print("步骤2: 删除归一化后Intensity为0的行...")
        df_nonzero = df_normalized[df_normalized['Intensity'] > 0].copy()
        zero_count = len(df_normalized) - len(df_nonzero)
        
        print(f"  归一化后总行数: {len(df_normalized)}")
        print(f"  删除的零强度行数: {zero_count}")
        print(f"  剩余有效行数: {len(df_nonzero)}")
        
        if len(df_nonzero) == 0:
            print("警告: 所有行的Intensity都为0")
            return None
        
        # 步骤3: 去除同位素峰
        print(f"\n{'='*60}")
        print(f"步骤3: 去除同位素峰 (质量容差: {mass_tolerance} Da)...")
        print("  在2Da范围内只保留最强的峰...")
        
        # 先按强度排序，确保先处理强峰
        df_sorted = df_nonzero.sort_values('Intensity', ascending=False).reset_index(drop=True)
        
        # 获取Mass和Intensity数组
        masses = df_sorted['Mass'].values
        intensities = df_sorted['Intensity'].values
        
        # 标记要保留的峰
        keep = np.ones(len(masses), dtype=bool)
        
        # 使用更严格的同位素峰检测
        for i in range(len(masses)):
            if not keep[i]:
                continue
                
            for j in range(i+1, len(masses)):
                if not keep[j]:
                    continue
                    
                # 计算质量差
                mass_diff = abs(masses[j] - masses[i])
                
                # 如果质量差在2Da以内，只保留强度更高的那个
                if mass_diff <= mass_tolerance:
                    if intensities[i] >= intensities[j]:
                        keep[j] = False
                        print(f"  移除弱峰: Mass={masses[j]:.6f} (强度={intensities[j]:.2f})")
                        print(f"    保留强峰: Mass={masses[i]:.6f} (强度={intensities[i]:.2f})")
                        print(f"    质量差: {mass_diff:.6f} Da")
                    else:
                        keep[i] = False
                        print(f"  移除弱峰: Mass={masses[i]:.6f} (强度={intensities[i]:.2f})")
                        print(f"    保留强峰: Mass={masses[j]:.6f} (强度={intensities[j]:.2f})")
                        print(f"    质量差: {mass_diff:.6f} Da")
                        break  # 当前i被移除，继续下一个i
        
        # 只保留标记为True的行
        df_filtered = df_sorted[keep].copy()
        
        # 按Mass排序
        df_filtered = df_filtered.sort_values('Mass').reset_index(drop=True)
        
        removed_isotope_count = len(df_nonzero) - len(df_filtered)
        print(f"\n同位素峰清理完成:")
        print(f"  清理前行数: {len(df_nonzero)}")
        print(f"  移除的同位素峰: {removed_isotope_count}")
        print(f"  保留的峰: {len(df_filtered)}")
        
        # 步骤4: 检查同位素峰去除效果
        print(f"\n{'='*60}")
        print("步骤4: 检查同位素峰去除效果...")
        check_passed = check_isotope_removal(df_filtered, mass_tolerance)
        
        # 步骤5: 删除强度小于1的峰
        print(f"\n{'='*60}")
        print(f"步骤5: 删除归一化后强度小于{min_intensity}的峰...")
        df_filtered_intensity = df_filtered[df_filtered['Intensity'] >= min_intensity].copy()
        
        removed_low_intensity_count = len(df_filtered) - len(df_filtered_intensity)
        print(f"  删除前峰数: {len(df_filtered)}")
        print(f"  删除的低强度峰数: {removed_low_intensity_count}")
        print(f"  删除后峰数: {len(df_filtered_intensity)}")
        
        if len(df_filtered_intensity) == 0:
            print("警告: 所有峰的强度都小于阈值")
            return None
        
        # 最终数据按Mass排序
        df_final = df_filtered_intensity.sort_values('Mass').reset_index(drop=True)
        
        # 显示清理后的数据预览
        print(f"\n最终数据预览 (前10行):")
        print(df_final.head(10))
        
        # 显示最终统计
        print(f"\n最终数据统计:")
        print(f"  总峰数: {len(df_final)}")
        print(f"  Mass范围: {df_final['Mass'].min():.4f} - {df_final['Mass'].max():.4f}")
        print(f"  Intensity范围: {df_final['Intensity'].min():.2f} - {df_final['Intensity'].max():.2f}")
        print(f"  平均Intensity: {df_final['Intensity'].mean():.2f}")
        
        # 步骤6: 保存到新的Excel文件
        print(f"\n{'='*60}")
        print(f"步骤6: 保存结果到 {output_file}...")
        
        # 保存为Excel
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 保存主数据（只有归一化的Intensity）
            df_final.to_excel(writer, sheet_name='Filtered Data', index=False)
            
            # 创建一个详细的处理过程工作表
            process_steps = {
                '处理步骤': [
                    '1. 数据读取与清洗',
                    '   - 原始总行数',
                    '   - 删除NaN后行数',
                    '2. Intensity归一化(0-100)',
                    '   - 原始强度范围',
                    '   - 归一化后强度范围',
                    '3. 删除零强度行',
                    '   - 删除行数',
                    '4. 同位素峰清理(2Da范围)',
                    '   - 清理前行数',
                    '   - 移除同位素峰数',
                    '5. 强度过滤(<1)',
                    '   - 删除低强度峰数',
                    '6. 最终结果',
                    '   - 总峰数',
                    '   - Mass范围',
                    '   - 强度范围'
                ],
                '数值/结果': [
                    '',
                    original_count,
                    len(df_clean),
                    '',
                    f"{df_clean['Intensity'].min():.4f} - {df_clean['Intensity'].max():.4f}",
                    '0.00 - 100.00',
                    '',
                    zero_count,
                    '',
                    len(df_nonzero),
                    removed_isotope_count,
                    '',
                    removed_low_intensity_count,
                    '',
                    len(df_final),
                    f"{df_final['Mass'].min():.4f} - {df_final['Mass'].max():.4f}",
                    f"{df_final['Intensity'].min():.2f} - {df_final['Intensity'].max():.2f}"
                ]
            }
            
            process_df = pd.DataFrame(process_steps)
            process_df.to_excel(writer, sheet_name='Process Steps', index=False)
            
            # 格式设置
            workbook = writer.book
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                
                # 设置列宽
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 30)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"\n✅ 处理完成!")
        print(f"输入文件: {input_file}")
        print(f"输出文件: {output_file}")
        print(f"原始总行数: {original_count}")
        print(f"最终保留行数: {len(df_final)}")
        
        # 创建简洁的报告
        print(f"\n{'='*60}")
        print("处理报告:")
        print(f"{'='*60}")
        print(f"1. 数据读取与清洗:")
        print(f"   - 原始总行数: {original_count}")
        print(f"   - 删除NaN行后: {len(df_clean)} 行")
        
        print(f"\n2. Intensity归一化:")
        print(f"   - 原始范围: {df_clean['Intensity'].min():.4f} - {df_clean['Intensity'].max():.4f}")
        print(f"   - 归一化后范围: 0.00 - 100.00")
        print(f"   - 删除零强度行: {zero_count}")
        
        print(f"\n3. 同位素峰清理:")
        print(f"   - 质量容差: {mass_tolerance} Da")
        print(f"   - 清理前行数: {len(df_nonzero)}")
        print(f"   - 移除同位素峰: {removed_isotope_count}")
        print(f"   - 检查结果: {'通过' if check_passed else '失败'}")
        
        print(f"\n4. 强度过滤:")
        print(f"   - 强度阈值: {min_intensity}")
        print(f"   - 删除低强度峰: {removed_low_intensity_count}")
        
        print(f"\n5. 最终结果:")
        print(f"   - 总峰数: {len(df_final)}")
        print(f"   - Mass范围: {df_final['Mass'].min():.4f} - {df_final['Mass'].max():.4f}")
        print(f"   - 强度范围: {df_final['Intensity'].min():.2f} - {df_final['Intensity'].max():.2f}")
        print(f"   - 平均强度: {df_final['Intensity'].mean():.2f}")
        
        print(f"\n6. 输出文件:")
        print(f"   - 文件: {output_file}")
        print(f"   - 'Filtered Data'工作表: 最终处理结果")
        print(f"   - 'Process Steps'工作表: 处理过程详细记录")
        print(f"{'='*60}")
        
        return df_final
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None

# 主程序
if __name__ == "__main__":
    print("="*70)
    print("Excel文件同位素峰清理与归一化系统")
    print("功能:")
    print("  1. 归一化Intensity到0-100范围")
    print("  2. 删除零强度行")
    print("  3. 在2Da范围内只保留最强的峰（严格同位素峰检测）")
    print("  4. 删除归一化后强度小于1的峰")
    print("="*70)
    print("\n开始处理...")
    
    # 使用默认设置处理
    input_file = "test-L1.xlsx"
    output_file = "test-L1-processed.xlsx"
    mass_tolerance = 2.0
    min_intensity = 1.0
    
    # 处理文件
    result = process_excel_file(input_file, output_file, mass_tolerance, min_intensity)
    
    if result is not None:
        print("\n✅ 处理完成!")
    else:
        print("\n❌ 处理失败!")
    
    # 等待用户按任意键退出
    input("\n按Enter键退出...")

# ==================== CELL 5 ====================
import pandas as pd
import numpy as np
import os

class RiskMatcher:
    def __init__(self, risk_db_file="risk_matching-1.xlsx", threshold=0.005, ion_mode=None):
        """
        Initialize risk matcher
        
        Parameters:
        risk_db_file: Path to risk level Excel file (default: risk_matching-1.xlsx)
        threshold: Threshold for Risk0 matching (default: 0.005 Da)
        ion_mode: Ion mode - 'positive' or 'negative'
        """
        self.risk_db_file = risk_db_file
        self.threshold = threshold
        self.ion_mode = ion_mode  # 'positive' or 'negative'
        self.risk_data = {}
        self.risk1_precise = []  # 存储风险1的原始四位小数精确值
        self.risk1_rounded = set()  # 存储风险1的四舍五入到两位小数值
        
        # 定义正离子和负离子的离子列
        self.positive_ion_columns = ['[M+H]+', '[M+Na]+', '[M+K]+']
        self.negative_ion_columns = ['[M-H]-']
        
    def get_user_ion_mode(self):
        """获取用户选择的离子模式"""
        print("\n" + "="*60)
        print("选择离子模式")
        print("="*60)
        print("1. 正离子模式 (positive ion mode)")
        print("2. 负离子模式 (negative ion mode)")
        print("="*60)
        
        while True:
            try:
                choice = input("请选择离子模式 (输入 1 或 2): ").strip()
                if choice == '1':
                    self.ion_mode = 'positive'
                    print(f"已选择: 正离子模式")
                    print(f"将处理的离子列: {', '.join(self.positive_ion_columns)}")
                    return True
                elif choice == '2':
                    self.ion_mode = 'negative'
                    print(f"已选择: 负离子模式")
                    print(f"将处理的离子列: {self.negative_ion_columns[0]}")
                    return True
                else:
                    print("输入无效，请输入 1 或 2")
            except KeyboardInterrupt:
                print("\n程序被用户中断")
                return False
            except Exception as e:
                print(f"输入错误: {e}")
    
    def load_risk_data(self):
        """Load risk level data"""
        try:
            print(f"\n加载风险数据库: {self.risk_db_file}")
            print(f"Risk0阈值: {self.threshold} Da")
            
            # Check if file exists
            if not os.path.exists(self.risk_db_file):
                print(f"错误: 文件 {self.risk_db_file} 不存在")
                return False
            
            # Read all sheets
            xls = pd.ExcelFile(self.risk_db_file)
            print(f"数据库中的工作表: {xls.sheet_names}")
            
            # Load risk1, risk2, risk3 data
            for sheet_name in ['风险1', '风险2', '风险3']:
                if sheet_name in xls.sheet_names:
                    df = pd.read_excel(self.risk_db_file, sheet_name=sheet_name)
                    print(f"\n加载 {sheet_name}: {len(df)} 行, {len(df.columns)} 列")
                    print(f"列名: {df.columns.tolist()}")
                    
                    # For 风险1, we need both precise and rounded values
                    if sheet_name == '风险1':
                        precise_values = []  # 原始四位小数
                        rounded_values = set()  # 两位小数
                        mz_count = 0
                        
                        # 根据离子模式选择要处理的列
                        if self.ion_mode == 'positive':
                            # 只处理正离子列
                            ion_columns = [col for col in self.positive_ion_columns if col in df.columns]
                            if not ion_columns:
                                print(f"  警告: 未找到正离子列 {self.positive_ion_columns}，将处理所有数值列")
                                ion_columns = df.columns
                        elif self.ion_mode == 'negative':
                            # 只处理负离子列
                            ion_columns = [col for col in self.negative_ion_columns if col in df.columns]
                            if not ion_columns:
                                print(f"  警告: 未找到负离子列 {self.negative_ion_columns}，将处理所有数值列")
                                ion_columns = df.columns
                        else:
                            # 默认处理所有列
                            ion_columns = df.columns
                        
                        print(f"  处理的列: {ion_columns}")
                        
                        # Iterate through selected columns to find all numerical data
                        for col in ion_columns:
                            for val in df[col].dropna():
                                try:
                                    # Try to convert to float
                                    mz = float(val)
                                    # Store precise value (4 decimal places)
                                    precise_values.append(mz)
                                    # Store rounded value (2 decimal places)
                                    mz_rounded = round(mz, 2)
                                    rounded_values.add(mz_rounded)
                                    mz_count += 1
                                except (ValueError, TypeError):
                                    # Skip if not numerical
                                    continue
                        
                        self.risk1_precise = precise_values
                        self.risk1_rounded = rounded_values
                        print(f"  风险1: {len(precise_values)} 精确值, {len(rounded_values)} 个唯一的两位小数近似值")
                        
                        # Show some examples
                        if precise_values:
                            print(f"  精确值示例: {precise_values[:5]}")
                            print(f"  近似值示例: {list(rounded_values)[:5]}")
                    
                    # For 风险2 and 风险3, we only need rounded values
                    else:
                        mz_values = set()
                        mz_count = 0
                        
                        # 根据离子模式选择要处理的列
                        if self.ion_mode == 'positive':
                            # 只处理正离子列
                            ion_columns = [col for col in self.positive_ion_columns if col in df.columns]
                            if not ion_columns:
                                print(f"  警告: 未找到正离子列 {self.positive_ion_columns}，将处理所有数值列")
                                ion_columns = df.columns
                        elif self.ion_mode == 'negative':
                            # 只处理负离子列
                            ion_columns = [col for col in self.negative_ion_columns if col in df.columns]
                            if not ion_columns:
                                print(f"  警告: 未找到负离子列 {self.negative_ion_columns}，将处理所有数值列")
                                ion_columns = df.columns
                        else:
                            # 默认处理所有列
                            ion_columns = df.columns
                        
                        print(f"  处理的列: {ion_columns}")
                        
                        # Iterate through selected columns to find all numerical data
                        for col in ion_columns:
                            for val in df[col].dropna():
                                try:
                                    # Try to convert to float
                                    mz = float(val)
                                    # Round to two decimal places
                                    mz_rounded = round(mz, 2)
                                    mz_values.add(mz_rounded)
                                    mz_count += 1
                                except (ValueError, TypeError):
                                    # Skip if not numerical
                                    continue
                        
                        self.risk_data[sheet_name] = mz_values
                        print(f"  {sheet_name}: {len(mz_values)} 个唯一的m/z值 ({mz_count} 个总数值)")
                else:
                    print(f"\n警告: 工作表 {sheet_name} 不存在")
                    if sheet_name == '风险1':
                        self.risk1_precise = []
                        self.risk1_rounded = set()
                    else:
                        self.risk_data[sheet_name] = set()
            
            return True
            
        except Exception as e:
            print(f"加载数据错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_mz_from_excel(self, excel_file="test-L1-processed.xlsx"):
        """Load m/z values from Excel file"""
        try:
            print(f"\n从Excel文件加载m/z值: {excel_file}")
            print(f"离子模式: {self.ion_mode}")
            
            if not os.path.exists(excel_file):
                print(f"错误: 文件 {excel_file} 不存在")
                return None, None, None
            
            # Read Excel file
            xls = pd.ExcelFile(excel_file)
            print(f"Excel文件中的工作表: {xls.sheet_names}")
            
            # Read first worksheet
            df = pd.read_excel(excel_file, sheet_name=0)
            print(f"数据形状: {df.shape}")
            print(f"所有列名: {df.columns.tolist()}")
            
            # 根据离子模式选择要处理的列
            if self.ion_mode == 'positive':
                # 查找正离子列
                available_columns = df.columns.tolist()
                print(f"寻找正离子列: {self.positive_ion_columns}")
                
                # 尝试找到匹配的正离子列
                ion_columns = []
                for ion_col in self.positive_ion_columns:
                    # 尝试精确匹配
                    if ion_col in available_columns:
                        ion_columns.append(ion_col)
                    else:
                        # 尝试模糊匹配
                        for col in available_columns:
                            if ion_col.lower() in str(col).lower():
                                ion_columns.append(col)
                                print(f"  找到匹配列: '{col}' 匹配 '{ion_col}'")
                                break
                
                if not ion_columns:
                    print(f"警告: 未找到正离子列，将使用所有数值列")
                    # 使用所有数值列
                    ion_columns = [col for col in available_columns if pd.api.types.is_numeric_dtype(df[col])]
                    if not ion_columns:
                        ion_columns = available_columns
            elif self.ion_mode == 'negative':
                # 查找负离子列
                available_columns = df.columns.tolist()
                print(f"寻找负离子列: {self.negative_ion_columns}")
                
                # 尝试找到匹配的负离子列
                ion_columns = []
                for ion_col in self.negative_ion_columns:
                    # 尝试精确匹配
                    if ion_col in available_columns:
                        ion_columns.append(ion_col)
                    else:
                        # 尝试模糊匹配
                        for col in available_columns:
                            if ion_col.lower() in str(col).lower():
                                ion_columns.append(col)
                                print(f"  找到匹配列: '{col}' 匹配 '{ion_col}'")
                                break
                
                if not ion_columns:
                    print(f"警告: 未找到负离子列，将使用所有数值列")
                    # 使用所有数值列
                    ion_columns = [col for col in available_columns if pd.api.types.is_numeric_dtype(df[col])]
                    if not ion_columns:
                        ion_columns = available_columns
            else:
                # 默认使用所有列
                ion_columns = df.columns.tolist()
            
            print(f"最终处理的列: {ion_columns}")
            
            # Extract m/z values from all selected columns
            mz_values_original = []  # 保存原始四位小数值
            mz_values_rounded = []   # 保存两位小数值
            intensity_values = []
            valid_count = 0
            
            # 处理每个选中的列
            for col_idx, col_name in enumerate(ion_columns):
                print(f"\n处理列 '{col_name}':")
                col_values_count = 0
                
                for row_idx, value in df[col_name].items():
                    try:
                        # 跳过空值
                        if pd.isna(value):
                            continue
                            
                        # 尝试转换为浮点数
                        mass = float(value)
                        mz_rounded = round(mass, 2)
                        mz_values_original.append(mass)      # 保持原始四位小数
                        mz_values_rounded.append(mz_rounded) # 存储两位小数用于匹配
                        
                        # Get intensity value (if available)
                        intensity = None
                        # 尝试查找对应的强度列
                        intensity_col_name = None
                        
                        # 常见强度列名模式
                        possible_intensity_names = [
                            f"{col_name} Intensity",
                            f"{col_name}_Intensity",
                            f"Intensity_{col_name}",
                            "Intensity",
                            "intensity",
                            "Abundance"
                        ]
                        
                        for possible_name in possible_intensity_names:
                            if possible_name in df.columns:
                                intensity_col_name = possible_name
                                break
                        
                        if intensity_col_name:
                            try:
                                intensity = float(df.at[row_idx, intensity_col_name])
                            except:
                                intensity = None
                        
                        intensity_values.append(intensity)
                        valid_count += 1
                        col_values_count += 1
                        
                        # Show first few values
                        if valid_count <= 5:
                            if intensity is not None:
                                print(f"  行 {row_idx+1}: m/z={mass:.6f} -> {mz_rounded:.2f}, 强度={intensity:.2f}")
                            else:
                                print(f"  行 {row_idx+1}: m/z={mass:.6f} -> {mz_rounded:.2f}")
                                
                    except (ValueError, TypeError) as e:
                        # 跳过非数值
                        continue
                
                if col_values_count > 0:
                    print(f"  列 '{col_name}': 提取了 {col_values_count} 个m/z值")
            
            if valid_count == 0:
                print("错误: 未提取到任何有效的m/z值")
                return None, None, None
            
            if valid_count > 5:
                print(f"  ... 还有 {valid_count - 5} 个更多值")
            
            print(f"\n总共加载了 {len(mz_values_original)} 个m/z值 (原始值保留4位小数)")
            
            # Return both original and rounded values
            return mz_values_original, mz_values_rounded, intensity_values
            
        except Exception as e:
            print(f"加载Excel文件错误: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def match_mz_exact(self, mz_value):
        """
        Match single m/z value risk level with new logic:
        1. First check for Risk0 (precise match to risk1 within threshold Da)
        2. Then check for Risk1 (rounded match to risk1, diff > threshold Da)
        3. Then check other risk levels
        """
        mz_rounded = round(mz_value, 2)
        
        # 1. Check for Risk0 (精确匹配到风险1，相差threshold Da以内)
        if self.risk1_precise:
            min_diff = min([abs(mz_value - db_value) for db_value in self.risk1_precise])
            if min_diff <= self.threshold:
                # Find the closest value in risk1 database
                closest_value = min(self.risk1_precise, key=lambda x: abs(x - mz_value))
                actual_risk = "Risk0"
                output_risk = "Risk1"  # 输出时显示为Risk1
                match_type = f"精确匹配 (阈值: {self.threshold} Da)"
                return actual_risk, output_risk, mz_rounded, closest_value, match_type, min_diff
        
        # 2. Check for Risk1 (四舍五入后匹配到风险1，但差值大于threshold Da)
        if mz_rounded in self.risk1_rounded:
            # Check the actual difference
            if self.risk1_precise:
                # Find the closest risk1 value that rounds to this value
                matching_precise_values = [v for v in self.risk1_precise if round(v, 2) == mz_rounded]
                if matching_precise_values:
                    closest_precise = min(matching_precise_values, key=lambda x: abs(x - mz_value))
                    diff = abs(mz_value - closest_precise)
                    if diff > self.threshold:  # 确保不是精确匹配的情况
                        actual_risk = "Risk1"
                        output_risk = "Risk1"  # 输出时显示为Risk1
                        match_type = "近似匹配 (两位小数相同)"
                        return actual_risk, output_risk, mz_rounded, closest_precise, match_type, diff
        
        # 3. Check risk2 (two decimal places)
        if mz_rounded in self.risk_data.get('风险2', set()):
            actual_risk = "Risk2"
            output_risk = "Risk2"  # 输出时显示为Risk2
            match_type = "两位小数匹配"
            return actual_risk, output_risk, mz_rounded, mz_rounded, match_type, 0.0
        
        # 4. Check risk3 (two decimal places)
        if mz_rounded in self.risk_data.get('风险3', set()):
            actual_risk = "Risk3"
            output_risk = "Risk3"  # 输出时显示为Risk3
            match_type = "两位小数匹配"
            return actual_risk, output_risk, mz_rounded, mz_rounded, match_type, 0.0
        
        # 5. No match
        actual_risk = "Low Risk"
        output_risk = "Low Risk"  # 输出时显示为Low Risk
        match_type = "无匹配"
        return actual_risk, output_risk, mz_rounded, None, match_type, None
    
    def match_mz_list(self, mz_list_original, mz_list_rounded, intensity_list=None):
        """Match risk levels for m/z list"""
        results = []
        
        print(f"\n开始对 {len(mz_list_original)} 个m/z值进行风险等级匹配...")
        print(f"离子模式: {self.ion_mode}")
        print(f"匹配逻辑:")
        print(f"  1. Risk0 (实际) / Risk1 (输出): 与风险1数据库值相差{self.threshold} Da以内")
        print(f"  2. Risk1 (实际) / Risk1 (输出): 与风险1数据库值相差{self.threshold} Da以外，但两位小数相同")
        print(f"  3. Risk2 (实际) / Risk2 (输出): 两位小数匹配到风险2数据库")
        print(f"  4. Risk3 (实际) / Risk3 (输出): 两位小数匹配到风险3数据库")
        print(f"  5. Low Risk (实际) / Low Risk (输出): 无匹配")
        print(f"风险数据库: {self.risk_db_file}")
        print("-" * 80)
        
        for i, (mz_original, mz_rounded) in enumerate(zip(mz_list_original, mz_list_rounded), 1):
            # 使用新的匹配逻辑
            actual_risk, output_risk, mz_rounded, mz_matched, match_type, diff = self.match_mz_exact(mz_original)
            
            # Get corresponding intensity value
            intensity = None
            if intensity_list is not None and i-1 < len(intensity_list):
                intensity = intensity_list[i-1]
            
            results.append({
                'Index': i,
                'Original m/z': mz_original,  # 保持原始四位/六位小数
                'Matched m/z': mz_rounded,
                'Matched to m/z': mz_matched if mz_matched is not None else "No match",
                'Match Type': match_type,
                'Difference (Da)': diff if diff is not None else "N/A",
                'Intensity': intensity if intensity is not None else "N/A",
                'Actual Risk': actual_risk,     # 第一列：实际Risk
                'Output Risk': output_risk      # 第二列：输出Risk
            })
            
            # Display matching result
            if mz_matched is not None:
                intensity_str = f", 强度={intensity:.2f}" if intensity is not None else ""
                if match_type.startswith("精确匹配"):
                    print(f"{i:3d}. m/z={mz_original:.6f} -> {mz_rounded:.2f} -> 匹配到 {mz_matched:.6f}")
                    print(f"     实际: {actual_risk}, 输出: {output_risk}, {match_type} (差值: {diff:.6f} Da){intensity_str}")
                elif match_type.startswith("近似匹配"):
                    print(f"{i:3d}. m/z={mz_original:.6f} -> {mz_rounded:.2f} -> 匹配到 {mz_matched:.6f}")
                    print(f"     实际: {actual_risk}, 输出: {output_risk}, {match_type} (差值: {diff:.6f} Da){intensity_str}")
                else:
                    print(f"{i:3d}. m/z={mz_original:.6f} -> {mz_rounded:.2f} -> 匹配到 {mz_rounded:.2f}")
                    print(f"     实际: {actual_risk}, 输出: {output_risk}, {match_type}{intensity_str}")
            else:
                intensity_str = f", 强度={intensity:.2f}" if intensity is not None else ""
                print(f"{i:3d}. m/z={mz_original:.6f} -> {mz_rounded:.2f}")
                print(f"     实际: {actual_risk}, 输出: {output_risk}{intensity_str}")
        
        return results
    
    def generate_summary(self, results):
        """Generate risk matching summary"""
        print("\n" + "="*80)
        print("风险匹配结果汇总")
        print("="*80)
        
        # Count each actual risk level
        actual_risk_counts = {}
        output_risk_counts = {}
        
        for result in results:
            actual_risk = result['Actual Risk']
            output_risk = result['Output Risk']
            actual_risk_counts[actual_risk] = actual_risk_counts.get(actual_risk, 0) + 1
            output_risk_counts[output_risk] = output_risk_counts.get(output_risk, 0) + 1
        
        # Display statistics
        print(f"\n离子模式: {self.ion_mode}")
        print(f"风险数据库: {self.risk_db_file}")
        print(f"Risk0阈值: {self.threshold} Da")
        print(f"总m/z值数量: {len(results)}")
        
        risk_levels_order = ['Risk0', 'Risk1', 'Risk2', 'Risk3', 'Low Risk']
        print("\n实际Risk统计:")
        for risk_level in risk_levels_order:
            count = actual_risk_counts.get(risk_level, 0)
            percentage = count / len(results) * 100 if len(results) > 0 else 0
            print(f"  {risk_level}: {count:4d} 个值 ({percentage:6.1f}%)")
        
        print("\n输出Risk统计:")
        for risk_level in ['Risk1', 'Risk2', 'Risk3', 'Low Risk']:
            count = output_risk_counts.get(risk_level, 0)
            percentage = count / len(results) * 100 if len(results) > 0 else 0
            print(f"  {risk_level}: {count:4d} 个值 ({percentage:6.1f}%)")
        
        # Show Risk0 and Risk1 details
        risk01_results = [r for r in results if r['Actual Risk'] in ['Risk0', 'Risk1']]
        if risk01_results:
            print(f"\nRisk0/Risk1 详情 ({len(risk01_results)} 个值):")
            print("-" * 80)
            for result in risk01_results:
                symbol = "★" if result['Actual Risk'] == 'Risk0' else "☆"
                intensity_str = f", 强度={result['Intensity']:.2f}" if result['Intensity'] != "N/A" else ""
                print(f"  {symbol} 索引 {result['Index']:3d}: m/z={result['Original m/z']:.6f}")
                print(f"     实际: {result['Actual Risk']}, 输出: {result['Output Risk']}, {result['Match Type']}")
                if result['Difference (Da)'] != "N/A":
                    print(f"     差值: {result['Difference (Da)']:.6f} Da, 匹配到: {result['Matched to m/z']:.6f}{intensity_str}")
                else:
                    print(f"     匹配到: {result['Matched to m/z']}{intensity_str}")
        
        # Show all risk data summary
        total_risk_data = len(results) - actual_risk_counts.get('Low Risk', 0)
        print(f"\n总风险数据: {total_risk_data} 个值 ({total_risk_data/len(results)*100:.1f}% of total)")
        
        return results, actual_risk_counts, output_risk_counts
    
    def save_results(self, results, output_file="test-L1-processed_risk_matching_results.xlsx"):
        """Save matching results to Excel file"""
        try:
            # 根据离子模式调整输出文件名
            base_name, ext = os.path.splitext(output_file)
            output_file = f"{base_name}_{self.ion_mode}{ext}"
            
            # Create DataFrame
            df_all = pd.DataFrame(results)
            
            # Format columns for display
            # Matched m/z: format to 2 decimal places
            df_all['Matched m/z'] = df_all['Matched m/z'].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "")
            
            # Matched to m/z: format to 6 decimal places if it's a number
            def format_matched_to(x):
                if isinstance(x, (int, float)) and not pd.isna(x):
                    return f"{x:.6f}"
                elif x == "No match":
                    return x
                else:
                    return str(x)
            
            df_all['Matched to m/z'] = df_all['Matched to m/z'].apply(format_matched_to)
            
            # Format difference
            def format_difference(x):
                if isinstance(x, (int, float)) and not pd.isna(x):
                    return f"{x:.6f}"
                else:
                    return str(x)
            
            df_all['Difference (Da)'] = df_all['Difference (Da)'].apply(format_difference)
            
            # Save to Excel
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Save all data
                sheet_name = f'All Results ({self.ion_mode})'
                df_all.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Create statistical summary
                actual_risk_counts = {}
                output_risk_counts = {}
                
                for result in results:
                    actual_risk = result['Actual Risk']
                    output_risk = result['Output Risk']
                    actual_risk_counts[actual_risk] = actual_risk_counts.get(actual_risk, 0) + 1
                    output_risk_counts[output_risk] = output_risk_counts.get(output_risk, 0) + 1
                
                # Calculate percentages
                total = len(results)
                summary_data = {
                    '统计项目': [
                        '总m/z数量',
                        f'实际Risk0 (精确匹配, 阈值={self.threshold} Da)',
                        '实际Risk1 (近似匹配, 两位小数相同)',
                        '实际Risk2',
                        '实际Risk3',
                        '实际Low Risk',
                        '输出Risk1 (包含Risk0和Risk1)',
                        '输出Risk2',
                        '输出Risk3',
                        '输出Low Risk',
                        '风险数据总数'
                    ],
                    '数量': [
                        total,
                        actual_risk_counts.get('Risk0', 0),
                        actual_risk_counts.get('Risk1', 0),
                        actual_risk_counts.get('Risk2', 0),
                        actual_risk_counts.get('Risk3', 0),
                        actual_risk_counts.get('Low Risk', 0),
                        output_risk_counts.get('Risk1', 0),
                        output_risk_counts.get('Risk2', 0),
                        output_risk_counts.get('Risk3', 0),
                        output_risk_counts.get('Low Risk', 0),
                        total - actual_risk_counts.get('Low Risk', 0)
                    ],
                    '百分比': [
                        '100%',
                        f"{actual_risk_counts.get('Risk0', 0)/total*100:.1f}%" if total > 0 else '0%',
                        f"{actual_risk_counts.get('Risk1', 0)/total*100:.1f}%" if total > 0 else '0%',
                        f"{actual_risk_counts.get('Risk2', 0)/total*100:.1f}%" if total > 0 else '0%',
                        f"{actual_risk_counts.get('Risk3', 0)/total*100:.1f}%" if total > 0 else '0%',
                        f"{actual_risk_counts.get('Low Risk', 0)/total*100:.1f}%" if total > 0 else '0%',
                        f"{output_risk_counts.get('Risk1', 0)/total*100:.1f}%" if total > 0 else '0%',
                        f"{output_risk_counts.get('Risk2', 0)/total*100:.1f}%" if total > 0 else '0%',
                        f"{output_risk_counts.get('Risk3', 0)/total*100:.1f}%" if total > 0 else '0%',
                        f"{output_risk_counts.get('Low Risk', 0)/total*100:.1f}%" if total > 0 else '0%',
                        f"{(total - actual_risk_counts.get('Low Risk', 0))/total*100:.1f}%" if total > 0 else '0%'
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Statistical Summary', index=False)
                
                # Add Risk0/Risk1 details
                risk01_results = [r for r in results if r['Actual Risk'] in ['Risk0', 'Risk1']]
                if risk01_results:
                    risk01_details = []
                    for result in risk01_results:
                        risk01_details.append({
                            'Index': result['Index'],
                            'Original m/z': result['Original m/z'],
                            'Matched to m/z': result['Matched to m/z'],
                            'Match Type': result['Match Type'],
                            'Difference (Da)': result['Difference (Da)'],
                            'Actual Risk': result['Actual Risk'],
                            'Output Risk': result['Output Risk'],
                            'Intensity': result['Intensity']
                        })
                    
                    df_risk01 = pd.DataFrame(risk01_details)
                    df_risk01.to_excel(writer, sheet_name='Risk0_Risk1 Details', index=False)
            
            print(f"\n匹配结果已保存到: {output_file}")
            print(f"总共 {len(results)} 个匹配结果")
            print(f"包含的工作表:")
            print(f"  1. {sheet_name} ({len(df_all)} 行)")
            print(f"  2. Risk0_Risk1 Details ({len(risk01_results) if 'risk01_results' in locals() else 0} 行)")
            print(f"  3. Statistical Summary")
            
            return True
        except Exception as e:
            print(f"保存结果错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self):
        """Run risk matching program"""
        print("="*80)
        print("风险等级匹配系统")
        print("="*80)
        
        # 获取离子模式
        if not self.get_user_ion_mode():
            return False
        
        print(f"\n输入文件: L1-475-1.xlsx")
        print(f"离子模式: {self.ion_mode}")
        print(f"风险数据库: {self.risk_db_file}")
        print(f"Risk0阈值: {self.threshold} Da")
        print(f"\n匹配逻辑:")
        print(f"  1. 实际Risk0 / 输出Risk1: 与风险1数据库值相差{self.threshold} Da以内")
        print(f"  2. 实际Risk1 / 输出Risk1: 与风险1数据库值相差{self.threshold} Da以外，但两位小数相同")
        print(f"  3. 实际Risk2 / 输出Risk2: 两位小数匹配到风险2数据库")
        print(f"  4. 实际Risk3 / 输出Risk3: 两位小数匹配到风险3数据库")
        print(f"  5. 实际Low Risk / 输出Low Risk: 无匹配")
        print("="*80)
        
        # Input file
        input_file = "test-L1-processed.xlsx"
        base_output_file = "test-L1-processed_risk_matching_results.xlsx"
        
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"❌ 错误: 输入文件 {input_file} 不存在")
            print("请确保 L1-475-1.xlsx 文件在当前目录中")
            return False
        
        # 1. Load risk data
        print("\n步骤 1: 加载风险数据库...")
        if not self.load_risk_data():
            print("❌ 加载风险数据失败")
            return False
        
        # 2. Load m/z data
        print("\n步骤 2: 加载m/z数据...")
        mz_list_original, mz_list_rounded, intensity_list = self.load_mz_from_excel(input_file)
        if mz_list_original is None:
            print("❌ 加载m/z数据失败")
            return False
        
        # 3. Perform matching
        print("\n步骤 3: 执行风险匹配...")
        results = self.match_mz_list(mz_list_original, mz_list_rounded, intensity_list)
        
        # 4. Generate summary
        print("\n步骤 4: 生成匹配汇总...")
        results, actual_counts, output_counts = self.generate_summary(results)
        
        # 5. Save results
        print("\n步骤 5: 保存结果文件...")
        success = self.save_results(results, base_output_file)
        
        if success:
            print("\n" + "="*80)
            print("✅ 风险匹配完成!")
            print("="*80)
            print(f"输入文件: {input_file}")
            print(f"离子模式: {self.ion_mode}")
            print(f"输出文件: L1-475-1_risk_matching_results_{self.ion_mode}.xlsx")
            print(f"风险数据库: {self.risk_db_file}")
            print(f"Risk0阈值: {self.threshold} Da")
            
            print("\n实际Risk统计:")
            for risk_level in ['Risk0', 'Risk1', 'Risk2', 'Risk3', 'Low Risk']:
                count = actual_counts.get(risk_level, 0)
                percentage = count / len(results) * 100 if len(results) > 0 else 0
                print(f"  {risk_level}: {count} 个值 ({percentage:.1f}%)")
            
            print("\n输出Risk统计:")
            for risk_level in ['Risk1', 'Risk2', 'Risk3', 'Low Risk']:
                count = output_counts.get(risk_level, 0)
                percentage = count / len(results) * 100 if len(results) > 0 else 0
                print(f"  {risk_level}: {count} 个值 ({percentage:.1f}%)")
            
            print("="*80)
            return True
        else:
            print("\n❌ 风险匹配失败!")
            return False

# Main program
if __name__ == "__main__":
    # Create matcher with threshold=0.005 Da
    matcher = RiskMatcher(threshold=0.005)
    
    # Run risk matching program
    matcher.run()

# ==================== CELL 8 ====================
import pandas as pd
import numpy as np
import os

class ExcelResultFormatter:
    def __init__(self, input_file="test-L1-processed_risk_matching_results_positive.xlsx"):
        """
        Initialize Excel result formatter
        
        Parameters:
        input_file: Path to existing risk matching results Excel file
        """
        self.input_file = input_file
        self.output_file = input_file.replace(".xlsx", "_formatted.xlsx")
        
    def format_results(self):
        """Format the All Matching Results sheet according to requirements"""
        try:
            # Check if file exists
            if not os.path.exists(self.input_file):
                print(f"错误: 文件 {self.input_file} 不存在")
                return False
            
            # Read Excel file
            df_all = pd.read_excel(self.input_file, sheet_name='All Results (positive)')
            
            # Check if Actual Risk column exists
            if 'Actual Risk' not in df_all.columns:
                print(f"错误: 文件中没有 'Actual Risk' 列")
                print(f"可用的列: {df_all.columns.tolist()}")
                return False
            
            # Create a new DataFrame for formatted results
            df_formatted = df_all.copy()
            
            # Add a new column for formatted output
            formatted_outputs = []
            
            for idx, row in df_formatted.iterrows():
                actual_risk = str(row.get('Actual Risk', ''))
                output_risk = str(row.get('Output Risk', ''))
                
                # Check actual risk level and create formatted output
                if actual_risk == 'Low Risk':
                    # For Low Risk: Output as Negative, Low Risk
                    formatted_output = 'Negative, Low Risk'
                elif actual_risk == 'Risk3':
                    # For Risk3: Output as Negative, Risk3
                    formatted_output = 'Negative, Risk3'
                elif actual_risk in ['Risk0', 'Risk1']:
                    # For Risk0 and Risk1: Output as Risk1 with warning
                    formatted_output = 'Risk1高风险，需要进行二级质谱筛查'
                elif actual_risk == 'Risk2':
                    # For Risk2: Output as Risk2 with warning
                    formatted_output = 'Risk2高风险，需要进行二级质谱筛查'
                else:
                    # For other cases, use Output Risk
                    formatted_output = output_risk
                
                formatted_outputs.append(formatted_output)
            
            # Add the new column
            df_formatted['Formatted Output'] = formatted_outputs
            
            # Save to Excel file
            df_formatted.to_excel(self.output_file, sheet_name='All Matching Results', index=False)
            
            # Display the formatted table as requested (show only selected columns)
            print("=" * 80)
            print(f"{'Index':<8} {'Original m/z':<18} {'Actual Risk':<12} {'Formatted Output':<40}")
            print("-" * 80)
            
            # Display first 20 rows with selected columns
            for i, row in df_formatted.head(20).iterrows():
                index = str(row.get('Index', ''))
                
                # Format original m/z
                original_mz = row.get('Original m/z', '')
                if isinstance(original_mz, (int, float)):
                    original_mz_str = f"{original_mz:.4f}"
                else:
                    original_mz_str = str(original_mz)
                
                actual_risk = str(row.get('Actual Risk', ''))
                formatted_output = str(row.get('Formatted Output', ''))
                
                # Format the display
                print(f"{index:<8} {original_mz_str:<18} {actual_risk:<12} {formatted_output:<40}")
            
            if len(df_formatted) > 20:
                print(f"... and {len(df_formatted) - 20} more rows")
            
            print("=" * 80)
            print(f"\nExcel文件已保存为: {self.output_file}")
            print("=" * 80)
            
            # Show summary
            print(f"\n格式化规则总结:")
            print(f"  1. Actual Risk = Low Risk → 输出: Negative, Low Risk")
            print(f"  2. Actual Risk = Risk3 → 输出: Negative, Risk3")
            print(f"  3. Actual Risk = Risk0/Risk1 → 输出: Risk1高风险，需要进行二级质谱筛查")
            print(f"  4. Actual Risk = Risk2 → 输出: Risk2高风险，需要进行二级质谱筛查")
            
            # Count formatted outputs
            output_counts = df_formatted['Formatted Output'].value_counts()
            print(f"\n格式化输出分布:")
            for output_type, count in output_counts.items():
                print(f"  {output_type}: {count}行")
            
            # Also show original Actual Risk distribution
            actual_counts = df_formatted['Actual Risk'].value_counts()
            print(f"\n原始Actual Risk分布:")
            for risk_type, count in actual_counts.items():
                print(f"  {risk_type}: {count}行")
            
            return True
            
        except Exception as e:
            print(f"格式化过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self):
        """Run the formatting process"""
        print("="*60)
        print("Excel Results Formatter")
        print("根据Actual Risk列格式化结果，新增Formatted Output列")
        print("="*60)
        
        print("格式化规则:")
        print("1. Actual Risk = Low Risk → 输出: Negative, Low Risk")
        print("2. Actual Risk = Risk3 → 输出: Negative, Risk3")
        print("3. Actual Risk = Risk0/Risk1 → 输出: Risk1高风险，需要进行二级质谱筛查")
        print("4. Actual Risk = Risk2 → 输出: Risk2高风险，需要进行二级质谱筛查")
        print("\n输出说明:")
        print("- 保留所有原始列，包括Actual Risk")
        print("- 新增Formatted Output列用于格式化输出")
        print("="*60)
        
        success = self.format_results()
        
        if success:
            print("\n" + "="*60)
            print("格式化完成!")
            print("="*60)
            print(f"输出文件: {self.output_file}")
            print(f"新增列: Formatted Output")
            print("="*60)
            return True
        else:
            print("\n" + "="*60)
            print("格式化失败!")
            print("="*60)
            return False


# Main program
if __name__ == "__main__":
    # Create formatter and run
    formatter = ExcelResultFormatter()
    formatter.run()

# ==================== CELL 11 ====================
import pandas as pd
import numpy as np
import os
import time

def remove_isotope_peaks(df, mass_tolerance=2.0):
    """
    去除同位素峰（在指定Da范围内只保留最强的峰）
    """
    if df.empty:
        return df
    
    df = df.sort_values('Mass').reset_index(drop=True)
    masses = df['Mass'].values
    intensities = df['Intensity'].values
    keep = np.ones(len(masses), dtype=bool)
    
    i = 0
    while i < len(masses):
        j = i + 1
        while j < len(masses) and masses[j] - masses[i] <= mass_tolerance:
            j += 1
        
        if j - i > 1:
            max_idx_in_range = i + np.argmax(intensities[i:j])
            for k in range(i, j):
                if k != max_idx_in_range:
                    keep[k] = False
            i = j
        else:
            i += 1
    
    df_filtered = df[keep].copy()
    return df_filtered.sort_values('Mass').reset_index(drop=True)

def normalize_intensity(df):
    """
    将Intensity归一化到0-100范围
    """
    if 'Intensity' not in df.columns:
        return df
    
    intensities = df['Intensity'].values
    min_val = intensities.min()
    max_val = intensities.max()
    
    if max_val == min_val:
        df['Normalized_Intensity'] = 0.0
    else:
        df['Normalized_Intensity'] = 100 * (intensities - min_val) / (max_val - min_val)
    
    return df

def format_mass_intensity_string(df, intensity_digits=2):
    """
    将DataFrame格式化为mass:intensity字符串，用逗号分隔
    """
    if df.empty:
        return ""
    
    df_sorted = df.sort_values('Mass').reset_index(drop=True)
    formatted_pairs = []
    
    for _, row in df_sorted.iterrows():
        mass = row['Mass']
        intensity = row['Intensity']
        mass_str = f"{mass:.4f}"
        intensity_str = f"{intensity:.{intensity_digits}f}"
        formatted_pairs.append(f"{mass_str}:{intensity_str}")
    
    return ",".join(formatted_pairs)

def check_max_mass_and_risk(risk_results_file):
    """
    检查最大Mass对应的风险等级：
    1. 从L2-475.xlsx中找到最大的Mass值
    2. 在风险匹配结果文件中找到对应的Original m/z行（容差0.005Da）
    3. 检查该行的Formatted Output是否为高风险需要筛查
    """
    print("\n" + "="*60)
    print("检查最大Mass对应的风险等级")
    print("="*60)
    
    try:
        # 1. 从L2-475-0.xlsx中找到最大的Mass值
        print("步骤1: 从L2-475.xlsx中找到最大的Mass值...")
        
        if not os.path.exists("L2-448.xlsx"):
            print("错误: 找不到文件 L2-448.xlsx")
            return "Error: L2-448.xlsx file not found"
            
        df_l2 = pd.read_excel("L2-448.xlsx", sheet_name=0)
        
        # 找到Mass列
        mass_col = None
        for col in df_l2.columns:
            col_str = str(col).lower()
            if 'mass' in col_str or 'm/z' in col_str or 'mz' in col_str:
                mass_col = col
                break
        
        if not mass_col and len(df_l2.columns) > 0:
            # 假设第一列是Mass
            mass_col = df_l2.columns[0]
        
        df_l2['Mass'] = pd.to_numeric(df_l2[mass_col], errors='coerce')
        df_l2 = df_l2.dropna(subset=['Mass'])
        
        if df_l2.empty:
            print("错误: L2文件中没有有效的Mass数据")
            return "Error: No valid Mass data in L2 file"
        
        # 找到最大的Mass值
        max_mass = df_l2['Mass'].max()
        
        print(f"  L2文件中最大Mass值: {max_mass:.6f} Da")
        
        # 2. 在风险匹配结果文件中找到对应的行（容差2Da）
        print(f"\n步骤2: 在风险结果文件中查找m/z ≈ {max_mass:.4f} Da 的行（容差2Da）...")
        
        if not os.path.exists(risk_results_file):
            print(f"错误: 找不到风险匹配结果文件 {risk_results_file}")
            return "Error: Risk results file not found"
        
        # 读取All Matching Results工作表
        df_risk = pd.read_excel(risk_results_file, sheet_name='All Matching Results')
        
        # 检查必要的列是否存在
        if 'Original m/z' not in df_risk.columns:
            print("错误: 风险结果文件中没有'Original m/z'列")
            return "Error: 'Original m/z' column not found"
        
        if 'Formatted Output' not in df_risk.columns:
            print("错误: 风险结果文件中没有'Formatted Output'列")
            return "Error: 'Formatted Output' column not found"
        
        # 查找匹配的Original m/z（容差0.005Da）
        matched_row = None
        matched_idx = None
        matched_mz = None
        
        for idx, row in df_risk.iterrows():
            original_mz = row.get('Original m/z', '')
            try:
                mz_value = float(original_mz)
                # 容差0.005Da
                if abs(mz_value - max_mass) < 2:
                    matched_row = row
                    matched_idx = idx
                    matched_mz = mz_value
                    break
            except:
                continue
        
        if matched_row is None:
            print(f"  ❌ 没有找到m/z ≈ {max_mass:.4f} Da 的匹配行（容差2Da）")
            # 显示一些可能接近的值以供参考
            print(f"  可能接近的m/z值:")
            for idx, row in df_risk.head(10).iterrows():
                original_mz = row.get('Original m/z', '')
                try:
                    mz_value = float(original_mz)
                    diff = abs(mz_value - max_mass)
                    print(f"    m/z={mz_value:.4f} Da, 差值={diff:.4f} Da")
                except:
                    continue
            return "Error: No matching m/z found in risk results within 2Da tolerance"
        
        # 3. 检查Formatted Output
        formatted_output = str(matched_row.get('Formatted Output', ''))
        original_mz = matched_row.get('Original m/z', '')
        diff = abs(matched_mz - max_mass)
        
        print(f"  ✓ 找到匹配行 (行索引: {matched_idx}):")
        print(f"    原始m/z: {original_mz}")
        print(f"    L2最大Mass: {max_mass:.4f} Da")
        print(f"    差值: {diff:.6f} Da (<2Da)")
        print(f"    Formatted Output: {formatted_output}")
        
        # 检查风险等级
        if "Negative, Low Risk" in formatted_output:
            print(f"\n  ⚠️ 风险等级: Negative, Low Risk")
            print(f"  ⚠️ 无需进行数据处理")
            return "Negative, Low Risk"
        elif "Negative, Risk3" in formatted_output:
            print(f"\n  ⚠️ 风险等级: Negative, Risk3")
            print(f"  ⚠️ 无需进行数据处理")
            return "Negative, Risk3"
        elif "Risk1高风险，需要进行二级质谱筛查" in formatted_output:
            print(f"\n  ✓ 风险等级: Risk1高风险，需要进行二级质谱筛查")
            print(f"  → 需要进行数据处理")
            return "PROCEED_WITH_PROCESSING"
        elif "Risk2高风险，需要进行二级质谱筛查" in formatted_output:
            print(f"\n  ✓ 风险等级: Risk2高风险，需要进行二级质谱筛查")
            print(f"  → 需要进行数据处理")
            return "PROCEED_WITH_PROCESSING"
        else:
            print(f"\n  ❌ 未知的风险等级: {formatted_output}")
            return f"Error: Unknown risk level: {formatted_output}"
            
    except Exception as e:
        print(f"检查过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

def process_l2_excel_file():
    """
    处理Excel文件，先归一化Intensity，再清理同位素峰，转换为mass:intensity格式
    """
    # 设置默认参数
    input_file = "L2-448.xlsx"
    output_excel = "L2-448-processed.xlsx"
    risk_results_file = "test-L1-processed_risk_matching_results_positive_formatted.xlsx"
    mass_tolerance = 2.0
    intensity_digits = 2
    
    print(f"处理Excel文件: {input_file}")
    print(f"输出文件: {output_excel}")
    print(f"风险匹配结果文件: {risk_results_file}")
    print(f"质量容差: {mass_tolerance} Da")
    print(f"Intensity小数位数: {intensity_digits}")
    print("-" * 60)
    
    # 首先检查是否需要处理
    check_result = check_max_mass_and_risk(risk_results_file)
    
    if check_result == "Negative, Low Risk":
        print(f"\n" + "="*60)
        print("最终结果: Negative, Low Risk")
        print("="*60)
        return "Negative, Low Risk"
    elif check_result == "Negative, Risk3":
        print(f"\n" + "="*60)
        print("最终结果: Negative, Risk3")
        print("="*60)
        return "Negative, Risk3"
    elif check_result != "PROCEED_WITH_PROCESSING":
        print(f"\n" + "="*60)
        print(f"错误: {check_result}")
        print("="*60)
        return check_result
    
    print(f"\n" + "="*60)
    print("开始数据处理...")
    print("="*60)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        
        possible_files = [f for f in os.listdir('.') if f.endswith(('.xlsx', '.xls'))]
        if possible_files:
            print(f"\n当前目录找到以下Excel文件:")
            for i, file in enumerate(possible_files, 1):
                print(f"  {i}. {file}")
        
        return False
    
    try:
        # 读取Excel文件
        xls = pd.ExcelFile(input_file)
        df = pd.read_excel(input_file, sheet_name=0)
        
        # 找到Mass和Intensity列
        mass_col = None
        intensity_col = None
        
        for col in df.columns:
            col_str = str(col).lower()
            if 'mass' in col_str or 'm/z' in col_str or 'mz' in col_str:
                mass_col = col
            if 'intensity' in col_str or 'int' in col_str:
                intensity_col = col
        
        # 使用默认列
        if not mass_col or not intensity_col:
            if len(df.columns) >= 2:
                mass_col = df.columns[0]
                intensity_col = df.columns[1]
            else:
                print("错误: 数据列不足")
                return False
        
        # 重命名列
        df = df.rename(columns={mass_col: 'Mass', intensity_col: 'Intensity'})
        df = df[['Mass', 'Intensity']].copy()
        
        # 删除NaN
        df_clean = df.dropna().copy()
        df_clean['Mass'] = pd.to_numeric(df_clean['Mass'], errors='coerce')
        df_clean['Intensity'] = pd.to_numeric(df_clean['Intensity'], errors='coerce')
        df_clean = df_clean.dropna().copy()
        
        original_count = len(df_clean)
        print(f"原始数据行数: {original_count}")
        
        # 步骤1: 归一化Intensity到0-100
        print(f"\n步骤1: Intensity归一化...")
        df_normalized = normalize_intensity(df_clean.copy())
        df_normalized['Intensity'] = df_normalized['Normalized_Intensity']
        df_normalized = df_normalized[['Mass', 'Intensity']].copy()
        
        # 步骤2: 删除零强度行
        df_nonzero = df_normalized[df_normalized['Intensity'] > 0].copy()
        zero_count = len(df_normalized) - len(df_nonzero)
        
        # 步骤3: 去除同位素峰
        print(f"步骤2: 去除同位素峰...")
        df_filtered = remove_isotope_peaks(df_nonzero, mass_tolerance)
        
        removed_count = len(df_nonzero) - len(df_filtered)
        df_final = df_filtered.sort_values('Mass').reset_index(drop=True)
        
        print(f"\n处理统计:")
        print(f"  原始行数: {original_count}")
        print(f"  删除零强度行: {zero_count}")
        print(f"  移除同位素峰: {removed_count}")
        print(f"  最终保留峰: {len(df_final)}")
        
        # 显示Mass范围
        print(f"  Mass范围: {df_final['Mass'].min():.4f} - {df_final['Mass'].max():.4f}")
        print(f"  最大Mass值: {df_final['Mass'].max():.4f}")
        
        # 步骤4: 格式化为mass:intensity字符串
        print(f"\n步骤3: 格式化为字符串...")
        formatted_string = format_mass_intensity_string(df_final, intensity_digits)
        
        # 显示预览
        if formatted_string:
            pairs = formatted_string.split(',')
            print(f"\n格式预览 (前5个):")
            for i in range(min(5, len(pairs))):
                print(f"  {pairs[i]}")
            if len(pairs) > 5:
                print(f"  ... and {len(pairs)-5} more")
        
        # 步骤5: 保存到Excel - 只生成一个工作表，列名为peaks
        print(f"\n步骤4: 保存到Excel...")
        
        # 创建DataFrame，列名为peaks
        data = {'peaks': [formatted_string]}
        df_output = pd.DataFrame(data)
        
        # 保存到Excel
        df_output.to_excel(output_excel, sheet_name='Formatted Output', index=False)
        
        print(f"\n✅ 数据预处理完成!")
        print(f"输出文件: {output_excel}")
        print(f"\nExcel文件内容:")
        print(f"  工作表: Formatted Output")
        print(f"  列名: peaks")
        print(f"  内容: 格式化字符串 (mass:intensity格式)")
        
        print("\n" + "="*60)
        print("最终结果: 数据处理完成")
        print("="*60)
        
        return "Data processing completed successfully"
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

def main():
    """
    主程序 - 自动运行，无需用户交互
    """
    print("="*60)
    print("L2数据预处理与风险等级检查")
    print("="*60)
    
    result = process_l2_excel_file()
    
    print(f"处理完成，最终状态: {result}")
    
    # 自动退出
    time.sleep(0.01)
    
    return result

if __name__ == "__main__":
    main()

# ==================== CELL 13 ====================
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
import warnings
warnings.filterwarnings('ignore')

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
    
    def check_characteristic_peaks_rule(self, ms_string):
        """
        检查特征峰匹配规则：
        1. 至少有3个m/z值与特征峰列表中的数值在小数点后两位一致
        2. 或者至少有2个m/z值与特征峰列表中的数值在小数点后三位一致
        
        返回: 如果满足条件返回True，否则返回False
        """
        if not ms_string or pd.isna(ms_string) or ms_string == 'nan' or ms_string.strip() == '':
            return False
        
        # 提取所有的m/z值
        peaks = ms_string.replace(';', ',').split(',')
        mz_values = []
        
        for peak in peaks:
            try:
                peak = peak.strip()
                if not peak:
                    continue
                
                parts = peak.split(':')
                if len(parts) >= 1:
                    mz = float(parts[0].strip())
                    mz_values.append(mz)
            except:
                continue
        
        if not mz_values:
            return False
        
        # 统计匹配情况
        match_two_decimals = 0  # 两位小数匹配计数
        match_three_decimals = 0  # 三位小数匹配计数
        matched_peaks = []  # 记录匹配到的峰
        
        # 检查每个m/z值
        for mz in mz_values:
            # 检查是否与特征峰匹配
            for characteristic_peak in self.nonafi_characteristic_peaks:
                # 检查两位小数匹配
                if round(mz, 2) == round(characteristic_peak, 2):
                    match_two_decimals += 1
                    matched_peaks.append((mz, characteristic_peak, "两位小数匹配"))
                    break
                # 检查三位小数匹配
                elif round(mz, 3) == round(characteristic_peak, 3):
                    match_three_decimals += 1
                    matched_peaks.append((mz, characteristic_peak, "三位小数匹配"))
                    break
        
        # 应用规则
        if match_two_decimals >= 3:
            return True
        elif match_three_decimals >= 2:
            return True
        
        return False
    
    def check_risk0_condition(self, risk_file_path, l2_file_path):
        """
        检查Risk0条件：
        1. 如果risk_file_path中的Actual Risk列是Risk0
        2. 且对应的Original m/z列格子里的数字就是l2_file_path中Mass列的最大数字
        则返回True，否则返回False
        """
        try:
            # 加载risk matching结果文件
            if risk_file_path.endswith('.csv'):
                risk_df = pd.read_csv(risk_file_path, encoding='utf-8')
            elif risk_file_path.endswith('.xlsx'):
                risk_df = pd.read_excel(risk_file_path)
            else:
                print(f"不支持的风险文件格式: {risk_file_path}")
                return False
            
            # 检查必要的列
            if 'Actual Risk' not in risk_df.columns or 'Original m/z' not in risk_df.columns:
                print("风险文件中缺少必要的列: 'Actual Risk' 或 'Original m/z'")
                return False
            
            # 加载L2-475-0文件
            if l2_file_path.endswith('.csv'):
                l2_df = pd.read_csv(l2_file_path, encoding='utf-8')
            elif l2_file_path.endswith('.xlsx'):
                l2_df = pd.read_excel(l2_file_path)
            else:
                print(f"不支持的L2文件格式: {l2_file_path}")
                return False
            
            # 检查Mass列
            if 'Mass' not in l2_df.columns:
                print("L2文件中缺少'Mass'列")
                return False
            
            # 获取L2文件中Mass列的最大值
            l2_mass_max = l2_df['Mass'].max()
            print(f"L2文件Mass列的最大值为: {l2_mass_max}")
            
            # 检查risk文件中的每一行
            for idx, row in risk_df.iterrows():
                output_risk = str(row['Actual Risk']).strip()
                original_mz = row['Original m/z']
                
                # 检查是否Risk0
                if output_risk == 'Risk0':
                    # 检查Original m/z是否等于L2 Mass的最大值
                    if not pd.isna(original_mz):
                        try:
                            original_mz_float = float(original_mz)
                            # 使用较小的容差进行比较（考虑浮点数精度）
                            if abs(original_mz_float - l2_mass_max) < 0.005:
                                print(f"发现Risk0条件匹配：行{idx+1}，Actual Risk={output_risk}，Original m/z={original_mz_float}，L2 Mass最大值={l2_mass_max}")
                                return True
                        except (ValueError, TypeError) as e:
                            print(f"解析Original m/z时出错: {e}")
                            continue
            
            print("未找到满足Risk0条件的记录")
            return False
            
        except Exception as e:
            print(f"检查Risk0条件时出错: {e}")
            return False
    
    def load_data_for_prediction(self, file_path, ms_column='peaks'):
        """
        仅加载数据用于预测
        返回: 如果数据满足特征峰规则，返回None；否则返回DataFrame
        """
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8')
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("仅支持.csv和.xlsx格式的文件")
        
        if ms_column not in df.columns:
            raise ValueError(f"列 '{ms_column}' 不存在")
        
        print(f"加载文件: {file_path}")
        print(f"总样本数: {len(df)}")
        
        # 首先检查是否满足特征峰规则
        print("\n=== 开始特征峰规则检查 ===")
        all_samples_satisfy_rule = True
        
        for i, row in df.iterrows():
            ms_str = str(row[ms_column]) if not pd.isna(row[ms_column]) else ""
            
            if pd.isna(ms_str) or ms_str == 'nan' or ms_str.strip() == '':
                print(f"样本 {i+1}: 质谱数据为空")
                all_samples_satisfy_rule = False
                continue
            
            print(f"\n样本 {i+1} 检查:")
            if self.check_characteristic_peaks_rule(ms_str):
                print(f"  样本 {i+1} 满足特征峰规则")
            else:
                print(f"  样本 {i+1} 不满足特征峰规则")
                all_samples_satisfy_rule = False
        
        print(f"\n=== 特征峰规则检查完成 ===")
        if all_samples_satisfy_rule and len(df) > 0:
            print(f"所有样本都满足特征峰规则，直接输出阳性结果")
            return None  # 返回None表示满足规则
        
        # 如果不满足规则，继续正常的数据处理流程
        print(f"不满足特征峰规则，继续正常预测流程")
        
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
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.model = None
        
        return self.model
    
    def predict_and_evaluate(self, use_rule_output=False):
        """
        进行预测和评估
        
        Parameters:
        use_rule_output: 如果为True，表示使用规则输出的结果
        """
        if use_rule_output:
            # 使用规则输出，直接给出阳性结果
            print(f"Positive,probability=1.0000")
            return None, None
        
        if self.model is None:
            return None, None
        
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
            print(f"Positive,probability={avg_prob:.4f}")
        else:
            # 如果是Negative，输出1-avg_prob
            negative_prob = 1.0 - avg_prob
            print(f"Negative,probability={negative_prob:.4f}")
        
        return y_pred, y_pred_prob


def main():
    """
    主函数：首先检查Risk0条件和特征峰规则，满足则直接输出阳性，否则按正常流程预测
    """
    # 初始化分类器
    classifier = SimplifiedAttentionClassifier(max_nodes=10, node_dim=10)
    
    # 文件路径
    external_file_path = "L2-448-processed.xlsx"
    risk_file_path = "test-L1-processed_risk_matching_results_positive_formatted.xlsx"  # 或.csv
    l2_file_path = "L2-448.xlsx"  # 或.csv
    
    # 第一步：检查Risk0条件
    print("=== 第一步：检查Risk0条件 ===")
    if classifier.check_risk0_condition(risk_file_path, l2_file_path):
        print("\nRisk0条件满足，直接输出阳性结果:")
        print("Positive,probability=1.0000")
        return
    
    print("Risk0条件不满足，继续检查特征峰规则...")
    
    # 第二步：检查特征峰规则
    try:
        print("\n=== 第二步：检查特征峰规则 ===")
        # 使用peaks列作为质谱数据
        print("开始处理文件...")
        external_df = classifier.load_data_for_prediction(external_file_path, ms_column='peaks')
        
        if external_df is None:
            # 如果返回None，表示满足特征峰规则，直接输出阳性
            print("\n特征峰规则触发!")
            print("输出结果:")
            classifier.predict_and_evaluate(use_rule_output=True)
            return
        
        # 如果不满足规则，继续正常流程
        if len(external_df) == 0:
            print("没有有效数据")
            return
        
    except Exception as e:
        print(f"加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 第三步：正常模型预测流程
    print("\n=== 第三步：正常模型预测 ===")
    
    # 加载预训练模型
    best_model_path = "251229.h5"
    print(f"加载预训练模型: {best_model_path}")
    classifier.load_best_model(best_model_path)
    
    if classifier.model is None:
        print("无法加载模型，退出")
        return
    
    print("\n开始模型预测...")
    
    # 进行预测和输出结果
    classifier.predict_and_evaluate(use_rule_output=False)


if __name__ == "__main__":
    main()

# ==================== CELL 16 ====================
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
