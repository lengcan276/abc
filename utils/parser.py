# utils/parser.py
import re
import os
import numpy as np
import logging

class GaussianParser:
    """
    用于解析 Gaussian 日志文件的工具类
    """
    
    @staticmethod
    def extract_energy(log_file):
        """从 Gaussian 日志文件中提取 HF 能量值"""
        if not os.path.exists(log_file):
            return None

        try:
            # 以二进制模式读取文件以处理潜在的编码问题
            with open(log_file, 'rb') as f:
                # 从文件末尾开始，以块为单位向后读取
                chunk_size = 20000  # 字节
                f.seek(0, os.SEEK_END)
                file_size = f.tell()

                # 如果文件较小，直接全部读取
                if file_size < chunk_size * 2:
                    f.seek(0)
                    content = f.read().decode('utf-8', errors='replace')
                else:
                    # 读取最后一块，HF 值可能在这里
                    f.seek(max(0, file_size - chunk_size))
                    content = f.read().decode('utf-8', errors='replace')

                # 查找 HF= 模式（主要方法）
                match = re.search(r'HF=(-?\d+\.\d+)', content)
                if match:
                    return float(match.group(1))

                # 如果在最后一块中未找到，尝试读取整个文件
                if file_size >= chunk_size * 2:
                    f.seek(0)
                    content = f.read().decode('utf-8', errors='replace')
                    match = re.search(r'HF=(-?\d+\.\d+)', content)
                    if match:
                        return float(match.group(1))

                # 如果未找到 HF=，尝试查找 SCF Done 作为备选
                matches = re.findall(r'SCF Done:\s+E\([^)]+\)\s+=\s+([-\d.]+)', content)
                if matches:
                    # 获取最后一个匹配项
                    return float(matches[-1])

        except Exception as e:
            logging.error(f"读取 {log_file} 时出错: {e}")

        return None
        
    @staticmethod
    def check_opt_success(log_file):
        """检查 Gaussian 优化是否成功完成"""
        if not os.path.exists(log_file):
            return False

        try:
            with open(log_file, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')

                # 检查优化是否成功完成
                if "Stationary point found" in content and "Optimization completed" in content:
                    return True
                # 有时还有其他成功的标志
                if "Normal termination" in content and "Optimized Parameters" in content:
                    return True
        except Exception as e:
            logging.error(f"检查优化状态 {log_file} 时出错: {e}")

        return False
        
    @staticmethod
    def check_imaginary_freq(log_file):
        """检查是否有虚频（负频率）"""
        if not os.path.exists(log_file):
            return None  # 无法确定

        try:
            with open(log_file, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')

                # 搜索频率部分
                freq_section = re.search(r'Frequencies\s+--\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)?\s+(-?\d+\.\d+)?', content)
                if freq_section:
                    # 提取频率值并检查是否有负值
                    freqs = []
                    for i in range(1, 4):
                        if freq_section.group(i):
                            freqs.append(float(freq_section.group(i)))

                    if freqs:
                        has_negative = any(f < 0 for f in freqs)
                        return has_negative

                # 如果未找到频率部分，可能没有进行频率计算
                if "Freq" not in content and "freq" not in content:
                    return None  # 无频率计算
        except Exception as e:
            logging.error(f"检查虚频 {log_file} 时出错: {e}")

        return None
        
    # utils/parser.py (继续)
    @staticmethod
    def extract_homo_lumo(log_file):
        """提取 HOMO 和 LUMO 能量值"""
        if not os.path.exists(log_file):
            return None, None

        try:
            with open(log_file, 'r', errors='replace') as f:
                content = f.read()

            # 方法1：搜索 Alpha 占据和虚轨道值
            alpha_occ_matches = re.findall(r'Alpha\s+occ\.\s+eigenvalues\s+--\s+([-\d\.\s]+)', content)
            alpha_virt_matches = re.findall(r'Alpha\s+virt\.\s+eigenvalues\s+--\s+([-\d\.\s]+)', content)

            if alpha_occ_matches and alpha_virt_matches:
                # 提取所有占据轨道能量
                occ_energies = []
                for match in alpha_occ_matches:
                    occ_energies.extend([float(x) for x in match.split()])

                # 提取所有虚轨道能量
                virt_energies = []
                for match in alpha_virt_matches:
                    virt_energies.extend([float(x) for x in match.split()])

                if occ_energies and virt_energies:
                    homo = occ_energies[-1]  # 最后一个占据轨道
                    lumo = virt_energies[0]  # 第一个虚轨道
                    return homo, lumo

            # 方法2：查找 HOMO/LUMO 标记
            orbital_section = re.search(r'Molecular Orbital Coefficients.*?(?:Density Matrix|Condensed)', content, re.DOTALL)
            if orbital_section:
                section_text = orbital_section.group(0)

                # 尝试找出 HOMO 和 LUMO
                homo_match = re.search(r'(\d+)\s+(\d+)\s+\w+\s+\w+\s+([-]?\d+\.\d+)\s+HOMO', section_text)
                lumo_match = re.search(r'(\d+)\s+(\d+)\s+\w+\s+\w+\s+([-]?\d+\.\d+)\s+LUMO', section_text)

                if homo_match and lumo_match:
                    homo = float(homo_match.group(3))
                    lumo = float(lumo_match.group(3))
                    return homo, lumo

            # 方法3：直接搜索关键字
            direct_homo = re.search(r'HOMO\s+=\s+([-]?\d+\.\d+)', content)
            direct_lumo = re.search(r'LUMO\s+=\s+([-]?\d+\.\d+)', content)

            if direct_homo and direct_lumo:
                homo = float(direct_homo.group(1))
                lumo = float(direct_lumo.group(1))
                return homo, lumo

        except Exception as e:
            logging.error(f"提取 HOMO-LUMO 值 {log_file} 时出错: {e}")

        return None, None
        
    @staticmethod
    def extract_dipole(log_file):
        """提取偶极矩"""
        if not os.path.exists(log_file):
            return None

        try:
            with open(log_file, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')

                # 查找偶极矩部分
                dipole_section = re.search(r'Dipole moment \(Debye\):(.*?)X=(.*?)Y=(.*?)Z=(.*?)Tot=(.*?)$',
                                        content, re.MULTILINE | re.DOTALL)
                if dipole_section:
                    try:
                        tot = float(re.search(r'Tot=\s*(\d+\.\d+)', dipole_section.group(0)).group(1))
                        return tot
                    except:
                        # 备用方法
                        dipole_matches = re.findall(r'Dipole moment \(Debye\):\s+X=\s+(-?\d+\.\d+)\s+Y=\s+(-?\d+\.\d+)\s+Z=\s+(-?\d+\.\d+)\s+Tot=\s+(\d+\.\d+)', content)
                        if dipole_matches:
                            # 返回最后一个匹配的总偶极矩
                            return float(dipole_matches[-1][3])
        except Exception as e:
            logging.error(f"提取偶极矩 {log_file} 时出错: {e}")

        return None
        
    @staticmethod
    def extract_charges(log_file):
        """提取 Mulliken 电荷分布"""
        if not os.path.exists(log_file):
            return None

        try:
            with open(log_file, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')

                # 查找 Mulliken 电荷部分
                mulliken_section = re.search(r'Mulliken charges:(.*?)Sum of Mulliken charges', content, re.DOTALL)
                if mulliken_section:
                    charges_text = mulliken_section.group(1)
                    charges = []
                    for line in charges_text.strip().split('\n'):
                        parts = line.split()
                        if len(parts) >= 3 and parts[0].isdigit():
                            charges.append(float(parts[2]))

                    if charges:
                        return {
                            'max_positive': max(charges),
                            'max_negative': min(charges),
                            'charge_spread': max(charges) - min(charges),
                            'avg_charge': sum(charges) / len(charges),
                            'std_charge': np.std(charges) if len(charges) > 1 else 0.0
                        }
        except Exception as e:
            logging.error(f"提取电荷分布 {log_file} 时出错: {e}")

        return None
        
    @staticmethod
    def extract_excitation_energy(log_file):
        """提取激发能量信息"""
        if not os.path.exists(log_file):
            return None, None, None

        try:
            with open(log_file, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')

                # 查找激发能量部分
                excitation_section = re.search(r'Excitation energies and oscillator strengths:(.*?)Leave Link', content, re.DOTALL)
                if excitation_section:
                    excitation_text = excitation_section.group(1)

                    # 提取第一个激发态（S1）
                    s1_match = re.search(r'Excited State\s+1:.*?(\d+\.\d+) eV', excitation_text)
                    # 提取振子强度
                    osc_match = re.search(r'Excited State\s+1:.*?f=\s*(\d+\.\d+)', excitation_text)
                    # 查找第一个三重态（T1）
                    t1_match = re.search(r'Excited State.*?Triplet.*?(\d+\.\d+) eV', excitation_text)

                    s1_energy = float(s1_match.group(1)) if s1_match else None
                    osc_strength = float(osc_match.group(1)) if osc_match else None
                    t1_energy = float(t1_match.group(1)) if t1_match else None

                    return s1_energy, osc_strength, t1_energy

        except Exception as e:
            logging.error(f"提取激发能量 {log_file} 时出错: {e}")

        return None, None, None


class CrestParser:
    """
    用于解析 CREST 结果文件的工具类
    """
    
    @staticmethod
    def extract_crest_results(results_file):
        """从 CREST 结果文件中提取能量和构象体数据"""
        if not os.path.exists(results_file):
            return None

        try:
            with open(results_file, 'r') as f:
                content = f.read()

            # 提取构象体数量
            num_match = re.search(r'Number of conformers: (\d+)', content)
            num_conformers = int(num_match.group(1)) if num_match else None

            # 提取能量范围
            range_match = re.search(r'Energy range: ([\d.]+) kcal/mol', content)
            energy_range = float(range_match.group(1)) if range_match else None

            # 提取构象体能量和分布
            conformer_energies = {}
            conformer_populations = {}

            energy_matches = re.findall(r'Conformer (\d+): ([\d.]+) kcal/mol, ([\d.]+)%', content)
            for conf_num, energy, population in energy_matches:
                conformer_energies[f"conf_{conf_num}"] = float(energy)
                conformer_populations[f"conf_{conf_num}"] = float(population)

            # 提取 CREST 总能量（可选，如果文件中包含）
            total_energy_match = re.search(r'CREST Total Energy: ([-\d.]+) Eh', content)
            total_energy = float(total_energy_match.group(1)) if total_energy_match else None

            return {
                'num_conformers': num_conformers,
                'energy_range': energy_range,
                'conformer_energies': conformer_energies,
                'conformer_populations': conformer_populations,
                'total_energy': total_energy
            }

        except Exception as e:
            logging.error(f"解析 CREST 结果文件时出错: {results_file}, {e}")
            return None
