# agents/data_agent.py
import os
import re
import pandas as pd
import numpy as np
from glob import glob
import logging
from tqdm import tqdm
import time
import traceback

class DataAgent:
    """
    Agent responsible for parsing Gaussian and CREST output files
    and extracting relevant molecular properties.
    """
    
    def __init__(self, base_dir='/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/logs'):
        """Initialize the DataAgent with the base directory containing molecular data."""
        self.base_dir = base_dir
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for the data agent."""
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                           filename='/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/logs/data_agent.log')
        self.logger = logging.getLogger('DataAgent')
        
    def extract_energy(self, log_file):
        """Extract HF energy from Gaussian log file."""
        if not os.path.exists(log_file):
            return None

        try:
            # Binary read to handle potential encoding issues
            with open(log_file, 'rb') as f:
                # Read from the end of file for efficiency
                chunk_size = 20000  # bytes
                f.seek(0, os.SEEK_END)
                file_size = f.tell()

                # If file is small, read it all
                if file_size < chunk_size * 2:
                    f.seek(0)
                    content = f.read().decode('utf-8', errors='replace')
                else:
                    # Read last chunk where HF value is likely to be
                    f.seek(max(0, file_size - chunk_size))
                    content = f.read().decode('utf-8', errors='replace')

                # Look for HF= pattern (primary method)
                match = re.search(r'HF=(-?\d+\.\d+)', content)
                if match:
                    return float(match.group(1))

                # If not found in last chunk, try reading entire file
                if file_size >= chunk_size * 2:
                    f.seek(0)
                    content = f.read().decode('utf-8', errors='replace')
                    match = re.search(r'HF=(-?\d+\.\d+)', content)
                    if match:
                        return float(match.group(1))

                # If HF= not found, try SCF Done as fallback
                matches = re.findall(r'SCF Done:\s+E\([^)]+\)\s+=\s+([-\d.]+)', content)
                if matches:
                    # Get the last match
                    return float(matches[-1])

        except Exception as e:
            self.logger.error(f"Error reading {log_file}: {e}")

        return None
    
    def check_opt_success(self, log_file):
        """Check if Gaussian optimization completed successfully."""
        if not os.path.exists(log_file):
            return False

        try:
            with open(log_file, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')

                # Check if optimization completed successfully
                if "Stationary point found" in content and "Optimization completed" in content:
                    return True
                # Sometimes there are other success indicators
                if "Normal termination" in content and "Optimized Parameters" in content:
                    return True
        except Exception as e:
            self.logger.error(f"Error checking optimization status {log_file}: {e}")

        return False
    
    def check_imaginary_freq(self, log_file):
        """Check for imaginary frequencies (negative frequencies)."""
        if not os.path.exists(log_file):
            return None  # Cannot determine

        try:
            with open(log_file, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')

                # Search for frequencies section
                freq_section = re.search(r'Frequencies\s+--\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)?\s+(-?\d+\.\d+)?', content)
                if freq_section:
                    # Extract frequency values and check for negative values
                    freqs = []
                    for i in range(1, 4):
                        if freq_section.group(i):
                            freqs.append(float(freq_section.group(i)))

                    if freqs:
                        has_negative = any(f < 0 for f in freqs)
                        return has_negative

                # If frequency section not found, may not have frequency calculation
                if "Freq" not in content and "freq" not in content:
                    return None  # No frequency calculation
        except Exception as e:
            self.logger.error(f"Error checking imaginary frequencies {log_file}: {e}")

        return None
    
    def extract_homo_lumo(self, log_file):
        """Extract HOMO and LUMO energy values."""
        if not os.path.exists(log_file):
            return None, None

        try:
            with open(log_file, 'r', errors='replace') as f:
                content = f.read()

            # Method 1: Search for Alpha occupied and virtual orbital values
            alpha_occ_matches = re.findall(r'Alpha\s+occ\.\s+eigenvalues\s+--\s+([-\d\.\s]+)', content)
            alpha_virt_matches = re.findall(r'Alpha\s+virt\.\s+eigenvalues\s+--\s+([-\d\.\s]+)', content)

            if alpha_occ_matches and alpha_virt_matches:
                # Extract all occupied orbital energies
                occ_energies = []
                for match in alpha_occ_matches:
                    # Fix: Add proper handling for values without spaces
                    match = match.strip()
                    # If the match contains values without spaces (like "-101.53533-101.53528")
                    if '-' in match[1:]:  # Check if there are negative signs after the first character
                        # Split the string into chunks that start with a negative sign
                        values = re.findall(r'-?\d+\.\d+', match)
                        occ_energies.extend([float(x) for x in values])
                    else:
                        # Normal case with proper spacing
                        occ_energies.extend([float(x) for x in match.split()])

                # Extract all virtual orbital energies
                virt_energies = []
                for match in alpha_virt_matches:
                    # Apply the same fix to virtual orbitals
                    match = match.strip()
                    if '-' in match[1:]:
                        values = re.findall(r'-?\d+\.\d+', match)
                        virt_energies.extend([float(x) for x in values])
                    else:
                        virt_energies.extend([float(x) for x in match.split()])

                if occ_energies and virt_energies:
                    homo = occ_energies[-1]  # Last occupied orbital
                    lumo = virt_energies[0]  # First virtual orbital
                    return homo, lumo

            # Method 2: Look for HOMO/LUMO tags
            orbital_section = re.search(r'Molecular Orbital Coefficients.*?(?:Density Matrix|Condensed)', content, re.DOTALL)
            if orbital_section:
                section_text = orbital_section.group(0)

                # Try to find HOMO and LUMO
                homo_match = re.search(r'(\d+)\s+(\d+)\s+\w+\s+\w+\s+([-]?\d+\.\d+)\s+HOMO', section_text)
                lumo_match = re.search(r'(\d+)\s+(\d+)\s+\w+\s+\w+\s+([-]?\d+\.\d+)\s+LUMO', section_text)

                if homo_match and lumo_match:
                    homo = float(homo_match.group(3))
                    lumo = float(lumo_match.group(3))
                    return homo, lumo

            # Method 3: Direct keyword search
            direct_homo = re.search(r'HOMO\s+=\s+([-]?\d+\.\d+)', content)
            direct_lumo = re.search(r'LUMO\s+=\s+([-]?\d+\.\d+)', content)

            if direct_homo and direct_lumo:
                homo = float(direct_homo.group(1))
                lumo = float(direct_lumo.group(1))
                return homo, lumo

        except Exception as e:
            self.logger.error(f"Error extracting HOMO-LUMO values {log_file}: {str(e)}")

        return None, None
    
    def extract_dipole(self, log_file):
        """Extract dipole moment."""
        if not os.path.exists(log_file):
            return None

        try:
            with open(log_file, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')

                # Search for dipole moment section
                dipole_section = re.search(r'Dipole moment \(Debye\):(.*?)X=(.*?)Y=(.*?)Z=(.*?)Tot=(.*?)$',
                                        content, re.MULTILINE | re.DOTALL)
                if dipole_section:
                    try:
                        tot = float(re.search(r'Tot=\s*(\d+\.\d+)', dipole_section.group(0)).group(1))
                        return tot
                    except:
                        # Fallback method
                        dipole_matches = re.findall(r'Dipole moment \(Debye\):\s+X=\s+(-?\d+\.\d+)\s+Y=\s+(-?\d+\.\d+)\s+Z=\s+(-?\d+\.\d+)\s+Tot=\s+(\d+\.\d+)', content)
                        if dipole_matches:
                            # Return the last matched total dipole moment
                            return float(dipole_matches[-1][3])
        except Exception as e:
            self.logger.error(f"Error extracting dipole moment {log_file}: {e}")

        return None
    
    def extract_charges(self, log_file):
        """Extract Mulliken charge distribution."""
        if not os.path.exists(log_file):
            return None

        try:
            with open(log_file, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')

                # Search for Mulliken charges section
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
            self.logger.error(f"Error extracting charge distribution {log_file}: {e}")

        return None
    
    def extract_excitation_energy(self, log_file):
        """Extract excitation energy information."""
        if not os.path.exists(log_file):
            return None, None, None

        try:
            with open(log_file, 'rb') as f:
                content = f.read().decode('utf-8', errors='replace')

                # Search for excitation energy section
                excitation_section = re.search(r'Excitation energies and oscillator strengths:(.*?)Leave Link', content, re.DOTALL)
                if excitation_section:
                    excitation_text = excitation_section.group(1)

                    # Extract first excited state (S1)
                    s1_match = re.search(r'Excited State\s+1:.*?(\d+\.\d+) eV', excitation_text)
                    # Extract oscillator strength
                    osc_match = re.search(r'Excited State\s+1:.*?f=\s*(\d+\.\d+)', excitation_text)
                    # Look for first triplet state (T1)
                    t1_match = re.search(r'Excited State.*?Triplet.*?(\d+\.\d+) eV', excitation_text)

                    s1_energy = float(s1_match.group(1)) if s1_match else None
                    osc_strength = float(osc_match.group(1)) if osc_match else None
                    t1_energy = float(t1_match.group(1)) if t1_match else None

                    return s1_energy, osc_strength, t1_energy

        except Exception as e:
            self.logger.error(f"Error extracting excitation energy {log_file}: {e}")

        return None, None, None
    
    def extract_all_excited_states(self, log_file, expected_multiplicity='singlet'):
        """
        提取所有激发态信息，包括高阶激发态
        
        Args:
            log_file: Gaussian log文件路径
            expected_multiplicity: 期望的多重度 ('singlet' 或 'triplet')
        """
        if not os.path.exists(log_file):
            return None
            
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 存储所有激发态
            states = {
                'singlets': [],
                'triplets': [],
                'all_states': []
            }
            
            # 正则表达式匹配激发态
            state_pattern = r'Excited State\s+(\d+):\s+(Singlet|Triplet)-?(\S*)\s+([\d.]+)\s+eV\s+([\d.]+)\s+nm\s+f=([\d.]+)'
            
            # 提取激发态详细信息
            for match in re.finditer(state_pattern, content):
                state_info = {
                    'number': int(match.group(1)),
                    'multiplicity': match.group(2),
                    'symmetry': match.group(3) if match.group(3) else 'A',
                    'energy_ev': float(match.group(4)),
                    'wavelength_nm': float(match.group(5)),
                    'osc_strength': float(match.group(6)),
                    'transitions': []
                }
                
                # 提取该激发态的轨道跃迁信息
                state_num = state_info['number']
                transition_section = re.search(
                    rf'Excited State\s+{state_num}:.*?(?=Excited State|\s*SavETr:|\Z)',
                    content,
                    re.DOTALL
                )
                
                if transition_section:
                    # 匹配轨道跃迁 (例如: 123 -> 124    0.70532)
                    trans_pattern = r'(\d+)\s*->\s*(\d+)\s+([-\d.]+)'
                    for trans_match in re.finditer(trans_pattern, transition_section.group(0)):
                        state_info['transitions'].append({
                            'from': int(trans_match.group(1)),
                            'to': int(trans_match.group(2)),
                            'coefficient': float(trans_match.group(3))
                        })
                
                # 分类存储
                states['all_states'].append(state_info)
                if state_info['multiplicity'] == 'Singlet':
                    states['singlets'].append(state_info)
                else:
                    states['triplets'].append(state_info)
            
            # 记录提取的激发态数量
            self.logger.info(f"从 {log_file} 提取了 {len(states['singlets'])} 个单激发态和 {len(states['triplets'])} 个三重激发态")
            
            return states
            
        except Exception as e:
            self.logger.error(f"Error extracting excited states from {log_file}: {e}")
            return None

    def calculate_all_st_gaps(self, singlets, triplets):
        """
        计算所有可能的S-T gap组合
        
        Args:
            singlets: 单激发态列表
            triplets: 三重激发态列表
            
        Returns:
            包含所有gap信息的字典
        """
        gaps = {}
        inverted_gaps = []
        
        # 计算所有S-T组合
        for i, s_state in enumerate(singlets):
            for j, t_state in enumerate(triplets):
                gap_name = f"s{i+1}_t{j+1}_gap"
                gap_value = s_state['energy_ev'] - t_state['energy_ev']
                
                gaps[gap_name] = gap_value
                gaps[f"{gap_name}_meV"] = gap_value * 1000
                
                # 如果是反转gap（S < T）
                if gap_value < 0:
                    # 计算跃迁相似度
                    similarity = self.calculate_transition_similarity(
                        s_state['transitions'], 
                        t_state['transitions']
                    )
                    
                    inverted_gaps.append({
                        'type': f"S{i+1}-T{j+1}",
                        'singlet_state': i + 1,
                        'triplet_state': j + 1,
                        'singlet_energy': s_state['energy_ev'],
                        'triplet_energy': t_state['energy_ev'],
                        'gap': gap_value,
                        'gap_meV': gap_value * 1000,
                        'singlet_symmetry': s_state.get('symmetry', 'A'),
                        'triplet_symmetry': t_state.get('symmetry', 'A'),
                        'transition_similarity': similarity,
                        'singlet_osc_strength': s_state['osc_strength']
                    })
        
        # 按能隙大小排序（最负的在前）
        inverted_gaps.sort(key=lambda x: x['gap'])
        
        return gaps, inverted_gaps

    def calculate_transition_similarity(self, trans1, trans2):
        """计算两个激发态之间的跃迁相似度"""
        if not trans1 or not trans2:
            return 0.0
        
        # 创建跃迁字典
        dict1 = {(t['from'], t['to']): abs(t['coefficient']) for t in trans1}
        dict2 = {(t['from'], t['to']): abs(t['coefficient']) for t in trans2}
        
        # 找到共同的跃迁
        common_transitions = set(dict1.keys()) & set(dict2.keys())
        
        if not common_transitions:
            return 0.0
        
        # 计算余弦相似度
        numerator = sum(dict1[t] * dict2[t] for t in common_transitions)
        denom1 = sum(dict1[t]**2 for t in dict1.keys())**0.5
        denom2 = sum(dict2[t]**2 for t in dict2.keys())**0.5
        
        if denom1 * denom2 == 0:
            return 0.0
            
        similarity = numerator / (denom1 * denom2)
        
        return similarity
    
    def extract_crest_results(self, results_file):
        """Extract energy and conformer data from CREST results file."""
        if not os.path.exists(results_file):
            return None

        try:
            with open(results_file, 'r') as f:
                content = f.read()

            # Extract number of conformers
            num_match = re.search(r'Number of conformers: (\d+)', content)
            num_conformers = int(num_match.group(1)) if num_match else None

            # Extract energy range
            range_match = re.search(r'Energy range: ([\d.]+) kcal/mol', content)
            energy_range = float(range_match.group(1)) if range_match else None

            # Extract conformer energies and distributions
            conformer_energies = {}
            conformer_populations = {}

            energy_matches = re.findall(r'Conformer (\d+): ([\d.]+) kcal/mol, ([\d.]+)%', content)
            for conf_num, energy, population in energy_matches:
                conformer_energies[f"conf_{conf_num}"] = float(energy)
                conformer_populations[f"conf_{conf_num}"] = float(population)

            # Extract CREST total energy (optional, if included in file)
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
            self.logger.error(f"Error parsing CREST results file: {results_file}, {e}")
            return None
            
    def extract_all_conformers(self, molecule_dir, state):
        """Extract properties for all conformers of a specific state."""
        state_path = os.path.join(molecule_dir, state)
        if not os.path.exists(state_path):
            self.logger.debug(f"State path does not exist: {state_path}")
            return []
         # 添加调试信息：显示state目录内容
        print(f"  Checking state directory: {state_path}")
        try:
            state_files = os.listdir(state_path)
            # 查找CREST相关文件
            crest_files = [f for f in state_files if 'crest' in f.lower() or f.endswith('.xyz')]
            if crest_files:
                print(f"    Found CREST-related files: {crest_files[:5]}")  # 显示前5个
            
            # 查找gaussian目录
            if 'gaussian' in state_files:
                print(f"    Found gaussian directory")
            else:
                print(f"    No gaussian directory found, looking for conformers directly")
        except Exception as e:
            print(f"    Error listing state directory: {e}")
        gaussian_path = os.path.join(state_path, 'gaussian')
        if not os.path.exists(gaussian_path):
            self.logger.debug(f"Gaussian path does not exist: {gaussian_path}")
            return []

        conformers = [d for d in os.listdir(gaussian_path) if d.startswith('conf_')]
        if not conformers:
            self.logger.debug(f"No conformers found: {gaussian_path}")
            return []

        all_conf_data = []

        # Find the lowest energy conformer (to mark as primary)
        lowest_energy = float('inf')
        lowest_conf = None

        # Import structure utils - surrounded by try/except to handle potential import errors
        try:
            from utils.structure_utils import StructureUtils
            # 判断是否有RDKit可用
            has_rdkit = False
            try:
                from rdkit import Chem
                from rdkit.Chem import AllChem
                has_rdkit = True
            except ImportError:
                self.logger.warning("RDKit not available, molecule structure analysis disabled")
        except ImportError as e:
            self.logger.error(f"Error importing StructureUtils: {str(e)}")
            # Create a simple placeholder if the import fails
            class SimpleStructureUtils:
                @staticmethod
                def load_molecule_from_gaussian(*args, **kwargs):
                    return None
                @staticmethod
                def calculate_structure_features(*args, **kwargs):
                    return {}
                @staticmethod
                def compare_structures(*args, **kwargs):
                    return {}
            StructureUtils = SimpleStructureUtils
            has_rdkit = False

        # First scan to find lowest energy conformer
        for conf in conformers:
            conf_path = os.path.join(gaussian_path, conf)
            log_file = os.path.join(conf_path, 'ground.log')

            if not os.path.exists(log_file):
                continue

            energy = self.extract_energy(log_file)
            if energy is not None and energy < lowest_energy:
                lowest_energy = energy
                lowest_conf = conf

        # Process all conformers
        for conf in conformers:
            conf_path = os.path.join(gaussian_path, conf)
            log_file = os.path.join(conf_path, 'ground.log')

            if not os.path.exists(log_file):
                continue

            # Extract energy and basic properties
            energy = self.extract_energy(log_file)
            if energy is None:
                continue

            # Determine if this is the lowest energy conformer
            is_primary = (conf == lowest_conf)

            # Collect basic properties
            opt_success = self.check_opt_success(log_file)
            has_imaginary = self.check_imaginary_freq(log_file)
            homo, lumo = self.extract_homo_lumo(log_file)
            dipole = self.extract_dipole(log_file)
            charges = self.extract_charges(log_file)

            # Convert extracted values to eV
            homo_ev = homo * 27.2114 if homo is not None else None
            lumo_ev = lumo * 27.2114 if lumo is not None else None
            homo_lumo_gap = (lumo - homo) * 27.2114 if homo is not None and lumo is not None else None

            conf_data = {
                'conformer': conf,
                'energy': energy,
                'opt_success': opt_success,
                'no_imaginary': False if has_imaginary else (None if has_imaginary is None else True),
                'homo': homo_ev,
                'lumo': lumo_ev,
                'homo_lumo_gap': homo_lumo_gap,
                'dipole': dipole,
                'is_primary': is_primary
            }

            # If there is charge data, add it to results
            if charges:
                conf_data.update({
                    'max_positive_charge': charges['max_positive'],
                    'max_negative_charge': charges['max_negative'],
                    'charge_spread': charges['charge_spread'],
                    'avg_charge': charges['avg_charge']
                })

            # 如果RDKit可用，尝试加载分子结构
            if has_rdkit:
                try:
                    # 尝试查找XYZ文件的多种可能路径
                    crest_xyz = None
                     # 首先检查state目录下的CREST文件（这是您的实际文件位置）
                    state_crest_files = [
                        'crest_conformers.xyz',
                        'crest_best.xyz', 
                        'crest_rotamers.xyz',
                        'struc.xyz',
                        'coords.xyz',
                        'crest.xyz'
                    ]
                    
                    # 在state目录下直接查找
                    for crest_file in state_crest_files:
                        potential_path = os.path.join(state_path, crest_file)
                        if os.path.exists(potential_path):
                            crest_xyz = potential_path
                            print(f"    Found CREST file in state directory: {crest_file}")
                            break
                    
                    # 如果没找到，尝试其他可能的路径
                    if crest_xyz is None:
                        possible_paths = [
                            # 原有的路径
                            os.path.join(molecule_dir, 'results', f'{state}_{conf}.xyz'),
                            os.path.join(molecule_dir, 'results', f'{state}_results.xyz'),
                            os.path.join(molecule_dir, state, 'crest_best.xyz'),
                            os.path.join(molecule_dir, state, 'crest', f'{conf}.xyz'),
                            os.path.join(molecule_dir, state, 'crest', 'crest_best.xyz'),
                            # 新增：直接在state目录下查找任何.xyz文件
                            os.path.join(state_path, f'{conf}.xyz'),
                            os.path.join(state_path, f'{state}_{conf}.xyz')
                        ]
                        
                        # 检查所有可能的路径
                        for path in possible_paths:
                            if os.path.exists(path):
                                crest_xyz = path
                                print(f"    Found CREST file at: {path}")
                                break
                                
                    # 如果还是没找到，列出state目录内容帮助调试
                    if crest_xyz is None:
                        print(f"    No CREST XYZ file found. State directory contents:")
                        try:
                            state_files = os.listdir(state_path)
                            xyz_files = [f for f in state_files if f.endswith('.xyz')]
                            if xyz_files:
                                print(f"      XYZ files in state dir: {xyz_files[:5]}")  # 只显示前5个
                                # 使用找到的第一个xyz文件
                                crest_xyz = os.path.join(state_path, xyz_files[0])
                                print(f"      Using first XYZ file found: {xyz_files[0]}")
                            else:
                                print(f"      No XYZ files found in state directory")
                                # 显示目录中的其他文件类型
                                other_files = [f for f in state_files if not f.startswith('.')][:10]
                                print(f"      Other files: {other_files}")
                        except Exception as e:
                            print(f"      Error listing directory: {e}")
                    
                    # 如果未找到，尝试查找任何XYZ文件
                    if crest_xyz is None:
                        state_crest_dir = os.path.join(molecule_dir, state, 'crest')
                        if os.path.exists(state_crest_dir):
                            from glob import glob
                            xyz_files = glob(os.path.join(state_crest_dir, "*.xyz"))
                            if xyz_files:
                                crest_xyz = xyz_files[0]
                                print(f"    Found XYZ file in crest subdirectory: {os.path.basename(crest_xyz)}")
                    
                    # 尝试两种方式加载分子
                    mol = None
                    
                    # 方法1: 从XYZ文件直接加载
                    if crest_xyz is not None:
                        try:
                            # 直接从XYZ文件读取
                            with open(crest_xyz, 'r') as f:
                                xyz_content = f.read()
                            
                            # 检查XYZ格式是否正确
                            lines = xyz_content.strip().split('\n')
                            if len(lines) > 2:
                                atom_count = int(lines[0].strip())
                                if len(lines) >= atom_count + 2:
                                    # 使用Chem.MolFromXYZBlock而不是MolFromXYZFile
                                    mol = Chem.MolFromXYZBlock(xyz_content)
                        except Exception as e:
                            self.logger.debug(f"Error loading from XYZ block: {str(e)}")
                    
                    # 方法2: 通过StructureUtils加载
                    if mol is None:
                        try:
                            # 传递state_path作为提示，帮助找到CREST文件
                            # 在state目录下查找可能的CREST文件
                            state_crest_file = None
                            crest_file_names = ['crest_conformers.xyz', 'crest_best.xyz', 'struc.xyz', 'coords.xyz']
                            
                            for crest_name in crest_file_names:
                                potential_crest = os.path.join(state_path, crest_name)
                                if os.path.exists(potential_crest):
                                    state_crest_file = potential_crest
                                    break
                            
                            # 调用时明确传递CREST文件路径
                            mol = StructureUtils.load_molecule_from_gaussian(
                                log_file, 
                                fallback_to_crest=True, 
                                crest_xyz_file=state_crest_file
                            )
                            
                            if mol is None and state_crest_file:
                                # 如果从Gaussian加载失败，直接尝试加载CREST文件
                                print(f"      Directly loading CREST file: {os.path.basename(state_crest_file)}")
                                mol = StructureUtils.load_molecule_from_xyz(state_crest_file)
                                
                        except Exception as e:
                            self.logger.debug(f"Error using StructureUtils: {str(e)}")
                    
                    # 手动强制处理RDKit分子
                    if mol is not None:
                        try:
                            # 确保环信息被初始化 - 多种方法尝试
                            if not mol.GetNumAtoms():
                                self.logger.debug("Molecule has no atoms, skipping")
                                mol = None
                            else:
                                # 方法1: 强制更新属性缓存
                                mol.UpdatePropertyCache(strict=False)
                                
                                # 自动推断键连接（针对从XYZ文件加载的分子）
                                if mol.GetNumBonds() == 0:
                                    try:
                                        from rdkit.Chem import rdDetermineBonds
                                        # 为分子添加连接信息
                                        rdDetermineBonds.DetermineBonds(mol)
                                        self.logger.debug(f"Added {mol.GetNumBonds()} bonds to molecule")
                                    except Exception as e:
                                        self.logger.debug(f"Failed to determine bonds: {str(e)}")
                                        # 备用方法：基于原子间距离推断键
                                        try:
                                            # 使用标准的共价半径来推断键
                                            from rdkit import Chem
                                            from rdkit.Chem import AllChem
                                            
                                            # 获取原子坐标
                                            conf = mol.GetConformer()
                                            num_atoms = mol.GetNumAtoms()
                                            
                                            # 基于距离添加键
                                            for i in range(num_atoms):
                                                for j in range(i + 1, num_atoms):
                                                    dist = AllChem.GetBondLength(conf, i, j)
                                                    atom1 = mol.GetAtomWithIdx(i)
                                                    atom2 = mol.GetAtomWithIdx(j)
                                                    
                                                    # 获取原子的共价半径（简化版本）
                                                    cov_radii = {
                                                        'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66,
                                                        'F': 0.57, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20,
                                                        'I': 1.39, 'P': 1.07, 'Si': 1.11, 'B': 0.84
                                                    }
                                                    
                                                    radius1 = cov_radii.get(atom1.GetSymbol(), 1.0)
                                                    radius2 = cov_radii.get(atom2.GetSymbol(), 1.0)
                                                    
                                                    # 如果距离小于共价半径和的1.3倍，认为有键
                                                    if dist < (radius1 + radius2) * 1.3:
                                                        mol.AddBond(i, j, Chem.BondType.SINGLE)
                                            
                                            # 更新分子属性
                                            mol.UpdatePropertyCache(strict=False)
                                            self.logger.debug(f"Manually added {mol.GetNumBonds()} bonds based on distances")
                                        except Exception as e2:
                                            self.logger.debug(f"Manual bond determination also failed: {str(e2)}")
                                
                                # 方法2: 尝试强制计算环信息
                                if hasattr(mol, 'RingInfo'):
                                    ri = mol.GetRingInfo()
                                    if hasattr(ri, 'Initialize'):
                                        ri.Initialize()
                                
                                # 方法3: 尝试轻量级清理
                                try:
                                    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_FINDRADICALS|
                                                        Chem.SanitizeFlags.SANITIZE_KEKULIZE|
                                                        Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|
                                                        Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|
                                                        Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|
                                                        Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                                                    catchErrors=True)
                                except Exception as e:
                                    self.logger.debug(f"Partial sanitization failed: {str(e)}")
                                
                                # 方法4: 创建新的分子拷贝
                                try:
                                    mol_copy = Chem.Mol(mol)
                                    if mol_copy.GetNumAtoms() > 0:
                                        # 确保环信息已计算
                                        mol_copy.GetSSSR()
                                        mol = mol_copy
                                except Exception as e:
                                    self.logger.debug(f"Molecule copy failed: {str(e)}")
                                
                                # 计算分子特征
                                try:
                                    # 使用简单结构特征计算
                                    num_atoms = mol.GetNumAtoms()
                                    num_bonds = mol.GetNumBonds()
                                    
                                    # 只提取基本特征，绕过RingInfo问题
                                    basic_features = {
                                        'num_atoms': num_atoms,
                                        'num_bonds': num_bonds,
                                        'formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
                                        'mol_weight': Chem.rdMolDescriptors.CalcExactMolWt(mol)
                                    }
                                    
                                    # 尝试计算其他标准特征
                                    try:
                                        basic_features['num_rotatable_bonds'] = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
                                    except:
                                        pass
                                        
                                    # 尝试计算总电荷
                                    try:
                                        total_formal_charge = sum([atom.GetFormalCharge() for atom in mol.GetAtoms()])
                                        basic_features['total_formal_charge'] = total_formal_charge
                                    except:
                                        pass
                                    
                                    # 更新数据
                                    conf_data.update(basic_features)
                                    
                                    # 如果StructureUtils可用，尝试使用它计算完整特征
                                    if hasattr(StructureUtils, 'calculate_structure_features'):
                                        structure_features = StructureUtils.calculate_structure_features(mol)
                                        if structure_features:
                                            conf_data.update(structure_features)
                                except Exception as e:
                                    self.logger.debug(f"Feature calculation failed: {str(e)}")
                        except Exception as e:
                            self.logger.debug(f"Molecule processing failed: {str(e)}")
                except Exception as e:
                    self.logger.debug(f"Error in molecular structure analysis: {str(e)}")
            
            # 处理激发态信息
            # neutral状态：从excited_singlet.log提取单激发态
            if state == 'neutral':
                excited_log = os.path.join(conf_path, 'excited_singlet.log')  # 修改文件名
                if os.path.exists(excited_log):
                    # 提取所有单激发态
                    singlet_states = self.extract_all_excited_states(excited_log, 'singlet')
                    
                    if singlet_states and singlet_states['singlets']:
                        conf_data['singlet_states'] = singlet_states['singlets']
                        conf_data['num_singlet_states'] = len(singlet_states['singlets'])
                        
                        # 保存S1信息（向后兼容）
                        if singlet_states['singlets']:
                            s1_state = singlet_states['singlets'][0]
                            conf_data['s1_energy_ev'] = s1_state['energy_ev']
                            conf_data['oscillator_strength'] = s1_state['osc_strength']
                        
                        # 保存所有单激发态能量
                        for i, s_state in enumerate(singlet_states['singlets']):
                            conf_data[f"s{i+1}_energy_ev"] = s_state['energy_ev']
                            conf_data[f"s{i+1}_oscillator"] = s_state['osc_strength']
                            conf_data[f"s{i+1}_wavelength_nm"] = s_state['wavelength_nm']
                    
                    # 提取激发态的其他属性
                    excited_energy = self.extract_energy(excited_log)
                    excited_opt_success = self.check_opt_success(excited_log)
                    excited_has_imaginary = self.check_imaginary_freq(excited_log)
                    excited_homo, excited_lumo = self.extract_homo_lumo(excited_log)
                    excited_dipole = self.extract_dipole(excited_log)
                    
                    # Convert to eV
                    excited_homo_ev = excited_homo * 27.2114 if excited_homo is not None else None
                    excited_lumo_ev = excited_lumo * 27.2114 if excited_lumo is not None else None
                    excited_homo_lumo_gap = (excited_lumo - excited_homo) * 27.2114 if excited_homo is not None and excited_lumo is not None else None

                    conf_data['excited_energy'] = excited_energy
                    conf_data['excited_opt_success'] = excited_opt_success
                    conf_data['excited_no_imaginary'] = False if excited_has_imaginary else (None if excited_has_imaginary is None else True)
                    conf_data['excited_homo'] = excited_homo_ev
                    conf_data['excited_lumo'] = excited_lumo_ev
                    conf_data['excited_homo_lumo_gap'] = excited_homo_lumo_gap
                    conf_data['excited_dipole'] = excited_dipole

                    # Calculate excitation energy directly from energy difference
                    if excited_energy is not None and energy is not None:
                        conf_data['excitation_energy_ev'] = (excited_energy - energy) * 27.2114
            
            # triplet状态：从excited_triplet.log提取三重激发态
            elif state == 'triplet':
                excited_log = os.path.join(conf_path, 'excited_triplet.log')  # 修改文件名
                if os.path.exists(excited_log):
                    # 提取所有三重激发态
                    triplet_states = self.extract_all_excited_states(excited_log, 'triplet')
                    
                    if triplet_states and triplet_states['triplets']:
                        conf_data['triplet_states'] = triplet_states['triplets']
                        conf_data['num_triplet_states'] = len(triplet_states['triplets'])
                        
                        # 保存T1信息（向后兼容）
                        if triplet_states['triplets']:
                            t1_state = triplet_states['triplets'][0]
                            conf_data['t1_energy_ev'] = t1_state['energy_ev']
                        
                        # 保存所有三重激发态能量
                        for i, t_state in enumerate(triplet_states['triplets']):
                            conf_data[f"t{i+1}_energy_ev"] = t_state['energy_ev']
                            conf_data[f"t{i+1}_wavelength_nm"] = t_state['wavelength_nm']
                    
                    # 提取三重激发态的其他属性
                    triplet_excited_energy = self.extract_energy(excited_log)
                    triplet_excited_opt_success = self.check_opt_success(excited_log)
                    triplet_excited_has_imaginary = self.check_imaginary_freq(excited_log)
                    triplet_excited_homo, triplet_excited_lumo = self.extract_homo_lumo(excited_log)
                    triplet_excited_dipole = self.extract_dipole(excited_log)
                    
                    # Convert to eV
                    triplet_excited_homo_ev = triplet_excited_homo * 27.2114 if triplet_excited_homo is not None else None
                    triplet_excited_lumo_ev = triplet_excited_lumo * 27.2114 if triplet_excited_lumo is not None else None
                    triplet_excited_homo_lumo_gap = (triplet_excited_lumo - triplet_excited_homo) * 27.2114 if triplet_excited_homo is not None and triplet_excited_lumo is not None else None

                    conf_data['triplet_excited_energy'] = triplet_excited_energy
                    conf_data['triplet_excited_opt_success'] = triplet_excited_opt_success
                    conf_data['triplet_excited_no_imaginary'] = False if triplet_excited_has_imaginary else (None if triplet_excited_has_imaginary is None else True)
                    conf_data['triplet_excited_homo'] = triplet_excited_homo_ev
                    conf_data['triplet_excited_lumo'] = triplet_excited_lumo_ev
                    conf_data['triplet_excited_homo_lumo_gap'] = triplet_excited_homo_lumo_gap
                    conf_data['triplet_excited_dipole'] = triplet_excited_dipole
                
            all_conf_data.append(conf_data)

        return all_conf_data

    def process_molecules(self):
        """Process all molecules in the base directory."""
        # Find all molecule directories in parent directory
        parent_dir = self.base_dir
        print(f"Scanning directory: {parent_dir}")

        # Get list of molecule directories (excluding non-molecules)
        molecule_dirs = [d for d in os.listdir(parent_dir)
                        if os.path.isdir(os.path.join(parent_dir, d))
                        and not d.startswith('.')
                        and not d in ['0330_analyse', '0424_1_analyses', '0425_tiqu']
                        and not any(d.endswith(ext) for ext in ['.out', '.py', '.sh', '.xyz'])]

        print(f"Found {len(molecule_dirs)} potential molecule directories")

        # Store results for all molecule conformers
        all_conformers_data = []
        processed_count = 0
        error_count = 0
        error_molecules = []

        # Create a more detailed error log file
        error_log_path = os.path.join(parent_dir, "error_log_details.txt")
        with open(error_log_path, "w") as error_log:
            error_log.write(f"DataAgent Processing Errors - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            error_log.write("="*80 + "\n\n")

            # Process each molecule directory
            for molecule in tqdm(molecule_dirs, desc="Processing molecules"):
                try:
                    print(f"Processing molecule: {molecule}")
                    molecule_path = os.path.join(parent_dir, molecule)

                    # Skip molecules with known problematic atoms (like BH2)
                    problematic_atoms = ['bh2']
                    if any(atom in molecule.lower() for atom in problematic_atoms):
                        print(f"  Skipping molecule {molecule} with problematic atoms")
                        error_log.write(f"SKIPPED: {molecule} - Contains problematic atoms\n\n")
                        error_count += 1
                        error_molecules.append(molecule)
                        continue

                    # Extract CREST results
                    crest_results = {}
                    for state in ['cation', 'neutral', 'triplet']:
                        results_file = os.path.join(molecule_path, 'results', f'{state}_results.txt')
                        crest_data = self.extract_crest_results(results_file)
                        if crest_data:
                            crest_results[state] = crest_data
                            print(f"  Extracted {molecule}/{state} CREST results")

                    # Extract all conformers for each state
                    mol_has_valid_data = False
                    
                    # 保存跨状态的激发态信息
                    molecule_singlet_states = []
                    molecule_triplet_states = []
                    
                    for state in ['neutral', 'cation', 'triplet']:
                        try:
                            conformers_data = self.extract_all_conformers(molecule_path, state)
                        except Exception as e:
                            error_msg = f"  Error processing {molecule}/{state}: {e}"
                            print(error_msg)
                            error_log.write(f"ERROR: {molecule}/{state}\n")
                            error_log.write(f"Exception: {str(e)}\n")
                            error_log.write(f"Traceback: {traceback.format_exc()}\n\n")
                            conformers_data = []

                        if conformers_data:
                            print(f"  Extracted {len(conformers_data)} conformers for {molecule}/{state}")
                            mol_has_valid_data = True

                            # Add molecule name and state info to each conformer
                            for conf_data in conformers_data:
                                conf_data['Molecule'] = molecule
                                conf_data['State'] = state

                                # Add CREST data (if available)
                                if state in crest_results:
                                    conf_name = conf_data['conformer']
                                    conf_data['crest_num_conformers'] = crest_results[state]['num_conformers']
                                    conf_data['crest_energy_range'] = crest_results[state]['energy_range']
                                    conf_data['crest_total_energy'] = crest_results[state]['total_energy']

                                    # If current conformer has energy and distribution data in CREST results, add them
                                    if conf_name in crest_results[state]['conformer_energies']:
                                        conf_data['crest_energy'] = crest_results[state]['conformer_energies'][conf_name]
                                        conf_data['crest_population'] = crest_results[state]['conformer_populations'][conf_name]

                                # 收集激发态信息（只使用primary构象）
                                if conf_data.get('is_primary', False):
                                    if state == 'neutral' and 'singlet_states' in conf_data:
                                        molecule_singlet_states = conf_data['singlet_states']
                                    elif state == 'triplet' and 'triplet_states' in conf_data:
                                        molecule_triplet_states = conf_data['triplet_states']

                                # Add to total dataset
                                all_conformers_data.append(conf_data)
                    
                    # 计算所有S-T gap（在收集完所有状态数据后）
                    if molecule_singlet_states and molecule_triplet_states:
                        print(f"  Calculating S-T gaps for {molecule}:")
                        print(f"    {len(molecule_singlet_states)} singlet states × {len(molecule_triplet_states)} triplet states")
                        
                        all_gaps, inverted_gaps = self.calculate_all_st_gaps(
                            molecule_singlet_states, 
                            molecule_triplet_states
                        )
                        
                        print(f"    Found {len(inverted_gaps)} inverted gaps")
                        
                        # 将gap信息添加到所有该分子的构象数据中
                        for conf_data in [d for d in all_conformers_data if d['Molecule'] == molecule]:
                            # 添加所有gap值
                            conf_data.update(all_gaps)
                            
                            # 添加反转gap信息
                            if inverted_gaps:
                                conf_data['inverted_gaps'] = inverted_gaps
                                conf_data['num_inverted_gaps'] = len(inverted_gaps)
                                
                                # 找到最重要的反转gap（能隙最负的）
                                primary_inversion = inverted_gaps[0]
                                conf_data['primary_inversion_type'] = primary_inversion['type']
                                conf_data['primary_inversion_gap'] = primary_inversion['gap']
                                conf_data['primary_inversion_gap_meV'] = primary_inversion['gap_meV']
                                
                                # 打印发现的反转gaps
                                for gap in inverted_gaps[:5]:  # 只显示前5个
                                    print(f"      - {gap['type']}: {gap['gap']:.4f} eV (f={gap['singlet_osc_strength']:.4f})")
                
                    if mol_has_valid_data:
                        processed_count += 1
                    else:
                        error_count += 1
                        error_molecules.append(molecule)
                        error_log.write(f"ERROR: {molecule} - No valid data extracted\n\n")
                
                except Exception as e:
                    error_count += 1
                    error_molecules.append(molecule)
                    error_msg = f"Error processing molecule {molecule}: {e}"
                    self.logger.error(error_msg)
                    print(f"  {error_msg}")
                    error_log.write(f"CRITICAL ERROR: {molecule}\n")
                    error_log.write(f"Exception: {str(e)}\n")
                    error_log.write(f"Traceback: {traceback.format_exc()}\n\n")
                    # Continue processing next molecule
                    continue

        # Create DataFrame and save to CSV
        if all_conformers_data:
            df = pd.DataFrame(all_conformers_data)
            
            # Save all conformer detailed data
            output_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/extracted'
            os.makedirs(output_dir, exist_ok=True)
            
            all_conf_file = os.path.join(output_dir, "all_conformers_data.csv")
            df.to_csv(all_conf_file, index=False)
            print(f"All conformer data saved to {all_conf_file}")
            
            # Write error summary
            error_summary_path = os.path.join(output_dir, "processing_errors.txt")
            with open(error_summary_path, "w") as f:
                f.write(f"Processing Summary - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
                f.write(f"Total molecules: {len(molecule_dirs)}\n")
                f.write(f"Successfully processed: {processed_count}\n")
                f.write(f"Failed to process: {error_count}\n\n")
                if error_molecules:
                    f.write("Failed molecules:\n")
                    for mol in error_molecules:
                        f.write(f"- {mol}\n")
                    f.write("\nSee error_log_details.txt for detailed error information.\n")
            
            # Create molecule summary table (taking lowest energy conformer for each molecule)
            self.create_molecule_summary(df, output_dir)
            
            print(f"Successfully processed {processed_count} molecules with {error_count} errors.")
            print(f"Error details saved to {error_log_path}")
            print(f"Error summary saved to {error_summary_path}")
            return all_conf_file
        else:
            print(f"No molecule data found. Processed {processed_count} molecules with {error_count} errors.")
            return None
            
    def create_molecule_summary(self, df, output_dir):
        """Create a summary table with one row per molecule, using primary conformers."""
        summary_data = []

        # Group by molecule and state
        for molecule in df['Molecule'].unique():
            molecule_data = {'Molecule': molecule}

            # Process each state
            for state in ['neutral', 'cation', 'triplet']:
                # Filter data for current molecule and state
                state_data = df[(df['Molecule'] == molecule) & (df['State'] == state)]

                # If there is data, extract primary (lowest energy) conformer info
                if not state_data.empty:
                    primary_conf = state_data[state_data['is_primary'] == True]
                    if not primary_conf.empty:
                        primary_conf = primary_conf.iloc[0]

                        # Add basic info to summary
                        for col in ['conformer', 'energy', 'opt_success', 'no_imaginary',
                                'homo', 'lumo', 'homo_lumo_gap', 'dipole']:
                            if col in primary_conf:
                                molecule_data[f'{state}_{col}'] = primary_conf[col]

                        # Add charge info
                        for charge_col in ['max_positive_charge', 'max_negative_charge',
                                        'charge_spread', 'avg_charge']:
                            if charge_col in primary_conf:
                                molecule_data[f'{state}_{charge_col}'] = primary_conf[charge_col]

                        # Add CREST info
                        for crest_col in [c for c in primary_conf.keys() if c.startswith('crest_')]:
                            if not pd.isna(primary_conf[crest_col]):
                                molecule_data[f'{state}_{crest_col}'] = primary_conf[crest_col]

                        # 添加激发态信息
                        if state == 'neutral':
                            # 添加单激发态信息
                            if 'num_singlet_states' in primary_conf:
                                molecule_data['num_singlet_states'] = primary_conf['num_singlet_states']
                            
                            # 添加所有单激发态能量
                            for i in range(1, 11):  # 最多10个单激发态
                                s_energy_col = f's{i}_energy_ev'
                                if s_energy_col in primary_conf and not pd.isna(primary_conf[s_energy_col]):
                                    molecule_data[s_energy_col] = primary_conf[s_energy_col]
                                    molecule_data[f's{i}_oscillator'] = primary_conf.get(f's{i}_oscillator', 0)
                        
                        elif state == 'triplet':
                            # 添加三重激发态信息
                            if 'num_triplet_states' in primary_conf:
                                molecule_data['num_triplet_states'] = primary_conf['num_triplet_states']
                            
                            # 添加所有三重激发态能量
                            for i in range(1, 11):  # 最多10个三重激发态
                                t_energy_col = f't{i}_energy_ev'
                                if t_energy_col in primary_conf and not pd.isna(primary_conf[t_energy_col]):
                                    molecule_data[t_energy_col] = primary_conf[t_energy_col]

                        # 保留原有的激发态信息字段（向后兼容）
                        for excited_col in [c for c in primary_conf.keys()
                                        if c.startswith('excited_') or
                                        c in ['s1_energy_ev', 'oscillator_strength',
                                                't1_energy_ev', 's1_t1_gap_ev', 's1_t1_gap',
                                                'excitation_energy_ev']]:
                            if excited_col in primary_conf and not pd.isna(primary_conf[excited_col]):
                                molecule_data[excited_col] = primary_conf[excited_col]
            
            # 添加所有S-T gap信息
            # 找到该分子的任意一个conformer来获取gap信息
            mol_data = df[df['Molecule'] == molecule].iloc[0]
            
            # 添加所有gap值
            for col in mol_data.index:
                if col.endswith('_gap') and not col.endswith('_gap_meV'):
                    if not pd.isna(mol_data[col]):
                        molecule_data[col] = mol_data[col]
                        molecule_data[f"{col}_meV"] = mol_data[col] * 1000
            
            # 添加反转gap统计信息
            if 'num_inverted_gaps' in mol_data:
                molecule_data['num_inverted_gaps'] = mol_data['num_inverted_gaps']
            
            if 'primary_inversion_type' in mol_data and not pd.isna(mol_data['primary_inversion_type']):
                molecule_data['primary_inversion_type'] = mol_data['primary_inversion_type']
                molecule_data['primary_inversion_gap'] = mol_data.get('primary_inversion_gap')
                molecule_data['primary_inversion_gap_meV'] = mol_data.get('primary_inversion_gap_meV')

            # Calculate ionization energy (if relevant data available)
            if 'neutral_energy' in molecule_data and 'cation_energy' in molecule_data:
                if molecule_data['neutral_energy'] is not None and molecule_data['cation_energy'] is not None:
                    molecule_data['ionization_energy_ev'] = (molecule_data['cation_energy'] -
                                                        molecule_data['neutral_energy']) * 27.2114
                    print(f"    {molecule} Ionization Energy: {molecule_data['ionization_energy_ev']} eV")

            # Calculate triplet gap (if relevant data available)
            if 'neutral_energy' in molecule_data and 'triplet_energy' in molecule_data:
                if molecule_data['neutral_energy'] is not None and molecule_data['triplet_energy'] is not None:
                    triplet_gap = (molecule_data['triplet_energy'] - molecule_data['neutral_energy']) * 27.2114
                    molecule_data['triplet_gap_ev'] = triplet_gap
                    print(f"    {molecule} Triplet Gap: {molecule_data['triplet_gap_ev']} eV")

            # Add to summary data
            summary_data.append(molecule_data)

        # Create summary DataFrame
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # 打印反转能隙统计
            if 'num_inverted_gaps' in summary_df.columns:
                molecules_with_inversions = (summary_df['num_inverted_gaps'] > 0).sum()
                total_inversions = summary_df['num_inverted_gaps'].sum()
                print(f"\nFound {molecules_with_inversions} molecules with inverted gaps")
                print(f"Total inverted gap pairs: {total_inversions}")
                
                # 统计不同类型的反转
                if 'primary_inversion_type' in summary_df.columns:
                    inversion_types = summary_df[summary_df['primary_inversion_type'].notna()]['primary_inversion_type'].value_counts()
                    print("\nPrimary inversion types:")
                    for inv_type, count in inversion_types.items():
                        print(f"  {inv_type}: {count} molecules")

            # Save summary data
            summary_file = os.path.join(output_dir, "molecular_properties_summary.csv")
            summary_df.to_csv(summary_file, index=False)
            print(f"Molecular property summary saved to {summary_file}")
            
            # 额外保存一个专门的反转能隙分析文件
            if 'primary_inversion_type' in summary_df.columns:
                inversion_df = summary_df[
                    (summary_df['num_inverted_gaps'] > 0) & 
                    (summary_df['primary_inversion_type'].notna())
                ]
                
                if not inversion_df.empty:
                    # 选择相关列
                    initial_cols = ['Molecule', 'primary_inversion_type', 'primary_inversion_gap', 
                                'primary_inversion_gap_meV', 'num_inverted_gaps']
                    
                    # 添加所有gap列，但排除已经在initial_cols中的列
                    gap_cols = [col for col in inversion_df.columns 
                            if col.endswith('_gap') 
                            and not col.endswith('_gap_meV')
                            and col not in initial_cols]  # 排除已经存在的列
                    
                    # 合并列名
                    cols_to_keep = initial_cols + gap_cols
                    
                    # 只保留存在的列
                    cols_to_keep = [col for col in cols_to_keep if col in inversion_df.columns]
                    
                    # 去除可能的重复列名（额外的安全措施）
                    cols_to_keep = list(dict.fromkeys(cols_to_keep))
                    
                    inversion_df = inversion_df[cols_to_keep].sort_values('primary_inversion_gap')
                    
                    inversion_file = os.path.join(output_dir, "inverted_gap_analysis.csv")
                    inversion_df.to_csv(inversion_file, index=False)
                    print(f"Inverted gap analysis saved to {inversion_file}")
            
            return summary_file
        
        return None