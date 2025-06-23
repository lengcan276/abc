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
    
<<<<<<< HEAD
    def __init__(self, base_dir='/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach/data/logs'):
=======
    def __init__(self, base_dir='/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach/data/logs'):
>>>>>>> 0181d62 (update excited)
        """Initialize the DataAgent with the base directory containing molecular data."""
        self.base_dir = base_dir
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for the data agent."""
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
<<<<<<< HEAD
                           filename='/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach/data/logs/data_agent.log')
=======
                           filename='/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach/data/logs/data_agent.log')
>>>>>>> 0181d62 (update excited)
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
    
    def extract_all_excited_states(self, log_file):
        """
        提取所有激发态信息，包括高阶激发态
        基于wB97X-D/def2-TZVP计算结果
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
            
            # 自动识别可能的反转能隙
            states['inverted_gaps'] = self.find_inverted_gaps(states)
            
            return states
            
        except Exception as e:
            self.logger.error(f"Error extracting excited states from {log_file}: {e}")
            return None

    def find_inverted_gaps(self, states):
        """
        寻找所有可能的反转单重态-三重态能隙
        不限于S1-T1，包括S2-T4, S3-T4等
        """
        inverted_pairs = []
        
        singlets = states['singlets']
        triplets = states['triplets']
        
        # 检查所有可能的单重态-三重态组合
        for i, singlet in enumerate(singlets):
            for j, triplet in enumerate(triplets):
                gap = singlet['energy_ev'] - triplet['energy_ev']
                
                # 如果单重态能量低于三重态（反转）
                if gap < 0:
                    # 计算跃迁相似度
                    similarity = self.calculate_transition_similarity(
                        singlet['transitions'], 
                        triplet['transitions']
                    )
                    
                    inverted_pairs.append({
                        'type': f"S{i+1}-T{j+1}",
                        'singlet_state': i + 1,
                        'triplet_state': j + 1,
                        'singlet_energy': singlet['energy_ev'],
                        'triplet_energy': triplet['energy_ev'],
                        'gap': gap,
                        'gap_meV': gap * 1000,  # 转换为meV
                        'singlet_symmetry': singlet['symmetry'],
                        'triplet_symmetry': triplet['symmetry'],
                        'transition_similarity': similarity,
                        'singlet_osc_strength': singlet['osc_strength']
                    })
        
        # 按能隙大小排序（最负的在前）
        inverted_pairs.sort(key=lambda x: x['gap'])
        
        return inverted_pairs

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
                    possible_paths = [
                        os.path.join(molecule_dir, 'results', f'{state}_{conf}.xyz'),
                        os.path.join(molecule_dir, 'results', f'{state}_results.xyz'),
                        os.path.join(molecule_dir, state, 'crest_best.xyz'),
                        os.path.join(molecule_dir, state, 'crest', f'{conf}.xyz'),
                        os.path.join(molecule_dir, state, 'crest', 'crest_best.xyz')
                    ]
                    
                    # 检查所有可能的路径
                    for path in possible_paths:
                        if os.path.exists(path):
                            crest_xyz = path
                            break
                            
                    # 如果未找到，尝试查找任何XYZ文件
                    if crest_xyz is None:
                        state_crest_dir = os.path.join(molecule_dir, state, 'crest')
                        if os.path.exists(state_crest_dir):
                            from glob import glob
                            xyz_files = glob(os.path.join(state_crest_dir, "*.xyz"))
                            if xyz_files:
                                crest_xyz = xyz_files[0]
                    
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
                            # 使用最简单的调用方式
                            mol = StructureUtils.load_molecule_from_gaussian(log_file)
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
            
            # For neutral state, also check excited state
            if state == 'neutral':
                excited_log = os.path.join(conf_path, 'excited.log')
                if os.path.exists(excited_log):
                    excited_energy = self.extract_energy(excited_log)
                    excited_opt_success = self.check_opt_success(excited_log)
                    excited_has_imaginary = self.check_imaginary_freq(excited_log)
                    excited_homo, excited_lumo = self.extract_homo_lumo(excited_log)
                    excited_dipole = self.extract_dipole(excited_log)
                    
                    # 使用新的全激发态提取方法
                    all_excited_states = self.extract_all_excited_states(excited_log)
                    
                    if all_excited_states:
                        # 保存完整的激发态信息
                        conf_data['all_excited_states'] = all_excited_states
                        
                        # 提取S1和T1信息（向后兼容）
                        if all_excited_states['singlets']:
                            s1_state = all_excited_states['singlets'][0]
                            conf_data['s1_energy_ev'] = s1_state['energy_ev']
                            conf_data['oscillator_strength'] = s1_state['osc_strength']
                        
                        if all_excited_states['triplets']:
                            t1_state = all_excited_states['triplets'][0]
                            conf_data['t1_energy_ev'] = t1_state['energy_ev']
                        
                        # 保存所有反转能隙信息
                        if all_excited_states['inverted_gaps']:
                            conf_data['inverted_gaps'] = all_excited_states['inverted_gaps']
                            
                            # 找到最重要的反转能隙（能隙最负的）
                            primary_inversion = all_excited_states['inverted_gaps'][0]
                            conf_data['primary_inversion_type'] = primary_inversion['type']
                            conf_data['primary_inversion_gap'] = primary_inversion['gap']
                            conf_data['primary_inversion_gap_meV'] = primary_inversion['gap_meV']
                            
                            # 如果主要反转是S1-T1，则保持向后兼容
                            if primary_inversion['type'] == 'S1-T1':
                                conf_data['s1_t1_gap_ev'] = primary_inversion['gap']
                                conf_data['s1_t1_gap'] = primary_inversion['gap']
                    else:
                        # 使用原始方法作为后备
                        s1_energy, osc_strength, t1_energy = self.extract_excitation_energy(excited_log)
                        
                        if s1_energy is not None:
                            conf_data['s1_energy_ev'] = s1_energy
                        if osc_strength is not None:
                            conf_data['oscillator_strength'] = osc_strength
                        if t1_energy is not None:
                            conf_data['t1_energy_ev'] = t1_energy
                        
                        # Calculate S1-T1 gap
                        if s1_energy is not None and t1_energy is not None:
                            conf_data['s1_t1_gap_ev'] = s1_energy - t1_energy
                            conf_data['s1_t1_gap'] = s1_energy - t1_energy

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
                    
                    # 激发态分子结构分析 - 仅当RDKit可用时进行
                    if has_rdkit:
                        try:
                            excited_mol = StructureUtils.load_molecule_from_gaussian(excited_log)
                            if excited_mol and excited_mol.GetNumAtoms() > 0:
                                # 计算基本特征
                                excited_features = {
                                    'excited_num_atoms': excited_mol.GetNumAtoms(),
                                    'excited_num_bonds': excited_mol.GetNumBonds(),
                                    'excited_mol_weight': Chem.rdMolDescriptors.CalcExactMolWt(excited_mol)
                                }
                                conf_data.update(excited_features)
                        except Exception as e:
                            self.logger.debug(f"Excited state structure analysis failed: {str(e)}")
                    
            # For neutral state with triplet data
            if state == 'neutral' and 'triplet' in os.listdir(molecule_dir):
                triplet_path = os.path.join(molecule_dir, 'triplet', 'gaussian', conf)
                if os.path.exists(triplet_path):
                    triplet_log = os.path.join(triplet_path, 'ground.log')
                    if os.path.exists(triplet_log):
                        triplet_energy = self.extract_energy(triplet_log)
                        # Calculate S1-T1 gap if triplet energy is available
                        if triplet_energy is not None and energy is not None:
                            t_s0_gap_ev = (triplet_energy - energy) * 27.2114
                            conf_data['triplet_gap_ev'] = t_s0_gap_ev
                            # Also copy to s1_t1_gap_ev variable if not already present
                            if 's1_t1_gap_ev' not in conf_data or conf_data['s1_t1_gap_ev'] is None:
                                # Only use this value if not already calculated via TD-DFT
                                conf_data['s1_t1_gap_ev'] = t_s0_gap_ev
                                conf_data['s1_t1_gap'] = t_s0_gap_ev
                        
                        # 三重态分子结构分析 - 仅当RDKit可用时进行
                        if has_rdkit:
                            try:
                                triplet_mol = StructureUtils.load_molecule_from_gaussian(triplet_log)
                                if triplet_mol and triplet_mol.GetNumAtoms() > 0:
                                    # 计算基本特征
                                    triplet_features = {
                                        'triplet_num_atoms': triplet_mol.GetNumAtoms(),
                                        'triplet_num_bonds': triplet_mol.GetNumBonds(),
                                        'triplet_mol_weight': Chem.rdMolDescriptors.CalcExactMolWt(triplet_mol)
                                    }
                                    conf_data.update(triplet_features)
                            except Exception as e:
                                self.logger.debug(f"Triplet state structure analysis failed: {str(e)}")
                
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

                                # Add to total dataset
                                all_conformers_data.append(conf_data)
                
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
<<<<<<< HEAD
            output_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach/data/extracted'
=======
            output_dir = '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach/data/extracted'
>>>>>>> 0181d62 (update excited)
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

                        # For neutral state, add excited state info
                        if state == 'neutral':
                            # 添加完整激发态信息 - 增加类型检查
                            if 'all_excited_states' in primary_conf:
                                all_states = primary_conf['all_excited_states']
                                # 确保 all_states 是字典类型
                                if isinstance(all_states, dict):
                                    # 保存激发态数量信息
                                    molecule_data['num_singlet_states'] = len(all_states.get('singlets', []))
                                    molecule_data['num_triplet_states'] = len(all_states.get('triplets', []))
                                    molecule_data['num_inverted_gaps'] = len(all_states.get('inverted_gaps', []))
                                else:
                                    # 如果不是字典，记录警告并跳过
                                    self.logger.warning(f"all_excited_states for {molecule} is not a dict but {type(all_states)}")
                            
                            # 保存主要反转信息
                            if 'primary_inversion_type' in primary_conf and not pd.isna(primary_conf['primary_inversion_type']):
                                molecule_data['primary_inversion_type'] = primary_conf['primary_inversion_type']
                                molecule_data['primary_inversion_gap'] = primary_conf.get('primary_inversion_gap')
                                molecule_data['primary_inversion_gap_meV'] = primary_conf.get('primary_inversion_gap_meV')
                            
                            # 保留原有的激发态信息字段
                            for excited_col in [c for c in primary_conf.keys()
                                            if c.startswith('excited_') or
                                            c in ['s1_energy_ev', 'oscillator_strength',
                                                    't1_energy_ev', 's1_t1_gap_ev', 's1_t1_gap',
                                                    'excitation_energy_ev']]:
                                if excited_col in primary_conf and not pd.isna(primary_conf[excited_col]):
                                    molecule_data[excited_col] = primary_conf[excited_col]

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
                    
                    # 确保我们同时有s1_t1_gap_ev值
                    if 's1_t1_gap_ev' not in molecule_data or molecule_data['s1_t1_gap_ev'] is None:
                        molecule_data['s1_t1_gap_ev'] = triplet_gap
                        molecule_data['s1_t1_gap'] = triplet_gap

            # 添加辅助检查，确保s1_t1_gap_ev存在
            if 's1_energy_ev' in molecule_data and 't1_energy_ev' in molecule_data:
                if molecule_data['s1_energy_ev'] is not None and molecule_data['t1_energy_ev'] is not None:
                    s1_t1_gap = molecule_data['s1_energy_ev'] - molecule_data['t1_energy_ev']
                    molecule_data['s1_t1_gap_ev'] = s1_t1_gap
                    molecule_data['s1_t1_gap'] = s1_t1_gap
                    print(f"    {molecule} S1-T1 Gap (from excitation): {s1_t1_gap} eV")

            # Add to summary data
            summary_data.append(molecule_data)

        # Create summary DataFrame
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            
            # 确保我们有s1_t1_gap_ev列，如果没有，尝试从其他列创建
            if 's1_t1_gap_ev' not in summary_df.columns:
                if 'triplet_gap_ev' in summary_df.columns:
                    summary_df['s1_t1_gap_ev'] = summary_df['triplet_gap_ev']
                    summary_df['s1_t1_gap'] = summary_df['triplet_gap_ev']
                    print("Created s1_t1_gap_ev column from triplet_gap_ev data")
                    
            # 最后检查，确保s1_t1_gap_ev列存在
            if 's1_t1_gap_ev' in summary_df.columns:
                # 打印负值计数，便于确认
                negative_count = (summary_df['s1_t1_gap_ev'] < 0).sum()
                print(f"Found {negative_count} molecules with negative S1-T1 gaps")
            
            # 打印反转能隙统计
            if 'num_inverted_gaps' in summary_df.columns:
                molecules_with_inversions = (summary_df['num_inverted_gaps'] > 0).sum()
                total_inversions = summary_df['num_inverted_gaps'].sum()
                print(f"Found {molecules_with_inversions} molecules with inverted gaps")
                print(f"Total inverted gap pairs: {total_inversions}")

            # Save summary data
            summary_file = os.path.join(output_dir, "molecular_properties_summary.csv")
            summary_df.to_csv(summary_file, index=False)
            print(f"Molecular property summary saved to {summary_file}")
            
            # 额外保存一个专门的反转能隙分析文件
            if 'primary_inversion_type' in summary_df.columns:
                # 添加额外的过滤条件，确保数据有效
                inversion_df = summary_df[
                    (summary_df['num_inverted_gaps'] > 0) & 
                    (summary_df['primary_inversion_type'].notna())
                ]
                
                if not inversion_df.empty:
                    # 选择相关列，并确保它们存在
                    cols_to_keep = ['Molecule']
                    for col in ['primary_inversion_type', 'primary_inversion_gap', 
                            'primary_inversion_gap_meV', 'num_inverted_gaps']:
                        if col in inversion_df.columns:
                            cols_to_keep.append(col)
                    
                    inversion_df = inversion_df[cols_to_keep].sort_values('primary_inversion_gap')
                    
                    inversion_file = os.path.join(output_dir, "inverted_gap_analysis.csv")
                    inversion_df.to_csv(inversion_file, index=False)
                    print(f"Inverted gap analysis saved to {inversion_file}")
            
            return summary_file
        
        return None