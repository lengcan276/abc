# utils/structure_utils.py
import numpy as np
import os
import re
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcNumRings, CalcNumAromaticRings
from rdkit.Chem.Scaffolds import MurckoScaffold
import logging
import glob

class StructureUtils:
    """用于分析和提取分子结构特征的工具类"""
    
    @staticmethod
    def load_molecule_from_gaussian(gaussian_log, fallback_to_crest=True, crest_xyz_file=None):
        """从Gaussian日志文件中提取分子结构，如果失败则尝试从CREST XYZ文件加载"""
        try:
            if not os.path.exists(gaussian_log):
                if not fallback_to_crest:
                    return None
                else:
                    # 如果高斯文件不存在，直接尝试从CREST文件加载
                    return StructureUtils.find_and_load_from_crest(gaussian_log, crest_xyz_file)
            
            # 尝试从Gaussian日志文件中提取最终优化的结构
            with open(gaussian_log, 'r', errors='replace') as f:
                content = f.read()
                
            # 寻找最后一个Standard orientation部分
            std_orient_sections = re.findall(r'Standard orientation:.*?---------------------------------------------------------------------\n(.*?)----', 
                                           content, re.DOTALL)
            
            if not std_orient_sections:
                # 尝试寻找Input orientation部分
                input_orient_sections = re.findall(r'Input orientation:.*?---------------------------------------------------------------------\n(.*?)----',
                                                 content, re.DOTALL)
                if not input_orient_sections:
                    if not fallback_to_crest:
                        return None
                    else:
                        # 无法从高斯文件提取结构，尝试从CREST文件加载
                        return StructureUtils.find_and_load_from_crest(gaussian_log, crest_xyz_file)
                orient_section = input_orient_sections[-1]  # 使用最后一个Input orientation
            else:
                orient_section = std_orient_sections[-1]  # 使用最后一个Standard orientation
                
            # 解析原子坐标
            atoms = []
            for line in orient_section.split('\n'):
                parts = line.split()
                if len(parts) >= 6 and parts[0].isdigit() and parts[1].isdigit():
                    atom_num = int(parts[1])
                    x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                    atoms.append((atom_num, x, y, z))
            
            if not atoms:
                if not fallback_to_crest:
                    return None
                else:
                    # 找不到原子数据，尝试从CREST文件加载
                    return StructureUtils.find_and_load_from_crest(gaussian_log, crest_xyz_file)
            
            # 创建RDKit分子对象
            mol = Chem.RWMol()
            
            # 添加原子
            atom_map = {}
            for i, (atom_num, x, y, z) in enumerate(atoms):
                # 将原子序数转换为元素符号
                atom_sym = StructureUtils._atomic_num_to_symbol(atom_num)
                atom = Chem.Atom(atom_sym)
                atom_idx = mol.AddAtom(atom)
                atom_map[i] = atom_idx
                
            # 创建构象并设置原子坐标
            conf = Chem.Conformer(len(atoms))
            for i, (atom_num, x, y, z) in enumerate(atoms):
                atom_idx = atom_map[i]
                conf.SetAtomPosition(atom_idx, (x, y, z))
            
            # 尝试推断键
            mol = mol.GetMol()
            mol.AddConformer(conf)
            
            # 使用RDKit的连接原子功能
            mol = AllChem.AssignBondOrdersFromTemplate(Chem.Mol(), mol)
            
            # 如果连接失败，尝试使用距离矩阵
            if mol.GetNumBonds() == 0:
                mol = StructureUtils._connect_atoms_by_distance(mol)
            
            # 如果仍然失败，尝试CREST
            if mol.GetNumBonds() == 0 and fallback_to_crest:
                return StructureUtils.find_and_load_from_crest(gaussian_log, crest_xyz_file)
                
            return mol
            
        except Exception as e:
            logging.error(f"从Gaussian日志提取分子结构时出错: {e}")
            if fallback_to_crest:
                # 尝试从CREST文件加载
                return StructureUtils.find_and_load_from_crest(gaussian_log, crest_xyz_file)
            return None
    
    @staticmethod
    def find_and_load_from_crest(gaussian_path, crest_xyz_file=None):
        """
        查找并加载CREST构型文件
        智能地尝试定位CREST文件，即使未明确提供
        """
        try:
            # 如果明确提供了CREST文件路径
            if crest_xyz_file and os.path.exists(crest_xyz_file):
                return StructureUtils.load_molecule_from_xyz(crest_xyz_file)
            
            # 否则，尝试从gaussian路径推断可能的CREST文件位置
            # 首先拆分路径以获取结构
            path_parts = gaussian_path.split(os.sep)
            
            # 假设路径类似：/path/to/molecule/state/gaussian/conf_1/ground.log
            if len(path_parts) >= 4:
                mol_name_idx = max(0, len(path_parts) - 5)
                molecule_name = path_parts[mol_name_idx]
                
                # 提取状态和构象名
                state = None
                conf_name = None
                
                for part in path_parts:
                    if part in ['neutral', 'cation', 'triplet']:
                        state = part
                    if part.startswith('conf_'):
                        conf_name = part
                
                if state and conf_name:
                    # 尝试几种可能的CREST文件位置
                    potential_paths = [
                        # 路径类型1: molecule/results/state_conf.xyz
                        os.path.join(*path_parts[:mol_name_idx+1], 'results', f"{state}_{conf_name}.xyz"),
                        # 路径类型2: molecule/results/conf/state_conf.xyz
                        os.path.join(*path_parts[:mol_name_idx+1], 'results', conf_name, f"{state}.xyz"),
                        # 路径类型3: molecule/state/crest/conf.xyz
                        os.path.join(*path_parts[:mol_name_idx+1], state, 'crest', f"{conf_name}.xyz"),
                    ]
                    
                    # 尝试更通用的搜索模式
                    results_dir = os.path.join(*path_parts[:mol_name_idx+1], 'results')
                    if os.path.exists(results_dir):
                        # 查找所有.xyz文件
                        xyz_files = glob.glob(os.path.join(results_dir, '**', '*.xyz'), recursive=True)
                        
                        # 添加到潜在路径
                        potential_paths.extend(xyz_files)
                    
                    # 尝试所有可能的路径
                    for path in potential_paths:
                        if os.path.exists(path):
                            print(f"找到CREST构型文件: {path}")
                            return StructureUtils.load_molecule_from_xyz(path)
            
            # 如果无法找到匹配的文件，尝试一般搜索
            # 查找以状态名（如neutral/cation/triplet）开头，以.xyz结尾的文件
            base_dir = os.path.dirname(gaussian_path)
            while base_dir and not os.path.basename(base_dir) in ['neutral', 'cation', 'triplet', 'gaussian']:
                base_dir = os.path.dirname(base_dir)
            
            if base_dir:
                results_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)), 'results')
                if os.path.exists(results_dir):
                    state = os.path.basename(base_dir)
                    xyz_files = glob.glob(os.path.join(results_dir, f"{state}*.xyz"))
                    
                    if xyz_files:
                        print(f"找到可能的CREST构型文件: {xyz_files[0]}")
                        return StructureUtils.load_molecule_from_xyz(xyz_files[0])
            
            print(f"无法找到匹配的CREST构型文件")
            return None
            
        except Exception as e:
            logging.error(f"查找CREST构型文件时出错: {e}")
            return None
    
    @staticmethod
    def load_molecule_from_xyz(xyz_file):
        """从XYZ文件加载分子结构"""
        if not os.path.exists(xyz_file):
            return None
            
        try:
            with open(xyz_file, 'r') as f:
                lines = f.readlines()
                
            # 检查是否是有效的XYZ文件
            if len(lines) < 2:
                return None
                
            try:
                num_atoms = int(lines[0].strip())
            except:
                return None
                
            # 跳过前两行（原子数和注释），解析原子坐标
            atoms = []
            for i in range(2, min(2 + num_atoms, len(lines))):
                parts = lines[i].strip().split()
                if len(parts) >= 4:
                    atom_sym = parts[0]
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    atoms.append((atom_sym, x, y, z))
            
            if not atoms:
                return None
                
            # 创建RDKit分子对象
            mol = Chem.RWMol()
            
            # 添加原子
            atom_map = {}
            for i, (atom_sym, x, y, z) in enumerate(atoms):
                atom = Chem.Atom(atom_sym)
                atom_idx = mol.AddAtom(atom)
                atom_map[i] = atom_idx
                
            # 设置原子坐标
            conf = Chem.Conformer(len(atoms))
            for i, (atom_sym, x, y, z) in enumerate(atoms):
                atom_idx = atom_map[i]
                conf.SetAtomPosition(atom_idx, (x, y, z))
                
            # 完成分子构建
            mol = mol.GetMol()
            mol.AddConformer(conf)
            
            # 使用RDKit的连接原子功能
            try:
                mol = AllChem.AssignBondOrdersFromTemplate(Chem.Mol(), mol)
            except:
                pass
                
            # 如果连接失败，尝试使用距离矩阵
            if mol.GetNumBonds() == 0:
                mol = StructureUtils._connect_atoms_by_distance(mol)
                
            return mol
            
        except Exception as e:
            logging.error(f"从XYZ文件加载分子结构时出错: {e}")
            return None
    
    @staticmethod
    def _connect_atoms_by_distance(mol):
        """基于原子间距离连接原子"""
        try:
            # 获取构象
            conf = mol.GetConformer()
            
            # 定义元素的共价半径（埃）
            covalent_radii = {
                'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
                'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Br': 1.20,
                'I': 1.39, 'B': 0.83
            }
            
            # 定义系数，用于宽容度
            tolerance = 1.3  # 允许最大共价半径之和的1.3倍
            
            # 创建可写分子对象
            rwmol = Chem.RWMol(mol)
            
            # 遍历所有原子对
            for i in range(mol.GetNumAtoms()):
                for j in range(i+1, mol.GetNumAtoms()):
                    atom_i = mol.GetAtomWithIdx(i)
                    atom_j = mol.GetAtomWithIdx(j)
                    
                    # 获取原子元素
                    elem_i = atom_i.GetSymbol()
                    elem_j = atom_j.GetSymbol()
                    
                    # 获取共价半径
                    radius_i = covalent_radii.get(elem_i, 0.7)  # 默认值
                    radius_j = covalent_radii.get(elem_j, 0.7)  # 默认值
                    
                    # 计算两原子之间的最大允许距离
                    max_dist = (radius_i + radius_j) * tolerance
                    
                    # 获取原子坐标
                    pos_i = conf.GetAtomPosition(i)
                    pos_j = conf.GetAtomPosition(j)
                    
                    # 计算距离
                    dist = ((pos_i.x - pos_j.x)**2 + 
                           (pos_i.y - pos_j.y)**2 + 
                           (pos_i.z - pos_j.z)**2)**0.5
                    
                    # 如果距离小于允许的最大距离，添加单键
                    if dist <= max_dist:
                        rwmol.AddBond(i, j, Chem.BondType.SINGLE)
            
            # 返回新分子
            return rwmol.GetMol()
            
        except Exception as e:
            logging.error(f"通过距离连接原子时出错: {e}")
            return mol
    
    @staticmethod
    def _atomic_num_to_symbol(atomic_num):
        """将原子序数转换为元素符号"""
        periodic_table = {
            1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
            9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
            16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 26: 'Fe', 35: 'Br', 53: 'I'
        }
        return periodic_table.get(atomic_num, 'C')  # 默认为C
    
    @staticmethod
    def calculate_structure_features(mol):
        """计算分子的结构特征，包括共轭、环系统、平面性等"""
        if mol is None:
            return {}
            
        features = {}
        
        try:
            # 基本特征
            features['num_atoms'] = mol.GetNumAtoms()
            features['num_heavy_atoms'] = mol.GetNumHeavyAtoms()
            features['num_bonds'] = mol.GetNumBonds()
            
            # 环系统
            features['num_rings'] = CalcNumRings(mol)
            features['num_aromatic_rings'] = CalcNumAromaticRings(mol)
            
            # 尝试生成SMILES字符串
            try:
                features['smiles'] = Chem.MolToSmiles(mol)
            except:
                features['smiles'] = ''
            
            # 共轭系统特征
            # 计算π键和共轭系统
            num_pi_bonds = 0
            num_conjugated_bonds = 0
            conjugated_systems = []
            
            # 标记共轭键
            conjugated = [False] * mol.GetNumBonds()
            
            # 首先标记所有可能参与共轭的键
            for bond_idx in range(mol.GetNumBonds()):
                bond = mol.GetBondWithIdx(bond_idx)
                if bond.GetBondType() == Chem.BondType.DOUBLE or bond.GetIsAromatic():
                    num_pi_bonds += 1
                    conjugated[bond_idx] = True
            
            # 识别共轭系统
            visited = [False] * mol.GetNumBonds()
            for start_idx in range(mol.GetNumBonds()):
                if visited[start_idx] or not conjugated[start_idx]:
                    continue
                    
                # 发现一个新的共轭系统
                system_size = 0
                stack = [start_idx]
                while stack:
                    bond_idx = stack.pop()
                    if visited[bond_idx]:
                        continue
                        
                    visited[bond_idx] = True
                    system_size += 1
                    
                    # 查找相邻的共轭键
                    bond = mol.GetBondWithIdx(bond_idx)
                    begin_atom = bond.GetBeginAtomIdx()
                    end_atom = bond.GetEndAtomIdx()
                    
                    for next_bond in mol.GetBonds():
                        next_idx = next_bond.GetIdx()
                        if visited[next_idx] or not conjugated[next_idx]:
                            continue
                            
                        # 检查是否相邻
                        next_begin = next_bond.GetBeginAtomIdx()
                        next_end = next_bond.GetEndAtomIdx()
                        
                        if (begin_atom == next_begin or begin_atom == next_end or
                            end_atom == next_begin or end_atom == next_end):
                            stack.append(next_idx)
                
                if system_size > 1:
                    num_conjugated_bonds += system_size
                    conjugated_systems.append(system_size)
            
            features['num_pi_bonds'] = num_pi_bonds
            features['num_conjugated_bonds'] = num_conjugated_bonds
            features['num_conjugated_systems'] = len(conjugated_systems)
            features['max_conjugated_system_size'] = max(conjugated_systems) if conjugated_systems else 0
            
            # 平面性分析 - 使用RMSD计算
            try:
                # 获取原子坐标
                conf = mol.GetConformer()
                coords = []
                for i in range(mol.GetNumAtoms()):
                    pos = conf.GetAtomPosition(i)
                    coords.append([pos.x, pos.y, pos.z])
                
                coords = np.array(coords)
                
                # 计算最佳拟合平面
                if len(coords) >= 3:
                    centroid = np.mean(coords, axis=0)
                    centered_coords = coords - centroid
                    
                    # 计算协方差矩阵
                    cov = np.cov(centered_coords, rowvar=False)
                    
                    # 计算特征值和特征向量
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    
                    # 最小特征值对应的特征向量就是平面法向量
                    normal = eigenvectors[:, 0]
                    
                    # 计算到平面的距离
                    distances = np.abs(np.dot(centered_coords, normal))
                    
                    # 计算RMSD作为平面度量
                    planarity_rmsd = np.sqrt(np.mean(distances**2))
                    
                    features['planarity_rmsd'] = planarity_rmsd
                    
                    # 添加更多平面性分析指标
                    features['planarity_max_dist'] = np.max(distances)  # 最大偏离距离
                    features['planarity_ratio'] = eigenvalues[0] / (eigenvalues[1] + 1e-10)  # 第一和第二特征值的比率，越小越平面
                else:
                    features['planarity_rmsd'] = 0
            except:
                features['planarity_rmsd'] = None
            
            # 扭曲分析 - 计算所有二面角
            try:
                torsions = []
                for i in range(mol.GetNumBonds()):
                    bond = mol.GetBondWithIdx(i)
                    if bond.IsInRing():
                        continue  # 跳过环内的键
                        
                    a1 = bond.GetBeginAtomIdx()
                    a2 = bond.GetEndAtomIdx()
                    
                    # 查找与a1相连的其他原子
                    a1_neighbors = [a.GetIdx() for a in mol.GetAtomWithIdx(a1).GetNeighbors() if a.GetIdx() != a2]
                    
                    # 查找与a2相连的其他原子
                    a2_neighbors = [a.GetIdx() for a in mol.GetAtomWithIdx(a2).GetNeighbors() if a.GetIdx() != a1]
                    
                    # 对所有可能的四元组计算二面角
                    for n1 in a1_neighbors:
                        for n2 in a2_neighbors:
                            try:
                                # 使用RDKit计算二面角
                                torsion = AllChem.GetDihedralDeg(mol.GetConformer(), n1, a1, a2, n2)
                                # 将角度标准化到0-90度范围
                                torsion = abs(torsion)
                                if torsion > 90:
                                    torsion = 180 - torsion
                                torsions.append(torsion)
                            except:
                                pass
                
                if torsions:
                    features['torsion_avg'] = np.mean(torsions)
                    features['torsion_max'] = np.max(torsions)
                    features['torsion_std'] = np.std(torsions)
                    # 计算平面扭曲指数（0表示完全平面，90表示极度扭曲）
                    features['twist_index'] = np.mean([t/90.0 for t in torsions])
            except:
                pass
            
            # 氢键分析
            # 识别潜在的氢键受体和供体
            num_h_donors = 0
            num_h_acceptors = 0
            
            for atom in mol.GetAtoms():
                # 氢键受体 (N, O, F 等电负性高的原子)
                if atom.GetSymbol() in ['N', 'O', 'F']:
                    num_h_acceptors += 1
                
                # 氢键供体 (连接到N, O的氢原子)
                if atom.GetSymbol() in ['N', 'O'] and atom.GetTotalNumHs() > 0:
                    num_h_donors += atom.GetTotalNumHs()
            
            features['num_h_donors'] = num_h_donors
            features['num_h_acceptors'] = num_h_acceptors
            
            # 添加更多特性
            features['molecular_weight'] = MolWt(mol)
            
            # TADF特定分析：官能团存在分析
            # 检测反转TADF分子中常见的基团
            tadf_groups = {
                'cyano': '[C]#N',  # 氰基
                'nitro': '[N+](=O)[O-]',  # 硝基
                'amino': 'N[H]',  # 氨基
                'carbonyl': 'C=O',  # 羰基
                'sulfone': 'S(=O)(=O)',  # 砜
                'triazine': 'c1ncncn1',  # 三嗪
                'boron': 'B',  # 硼
                'phosphorus': 'P',  # 磷
                'halogen': '[F,Cl,Br,I]',  # 卤素
                'methyl': 'C[H]',  # 甲基
                'trifluoromethyl': 'C(F)(F)F'  # 三氟甲基
            }
            
            for group_name, smarts in tadf_groups.items():
                try:
                    pattern = Chem.MolFromSmarts(smarts)
                    if pattern:
                        matches = mol.GetSubstructMatches(pattern)
                        features[f'has_{group_name}'] = 1 if matches else 0
                        features[f'count_{group_name}'] = len(matches)
                except:
                    features[f'has_{group_name}'] = 0
                    features[f'count_{group_name}'] = 0
            
            # 特殊环系统分析
            ring_systems = {
                'phenyl': 'c1ccccc1',  # 苯环
                'pyridine': 'c1ccncc1',  # 吡啶
                'triazine': 'c1ncncn1',  # 三嗪
                'thiophene': 'c1sccc1',  # 噻吩
                'furan': 'c1occc1',  # 呋喃
                'pyrrole': 'c1[nH]ccc1',  # 吡咯
                'carbazole': 'c1ccc2c(c1)c3ccccc3[nH]2',  # 咔唑
                'triphenylamine': 'N(c1ccccc1)(c2ccccc2)c3ccccc3'  # 三苯胺
            }
            
            for ring_name, smarts in ring_systems.items():
                try:
                    pattern = Chem.MolFromSmarts(smarts)
                    if pattern:
                        matches = mol.GetSubstructMatches(pattern)
                        features[f'has_{ring_name}'] = 1 if matches else 0
                        features[f'count_{ring_name}'] = len(matches)
                except:
                    features[f'has_{ring_name}'] = 0
                    features[f'count_{ring_name}'] = 0
            
            return features
            
        except Exception as e:
            logging.error(f"计算分子结构特征时出错: {e}")
            return {}
    
    @staticmethod
    def compare_structures(mol1, mol2):
        """比较两个分子结构（如基态和激发态）的变化"""
        if mol1 is None or mol2 is None:
            return {}
            
        changes = {}
        
        try:
            # 确保分子具有相同数量的原子
            if mol1.GetNumAtoms() != mol2.GetNumAtoms():
                return {'structure_mismatch': True}
                
            # 获取构象
            conf1 = mol1.GetConformer()
            conf2 = mol2.GetConformer()
            
            # 计算RMSD
            coords1 = []
            coords2 = []
            for i in range(mol1.GetNumAtoms()):
                pos1 = conf1.GetAtomPosition(i)
                pos2 = conf2.GetAtomPosition(i)
                coords1.append([pos1.x, pos1.y, pos1.z])
                coords2.append([pos2.x, pos2.y, pos2.z])
                
            coords1 = np.array(coords1)
            coords2 = np.array(coords2)
            
            # 计算RMSD
            rmsd = np.sqrt(np.mean(np.sum((coords1 - coords2)**2, axis=1)))
            changes['structure_rmsd'] = rmsd
            
            # 计算键长变化
            bond_lengths1 = {}
            bond_lengths2 = {}
            
            for bond in mol1.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                
                pos1_begin = conf1.GetAtomPosition(begin_idx)
                pos1_end = conf1.GetAtomPosition(end_idx)
                
                # 计算键长
                length = np.sqrt((pos1_begin.x - pos1_end.x)**2 + 
                                 (pos1_begin.y - pos1_end.y)**2 + 
                                 (pos1_begin.z - pos1_end.z)**2)
                
                bond_lengths1[(begin_idx, end_idx)] = length
                
            for bond in mol2.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                
                pos2_begin = conf2.GetAtomPosition(begin_idx)
                pos2_end = conf2.GetAtomPosition(end_idx)
                
                # 计算键长
                length = np.sqrt((pos2_begin.x - pos2_end.x)**2 + 
                                 (pos2_begin.y - pos2_end.y)**2 + 
                                 (pos2_begin.z - pos2_end.z)**2)
                
                bond_lengths2[(begin_idx, end_idx)] = length
            
            # 计算平均键长变化
            bond_length_diffs = []
            for key in bond_lengths1:
                if key in bond_lengths2:
                    diff = bond_lengths2[key] - bond_lengths1[key]
                    bond_length_diffs.append(diff)
            
            if bond_length_diffs:
                changes['mean_bond_length_change'] = np.mean(bond_length_diffs)
                changes['max_bond_length_change'] = np.max(np.abs(bond_length_diffs))
            
            # 比较分子特征变化
            features1 = StructureUtils.calculate_structure_features(mol1)
            features2 = StructureUtils.calculate_structure_features(mol2)
            
            for key in features1:
                if key in features2 and isinstance(features1[key], (int, float)) and isinstance(features2[key], (int, float)):
                    changes[f'{key}_change'] = features2[key] - features1[key]
            
            # 分析TADF相关的结构变化
            # 比较平面性变化，对反转TADF分子尤为重要
            if 'planarity_rmsd' in features1 and 'planarity_rmsd' in features2:
                changes['planarity_rmsd_change'] = features2['planarity_rmsd'] - features1['planarity_rmsd']
                # 平面性变化百分比
                if features1['planarity_rmsd'] > 0:
                    changes['planarity_change_percent'] = (features2['planarity_rmsd'] - features1['planarity_rmsd']) / features1['planarity_rmsd'] * 100
            
            # 比较二面角变化，检测扭曲
            if 'torsion_avg' in features1 and 'torsion_avg' in features2:
                changes['torsion_avg_change'] = features2['torsion_avg'] - features1['torsion_avg']
            
            # 比较共轭系统变化
            if 'num_conjugated_bonds' in features1 and 'num_conjugated_bonds' in features2:
                changes['conjugation_change'] = features2['num_conjugated_bonds'] - features1['num_conjugated_bonds']
            
            return changes
            
        except Exception as e:
            logging.error(f"比较分子结构时出错: {e}")
            return {}
