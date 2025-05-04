# utils/structure_utils.py
import numpy as np
import re
import os
import logging
from math import sqrt, acos, degrees
from collections import defaultdict

class StructureUtils:
    """用于提取和分析分子结构特征的工具类"""
    
    @staticmethod
    def load_molecule_from_gaussian(log_file, fallback_to_crest=True, crest_xyz_file=None):
        """从Gaussian日志文件加载分子结构
        
        Args:
            log_file: Gaussian日志文件路径
            fallback_to_crest: 如果无法从Gaussian加载，是否尝试CREST结构
            crest_xyz_file: CREST XYZ文件路径（如果有）
            
        Returns:
            成功返回包含分子结构信息的字典，失败返回None
        """
        try:
            # 尝试从Gaussian日志文件中提取最后一个优化的结构
            with open(log_file, 'r', errors='replace') as f:
                content = f.read()
                
            # 查找最后一个标准取向部分
            standard_orientations = re.findall(
                r'Standard orientation:.*?'
                r'-+\n'  # 分隔线
                r'(.*?)'  # 原子坐标
                r'\s*-+\n',
                content, re.DOTALL
            )
            
            if standard_orientations:
                # 获取最后一个优化结构
                orientation = standard_orientations[-1]
                
                # 解析原子坐标
                atoms = []
                atom_types = []
                coordinates = []
                
                for line in orientation.strip().split('\n'):
                    parts = line.split()
                    if len(parts) >= 6 and parts[0].isdigit():
                        atom_num = int(parts[0])
                        atom_type = int(parts[1])
                        atom_types.append(atom_type)
                        
                        # 转换原子编号到元素符号
                        if atom_type == 1:
                            atom = 'H'
                        elif atom_type == 6:
                            atom = 'C'
                        elif atom_type == 7:
                            atom = 'N'
                        elif atom_type == 8:
                            atom = 'O'
                        elif atom_type == 9:
                            atom = 'F'
                        elif atom_type == 16:
                            atom = 'S'
                        elif atom_type == 5:
                            atom = 'B'
                        else:
                            atom = f'X{atom_type}'  # 未知原子类型
                            
                        atoms.append(atom)
                        
                        # 提取原子坐标
                        x, y, z = float(parts[3]), float(parts[4]), float(parts[5])
                        coordinates.append([x, y, z])
                
                if atoms and coordinates:
                    return {
                        'atoms': atoms,
                        'atom_types': atom_types,
                        'coordinates': np.array(coordinates),
                        'source': 'gaussian'
                    }
            
            # 如果从Gaussian未能提取结构且允许回退到CREST
            if fallback_to_crest and crest_xyz_file and os.path.exists(crest_xyz_file):
                return StructureUtils.load_molecule_from_xyz(crest_xyz_file)
                
            return None
            
        except Exception as e:
            logging.error(f"从Gaussian加载分子结构时出错: {e}")
            
            # 如果从Gaussian未能提取结构且允许回退到CREST
            if fallback_to_crest and crest_xyz_file and os.path.exists(crest_xyz_file):
                return StructureUtils.load_molecule_from_xyz(crest_xyz_file)
                
            return None
    
    @staticmethod
    def load_molecule_from_xyz(xyz_file):
        """从XYZ文件加载分子结构
        
        Args:
            xyz_file: XYZ文件路径
            
        Returns:
            成功返回包含分子结构信息的字典，失败返回None
        """
        try:
            atoms = []
            atom_types = []
            coordinates = []
            
            with open(xyz_file, 'r') as f:
                lines = f.readlines()
                
            # 跳过前两行（原子数和注释）
            for line in lines[2:]:
                parts = line.strip().split()
                if len(parts) >= 4:
                    atom = parts[0]
                    atoms.append(atom)
                    
                    # 转换元素符号到原子类型编号
                    if atom == 'H':
                        atom_type = 1
                    elif atom == 'C':
                        atom_type = 6
                    elif atom == 'N':
                        atom_type = 7
                    elif atom == 'O':
                        atom_type = 8
                    elif atom == 'F':
                        atom_type = 9
                    elif atom == 'S':
                        atom_type = 16
                    elif atom == 'B':
                        atom_type = 5
                    else:
                        # 尝试从元素周期表符号获取编号
                        atom_type = 0  # 未知
                        
                    atom_types.append(atom_type)
                    
                    # 提取原子坐标
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    coordinates.append([x, y, z])
            
            if atoms and coordinates:
                return {
                    'atoms': atoms,
                    'atom_types': atom_types,
                    'coordinates': np.array(coordinates),
                    'source': 'xyz'
                }
                
            return None
            
        except Exception as e:
            logging.error(f"从XYZ文件加载分子结构时出错: {e}")
            return None
    
    @staticmethod
    def calculate_bond_length(coord1, coord2):
        """计算两点之间的距离（键长）"""
        return sqrt(sum((a - b)**2 for a, b in zip(coord1, coord2)))
    
    @staticmethod
    def calculate_bond_angle(coord1, coord2, coord3):
        """计算三点形成的键角（度）"""
        # 计算向量
        v1 = coord1 - coord2
        v2 = coord3 - coord2
        
        # 计算向量长度
        v1_len = np.linalg.norm(v1)
        v2_len = np.linalg.norm(v2)
        
        # 如果任一长度为零，返回None
        if v1_len < 1e-6 or v2_len < 1e-6:
            return None
        
        # 计算向量间夹角的余弦值
        cos_angle = np.dot(v1, v2) / (v1_len * v2_len)
        
        # 处理数值误差，确保cos_angle在[-1, 1]范围内
        cos_angle = max(-1.0, min(1.0, cos_angle))
        
        # 将余弦值转换为角度
        return degrees(acos(cos_angle))
    
    @staticmethod
    def calculate_dihedral_angle(coord1, coord2, coord3, coord4):
        """计算四点形成的二面角（度）"""
        # 计算法向量
        b1 = coord2 - coord1
        b2 = coord3 - coord2
        b3 = coord4 - coord3
        
        # 检查共线情况
        if np.linalg.norm(b2) < 1e-6:
            return None
            
        # 计算法平面的法向量
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        
        # 如果任一法向量接近零向量，返回None
        if np.linalg.norm(n1) < 1e-6 or np.linalg.norm(n2) < 1e-6:
            return None
        
        # 将法向量归一化
        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)
        
        # 计算法向量间的夹角
        cos_angle = np.dot(n1, n2)
        
        # 处理数值误差，确保cos_angle在[-1, 1]范围内
        cos_angle = max(-1.0, min(1.0, cos_angle))
        
        # 计算二面角值
        angle = degrees(acos(cos_angle))
        
        # 确定符号（使用叉积方向判断）
        if np.dot(np.cross(n1, n2), b2) < 0:
            angle = -angle
            
        return angle
    
    @staticmethod
    def build_bonds(mol, distance_threshold=1.7):
        """根据原子间距离构建分子中的化学键
        
        Args:
            mol: 包含分子信息的字典
            distance_threshold: 判断化学键的最大距离阈值
            
        Returns:
            包含键信息的列表
        """
        if not mol:
            return []
            
        bonds = []
        coords = mol['coordinates']
        atoms = mol['atoms']
        n_atoms = len(atoms)
        
        # 为不同原子对设置不同的距离阈值
        distance_thresholds = {
            ('H', 'H'): 1.0,
            ('H', 'C'): 1.2,
            ('H', 'N'): 1.1,
            ('H', 'O'): 1.1,
            ('H', 'F'): 1.0,
            ('H', 'S'): 1.4,
            ('H', 'B'): 1.3,
            ('C', 'C'): 1.7,
            ('C', 'N'): 1.6,
            ('C', 'O'): 1.5,
            ('C', 'F'): 1.4,
            ('C', 'S'): 1.9,
            ('C', 'B'): 1.7,
            ('N', 'N'): 1.5,
            ('N', 'O'): 1.5,
            ('N', 'F'): 1.4,
            ('N', 'S'): 1.8,
            ('N', 'B'): 1.6,
            ('O', 'O'): 1.5,
            ('O', 'F'): 1.4,
            ('O', 'S'): 1.8,
            ('O', 'B'): 1.5,
            ('F', 'F'): 1.4,
            ('F', 'S'): 1.7,
            ('F', 'B'): 1.4,
            ('S', 'S'): 2.2,
            ('S', 'B'): 2.0,
            ('B', 'B'): 1.8
        }
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                atom_i = atoms[i]
                atom_j = atoms[j]
                
                # 获取这对原子的距离阈值
                # 确保原子对正确排序（按字母顺序）
                atom_pair = tuple(sorted([atom_i, atom_j]))
                threshold = distance_thresholds.get(atom_pair, distance_threshold)
                
                # 计算距离
                distance = StructureUtils.calculate_bond_length(coords[i], coords[j])
                
                # 如果距离小于阈值，认为有化学键
                if distance < threshold:
                    bonds.append((i, j, distance))
        
        return bonds
    
    @staticmethod
    def identify_rings(mol, bonds):
        """识别分子中的环结构
        
        简化的环识别算法，仅识别常见环大小(3-8)
        
        Args:
            mol: 包含分子信息的字典
            bonds: 化学键信息
            
        Returns:
            环列表，每个环是组成环的原子索引列表
        """
        if not mol or not bonds:
            return []
            
        # 创建邻接列表
        neighbors = defaultdict(list)
        for i, j, _ in bonds:
            neighbors[i].append(j)
            neighbors[j].append(i)
            
        # 识别环
        rings = []
        atoms = mol['atoms']
        
        # 对每个原子进行深度优先搜索
        for start_atom in range(len(atoms)):
            # 对环大小3-8进行搜索
            for ring_size in range(3, 9):
                # 深度优先搜索函数
                def dfs(current, path, visited, depth):
                    if depth == ring_size:
                        # 检查是否能回到起点
                        if start_atom in neighbors[current]:
                            new_ring = sorted(path)
                            if new_ring not in rings:
                                rings.append(new_ring)
                        return
                    
                    # 尝试访问邻居
                    for neighbor in neighbors[current]:
                        if neighbor != start_atom and neighbor not in visited:
                            dfs(neighbor, path + [neighbor], visited | {neighbor}, depth + 1)
                
                # 从起点开始搜索
                dfs(start_atom, [start_atom], {start_atom}, 1)
        
        return rings
    
    @staticmethod
    def is_aromatic_ring(mol, ring, bonds):
        """判断环是否为芳香环
        
        简化判断：6元环且所有键长相近
        
        Args:
            mol: 包含分子信息的字典
            ring: 环原子索引列表
            bonds: 化学键信息
            
        Returns:
            如果是芳香环返回True，否则返回False
        """
        if len(ring) != 6:  # 仅考虑6元环
            return False
            
        # 获取环中所有键的长度
        ring_bonds = []
        for i in range(len(ring)):
            j = (i + 1) % len(ring)
            atom_i = ring[i]
            atom_j = ring[j]
            
            # 找到相应的键
            bond_found = False
            for bond in bonds:
                if (bond[0] == atom_i and bond[1] == atom_j) or (bond[0] == atom_j and bond[1] == atom_i):
                    ring_bonds.append(bond[2])  # 添加键长
                    bond_found = True
                    break
            
            if not bond_found:
                return False  # 如果环中不是所有相邻原子都有键，则不是环
        
        # 检查键长是否相近（芳香环应该具有相似的键长）
        if ring_bonds:
            avg_bond_length = sum(ring_bonds) / len(ring_bonds)
            # 所有键长与平均值的差异都应小于阈值
            return all(abs(length - avg_bond_length) < 0.1 for length in ring_bonds)
        
        return False
    
    @staticmethod
    def identify_conjugated_paths(mol, bonds):
        """识别分子中的共轭路径
        
        寻找sp2杂化原子形成的共轭系统
        
        Args:
            mol: 包含分子信息的字典
            bonds: 化学键信息
            
        Returns:
            共轭路径列表
        """
        if not mol or not bonds:
            return []
            
        atoms = mol['atoms']
        
        # 识别可能参与共轭的原子（C, N, O等sp2杂化的原子）
        conjugation_candidates = []
        for i, atom in enumerate(atoms):
            if atom in ['C', 'N', 'O', 'S']:
                # 计算该原子的键数
                bond_count = sum(1 for bond in bonds if i in bond[:2])
                if bond_count > 1:  # 至少有两个键才可能参与共轭
                    conjugation_candidates.append(i)
        
        # 创建邻接列表
        neighbors = defaultdict(list)
        for i, j, _ in bonds:
            neighbors[i].append(j)
            neighbors[j].append(i)
        
        # 查找共轭路径
        visited = set()
        conjugated_paths = []
        
        # 对每个候选原子进行广度优先搜索
        for start_atom in conjugation_candidates:
            if start_atom in visited:
                continue
                
            # 进行广度优先搜索
            queue = [start_atom]
            path = [start_atom]
            visited.add(start_atom)
            
            while queue:
                current = queue.pop(0)
                
                for neighbor in neighbors[current]:
                    if neighbor not in visited and neighbor in conjugation_candidates:
                        queue.append(neighbor)
                        path.append(neighbor)
                        visited.add(neighbor)
            
            if len(path) > 2:  # 有意义的共轭路径至少要有3个原子
                conjugated_paths.append(path)
        
        return conjugated_paths
    
    @staticmethod
    def identify_h_bonds(mol):
        """识别分子中的氢键
        
        Args:
            mol: 包含分子信息的字典
            
        Returns:
            氢键列表，每个氢键是(donor, hydrogen, acceptor, strength)形式
        """
        if not mol:
            return []
            
        atoms = mol['atoms']
        coords = mol['coordinates']
        
        # 识别可能的H键受体（O, N, F）
        acceptors = [i for i, atom in enumerate(atoms) if atom in ['O', 'N', 'F']]
        
        # 识别可能的H键供体（连接到N或O的H）
        donors = []
        hydrogens = []
        
        # 对每个H原子，检查它是否连接到N或O
        for i, atom in enumerate(atoms):
            if atom == 'H':
                # 寻找最近的N或O原子
                nearest_donor = None
                min_distance = float('inf')
                
                for j, donor_atom in enumerate(atoms):
                    if donor_atom in ['N', 'O'] and j != i:
                        distance = StructureUtils.calculate_bond_length(coords[i], coords[j])
                        if distance < 1.2 and distance < min_distance:  # 典型的NH或OH键长小于1.2
                            min_distance = distance
                            nearest_donor = j
                
                if nearest_donor is not None:
                    donors.append(nearest_donor)
                    hydrogens.append(i)
        
        # 寻找氢键
        h_bonds = []
        for donor, hydrogen in zip(donors, hydrogens):
            for acceptor in acceptors:
                # 避免同一原子
                if donor == acceptor:
                    continue
                
                # 计算H...A距离
                h_a_distance = StructureUtils.calculate_bond_length(coords[hydrogen], coords[acceptor])
                
                # 氢键典型距离为1.5-2.5埃
                if 1.5 <= h_a_distance <= 2.5:
                    # 计算D-H...A角度
                    angle = StructureUtils.calculate_bond_angle(coords[donor], coords[hydrogen], coords[acceptor])
                    
                    # 氢键通常是接近线性的
                    if angle and angle > 120:
                        # 估计氢键强度（简化模型：距离越短，角度越接近180度，强度越高）
                        distance_factor = 1.0 - (h_a_distance - 1.5) / 1.0  # 1.5->1.0, 2.5->0.0
                        angle_factor = (angle - 120) / 60.0  # 120->0.0, 180->1.0
                        strength = distance_factor * angle_factor
                        
                        h_bonds.append((donor, hydrogen, acceptor, strength))
        
        return h_bonds
    
    @staticmethod
    def calculate_planarity(coords):
        """计算一组原子的平面性
        
        Args:
            coords: 原子坐标数组
            
        Returns:
            平面性指数(0-1)，1表示完全平面
        """
        if len(coords) < 4:
            return 1.0  # 少于4个点总是共面的
            
        # 前3点确定一个平面
        p1 = coords[0]
        p2 = coords[1]
        p3 = coords[2]
        
        # 计算法向量
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        
        # 归一化法向量
        normal_length = np.linalg.norm(normal)
        if normal_length < 1e-6:
            # 前三点共线，尝试其他点组合
            if len(coords) > 3:
                return StructureUtils.calculate_planarity(coords[1:])
            return 1.0
            
        normal = normal / normal_length
        
        # 计算平面方程: ax + by + cz + d = 0
        a, b, c = normal
        d = -np.dot(normal, p1)
        
        # 计算每个点到平面的距离
        total_distance = 0.0
        for coord in coords:
            distance = abs(np.dot(normal, coord) + d) / np.sqrt(a*a + b*b + c*c)
            total_distance += distance
        
        # 计算平均距离
        avg_distance = total_distance / len(coords)
        
        # 将平均距离转换为平面性指数
        # 距离为0表示完全平面(指数=1)
        # 使用指数衰减函数将距离映射到[0-1]区间
        planarity = np.exp(-avg_distance * 5)  # 参数5控制衰减速率
        
        return planarity
    
    @staticmethod
    def calculate_structure_features(mol):
        """计算分子的结构特征
        
        Args:
            mol: 包含分子信息的字典
            
        Returns:
            结构特征字典
        """
        if not mol:
            return {}
            
        try:
            # 获取基本信息
            atoms = mol['atoms']
            coords = mol['coordinates']
            
            # 构建键信息
            bonds = StructureUtils.build_bonds(mol)
            
            # 识别环
            rings = StructureUtils.identify_rings(mol, bonds)
            
            # 识别芳香环
            aromatic_rings = [ring for ring in rings if StructureUtils.is_aromatic_ring(mol, ring, bonds)]
            
            # 识别共轭路径
            conjugated_paths = StructureUtils.identify_conjugated_paths(mol, bonds)
            
            # 计算二面角
            dihedral_angles = []
            for i in range(len(atoms) - 3):
                for j in range(i+1, len(atoms) - 2):
                    for k in range(j+1, len(atoms) - 1):
                        for l in range(k+1, len(atoms)):
                            angle = StructureUtils.calculate_dihedral_angle(
                                coords[i], coords[j], coords[k], coords[l]
                            )
                            if angle is not None:
                                dihedral_angles.append(abs(angle))
            
            # 识别氢键
            h_bonds = StructureUtils.identify_h_bonds(mol)
            
            # 计算全分子平面性
            planarity = StructureUtils.calculate_planarity(coords)
            
            # 计算扭曲键（二面角大于40度）
            twisted_bonds_count = sum(1 for angle in dihedral_angles if abs(angle) > 40 and abs(angle) < 140)
            
            # 整理特征
            features = {
                'conjugation_path_count': len(conjugated_paths),
                'max_conjugation_length': max([len(path) for path in conjugated_paths]) if conjugated_paths else 0,
                'aromatic_rings_count': len(aromatic_rings),
                'dihedral_angles_count': len(dihedral_angles),
                'max_dihedral_angle': max(dihedral_angles) if dihedral_angles else 0,
                'avg_dihedral_angle': sum(dihedral_angles) / len(dihedral_angles) if dihedral_angles else 0,
                'twisted_bonds_count': twisted_bonds_count,
                'twist_ratio': twisted_bonds_count / len(dihedral_angles) if dihedral_angles else 0,
                'hydrogen_bonds_count': len(h_bonds),
                'avg_h_bond_strength': sum(hb[3] for hb in h_bonds) / len(h_bonds) if h_bonds else 0,
                'max_h_bond_strength': max(hb[3] for hb in h_bonds) if h_bonds else 0,
                'planarity': planarity
            }
            
            return features
            
        except Exception as e:
            logging.error(f"计算结构特征时出错: {e}")
            return {}
    
    @staticmethod
    def compare_structures(mol1, mol2):
        """比较两个分子结构的差异
        
        Args:
            mol1: 第一个分子结构
            mol2: 第二个分子结构
            
        Returns:
            结构差异特征字典
        """
        if not mol1 or not mol2:
            return {}
            
        try:
            # 确保两个分子有相同数量的原子
            if len(mol1['atoms']) != len(mol2['atoms']):
                return {'structure_match': False}
                
            # 计算坐标均方根偏差(RMSD)
            coords1 = mol1['coordinates']
            coords2 = mol2['coordinates']
            
            squared_diff = np.sum((coords1 - coords2) ** 2)
            rmsd = np.sqrt(squared_diff / len(coords1))
            
            # 计算原子间距离变化
            distance_changes = []
            for i in range(len(coords1)):
                for j in range(i+1, len(coords1)):
                    dist1 = StructureUtils.calculate_bond_length(coords1[i], coords1[j])
                    dist2 = StructureUtils.calculate_bond_length(coords2[i], coords2[j])
                    distance_changes.append(dist2 - dist1)
            
            # 计算平均距离变化和最大距离变化
            avg_distance_change = sum(distance_changes) / len(distance_changes) if distance_changes else 0
            max_distance_change = max(abs(change) for change in distance_changes) if distance_changes else 0
            
            # 计算两个结构的平面性变化
            planarity1 = StructureUtils.calculate_planarity(coords1)
            planarity2 = StructureUtils.calculate_planarity(coords2)
            planarity_change = planarity2 - planarity1
            
            # 返回结构变化特征
            features = {
                'structure_match': True,
                'structure_rmsd': rmsd,
                'avg_distance_change': avg_distance_change,
                'max_distance_change': max_distance_change,
                'planarity_change': planarity_change
            }
            
            return features
            
        except Exception as e:
            logging.error(f"比较分子结构时出错: {e}")
            return {'structure_match': False}
