# utils/molecular_design.py
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, Lipinski, Fragments
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import BRICS
import numpy as np
import pandas as pd
import random
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MoleculeGenerator:
    """分子生成器，用于设计新的反向TADF分子"""
    
    def __init__(self, seed_smiles=None):
        """初始化分子生成器"""
        self.seed_smiles = seed_smiles
        if seed_smiles:
            self.seed_mol = Chem.MolFromSmiles(seed_smiles)
        else:
            # 默认使用calicene作为种子分子
            self.seed_smiles = "C1=CC=CC=1C=C1C=C1"
            self.seed_mol = Chem.MolFromSmiles(self.seed_smiles)
            
        # 初始化常用取代基和反应
        self.initialize_common_groups()
        
    def initialize_common_groups(self):
        """初始化常用的取代基"""
        # 强吸电子基团（有助于降低HOMO能级）
        self.electron_withdrawing_groups = {
            'cyano': 'C#N',
            'nitro': '[N+](=O)[O-]',
            'trifluoromethyl': 'C(F)(F)F',
            'carbonyl': 'C(=O)',
            'sulfonyl': 'S(=O)(=O)',
            'ester': 'C(=O)O',
        }
        
        # 强给电子基团（有助于提高LUMO能级）
        self.electron_donating_groups = {
            'amino': 'N',
            'dimethylamino': 'N(C)C',
            'trimethylamino': '[N+](C)(C)C',
            'hydroxy': 'O',
            'methoxy': 'OC',
            'alkyl': 'C',
        }
        
        # 常见环系统
        self.ring_systems = {
            'cyclopentadiene': 'C1=CC=CC1',
            'cyclopropene': 'C1=CC1',
            'cycloheptatriene': 'C1=CC=CC=CC1',
            'benzene': 'c1ccccc1',
            'furan': 'o1cccc1',
            'thiophene': 's1cccc1',
            'pyrrole': '[nH]1cccc1',
        }
        
    def mutate_molecule(self, mol=None, mutation_type=None, position=None):
        """变异分子结构"""
        if mol is None:
            mol = self.seed_mol
            
        if mol is None:
            return None
            
        # 如果未指定变异类型，随机选择
        mutation_types = [
            'add_electron_withdrawing',
            'add_electron_donating',
            'swap_groups',
            'add_ring',
            'remove_group',
            'substitute_atom'
        ]
        
        if mutation_type is None:
            mutation_type = random.choice(mutation_types)
            
        # 执行变异
        if mutation_type == 'add_electron_withdrawing':
            return self.add_group(mol, random.choice(list(self.electron_withdrawing_groups.values())), position)
        elif mutation_type == 'add_electron_donating':
            return self.add_group(mol, random.choice(list(self.electron_donating_groups.values())), position)
        elif mutation_type == 'swap_groups':
            return self.swap_groups(mol)
        elif mutation_type == 'add_ring':
            return self.add_ring(mol, random.choice(list(self.ring_systems.values())))
        elif mutation_type == 'remove_group':
            return self.remove_group(mol)
        elif mutation_type == 'substitute_atom':
            return self.substitute_atom(mol)
        else:
            return mol
            
    def add_group(self, mol, group_smarts, position=None):
        """向分子添加取代基"""
        # 这是一个简化的实现，实际中应该使用更复杂的反应规则
        try:
            # 获取可修改的位置
            if position is None:
                atoms = list(range(mol.GetNumAtoms()))
                random.shuffle(atoms)
                position = atoms[0] if atoms else 0
                
            # 创建编辑对象
            em = Chem.EditableMol(mol)
            
            # 添加取代基（这是一个简化实现）
            group_mol = Chem.MolFromSmiles(group_smarts)
            if group_mol is None:
                return mol
                
            # 获取连接点
            # 在实际应用中，需要更复杂的逻辑来确定正确的连接点
            em.ReplaceAtom(position, group_mol.GetAtomWithIdx(0))
            
            # 获取修改后的分子
            modified_mol = em.GetMol()
            
            # 清理分子
            Chem.SanitizeMol(modified_mol)
            
            return modified_mol
        except Exception as e:
            logger.error(f"添加取代基时出错: {e}")
            return mol
            
    def swap_groups(self, mol):
        """交换分子中的两个取代基的位置"""
        # 简化实现，实际中可能需要更复杂的逻辑
        return mol
        
    def add_ring(self, mol, ring_smarts):
        """向分子添加环系统"""
        # 简化实现，实际中需要更复杂的反应规则
        return mol
        
    def remove_group(self, mol):
        """移除分子中的一个取代基"""
        # 简化实现
        return mol
        
    def substitute_atom(self, mol):
        """替换分子中的一个原子"""
        # 简化实现
        return mol
        
    def generate_molecules(self, n_molecules=10, n_mutations=3):
        """生成一组新分子"""
        if self.seed_mol is None:
            return []
            
        generated_mols = []
        
        for _ in range(n_molecules):
            # 从种子分子开始
            mol = Chem.Mol(self.seed_mol)
            
            # 应用多次变异
            for _ in range(n_mutations):
                mol = self.mutate_molecule(mol)
                
                # 确保分子有效
                if mol is None:
                    break
                
                try:
                    # 清理分子
                    Chem.SanitizeMol(mol)
                except:
                    # 如果分子无效，回退到上一状态
                    mol = Chem.Mol(self.seed_mol)
                    
            # 只保留有效分子
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                if smiles and Chem.MolFromSmiles(smiles) is not None:
                    generated_mols.append(mol)
                    
        # 将分子转换为SMILES
        smiles_list = [Chem.MolToSmiles(mol) for mol in generated_mols]
        
        # 去重并返回
        unique_smiles = list(set(smiles_list))
        
        return unique_smiles
        
    def evaluate_molecules(self, smiles_list, predictor):
        """评估生成的分子"""
        results = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                    
                # 预测S1-T1能隙
                predicted_gap = predictor.predict(smiles)
                
                # 计算分子特性
                properties = {
                    'smiles': smiles,
                    'predicted_gap': predicted_gap,
                    'molecular_weight': Descriptors.MolWt(mol),
                    'logP': Descriptors.MolLogP(mol),
                    'TPSA': Descriptors.TPSA(mol),
                    'HBA': Descriptors.NumHAcceptors(mol),
                    'HBD': Descriptors.NumHDonors(mol),
                    'RotBonds': Descriptors.NumRotatableBonds(mol),
                    'RingCount': Descriptors.RingCount(mol),
                    'AromaticRings': Descriptors.NumAromaticRings(mol),
                }
                
                results.append(properties)
            except Exception as e:
                logger.error(f"评估分子 {smiles} 时出错: {e}")
                
        # 创建数据框
        if results:
            return pd.DataFrame(results)
        else:
            return None

class CaliceneDesigner:
    """专门用于设计基于Calicene的反向TADF分子"""
    
    def __init__(self):
        """初始化calicene设计器"""
        # Calicene的SMILES表示
        self.calicene_smiles = "C1=CC=CC=1C=C1C=C1"
        self.calicene = Chem.MolFromSmiles(self.calicene_smiles)
        
        # 初始化取代位置信息
        self.positions = {
            'R1': 6,  # 3-membered ring position (acceptor site)
            'R2': 2,  # 5-membered ring position (donor site)
            'R3': 0   # secondary 5-membered ring position
        }
        
        # 初始化取代基
        self.acceptors = {
            'CN': 'C#N',
            'NO2': '[N+](=O)[O-]',
            'CF3': 'C(F)(F)F',
            'SO2Me': 'S(=O)(=O)C',
            'COOMe': 'C(=O)OC',
            'CHO': 'C=O',
        }
        
        self.donors = {
            'NH2': 'N',
            'OH': 'O',
            'OMe': 'OC',
            'NMe2': 'N(C)C',
            'NPMe3': 'N(P(C)(C)C)',
            'SMe': 'SC',
        }
        
    def create_substituted_calicene(self, r1=None, r2=None, r3=None):
        """创建特定取代模式的calicene衍生物"""
        try:
            # 克隆基础calicene
            mol = Chem.Mol(self.calicene)
            em = Chem.EditableMol(mol)
            
            # 在R1位置添加吸电子基团
            if r1 is not None:
                r1_group = r1 if r1 in self.acceptors.values() else self.acceptors.get(r1)
                if r1_group:
                    self._add_group(em, r1_group, self.positions['R1'])
                    
            # 在R2位置添加给电子基团
            if r2 is not None:
                r2_group = r2 if r2 in self.donors.values() else self.donors.get(r2)
                if r2_group:
                    self._add_group(em, r2_group, self.positions['R2'])
                    
            # 在R3位置添加额外取代基
            if r3 is not None:
                r3_group = None
                if r3 in self.donors.values() or r3 in self.acceptors.values():
                    r3_group = r3
                elif r3 in self.donors:
                    r3_group = self.donors[r3]
                elif r3 in self.acceptors:
                    r3_group = self.acceptors[r3]
                    
                if r3_group:
                    self._add_group(em, r3_group, self.positions['R3'])
                    
            # 获取修改后的分子
            modified_mol = em.GetMol()
            
            # 清理分子
            Chem.SanitizeMol(modified_mol)
            
            return modified_mol
        except Exception as e:
            logger.error(f"创建取代calicene时出错: {e}")
            return None
            
    def _add_group(self, editable_mol, group_smarts, position):
        """向分子的特定位置添加取代基"""
        # 注意：这是一个简化实现，真实场景需要更复杂的反应逻辑
        pass
        
    def generate_candidates(self, n_candidates=10):
        """生成一组calicene候选分子"""
        candidates = []
        
        # 使用文献中提到的最佳取代模式
        combinations = [
            # r1, r2, r3
            ('CN', 'NMe2', None),      # 吸电子 + 给电子
            ('CN', 'NPMe3', None),     # 强吸电子 + 强给电子
            ('NO2', 'NMe2', None),     # 吸电子 + 给电子
            ('CF3', 'NMe2', None),     # 吸电子 + 给电子
            ('SO2Me', 'NMe2', None),   # 吸电子 + 给电子
            ('CN', 'NMe2', 'OH'),      # 组合取代
            ('CN', 'NMe2', 'OMe'),     # 组合取代
            # 添加更多组合
        ]
        
        # 创建基本组合
        for r1, r2, r3 in combinations:
            mol = self.create_substituted_calicene(r1, r2, r3)
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                candidates.append(smiles)
                
        # 如果需要更多分子，随机组合取代基
        while len(candidates) < n_candidates:
            r1 = random.choice(list(self.acceptors.keys()))
            r2 = random.choice(list(self.donors.keys()))
            r3 = random.choice([None] + list(self.donors.keys()) + list(self.acceptors.keys()))
            
            mol = self.create_substituted_calicene(r1, r2, r3)
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                if smiles not in candidates:
                    candidates.append(smiles)
                    
        return candidates[:n_candidates]
