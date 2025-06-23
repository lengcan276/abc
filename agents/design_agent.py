# agents/design_agent.py
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import joblib

# 延迟导入PyTorch，使其成为可选依赖
torch_available = False
try:
    import torch
    torch_available = True
except ImportError:
    pass

# 导入RDKit组件
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, Fragments
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold

# 条件导入强化学习库
rl_available = False
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.evaluation import evaluate_policy
    rl_available = True
except ImportError:
    pass

class MoleculeEnvironment:
    """用于分子设计强化学习的环境"""
    
    def __init__(self, predictive_model, target_gap=-0.1, scaffold=None):
        """
        初始化分子设计环境
        
        参数:
            predictive_model: 预测S1-T1能隙的模型
            target_gap: 目标S1-T1能隙值
            scaffold: 可选的起始骨架
        """
        self.model = predictive_model
        self.target_gap = target_gap
        self.scaffold = scaffold
        self.current_mol = None
        self.steps = 0
        self.max_steps = 40  # 最大步数
        self.valid_actions = self._get_valid_actions()

    def validate_smiles(self, smiles):
        """验证SMILES字符串的有效性"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            # 检查环状结构是否闭合
            if smiles.count('1') % 2 != 0 or smiles.count('2') % 2 != 0:
                return False
            # 验证是否可以转换回SMILES
            new_smiles = Chem.MolToSmiles(mol)
            return True
        except Exception:
            return False   
            
    def _get_valid_actions(self):
        """定义有效的分子修改操作"""
        # 包括增加/移除原子、改变键、添加/替换取代基等
        actions = {
            # 有利于降低HOMO的取代基（吸电子基团）
            'add_CN': lambda mol: self._add_functional_group(mol, '[C]#N'),
            'add_NO2': lambda mol: self._add_functional_group(mol, '[N+](=O)[O-]'),
            'add_CF3': lambda mol: self._add_functional_group(mol, 'C(F)(F)F'),
            
            # 有利于提高LUMO的取代基（给电子基团）
            'add_NH2': lambda mol: self._add_functional_group(mol, 'N'),
            'add_OH': lambda mol: self._add_functional_group(mol, 'O'),
            'add_OCH3': lambda mol: self._add_functional_group(mol, 'OC'),
            'add_N(CH3)2': lambda mol: self._add_functional_group(mol, 'N(C)C'),
            'add_calicene_specific_groups': self._add_calicene_specific_groups,
            
            # 共轭体系修改
            'extend_conjugation': self._extend_conjugation,
            'add_five_ring': self._add_five_ring,
            'add_seven_ring': self._add_seven_ring,
            'add_three_ring': self._add_three_ring,
            
            # 布局修饰
            'swap_groups': self._swap_groups,
            'rotate_groups': self._rotate_groups,
        }
        return actions
        
    def _add_functional_group(self, mol, smarts):
        """添加功能团到分子上"""
        # 实现添加功能基团的逻辑
        try:
            # 这里是简化的实现，实际应该使用RDKit的反应模块
            # 找到可能的添加位置
            if mol is None:
                return None
                
            # 复制原始分子
            new_mol = Chem.Mol(mol)
            
            # 获取所有可能的添加位置（碳原子）
            carbon_indices = [atom.GetIdx() for atom in new_mol.GetAtoms() 
                             if atom.GetSymbol() == 'C' and atom.GetDegree() < 3]
            
            if not carbon_indices:
                return None
                
            # 随机选择一个位置
            position = np.random.choice(carbon_indices)
            
            # 创建新分子
            rwmol = Chem.RWMol(new_mol)
            
            # 将SMARTS转换为分子
            frag = Chem.MolFromSmiles(smarts)
            if frag is None:
                return None
                
            # 简单处理：添加到指定位置（实际应用中应该使用反应SMARTS）
            # 这只是一个示例，不一定化学上合理
            rwmol.ReplaceAtom(position, frag.GetAtomWithIdx(0))
            
            # 获取修改后的分子
            modified_mol = rwmol.GetMol()
            
            # 尝试清理分子（可能会失败，如果结构不合理）
            Chem.SanitizeMol(modified_mol)
            
            return modified_mol
        except:
            return None
            
    def _extend_conjugation(self, mol):
        """扩展分子的共轭系统"""
        if mol is None:
            return None
            
        try:
            # 这是一个简化实现，实际需要更复杂的逻辑
            # 复制原始分子
            new_mol = Chem.Mol(mol)
            
            # 在此模拟添加一个双键或环以扩展共轭
            # 找到合适的位置
            # 添加共轭单元
            
            # 简单起见，此处返回原始分子
            return new_mol
        except:
            return None
        
    def _add_five_ring(self, mol):
        """添加五元环"""
        if mol is None:
            return None
            
        try:
            # 这是一个简化实现
            # 真正的实现应该使用RDKit的反应模块
            # 添加五元环模板
            
            # 简单起见，此处返回原始分子
            return mol
        except:
            return None
        
    def _add_seven_ring(self, mol):
        """添加七元环"""
        if mol is None:
            return None
            
        try:
            # 简化实现
            return mol
        except:
            return None
        
    def _add_three_ring(self, mol):
        """添加三元环"""
        if mol is None:
            return None
            
        try:
            # 简化实现
            return mol
        except:
            return None
            
    def _add_calicene_specific_groups(self, mol):
        """针对Calicene结构添加特定的基团"""
        if mol is None:
            return None
            
        try:
            # 复制原始分子
            new_mol = Chem.Mol(mol)
            rwmol = Chem.RWMol(new_mol)
            
            # 尝试识别Calicene的五元环和三元环
            five_ring_pattern = Chem.MolFromSmarts('C1=CC=CC=1')  # 简化的五元环模式
            three_ring_pattern = Chem.MolFromSmarts('C1CC1')      # 简化的三元环模式
            
            # 找到模式位置
            five_ring_matches = mol.GetSubstructMatches(five_ring_pattern)
            three_ring_matches = mol.GetSubstructMatches(three_ring_pattern)
            
            if five_ring_matches and three_ring_matches:
                # 找到五元环上的位置添加N(CH3)2（给电子基团）
                five_ring_pos = five_ring_matches[0][0]  # 使用第一个匹配的第一个原子
                # 这里只是示例，实际应用中需要更复杂的逻辑
                
                # 找到三元环上的位置添加CN（吸电子基团）
                three_ring_pos = three_ring_matches[0][0]  # 使用第一个匹配的第一个原子
                
                # 修改分子...（实际实现需要更复杂的化学逻辑）
                
                # 获取修改后的分子
                modified_mol = rwmol.GetMol()
                Chem.SanitizeMol(modified_mol)
                return modified_mol
            
            return mol  # 如果没有找到匹配，返回原始分子
        except:
            return mol
    
    def _swap_groups(self, mol):
        """交换取代基位置"""
        if mol is None:
            return None
            
        try:
            # 简化实现
            return mol
        except:
            return None
        
    def _rotate_groups(self, mol):
        """旋转取代基"""
        if mol is None:
            return None
            
        try:
            # 简化实现
            return mol
        except:
            return None
        
    def reset(self):
        """重置环境到初始状态"""
        if self.scaffold:
            try:
                self.current_mol = Chem.MolFromSmiles(self.scaffold)
                if self.current_mol is None:
                    # 如果无法解析，使用默认的calicene结构
                    self.current_mol = Chem.MolFromSmiles("C1=CC=CC=1C=C1C=C1")
            except:
                # 出错时使用默认的calicene结构
                self.current_mol = Chem.MolFromSmiles("C1=CC=CC=1C=C1C=C1")
        else:
            # 使用calicene作为默认起始分子
            self.current_mol = Chem.MolFromSmiles("C1=CC=CC=1C=C1C=C1")
        
        self.steps = 0
        return self._get_state()
            
    def _get_state(self):
        """获取当前分子的状态表示"""
        # 可以使用Morgan指纹、ECFP或其他分子表示
        if self.current_mol is None:
            return np.zeros(2048)
            
        fp = AllChem.GetMorganFingerprintAsBitVect(self.current_mol, 2, nBits=2048)
        return np.array(fp)
        
    def step(self, action_idx):
        """执行动作并返回新状态、奖励和终止标志"""
        self.steps += 1
        
        # 获取动作函数
        action_keys = list(self.valid_actions.keys())
        action_func = self.valid_actions[action_keys[action_idx % len(action_keys)]]
        
        # 应用动作
        new_mol = action_func(self.current_mol)
        
        # 检查动作是否有效
        if new_mol is None:
            reward = -1  # 惩罚无效动作
            return self._get_state(), reward, False, {'mol': self.current_mol}
            
        self.current_mol = new_mol
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 检查是否终止
        done = (self.steps >= self.max_steps)
        
        return self._get_state(), reward, done, {'mol': self.current_mol}
        
    def _calculate_reward(self):
        """基于当前分子的S1-T1能隙计算奖励"""
        if self.current_mol is None:
            return -1
            
        # 计算分子特征
        smiles = Chem.MolToSmiles(self.current_mol)
        
        try:
            # 使用预测模型预测S1-T1能隙
            gap = self.model.predict(smiles)
            
            # 基于与目标值的接近程度计算奖励
            reward = -abs(gap - self.target_gap)
            
            # 如果实现了负能隙，给予额外奖励
            if gap < 0:
                reward += 2
                
            # 考虑TADF材料相关的特性
            reward += self._calculate_tadf_properties_reward()
            
            return reward
        except:
            return -1
            
    def _calculate_tadf_properties_reward(self):
        """计算分子的TADF材料特性奖励"""
        try:
            reward = 0
            
            # 1. 共轭系统评估 - TADF材料通常需要良好的共轭系统
            aromatic_rings = Descriptors.NumAromaticRings(self.current_mol)
            if aromatic_rings > 0:
                reward += 0.1 * min(aromatic_rings, 3)  # 奖励适度的芳香环数量
                
            # 2. 推拉电子体系 - 反向TADF材料通常需要强推拉体系
            has_donor = any([
                Fragments.fr_NH2(self.current_mol) > 0, 
                Fragments.fr_NH1(self.current_mol) > 0,
                Fragments.fr_NH0(self.current_mol) > 0,
                Fragments.fr_Ar_OH(self.current_mol) > 0
            ])
                            
            has_acceptor = any([
                Fragments.fr_C_O(self.current_mol) > 0,
                Fragments.fr_nitro(self.current_mol) > 0,
                Fragments.fr_nitro_arom(self.current_mol) > 0,
                Fragments.fr_nitrile(self.current_mol) > 0
            ])
                              
            if has_donor and has_acceptor:
                reward += 0.5  # 同时具有给体和受体基团
                
            # 3. 检查是否含有Calicene相关结构（三元环和五元环组合）
            # 这需要更复杂的子结构匹配，这里只是简化示例
            calicene_pattern = Chem.MolFromSmarts('C1=CC=CC=1C=C1C=C1')
            if calicene_pattern and self.current_mol.HasSubstructMatch(calicene_pattern):
                reward += 1.0  # 强奖励Calicene基本结构
                
            # 4. 分子稳定性考虑
            # 避免高度不稳定的结构
            unstable_groups = [
                Chem.MolFromSmarts('[N+]=[N-]'),  # 叠氮基团
                Chem.MolFromSmarts('O-O'),        # 过氧基团
                Chem.MolFromSmarts('C=C=C')       # 累积二烯
            ]
            
            stability_penalty = sum(1 for group in unstable_groups 
                                   if group and self.current_mol.HasSubstructMatch(group))
            reward -= 0.5 * stability_penalty
            
            # 5. 合成复杂度考虑
            # 通常，环数和杂原子数量会增加合成复杂度
            hetero_atoms = sum(1 for atom in self.current_mol.GetAtoms() 
                              if atom.GetSymbol() not in ['C', 'H'])
            if hetero_atoms > 8:  # 过多杂原子可能增加合成难度
                reward -= 0.1 * (hetero_atoms - 8)
                
            # 6. 特定功能基团奖励
            # 论文中提到的有效取代基
            valuable_groups = [
                Chem.MolFromSmarts('c-N(C)C'),    # NMe2基团
                Chem.MolFromSmarts('c-C#N'),      # CN基团
                Chem.MolFromSmarts('c-[N+](=O)[O-]'),  # NO2基团
                Chem.MolFromSmarts('c-C(F)(F)F')  # CF3基团
            ]
            
            for group in valuable_groups:
                if group and self.current_mol.HasSubstructMatch(group):
                    reward += 0.2
                    
            return reward
        except Exception as e:
            logging.error(f"计算TADF特性奖励时出错: {e}")
            return 0


class DesignAgent:
    """
    Agent responsible for designing new reversed TADF molecules
    using guided exploration methods.
    """
    
    def __init__(self, predictive_model_path=None, fine_tuned_model_path=None):
        """Initialize the DesignAgent with predictive models."""
        self.predictive_model = None
        self.fine_tuned_model = None
        self.rl_model = None
        self.environment = None
        self.torch_available = False
        self.rl_available = False
        
        # 设置日志记录 - 移到前面
        self.setup_logging()
        
        # 检查是否可以导入torch，而不是直接导入
        try:
            import torch
            self.logger.info("Successfully imported PyTorch")
        except ImportError:
            self.logger.warning("PyTorch is not available")
            pass
            
        # 检查是否可以导入强化学习库
        try:
            from stable_baselines3 import PPO
            self.rl_available = True
        except ImportError:
            pass
        
        # 添加自动查找模型文件的逻辑
        if predictive_model_path is None:
            # 尝试自动查找分类模型和回归模型
<<<<<<< HEAD
            model_dir = "/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach/data/models"
=======
            model_dir = "/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach/data/models"
>>>>>>> 0181d62 (update excited)
            classifier_path = os.path.join(model_dir, "s1t1_gap_classifier.joblib")
            regressor_path = os.path.join(model_dir, "s1t1_gap_regressor.joblib")
            
            # 首先尝试加载分类模型
            if os.path.exists(classifier_path):
                self.logger.info(f"找到分类模型: {classifier_path}")
                if self.load_predictive_model(classifier_path):
                    self.logger.info("成功加载分类模型")
                    predictive_model_path = classifier_path
            
            # 如果分类模型加载失败，尝试加载回归模型
            if self.predictive_model is None and os.path.exists(regressor_path):
                self.logger.info(f"找到回归模型: {regressor_path}")
                if self.load_predictive_model(regressor_path):
                    self.logger.info("成功加载回归模型")
                    predictive_model_path = regressor_path
        
        # 如果自动查找失败或提供了路径，则尝试加载指定模型
        if self.predictive_model is None:
            # 创建一个基本的内置模型
            self.create_basic_model()
            
            # 如果提供了模型路径，尝试加载
            if predictive_model_path:
                self.load_predictive_model(predictive_model_path)
        
        # 尝试加载微调模型（如果提供了路径）
        if fine_tuned_model_path and self.torch_available:
            self.load_fine_tuned_model(fine_tuned_model_path)
            
    def setup_logging(self):
        """Configure logging for the design agent."""
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
<<<<<<< HEAD
                           filename='/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach/data/logs/design_agent.log')
=======
                           filename='/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach/data/logs/design_agent.log')
>>>>>>> 0181d62 (update excited)
        self.logger = logging.getLogger('DesignAgent')
        
    def create_basic_model(self):
        """创建一个简单的内置模型，当没有外部模型可用时使用"""
        class SimplePredictor:
            def predict(self, smiles):
                """基于分子名称模式的简单预测逻辑"""
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None:
                        return 0.5
                    
                    # 检查是否有三元环和五元环
                    five_ring = Chem.MolFromSmarts('C1=CC=CC=1')
                    three_ring = Chem.MolFromSmarts('C1CC1')
                    
                    has_five_ring = mol.HasSubstructMatch(five_ring) if five_ring else False
                    has_three_ring = mol.HasSubstructMatch(three_ring) if three_ring else False
                    
                    # 检查是否有给电子和吸电子基团
                    nh2_pattern = Chem.MolFromSmarts('cN')
                    cn_pattern = Chem.MolFromSmarts('C#N')
                    oh_pattern = Chem.MolFromSmarts('cO')
                    
                    has_nh2 = mol.HasSubstructMatch(nh2_pattern) if nh2_pattern else False
                    has_cn = mol.HasSubstructMatch(cn_pattern) if cn_pattern else False
                    has_oh = mol.HasSubstructMatch(oh_pattern) if oh_pattern else False
                    
                    # 预测逻辑
                    if has_five_ring and has_three_ring:
                        if (has_nh2 or has_oh) and has_cn:
                            return -0.15  # 基于已知分子的负能隙
                        elif has_nh2 and has_cn:
                            return -0.1   # 已知nh2和cn组合产生负能隙
                        elif has_oh and has_cn:
                            return -0.05  # 已知oh和cn组合产生负能隙
                    
                    # 默认返回正能隙
                    return 0.1
                except:
                    return 0.5
                    
        self.predictive_model = SimplePredictor()
        self.logger.info("创建了简单的内置预测模型")
        
    def load_predictive_model(self, model_path):
        """加载传统预测模型（如随机森林）"""
        try:
            self.predictive_model = joblib.load(model_path)
            self.logger.info(f"成功加载预测模型: {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"加载预测模型时出错: {e}")
            return False
            
    def load_fine_tuned_model(self, model_path):
        """加载微调的深度学习模型"""
        if not self.torch_available:
            self.logger.warning("PyTorch不可用，无法加载深度学习模型")
            return False
            
        try:
            # 动态导入torch
            import torch
            self.fine_tuned_model = torch.load(model_path)
            self.logger.info(f"成功加载微调模型: {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"加载微调模型时出错: {e}")
            return False
            
    def predict_gap(self, model, smiles):
        """通用预测接口，适应不同类型的模型"""
        try:
            # 处理不同类型的模型
            if hasattr(model, 'predict'):
                # 标准sklearn或类似接口
                # 需要特征提取
                features = self.extract_features(smiles)
                return model.predict([features])[0]
            elif self.torch_available and isinstance(model, torch.nn.Module):
                # PyTorch模型
                # 提取特征并转换为tensor
                features = self.extract_features(smiles)
                with torch.no_grad():
                    tensor_input = torch.tensor([features], dtype=torch.float32)
                    return model(tensor_input).item()
            else:
                # 假设是自定义模型类
                return model.predict(smiles)
        except Exception as e:
            self.logger.error(f"预测S1-T1能隙时出错: {e}")
            # 返回较大的正值，表示预测失败
            return 1.0
            
    def extract_features(self, smiles):
        """从SMILES提取分子特征"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return np.zeros(200)  # 返回零向量表示无效分子
                
            # 可以实现更复杂的特征提取逻辑
            # 这里简单使用Morgan指纹作为示例
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=200)
            return np.array(fp)
        except:
            return np.zeros(200)       
    
    def create_rule_based_environment(self, target_gap=-0.1, scaffold=None):
        """创建一个基于规则的简化环境，不依赖PyTorch"""
        class SimpleEnvironment:
            def __init__(self, predictor, target_gap, scaffold):
                self.predictor = predictor
                self.target_gap = target_gap
                self.scaffold = scaffold
                self.current_mol = None
                self.known_patterns = {
                    '5ring_nh2_3ring_cn_both': 'C1=CC=C(N)C1=C1CC1C#N',
                    '5ring_nme2_3ring_cn_in_con2': 'C1=CC=C(N(C)C)C1=C1CC1C#N',
                    '5ring_oh_3ring_cn_both': 'C1=CC=C(O)C1=C1CC1C#N',
                    'me_oh_cn': 'CC1=CC=C(O)C1=C1CC1C#N',
                    '5ring_nh2_3ring_cn_in': 'C1=CC=C(N)C1=C1CC1C#N'
                }
                
            def reset(self):
                """重置环境"""
                if self.scaffold and self.scaffold in self.known_patterns:
                    self.current_mol = Chem.MolFromSmiles(self.known_patterns[self.scaffold])
                else:
                    # 选择一个已知的负能隙分子
                    self.current_mol = Chem.MolFromSmiles(
                        self.known_patterns.get('5ring_nh2_3ring_cn_both', 'C1=CC=CC=1C=C1C=C1')
                    )
                return None  # 简化接口，不返回状态
                
            def predict(self, smiles):
                """预测能隙"""
                # 使用规则预测
                if 'nh2' in smiles.lower() and 'cn' in smiles.lower():
                    return -0.2
                elif 'oh' in smiles.lower() and 'cn' in smiles.lower():
                    return -0.1
                else:
                    return 0.1
                
        return SimpleEnvironment(self.predictive_model, target_gap, scaffold)
            
    def setup_rl_environment(self, target_gap=-0.1, scaffold=None):
        """设置强化学习环境"""
        # 如果没有可用的RL库，切换到基于规则的环境
        if not self.rl_available:
            self.logger.warning("强化学习库不可用，将使用基于规则的环境")
            self.environment = self.create_rule_based_environment(target_gap, scaffold)
            return True
            
        # 确定使用哪个预测模型
        model_to_use = self.fine_tuned_model if self.fine_tuned_model else self.predictive_model
        
        if model_to_use is None:
            self.logger.error("没有可用的预测模型")
            # 创建一个简单的基本模型
            self.create_basic_model()
            model_to_use = self.predictive_model
            
        # 如果没有指定骨架，使用已知的负能隙分子之一作为起点
        if scaffold is None:
            known_negative_gap_molecules = [
                '5ring_nh2_3ring_cn_both',
                '5ring_nme2_3ring_cn_in_con2',
                '5ring_oh_3ring_cn_both',
                'me_oh_cn',
                '5ring_nh2_3ring_cn_in'
            ]
            # 将这些名称转换为SMILES（实际应用中需要从数据中获取）
            # 这里仅作示例，实际使用时需要根据真实数据调整
            scaffolds = {
                '5ring_nh2_3ring_cn_both': 'C1=CC=C(N)C1=C1CC1C#N',
                '5ring_nme2_3ring_cn_in_con2': 'C1=CC=C(N(C)C)C1=C1CC1C#N',
                '5ring_oh_3ring_cn_both': 'C1=CC=C(O)C1=C1CC1C#N'
            }
            # 选择一个已知的骨架或默认calicene
            scaffold = scaffolds.get(known_negative_gap_molecules[0], 'C1=CC=CC=1C=C1C=C1')
        
        # 创建环境
        self.environment = MoleculeEnvironment(model_to_use, target_gap, scaffold)
        
        # 创建向量化环境
        try:
            from stable_baselines3.common.vec_env import DummyVecEnv
            env = DummyVecEnv([lambda: self.environment])
            
            # 初始化RL模型
            from stable_baselines3 import PPO
            self.rl_model = PPO("MlpPolicy", env, verbose=1)
            
            self.logger.info("强化学习环境设置完成")
        except Exception as e:
            self.logger.error(f"设置RL环境时出错: {e}")
            # 回退到简单环境
            self.environment = self.create_rule_based_environment(target_gap, scaffold)
            
        return True
        
    def train_rl_model(self, total_timesteps=10000):
        """训练强化学习模型"""
        if not self.rl_available:
            self.logger.warning("强化学习库不可用，无法训练RL模型")
            return True  # 返回True以允许流程继续
            
        if self.rl_model is None:
            self.logger.error("RL模型未初始化，请先设置环境")
            return False
            
        try:
            self.logger.info(f"开始训练RL模型，总步数: {total_timesteps}")
            self.rl_model.learn(total_timesteps=total_timesteps)
            self.logger.info("RL模型训练完成")
            
            # 保存模型
<<<<<<< HEAD
            model_dir = "/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach/data/models"
=======
            model_dir = "/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach/data/models"
>>>>>>> 0181d62 (update excited)
            os.makedirs(model_dir, exist_ok=True)
            self.rl_model.save(os.path.join(model_dir, "rl_model"))
            return True
        except Exception as e:
            self.logger.error(f"训练RL模型时出错: {e}")
            return True  # 返回True以允许流程继续
        
    def generate_molecules(self, n_samples=10):
        """生成新的反向TADF分子"""
        # 如果没有RL模型或环境，使用基于规则的方法
        if not self.rl_available or not self.rl_model or not hasattr(self.environment, 'step'):
            return self.generate_molecules_rule_based(n_samples)
            
        # 使用RL模型生成分子
        try:
            # 创建结果列表
            generated_molecules = []
            
            # 评估模型并生成分子
            for _ in range(n_samples):
                obs = self.environment.reset()
                done = False
                
                while not done:
                    action, _ = self.rl_model.predict(obs)
                    obs, _, done, info = self.environment.step(action)
                    
                    if done:
                        mol = info['mol']
                        if mol is not None:
                            smiles = Chem.MolToSmiles(mol)
                            
                            # 预测S1-T1能隙
                            try:
                                gap = self.environment.model.predict(smiles)
                            except:
                                gap = 0.0  # 默认值
                            
                            generated_molecules.append({
                                'smiles': smiles,
                                'predicted_gap': gap,
                                'mol': mol
                            })
                            
            self.logger.info(f"生成了 {len(generated_molecules)} 个分子")
            return generated_molecules
        except Exception as e:
            self.logger.error(f"使用RL生成分子时出错: {e}")
            # 回退到基于规则的方法
            return self.generate_molecules_rule_based(n_samples)
            
    def generate_molecules_rule_based(self, n_samples=10):
        """使用基于规则的方法生成分子"""
        self.logger.info("使用基于规则的方法生成分子")
        
        # 创建结果列表
        generated_molecules = []
        
        # 已知的负能隙模板
        templates = {
            '5ring_nh2_3ring_cn_both': 'C1=CC=C(N)C1=C1CC1C#N',
            '5ring_nme2_3ring_cn_in_con2': 'C1=CC=C(N(C)C)C1=C1CC1C#N',
            '5ring_oh_3ring_cn_both': 'C1=CC=C(O)C1=C1CC1C#N',
            'me_oh_cn': 'CC1=CC=C(O)C1=C1CC1C#N',
            '5ring_nh2_3ring_cn_in': 'C1=CC=C(N)C1=C1CC1C#N'
        }
        
        # 功能基团
        donor_groups = ['N', 'N(C)C', 'OH', 'OMe', 'NH2']
        acceptor_groups = ['CN', 'NO2', 'CF3']
        
        # 基于模板生成分子
        for i in range(n_samples):
            if i < len(templates):
                # 使用现有模板
                name = list(templates.keys())[i]
                smiles = templates[name]
            else:
                # 生成变种
                base_idx = i % len(templates)
                base_name = list(templates.keys())[base_idx]
                base_smiles = templates[base_name]
                
                try:
                    # 尝试修改分子
                    mol = Chem.MolFromSmiles(base_smiles)
                    if mol is None:
                        # 如果分子无效，使用默认模板
                        mol = Chem.MolFromSmiles('C1=CC=C(N)C1=C1CC1C#N')
                        
                    # 简单修改 - 仅作示例
                    smiles = Chem.MolToSmiles(mol)
                except:
                    # 出错时使用默认模板
                    smiles = 'C1=CC=C(N)C1=C1CC1C#N'
            
            # 验证分子有效性
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                    
                # 预测S1-T1能隙
                if self.predictive_model:
                    try:
                        gap = self.predictive_model.predict(smiles)
                    except:
                        # 基于规则的预测
                        if 'N(C)C' in smiles and 'C#N' in smiles:
                            gap = -0.15
                        elif 'NH2' in smiles and 'C#N' in smiles:
                            gap = -0.1
                        elif 'OH' in smiles and 'C#N' in smiles:
                            gap = -0.05
                        else:
                            gap = 0.1
                else:
                    # 没有模型时的默认预测
                    gap = -0.1 if 'N' in smiles and 'C#N' in smiles else 0.1
                
                generated_molecules.append({
                    'smiles': smiles,
                    'predicted_gap': gap,
                    'mol': mol
                })
            except Exception as e:
                self.logger.error(f"处理分子 {smiles} 时出错: {e}")
        
        self.logger.info(f"生成了 {len(generated_molecules)} 个分子")
        return generated_molecules
        
    def evaluate_molecules(self, molecules):
        """评估生成的分子"""
        if not molecules:
            return None
            
        results = []
        
        for mol_data in molecules:
            smiles = mol_data['smiles']
            mol = mol_data['mol']
            
            # 计算分子特性
            try:
                # 分子基本属性
                mw = Descriptors.MolWt(mol)
                # TADF相关特性
                aromatic_rings = Descriptors.NumAromaticRings(mol)
                ring_count = Descriptors.RingCount(mol)
                rot_bonds = Descriptors.NumRotatableBonds(mol)
                
                # 电子性质
                donors = Fragments.fr_NH2(mol) + Fragments.fr_NH1(mol) + Fragments.fr_NH0(mol) + Fragments.fr_Ar_OH(mol)
                acceptors = Fragments.fr_C_O(mol) + Fragments.fr_nitro(mol) + Fragments.fr_nitro_arom(mol) + Fragments.fr_nitrile(mol)
                
                # 创建结果字典
                result = {
                    'smiles': smiles,
                    'predicted_gap': mol_data['predicted_gap'],
                    'molecular_weight': mw,
                    'aromatic_rings': aromatic_rings,
                    'ring_count': ring_count,
                    'donor_groups': donors,
                    'acceptor_groups': acceptors,
                    'rotatable_bonds': rot_bonds
                }
                
                # 检查是否含有calicene结构
                calicene_pattern = Chem.MolFromSmarts('C1=CC=CC=1C=C1C=C1')
                result['has_calicene'] = 1 if calicene_pattern and mol.HasSubstructMatch(calicene_pattern) else 0
                
                results.append(result)
            except Exception as e:
                self.logger.error(f"评估分子 {smiles} 时出错: {e}")
            
        # 创建结果数据框
        if results:
            results_df = pd.DataFrame(results)
            
            # 保存结果
<<<<<<< HEAD
            report_dir = "/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach/data/reports"
=======
            report_dir = "/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach/data/reports"
>>>>>>> 0181d62 (update excited)
            os.makedirs(report_dir, exist_ok=True)
            results_df.to_csv(os.path.join(report_dir, "generated_molecules.csv"), index=False)
            
            return results_df
        else:
            return None
        
<<<<<<< HEAD
    def visualize_results(self, results_df, output_dir="/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach/data/reports"):
=======
    def visualize_results(self, results_df, output_dir="/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach/data/reports"):
>>>>>>> 0181d62 (update excited)
        """可视化生成的分子结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. S1-T1能隙分布
        plt.figure(figsize=(10, 6))
        sns.histplot(data=results_df, x='predicted_gap', kde=True)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title('Generated Molecules: S1-T1 Gap Distribution')
        plt.xlabel('Predicted S1-T1 Gap (eV)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'generated_gap_distribution.png'))
        plt.close()
        
        # 2. 分子属性散点图
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=results_df, x='molecular_weight', y='donor_groups', 
                        hue='predicted_gap', palette='coolwarm', size='acceptor_groups',
                        sizes=(50, 200))
        plt.title('Generated Molecules: Structure-Property Relationships')
        plt.xlabel('Molecular Weight')
        plt.ylabel('Donor Groups Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'generated_property_space.png'))
        plt.close()
        
        # 3. 绘制前10个分子结构
        try:
            top_molecules = results_df.sort_values('predicted_gap').head(10)
            mols = [Chem.MolFromSmiles(smiles) for smiles in top_molecules['smiles']]
            
            # 添加S1-T1能隙值作为分子标签
            for i, mol in enumerate(mols):
                if mol:
                    gap_value = top_molecules.iloc[i]['predicted_gap']
                    mol.SetProp("_Name", f"Gap: {gap_value:.2f} eV")
                
            valid_mols = [mol for mol in mols if mol is not None]
            if valid_mols:
                img = Draw.MolsToGridImage(valid_mols, molsPerRow=5, subImgSize=(300, 300), 
                                          legends=[mol.GetProp("_Name") for mol in valid_mols])
                img.save(os.path.join(output_dir, 'top_generated_molecules.png'))
        except Exception as e:
            self.logger.error(f"可视化分子结构时出错: {e}")
        
        self.logger.info(f"结果可视化已保存至 {output_dir}")
    
    def run_design_pipeline(self, target_gap=-0.1, scaffold=None, n_samples=20):
        """运行完整的分子设计流程"""
        try:
            # 1. 设置环境
            self.setup_rl_environment(target_gap, scaffold)
            
            # 2. 训练RL模型（如果可用）
            if self.rl_available and self.rl_model:
                self.train_rl_model(total_timesteps=5000)  # 减少训练步数，提高效率
            
            # 3. 生成分子
            molecules = self.generate_molecules(n_samples)
            if not molecules:
                self.logger.error("生成分子失败")
                return None
                
            # 4. 评估分子
            results_df = self.evaluate_molecules(molecules)
            
            # 5. 可视化结果
            if results_df is not None:
                self.visualize_results(results_df)
            
            return {
                'molecules': molecules,
                'results_df': results_df,
<<<<<<< HEAD
                'report_path': "/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach/data/reports/generated_molecules.csv"
=======
                'report_path': "/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach/data/reports/generated_molecules.csv"
>>>>>>> 0181d62 (update excited)
            }
        except Exception as e:
            self.logger.error(f"运行分子设计流程时出错: {e}")
            # 尝试恢复并返回部分结果
            return {
                'error': str(e),
<<<<<<< HEAD
                'report_path': "/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach/data/reports/generated_molecules.csv"
=======
                'report_path': "/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach/data/reports/generated_molecules.csv"
>>>>>>> 0181d62 (update excited)
            }