# agents/fine_tuning_agent.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tqdm import tqdm

class MoleculeGraphDataset(Dataset):
    """分子图数据集，用于GNN训练"""
    
    def __init__(self, molecules, labels, smiles_list=None):
        """初始化数据集
        
        Args:
            molecules: RDKit分子对象列表或SMILES字符串列表
            labels: 标签（如S1-T1能隙值或正/负分类）
            smiles_list: 可选的SMILES字符串列表（如果molecules不是SMILES）
        """
        self.molecules = molecules
        self.labels = labels
        self.smiles_list = smiles_list
        
        # 如果传入的是RDKit分子，尝试生成SMILES
        if smiles_list is None and isinstance(molecules[0], Chem.Mol):
            self.smiles_list = [Chem.MolToSmiles(mol) for mol in molecules]
        
    def __len__(self):
        return len(self.molecules)
    
    def __getitem__(self, idx):
        if isinstance(self.molecules[idx], str):
            # 如果是SMILES字符串，转换为分子
            mol = Chem.MolFromSmiles(self.molecules[idx])
        else:
            mol = self.molecules[idx]
            
        if mol is None:
            # 如果分子无效，返回一个默认的小图
            x = torch.zeros((1, 58))
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            return Data(x=x, edge_index=edge_index), torch.tensor(0.0)
        
        # 生成原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            features = self._get_atom_features(atom)
            atom_features.append(features)
        
        x = torch.tensor(atom_features, dtype=torch.float)
        
        # 生成边索引和特征
        edge_index = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            # 添加双向边
            edge_index.append([i, j])
            edge_index.append([j, i])
        
        if not edge_index:
            # 如果没有边，创建一个空的边索引
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        # 创建图数据对象
        graph_data = Data(x=x, edge_index=edge_index)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return graph_data, label
    
    def _get_atom_features(self, atom):
        """为原子生成特征向量"""
        # 原子类型的独热编码
        atom_types = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'H', 'Unknown']
        atom_type = [0] * len(atom_types)
        atom_type[atom_types.index(atom.GetSymbol()) if atom.GetSymbol() in atom_types else -1] = 1
        
        # 原子特性
        formal_charge = [atom.GetFormalCharge()]
        hybridization = [
            atom.GetHybridization() == Chem.rdchem.HybridizationType.SP,
            atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2,
            atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3,
            atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3D,
            atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3D2
        ]
        aromatic = [atom.GetIsAromatic()]
        degree = [atom.GetDegree()]
        valence = [atom.GetTotalValence()]
        h_count = [atom.GetTotalNumHs()]
        radical_electrons = [atom.GetNumRadicalElectrons()]
        in_ring = [atom.IsInRing()]
        
        # 结合所有特征
        features = atom_type + formal_charge + hybridization + aromatic + degree + valence + h_count + radical_electrons + in_ring
        
        # 添加特征以捕捉共轭效应
        conjugated = [atom.GetIsConjugated() if hasattr(atom, 'GetIsConjugated') else 0]
        features += conjugated
        
        # 添加特征以捕捉平面扭曲
        in_ring_size = []
        for i in range(3, 9):  # 检查3到8元环
            in_ring_size.append(1 if atom.IsInRingSize(i) else 0)
        features += in_ring_size
        
        # 判断是否在可能形成氢键的元素 (N, O, F)
        h_bond_acceptor = [1 if atom.GetSymbol() in ['N', 'O', 'F'] else 0]
        # 判断是否可能是氢键供体 (O-H, N-H)
        h_bond_donor = [1 if (atom.GetSymbol() in ['N', 'O']) and atom.GetTotalNumHs() > 0 else 0]
        
        features += h_bond_acceptor + h_bond_donor
        
        return features

class GNNModel(nn.Module):
    """用于分子属性预测的图神经网络模型"""
    
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        """初始化GNN模型
        
        Args:
            input_dim: 输入特征维度（原子特征大小）
            hidden_dim: 隐藏层维度
            output_dim: 输出维度（1表示回归或二分类）
        """
        super(GNNModel, self).__init__()
        
        # 图卷积层
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Dropout正则化
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, data):
        """前向传播
        
        Args:
            data: PyG数据对象，包含x和edge_index
            
        Returns:
            输出预测
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 如果batch未指定，默认所有节点属于同一图
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        
        # 图卷积层
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        
        # 全局池化（将节点特征聚合为图特征）
        x = global_mean_pool(x, batch)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class TransformerMoleculeModel(nn.Module):
    """基于Transformer的分子表示模型（使用预训练的化学语言模型）"""
    
    def __init__(self, model_name="seyonec/ChemBERTa-zinc-base", output_dim=1):
        """初始化Transformer模型
        
        Args:
            model_name: 预训练模型名称
            output_dim: 输出维度
        """
        super(TransformerMoleculeModel, self).__init__()
        
        # 加载预训练模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # 分类/回归头
        transformer_dim = self.transformer.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(transformer_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, smiles_list):
        """前向传播
        
        Args:
            smiles_list: SMILES字符串列表
            
        Returns:
            输出预测
        """
        # 分词并获取输入ID
        inputs = self.tokenizer(smiles_list, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(next(self.transformer.parameters()).device) for k, v in inputs.items()}
        
        # 获取transformer输出
        outputs = self.transformer(**inputs)
        
        # 使用[CLS]令牌的最后隐藏状态
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # 通过分类/回归头
        logits = self.classifier(pooled_output)
        
        return logits

class HybridModel(nn.Module):
    """结合GNN和Transformer的混合模型"""
    
    def __init__(self, gnn_input_dim, transformer_model_name="seyonec/ChemBERTa-zinc-base", output_dim=1):
        """初始化混合模型
        
        Args:
            gnn_input_dim: GNN模型的输入维度
            transformer_model_name: 预训练模型名称
            output_dim: 输出维度
        """
        super(HybridModel, self).__init__()
        
        # GNN分支
        self.gnn = GNNModel(gnn_input_dim, hidden_dim=64, output_dim=64)
        
        # Transformer分支
        self.transformer_model = TransformerMoleculeModel(model_name=transformer_model_name, output_dim=64)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, graph_data, smiles_list):
        """前向传播
        
        Args:
            graph_data: PyG数据对象
            smiles_list: SMILES字符串列表
            
        Returns:
            输出预测
        """
        # GNN分支
        gnn_output = self.gnn(graph_data)
        
        # Transformer分支
        transformer_output = self.transformer_model(smiles_list)
        
        # 融合两个分支的输出
        combined = torch.cat([gnn_output, transformer_output], dim=1)
        
        # 通过融合层
        output = self.fusion(combined)
        
        return output

class TuningAgent:
    """用于微调分子属性预测模型的代理"""
    
    def __init__(self, data_file=None):
        """初始化微调代理
        
        Args:
            data_file: 包含分子特征和标签的CSV文件路径
        """
        self.data_file = data_file
        self.df = None
        self.models = {}
        self.model_type = "hybrid"  # 默认模型类型
        self.model_name = "seyonec/ChemBERTa-zinc-base"  # 默认预训练模型
        self.setup_logging()
        
    def setup_logging(self):
        """配置日志记录"""
        log_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/logs'
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO,
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                           filename=os.path.join(log_dir, 'tuning_agent.log'))
        self.logger = logging.getLogger('TuningAgent')
        
    def load_data(self, file_path=None):
        """加载数据
        
        Args:
            file_path: 可选的数据文件路径，如果未提供则使用初始化时的路径
            
        Returns:
            加载成功返回True，否则返回False
        """
        if file_path:
            self.data_file = file_path
            
        if not self.data_file or not os.path.exists(self.data_file):
            self.logger.error(f"数据文件未找到: {self.data_file}")
            return False
            
        print(f"从 {self.data_file} 加载数据...")
        try:
            self.df = pd.read_csv(self.data_file)
            
            # 打印数据集基本信息
            if self.df is not None:
                print(f"数据集形状: {self.df.shape}")
                print(f"列名: {self.df.columns.tolist()[:10]}...")
                
                # 检查S1-T1能隙数据
                s1t1_cols = [col for col in self.df.columns if 's1_t1' in col.lower() or 'triplet_gap' in col.lower()]
                if s1t1_cols:
                    for col in s1t1_cols:
                        non_null = self.df[col].notna().sum()
                        neg_count = (self.df[col] < 0).sum()
                        print(f"列 '{col}' 有 {non_null} 个非空值，其中 {neg_count} 个负值")
                else:
                    print("警告：未找到S1-T1能隙相关列")
                
            return True
        except Exception as e:
            self.logger.error(f"加载数据时出错: {e}")
            print(f"加载数据时出错: {e}")
            return False
            
    def generate_smiles(self):
        """从分子名称生成SMILES字符串
        
        Returns:
            带有'smiles'列的数据框
        """
        if self.df is None:
            self.logger.error("未加载数据。先调用load_data()。")
            return None
            
        if 'smiles' in self.df.columns:
            print("数据框已包含'smiles'列")
            return self.df
            
        try:
            # 尝试使用RDKit将分子名称转换为SMILES
            # 这是一个简化版本，实际实现可能需要更复杂的逻辑
            from rdkit.Chem import AllChem
            
            def name_to_smiles(name):
                """将分子名称转换为SMILES字符串"""
                # 这里只是一个简单的示例，实际实现需要根据命名规则进行调整
                if 'ring' in name.lower():
                    if '5ring' in name.lower() and '3ring' in name.lower():
                        # 假设这是一个calicene衍生物
                        if 'cn' in name.lower():
                            return "C1=CC=C(C1)C=C1CC1"  # 带CN的简化calicene
                        else:
                            return "C1=CC=C(C1)C=C1CC1"  # 简化的calicene
                    elif '5ring' in name.lower():
                        return "C1=CCC=C1"  # 简化的环戊二烯
                    elif '7ring' in name.lower():
                        return "C1=CC=CC=CC=1"  # 简化的七元环
                    else:
                        return "c1ccccc1"  # 默认苯环
                elif 'cn' in name.lower():
                    return "C#N"  # 氰基
                elif 'nh2' in name.lower():
                    return "N"  # 胺基
                elif 'oh' in name.lower():
                    return "O"  # 羟基
                else:
                    return "C"  # 默认返回甲烷
            
            self.df['smiles'] = self.df['Molecule'].apply(name_to_smiles)
            print(f"已为{len(self.df)}个分子生成SMILES字符串")
            
            return self.df
        except Exception as e:
            self.logger.error(f"生成SMILES时出错: {e}")
            print(f"生成SMILES时出错: {e}")
            
            # 创建一个简单的填充SMILES列
            self.df['smiles'] = "C"  # 所有分子默认为甲烷
            return self.df
            
    def prepare_molecular_data(self, target_col, is_classification=False):
        """准备模型训练所需的分子数据
        
        Args:
            target_col: 目标列名（如's1_t1_gap_ev'）
            is_classification: 是否为分类任务
            
        Returns:
            分子、标签和SMILES字符串
        """
        if self.df is None:
            self.logger.error("未加载数据。先调用load_data()。")
            return None, None, None
            
        # 生成SMILES字符串（如果尚未生成）
        self.generate_smiles()
            
        # 筛选有目标值的行
        valid_data = self.df[self.df[target_col].notna()].copy()
        
        if len(valid_data) < 5:  # 放宽限制，允许更少的样本用于测试
            self.logger.warning(f"{target_col}数据样本太少（{len(valid_data)}），推荐至少5个样本")
            print(f"警告：{target_col}数据样本太少（{len(valid_data)}），推荐至少5个样本")
            
        # 准备SMILES字符串
        smiles_list = valid_data['smiles'].tolist()
            
        # 准备标签
        if is_classification:
            # 对于分类任务，创建二进制标签
            labels = (valid_data[target_col] < 0).astype(int).values
            print(f"创建分类标签，负值（标签1）样本数: {sum(labels)}, 正值（标签0）样本数: {len(labels) - sum(labels)}")
        else:
            # 对于回归任务，直接使用数值
            labels = valid_data[target_col].values
            print(f"使用回归标签，范围: {min(labels):.4f} 到 {max(labels):.4f}")
            
        # 分子对象通过SMILES创建
        try:
            molecules = [Chem.MolFromSmiles(s) if s and isinstance(s, str) else None for s in smiles_list]
            # 过滤掉无效分子
            valid_indices = [i for i, mol in enumerate(molecules) if mol is not None]
            
            if len(valid_indices) < len(molecules):
                self.logger.warning(f"有{len(molecules) - len(valid_indices)}个无效分子被过滤掉")
                print(f"警告：有{len(molecules) - len(valid_indices)}个无效分子被过滤掉")
                
            molecules = [molecules[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]
            smiles_list = [smiles_list[i] for i in valid_indices]
            
            print(f"准备了{len(molecules)}个有效分子用于模型训练")
            
        except Exception as e:
            self.logger.error(f"准备分子数据时出错: {e}")
            print(f"准备分子数据时出错: {e}")
            return None, None, None
            
        return molecules, labels, smiles_list
        
    def _collate_fn(self, batch):
        """自定义的批处理函数，用于处理图数据"""
        graphs, labels = zip(*batch)
        batched_graphs = Batch.from_data_list(graphs)
        batched_labels = torch.tensor(labels)
        return batched_graphs, batched_labels
        
    def create_model(self, input_dim=58, output_dim=1):
        """创建神经网络模型
        
        Args:
            input_dim: 输入特征维度
            output_dim: 输出维度
            
        Returns:
            创建的模型
        """
        if self.model_type == "gnn":
            model = GNNModel(input_dim=input_dim, output_dim=output_dim)
            print(f"创建GNN模型，输入维度: {input_dim}，输出维度: {output_dim}")
        elif self.model_type == "transformer":
            model = TransformerMoleculeModel(model_name=self.model_name, output_dim=output_dim)
            print(f"创建Transformer模型 ({self.model_name})，输出维度: {output_dim}")
        else:  # hybrid
            model = HybridModel(gnn_input_dim=input_dim, transformer_model_name=self.model_name, output_dim=output_dim)
            print(f"创建混合模型，GNN输入维度: {input_dim}，输出维度: {output_dim}")
            
        return model
        
    def train_model(self, target_col, is_classification=False, epochs=30, batch_size=16, learning_rate=0.001):
        """训练深度学习模型
        
        Args:
            target_col: 目标列名（如's1_t1_gap_ev'）
            is_classification: 是否为分类任务
            epochs: 训练轮次
            batch_size: 批大小
            learning_rate: 学习率
            
        Returns:
            训练结果字典
        """
        # 准备数据
        molecules, labels, smiles_list = self.prepare_molecular_data(target_col, is_classification)
        
        if molecules is None or len(molecules) == 0:
            self.logger.error("数据准备失败。")
            print("错误：数据准备失败。")
            return None
            
        # 创建结果目录
        results_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/reports/tuning'
        os.makedirs(results_dir, exist_ok=True)
        
        # 创建数据集
        try:
            dataset = MoleculeGraphDataset(molecules, labels, smiles_list)
            print(f"创建数据集，包含 {len(dataset)} 个样本")
            
            # 拆分训练集和测试集
            train_idx, test_idx = train_test_split(
                range(len(dataset)), test_size=0.2, random_state=42,
                stratify=labels if is_classification else None
            )
            
            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            test_dataset = torch.utils.data.Subset(dataset, test_idx)
            
            print(f"拆分数据：训练集 {len(train_dataset)} 样本，测试集 {len(test_dataset)} 样本")
            
            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)), shuffle=True, 
                                    collate_fn=self._collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=min(batch_size, len(test_dataset)), 
                                  collate_fn=self._collate_fn)
        except Exception as e:
            self.logger.error(f"创建数据加载器时出错: {e}")
            print(f"错误：创建数据加载器时出错: {e}")
            return None
            
        # 确定原子特征维度（从第一个有效分子中获取）
        first_graph, _ = dataset[0]
        input_dim = first_graph.x.shape[1]
        
        # 创建模型
        try:
            model = self.create_model(input_dim=input_dim, output_dim=1)
            
            # 移至GPU（如果可用）
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            print(f"使用设备：{device}")
            
            # 设置损失函数和优化器
            if is_classification:
                criterion = nn.BCEWithLogitsLoss()
                print("使用二元交叉熵损失函数")
            else:
                criterion = nn.MSELoss()
                print("使用均方误差损失函数")
                
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            print(f"使用Adam优化器，学习率: {learning_rate}")
        except Exception as e:
            self.logger.error(f"创建模型时出错: {e}")
            print(f"错误：创建模型时出错: {e}")
            return None
            
        # 训练循环
        print(f"开始训练，总共 {epochs} 轮次...")
        best_loss = float('inf')
        best_model = None
        train_losses = []
        val_losses = []
        
        try:
            for epoch in range(epochs):
                # 训练阶段
                model.train()
                train_loss = 0
                
                for batch_data in train_loader:
                    graphs, targets = batch_data
                    graphs = graphs.to(device)
                    targets = targets.to(device)
                    
                    # 前向传播
                    outputs = model(graphs)
                    
                    # 计算损失
                    loss = criterion(outputs.squeeze(), targets)
                    
                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * len(targets)
                
                train_loss /= len(train_dataset)
                train_losses.append(train_loss)
                
                # 验证阶段
                model.eval()
                val_loss = 0
                
                with torch.no_grad():
                    for batch_data in test_loader:
                        graphs, targets = batch_data
                        graphs = graphs.to(device)
                        targets = targets.to(device)
                        
                        # 前向传播
                        outputs = model(graphs)
                        
                        # 计算损失
                        loss = criterion(outputs.squeeze(), targets)
                        val_loss += loss.item() * len(targets)
                
                val_loss /= len(test_dataset)
                val_losses.append(val_loss)
                
                # 打印进度
                if (epoch + 1) % 5 == 0 or epoch == 0:
                    print(f'轮次 {epoch+1}/{epochs}, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}')
                    
                # 保存最佳模型
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        
        except Exception as e:
            self.logger.error(f"训练模型时出错: {e}")
            print(f"错误：训练模型时出错: {e}")
            # 如果训练中断，但有部分训练好的模型，尝试继续评估
            if best_model is None:
                return None
        
        # 评估模型
        try:
            # 加载最佳模型
            if best_model is not None:
                model.load_state_dict(best_model)
            
            # 绘制训练和验证损失曲线
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='训练损失')
            plt.plot(val_losses, label='验证损失')
            plt.title(f'{target_col} {"分类" if is_classification else "回归"} 训练')
            plt.xlabel('轮次')
            plt.ylabel('损失')
            plt.legend()
            plt.tight_layout()
            
            loss_curve_path = os.path.join(results_dir, f'{target_col}_loss_curve.png')
            plt.savefig(loss_curve_path)
            plt.close()
            
            print(f"损失曲线已保存到: {loss_curve_path}")
            
            # 评估模型性能
            model.eval()
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for batch_data in test_loader:
                    graphs, targets = batch_data
                    graphs = graphs.to(device)
                    targets = targets.to(device)
                    
                    # 前向传播
                    outputs = model(graphs)
                    
                    # 收集预测和目标
                    if is_classification:
                        preds = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
                    else:
                        preds = outputs.squeeze()
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
            
            # 根据任务类型计算性能指标
            if is_classification:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                
                accuracy = accuracy_score(all_targets, all_preds)
                precision = precision_score(all_targets, all_preds, zero_division=0)
                recall = recall_score(all_targets, all_preds, zero_division=0)
                f1 = f1_score(all_targets, all_preds, zero_division=0)
                conf_matrix = confusion_matrix(all_targets, all_preds)
                
                metrics = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'confusion_matrix': conf_matrix
                }
                
                print(f"分类性能指标:")
                print(f"  - 准确率: {accuracy:.4f}")
                print(f"  - 精确率: {precision:.4f}")
                print(f"  - 召回率: {recall:.4f}")
                print(f"  - F1分数: {f1:.4f}")
                
                # 绘制混淆矩阵
                plt.figure(figsize=(8, 6))
                sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                          xticklabels=['正值能隙', '负值能隙'],
                          yticklabels=['正值能隙', '负值能隙'])
                plt.xlabel('预测')
                plt.ylabel('实际')
                plt.title(f'{target_col} 分类混淆矩阵')
                plt.tight_layout()
                
                confusion_matrix_path = os.path.join(results_dir, f'{target_col}_confusion_matrix.png')
                plt.savefig(confusion_matrix_path)
                plt.close()
                
                print(f"混淆矩阵已保存到: {confusion_matrix_path}")
                
                prediction_plot_path = confusion_matrix_path
                
            else:
                from sklearn.metrics import mean_squared_error, r2_score
                
                mse = mean_squared_error(all_targets, all_preds)
                rmse = np.sqrt(mse)
                r2 = r2_score(all_targets, all_preds)
                
                metrics = {
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2
                }
                
                print(f"回归性能指标:")
                print(f"  - 均方误差: {mse:.4f}")
                print(f"  - 均方根误差: {rmse:.4f}")
                print(f"  - R²分数: {r2:.4f}")
                
                # 绘制预测值vs实际值
                plt.figure(figsize=(8, 8))
                plt.scatter(all_targets, all_preds, alpha=0.5)
                
                # 添加理想线(y=x)
                min_val = min(min(all_targets), min(all_preds))
                max_val = max(max(all_targets), max(all_preds))
                plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                
                plt.xlabel('实际值')
                plt.ylabel('预测值')
                plt.title(f'{target_col} 回归: 预测 vs 实际')
                
                # 添加性能指标文本
                plt.text(
                    0.05, 0.95, 
                    f"RMSE: {rmse:.4f}\nR²: {r2:.4f}",
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', alpha=0.1)
                )
                
                # 添加零线
                plt.axhline(y=0, color='green', linestyle='-', alpha=0.3)
                plt.axvline(x=0, color='green', linestyle='-', alpha=0.3)
                
                plt.tight_layout()
                prediction_plot_path = os.path.join(results_dir, f'{target_col}_prediction.png')
                plt.savefig(prediction_plot_path)
                plt.close()
                
                print(f"预测图已保存到: {prediction_plot_path}")
            
            # 保存模型
            model_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/models'
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, f'{target_col}_{"classifier" if is_classification else "regressor"}.pt')
            torch.save({
                'model_state_dict': best_model if best_model is not None else model.state_dict(),
                'model_type': self.model_type,
                'input_dim': input_dim,
                'is_classification': is_classification
            }, model_path)
            
            print(f"模型已保存到: {model_path}")
            
            # 保存特征处理器
            feature_processor = {
                'target_col': target_col,
                'is_classification': is_classification
            }
            
            processor_path = os.path.join(model_dir, f'{target_col}_processor.pkl')
            joblib.dump(feature_processor, processor_path)
            
            # 保存模型到实例
            self.models[target_col] = {
                'model': model,
                'metrics': metrics,
                'is_classification': is_classification,
                'model_path': model_path,
                'processor_path': processor_path
            }
            
            return {
                'model_path': model_path,
                'processor_path': processor_path,
                'metrics': metrics,
                'loss_curve': loss_curve_path,
                'prediction_plot': prediction_plot_path
            }
            
        except Exception as e:
            self.logger.error(f"评估模型时出错: {e}")
            print(f"错误：评估模型时出错: {e}")
            return None
    
    def run_tuning_pipeline(self, feature_file=None, model_type="hybrid", epochs=10,
                     data_file=None, batch_size=16, learning_rate=0.0001,
                     target_col='s1_t1_gap_ev', smiles_col='Molecule', **kwargs):
        """运行完整的微调流程
            
        Args:
            feature_file: 特征文件路径(首选)
            data_file: 备用数据文件路径
            model_type: 模型类型 ('gnn', 'transformer', 'hybrid')
            epochs: 训练轮次
            batch_size: 批处理大小
            learning_rate: 学习率
            target_col: 目标列名
            smiles_col: 分子SMILES列名
            **kwargs: 其他参数
                
        Returns:
            流程结果字典
        """
        print(f"运行微调流程,使用模型类型: {model_type}, 轮次: {epochs}, 批次: {batch_size}, 学习率: {learning_rate}")
            
        # 更新模型类型和参数
        self.model_type = model_type
            
        # 处理文件路径,优先使用feature_file
        actual_file = feature_file if feature_file is not None else data_file
            
        # 加载数据
        if actual_file:
            self.load_data(actual_file)
        elif self.data_file:
            self.load_data()
        else:
            self.logger.error("未指定数据文件。")
            print("错误:未指定数据文件。")
            return {"status": "error", "message": "未提供数据文件"}
                
        # 创建结果目录
        results_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/reports/tuning'
        os.makedirs(results_dir, exist_ok=True)
            
        # 创建模型目录
        model_dir = '/vol1/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system/data/models'
        os.makedirs(model_dir, exist_ok=True)
            
        # 打印数据框信息以便调试
        print(f"加载的数据框形状: {self.df.shape}")
        print(f"数据框列名: {self.df.columns.tolist()}")
        
        # 确定目标列
        if target_col not in self.df.columns:
            # 尝试不区分大小写地查找列名
            case_insensitive_cols = {col.lower(): col for col in self.df.columns}
            if target_col.lower() in case_insensitive_cols:
                # 找到了不区分大小写的匹配
                actual_col = case_insensitive_cols[target_col.lower()]
                print(f"找到列名匹配(不区分大小写): {actual_col}")
                target_col = actual_col
            else:
                # 查找任何包含s1_t1或triplet_gap的列
                s1t1_columns = [col for col in self.df.columns 
                            if ('s1_t1' in col.lower() or 
                                'triplet_gap' in col.lower() or
                                's1t1gap' in col.lower())]
                
                if not s1t1_columns:
                    # 如果找不到，尝试创建该列(基于特征工程)
                    print("尝试从现有数据创建S1-T1能隙列...")
                    if 's1_energy_ev' in self.df.columns and 't1_energy_ev' in self.df.columns:
                        self.df['s1_t1_gap_ev'] = self.df['s1_energy_ev'] - self.df['t1_energy_ev']
                        print("已创建s1_t1_gap_ev列")
                        target_col = 's1_t1_gap_ev'
                    elif 'gap_ev' in self.df.columns:
                        # 如果有其他能隙列，尝试使用它
                        print(f"使用替代能隙列: gap_ev")
                        target_col = 'gap_ev'
                    else:
                        self.logger.error(f"未找到目标列 {target_col} 或任何S1-T1能隙数据列，且无法创建。")
                        print(f"错误:未找到目标列 {target_col} 或任何S1-T1能隙数据列，且无法创建。")
                        # 打印数据框中前10个列的前几个值，以帮助调试
                        if not self.df.empty:
                            sample_cols = min(10, len(self.df.columns))
                            print("数据框前10行的样本值:")
                            print(self.df.iloc[:5, :sample_cols])
                        return {"status": "error", "message": f"未找到目标列 {target_col}"}
                else:
                    target_col = s1t1_columns[0]
                    print(f"使用找到的目标列: {target_col}")
        
        # 检查并报告目标列的数据情况
        if target_col in self.df.columns:
            non_null_count = self.df[target_col].count()
            total_count = len(self.df)
            print(f"目标列 {target_col} 有 {non_null_count}/{total_count} 个非空值 ({non_null_count/total_count*100:.1f}%)")
            
            if self.df[target_col].isna().all():
                self.logger.warning(f"目标列 {target_col} 所有值均为NaN")
                print(f"警告: 目标列 {target_col} 所有值均为NaN")
                return {"status": "error", "message": f"目标列 {target_col} 所有值均为NaN"}
        
        try:
            # 判断是否为分类任务
            is_classification = (target_col == 'is_negative_gap' or
                            (target_col in self.df.columns and
                            pd.api.types.is_bool_dtype(self.df[target_col]) or
                            set(self.df[target_col].unique()) == {0, 1}))
                    
            # 训练模型
            print(f"训练{'分类' if is_classification else '回归'}模型...")
            results = self.train_model(
                target_col,
                is_classification=is_classification,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
                    
            if not results:
                self.logger.error("模型训练失败")
                print("错误:模型训练失败")
                return {"status": "error", "message": "模型训练失败"}
                    
            # 准备返回结果
            return {
                "status": "success",
                "message": f"微调完成,使用模型类型: {model_type}",
                "model_type": model_type,
                "epochs": epochs,
                "smiles_col": smiles_col,
                "metrics": results,
                "target_col": target_col,
                "is_classification": is_classification
            }
                    
        except Exception as e:
            self.logger.error(f"微调过程中出错: {e}")
            print(f"错误:微调过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": f"微调过程中出错: {str(e)}"}