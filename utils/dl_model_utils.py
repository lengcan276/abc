# utils/dl_model_utils.py
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from transformers import AutoTokenizer
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MolecularFeatureExtractor:
    """分子特征提取器，用于深度学习模型"""
    
    @staticmethod
    def smiles_to_fingerprint(smiles, radius=2, nbits=2048):
        """将SMILES转换为Morgan指纹"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits))
        except:
            return None
    
    @staticmethod
    def smiles_to_descriptors(smiles):
        """将SMILES转换为RDKit描述符"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
                
            # 计算常用描述符
            descriptors = {}
            descriptors['MolWt'] = Descriptors.MolWt(mol)
            descriptors['LogP'] = Descriptors.MolLogP(mol)
            descriptors['NumHDonors'] = Descriptors.NumHDonors(mol)
            descriptors['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
            descriptors['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
            descriptors['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
            descriptors['TPSA'] = Descriptors.TPSA(mol)
            
            return list(descriptors.values())
        except:
            return None
            
    @staticmethod
    def smiles_to_tokens(smiles, tokenizer):
        """使用预训练的分词器将SMILES转换为token ID"""
        try:
            encoding = tokenizer(
                smiles, 
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=128
            )
            return encoding
        except Exception as e:
            logger.error(f"分词错误: {e}")
            return None
            
    @staticmethod
    def concat_features(fingerprints, descriptors):
        """连接指纹和描述符特征"""
        if fingerprints is None or descriptors is None:
            return None
        return fingerprints + descriptors

class DLModelInference:
    """深度学习模型推理类"""
    
    def __init__(self, model, tokenizer=None, device=None):
        """初始化模型推理类"""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, smiles, batch_size=32):
        """批量预测SMILES的属性"""
        if isinstance(smiles, str):
            smiles = [smiles]
            
        # 分批处理
        predictions = []
        for i in range(0, len(smiles), batch_size):
            batch = smiles[i:i+batch_size]
            batch_predictions = self._predict_batch(batch)
            predictions.extend(batch_predictions)
            
        return predictions if len(smiles) > 1 else predictions[0]
        
    def _predict_batch(self, smiles_batch):
        """预测一个批次的SMILES"""
        # 分词
        encodings = [MolecularFeatureExtractor.smiles_to_tokens(s, self.tokenizer) for s in smiles_batch]
        valid_indices = [i for i, e in enumerate(encodings) if e is not None]
        valid_encodings = [encodings[i] for i in valid_indices]
        
        if not valid_encodings:
            return [np.nan] * len(smiles_batch)
            
        # 准备数据
        input_ids = torch.cat([e['input_ids'] for e in valid_encodings]).to(self.device)
        attention_mask = torch.cat([e['attention_mask'] for e in valid_encodings]).to(self.device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.squeeze(-1).cpu().numpy()
            
        # 重建完整预测结果
        full_predictions = [np.nan] * len(smiles_batch)
        for idx, pred in zip(valid_indices, predictions):
            full_predictions[idx] = pred
            
        return full_predictions

def load_fine_tuned_model(model_path, tokenizer_path=None):
    """加载微调的模型"""
    try:
        # 加载模型
        model = torch.load(model_path)
        
        # 加载分词器
        tokenizer = None
        if tokenizer_path:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        elif tokenizer_path is None and os.path.exists(os.path.join(model_path, 'tokenizer')):
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, 'tokenizer'))
            
        # 创建推理对象
        inference = DLModelInference(model, tokenizer)
        
        return inference
    except Exception as e:
        logger.error(f"加载模型时出错: {e}")
        return None
