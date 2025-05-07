# utils/smiles_utils.py
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import pandas as pd
import numpy as np
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_smiles(smiles):
    """验证SMILES字符串"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def canonicalize_smiles(smiles):
    """规范化SMILES字符串"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        return None
    except:
        return None

def smiles_to_mol(smiles):
    """将SMILES转换为RDKit分子对象"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except:
        return None

def mol_to_smiles(mol):
    """将RDKit分子对象转换为SMILES"""
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        return None

def draw_molecule(smiles, molSize=(400, 300), filename=None):
    """绘制分子结构"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        img = Draw.MolToImage(mol, size=molSize)
        
        if filename:
            img.save(filename)
            
        return img
    except Exception as e:
        logger.error(f"绘制分子时出错: {e}")
        return None

def draw_molecules_grid(smiles_list, labels=None, molsPerRow=3, subImgSize=(200, 200), filename=None):
    """绘制多个分子的网格图"""
    try:
        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        mols = [m for m in mols if m is not None]
        
        if not mols:
            return None
            
        # 如果提供了标签，将其设置为分子属性
        if labels:
            for mol, label in zip(mols, labels):
                mol.SetProp("_Name", str(label))
                
        # 创建网格图像
        img = Draw.MolsToGridImage(
            mols, 
            molsPerRow=molsPerRow, 
            subImgSize=subImgSize,
            legends=[mol.GetProp("_Name") if mol.HasProp("_Name") else "" for mol in mols]
        )
        
        if filename:
            img.save(filename)
            
        return img
    except Exception as e:
        logger.error(f"绘制分子网格时出错: {e}")
        return None

def fragment_molecule(smiles):
    """使用BRICS将分子分解为片段"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
            
        fragments = list(Chem.BRICS.BRICSDecompose(mol))
        return fragments
    except Exception as e:
        logger.error(f"分解分子时出错: {e}")
        return []

def get_scaffold(smiles):
    """获取分子的Murcko骨架"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        scaffold = Chem.Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold)
        
        return scaffold_smiles
    except Exception as e:
        logger.error(f"获取骨架时出错: {e}")
        return None

def smiles_from_name(name, try_online=False):
    """尝试从化合物名称获取SMILES"""
    # 这里可以添加一个本地名称-SMILES映射字典
    name_to_smiles = {
        'calicene': 'C1=CC=CC=1C=C1C=C1',
        'azulene': 'c1ccc2cccc-2cc1',
        'heptazine': 'c1nc2nc3nc(nc3nc2n1)n1c2nc3nc(nc3nc2n1)n1c2nc3nc(nc3nc2n1)n1',
        # 添加更多的反向TADF相关分子
    }
    
    # 先查本地字典
    if name.lower() in name_to_smiles:
        return name_to_smiles[name.lower()]
        
    # 如果允许在线查询且本地找不到
    if try_online:
        try:
            # 这里可以添加使用PubChem或ChemSpider API的代码
            # 例如：
            # from pubchempy import get_compounds
            # compounds = get_compounds(name, 'name')
            # if compounds:
            #     return compounds[0].canonical_smiles
            pass
        except Exception as e:
            logger.error(f"在线查询SMILES时出错: {e}")
            
    return None
