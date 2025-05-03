# workflows/router.py
import logging
from workflows.task_chain import TaskChain

class Router:
    """
    控制流程入口类，用于协调不同任务路径和调度任务链
    """
    
    def __init__(self):
        """初始化路由器"""
        self.logger = logging.getLogger('Router')
        self.task_chain = TaskChain()
        self.pipeline_results = {}
        
    def route_task(self, task_type, input_data=None):
        """
        根据任务类型路由到相应的处理流程
        
        参数:
            task_type: 任务类型（数据提取、特征工程、探索分析、建模、洞察生成）
            input_data: 任务输入数据
            
        返回:
            执行结果
        """
        self.logger.info(f"路由任务: {task_type}")
        
        if task_type == "数据提取":
            return self.route_data_extraction(input_data)
        elif task_type == "特征工程":
            return self.route_feature_engineering(input_data)
        elif task_type == "探索分析":
            return self.route_exploration_analysis(input_data)
        elif task_type == "预测建模":
            return self.route_predictive_modeling(input_data)
        elif task_type == "洞察生成":
            return self.route_insight_generation(input_data)
        elif task_type == "完整流程":
            return self.route_complete_pipeline(input_data)
        else:
            self.logger.error(f"未知任务类型: {task_type}")
            return {"status": "错误", "message": f"未知任务类型: {task_type}"}
            
    def route_data_extraction(self, input_data):
        """路由数据提取任务"""
        try:
            base_dir = input_data.get('base_dir')
            if not base_dir:
                return {"status": "错误", "message": "未提供数据目录"}
                
            result = self.task_chain.execute_data_extraction(base_dir)
            
            if result:
                self.pipeline_results['data_extraction'] = result
                return {"status": "成功", "data": result}
            else:
                return {"status": "错误", "message": "数据提取失败"}
        except Exception as e:
            self.logger.error(f"数据提取路由错误: {e}")
            return {"status": "错误", "message": str(e)}
            
    def route_feature_engineering(self, input_data):
        """路由特征工程任务"""
        try:
            data_file = input_data.get('data_file')
            
            # 如果没有提供文件但有之前的结果，使用之前的结果
            if not data_file and 'data_extraction' in self.pipeline_results:
                data_file = self.pipeline_results['data_extraction']
                
            if not data_file:
                return {"status": "错误", "message": "未提供数据文件"}
                
            result = self.task_chain.execute_feature_engineering(data_file)
            
            if result:
                self.pipeline_results['feature_engineering'] = result
                return {"status": "成功", "data": result}
            else:
                return {"status": "错误", "message": "特征工程失败"}
        except Exception as e:
            self.logger.error(f"特征工程路由错误: {e}")
            return {"status": "错误", "message": str(e)}
            
    def route_exploration_analysis(self, input_data):
        """路由探索分析任务"""
        try:
            gap_data = input_data.get('gap_data')
            
            # 如果没有提供间隙数据但有之前的特征工程结果，使用之前的结果
            if not gap_data and 'feature_engineering' in self.pipeline_results:
                gap_data = self.pipeline_results['feature_engineering'].get('gap_data')
                
            if not gap_data:
                return {"status": "错误", "message": "未提供间隙数据"}
                
            result = self.task_chain.execute_exploration_analysis(gap_data)
            
            if result:
                self.pipeline_results['exploration_analysis'] = result
                return {"status": "成功", "data": result}
            else:
                return {"status": "错误", "message": "探索分析失败"}
        except Exception as e:
            self.logger.error(f"探索分析路由错误: {e}")
            return {"status": "错误", "message": str(e)}
            
    def route_predictive_modeling(self, input_data):
        """路由预测建模任务"""
        try:
            feature_file = input_data.get('feature_file')
            
            # 如果没有提供特征文件但有之前的特征工程结果，使用之前的结果
            if not feature_file and 'feature_engineering' in self.pipeline_results:
                feature_file = self.pipeline_results['feature_engineering'].get('feature_file')
                
            if not feature_file:
                return {"status": "错误", "message": "未提供特征文件"}
                
            result = self.task_chain.execute_predictive_modeling(feature_file)
            
            if result:
                self.pipeline_results['predictive_modeling'] = result
                return {"status": "成功", "data": result}
            else:
                return {"status": "错误", "message": "预测建模失败"}
        except Exception as e:
            self.logger.error(f"预测建模路由错误: {e}")
            return {"status": "错误", "message": str(e)}
            
    def route_insight_generation(self, input_data):
        """路由洞察生成任务"""
        try:
            modeling_results = input_data.get('modeling_results')
            exploration_results = input_data.get('exploration_results')
            
            # 如果没有提供建模结果但有之前的建模结果，使用之前的结果
            if not modeling_results and 'predictive_modeling' in self.pipeline_results:
                modeling_results = self.pipeline_results['predictive_modeling']
                
            # 如果没有提供探索结果但有之前的探索结果，使用之前的结果
            if not exploration_results and 'exploration_analysis' in self.pipeline_results:
                exploration_results = self.pipeline_results['exploration_analysis']
                
            if not modeling_results and not exploration_results:
                return {"status": "错误", "message": "未提供建模或探索结果"}
                
            result = self.task_chain.execute_insight_generation({
                'modeling_results': modeling_results,
                'exploration_results': exploration_results
            })
            
            if result:
                self.pipeline_results['insight_generation'] = result
                return {"status": "成功", "data": result}
            else:
                return {"status": "错误", "message": "洞察生成失败"}
        except Exception as e:
            self.logger.error(f"洞察生成路由错误: {e}")
            return {"status": "错误", "message": str(e)}
            
    def route_complete_pipeline(self, input_data):
        """路由完整流程任务"""
        try:
            base_dir = input_data.get('base_dir')
            if not base_dir:
                return {"status": "错误", "message": "未提供数据目录"}
                
            result = self.task_chain.run_complete_pipeline(base_dir)
            
            if result:
                self.pipeline_results = result
                return {"status": "成功", "data": result}
            else:
                return {"status": "错误", "message": "完整流程执行失败"}
        except Exception as e:
            self.logger.error(f"完整流程路由错误: {e}")
            return {"status": "错误", "message": str(e)}
            
    def get_results(self):
        """获取所有阶段的管道结果"""
        return self.pipeline_results
