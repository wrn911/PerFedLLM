"""
经典模型分布式训练器 - 专门用于训练传统时间序列预测模型
支持ARIMA, Lasso, SVR, Prophet, LSTM, TFT等模型的分布式训练
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import logging
import os
import json
import time
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class ClassicalModelTrainer:
    """经典模型分布式训练器"""

    def __init__(self, args, logger: logging.Logger):
        self.args = args
        self.logger = logger
        self.device = args.device if hasattr(args, 'device') else 'cpu'

        # 设置随机种子
        self._set_seed()

        # 初始化组件
        self.federated_data = None
        self.model = None
        self.is_neural_model = False

    def _set_seed(self):
        """设置随机种子"""
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def setup_data(self, federated_data: Dict, data_loader_factory):
        """设置数据"""
        self.federated_data = federated_data
        self.data_loader_factory = data_loader_factory
        self.logger.info(f"数据设置完成，客户端数量: {len(federated_data['clients'])}")

    def setup_model(self, model_class, model_args):
        """设置模型"""
        self.model = model_class(model_args)

        # 判断是否为神经网络模型
        self.is_neural_model = isinstance(self.model, nn.Module)

        if self.is_neural_model:
            self.model = self.model.to(self.device)

            # 打印模型信息
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            self.logger.info(f"神经网络模型设置完成:")
            self.logger.info(f"  总参数: {total_params:,}")
            self.logger.info(f"  可训练参数: {trainable_params:,}")
            if total_params > 0:
                self.logger.info(f"  可训练比例: {100 * trainable_params / total_params:.2f}%")
            else:
                self.logger.info(f"  可训练比例: N/A (无参数模型)")
        else:
            self.logger.info(f"传统机器学习模型设置完成: {model_class.__name__}")

    def train_neural_model_client(self, client_id: str, client_data: Dict) -> Dict[str, float]:
        """训练单个客户端的神经网络模型"""
        self.logger.info(f"开始训练客户端 {client_id} (神经网络模式)")

        # 创建数据加载器
        data_loaders = self.data_loader_factory.create_data_loaders(
            client_data['sequences'],
            batch_size=self.args.local_bs
        )

        train_loader = data_loaders['train']
        test_loader = data_loaders.get('test', train_loader)

        # 设置优化器和损失函数
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=getattr(self.args, 'weight_decay', 1e-5)
        )
        criterion = nn.MSELoss()

        # 训练循环
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.args.epochs):
            epoch_loss = 0.0
            epoch_batches = 0

            for batch_idx, batch_data in enumerate(train_loader):
                optimizer.zero_grad()

                # 解包数据
                x_enc, y_true, x_mark, y_mark = batch_data
                x_enc = x_enc.to(self.device)
                y_true = y_true.to(self.device)
                x_mark = x_mark.to(self.device)
                y_mark = y_mark.to(self.device)

                # 前向传播
                if hasattr(self.args, 'label_len') and self.args.label_len > 0:
                    # 为需要decoder输入的模型准备数据
                    batch_size = x_enc.size(0)
                    x_dec = torch.zeros(batch_size, self.args.label_len + self.args.pred_len,
                                      x_enc.size(-1)).to(self.device)
                    x_dec[:, :self.args.label_len, :] = x_enc[:, -self.args.label_len:, :]
                    x_mark_dec = torch.cat([x_mark[:, -self.args.label_len:, :], y_mark], dim=1)

                    y_pred = self.model(x_enc, x_mark, x_dec, x_mark_dec)
                else:
                    # 简化的前向传播
                    y_pred = self.model(x_enc, x_mark, None, y_mark)

                # 计算损失
                loss = criterion(y_pred, y_true)

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_batches += 1

            if epoch % 10 == 0:
                avg_loss = epoch_loss / max(epoch_batches, 1)
                self.logger.info(f"客户端 {client_id} Epoch {epoch}/{self.args.epochs}, 损失: {avg_loss:.6f}")

            total_loss += epoch_loss
            num_batches += epoch_batches

        # 测试评估
        test_metrics = self._evaluate_neural_model(test_loader)

        avg_train_loss = total_loss / max(num_batches, 1)
        self.logger.info(f"客户端 {client_id} 训练完成")
        self.logger.info(f"  训练损失: {avg_train_loss:.6f}")
        self.logger.info(f"  测试MSE: {test_metrics['mse']:.6f}")
        self.logger.info(f"  测试MAE: {test_metrics['mae']:.6f}")

        return test_metrics

    def train_traditional_model_client(self, client_id: str, client_data: Dict) -> Dict[str, float]:
        """训练单个客户端的传统机器学习模型"""
        self.logger.info(f"开始训练客户端 {client_id} (传统模型模式)")

        # 获取训练和测试数据
        train_sequences = client_data['sequences']['train']
        test_sequences = client_data['sequences']['test']

        # 转换为numpy数组
        train_history = train_sequences['history']  # [N, seq_len, 1]
        train_target = train_sequences['target']    # [N, pred_len, 1]
        test_history = test_sequences['history']
        test_target = test_sequences['target']

        # 选择一个代表性的序列进行训练（或合并所有序列）
        if len(train_history) > 0:
            # 使用最后一个训练样本作为训练数据
            # 修复squeeze问题：先检查维度
            if train_history.ndim == 3 and train_history.shape[-1] == 1:
                train_data = train_history[-1].squeeze(-1)  # [seq_len]
            else:
                train_data = train_history[-1]  # 如果不是3维或最后一维不是1，直接使用

            # 对于传统模型，我们使用整个历史序列进行训练
            combined_data = []
            for i in range(len(train_history)):
                if train_history[i].ndim == 3 and train_history[i].shape[-1] == 1:
                    hist = train_history[i].squeeze(-1)
                else:
                    hist = train_history[i]

                if train_target[i].ndim == 3 and train_target[i].shape[-1] == 1:
                    target = train_target[i].squeeze(-1)
                else:
                    target = train_target[i]

                # 确保hist和target都是1维的
                if hist.ndim > 1:
                    hist = hist.flatten()
                if target.ndim > 1:
                    target = target.flatten()

                combined = np.concatenate([hist, target])
                combined_data.append(combined)

            # 使用最近的数据作为训练序列
            if len(combined_data) > 0:
                train_data = combined_data[-1]  # 使用最新的完整序列
            else:
                # 确保train_data是1维的
                if train_data.ndim > 1:
                    train_data = train_data.flatten()

            # 重塑为 [seq_len, features]
            if train_data.ndim == 1:
                train_data = train_data.reshape(-1, 1)

            try:
                # 使用模型的fit_and_predict方法
                predictions = self.model.fit_and_predict(train_data, self.args.pred_len)

                # 计算测试指标
                test_predictions = []
                test_targets = []

                for i in range(len(test_history)):
                    # 处理测试历史数据
                    if test_history[i].ndim == 3 and test_history[i].shape[-1] == 1:
                        test_seq = test_history[i].squeeze(-1)
                    else:
                        test_seq = test_history[i]

                    if test_seq.ndim > 1:
                        test_seq = test_seq.flatten()

                    test_seq = test_seq.reshape(-1, 1)
                    pred = self.model.fit_and_predict(test_seq, self.args.pred_len)

                    # 处理测试目标数据
                    if test_target[i].ndim == 3 and test_target[i].shape[-1] == 1:
                        target = test_target[i].squeeze(-1)
                    else:
                        target = test_target[i]

                    if target.ndim > 1:
                        target = target.flatten()

                    test_predictions.append(pred.flatten())
                    test_targets.append(target)

                if len(test_predictions) > 0:
                    test_pred_array = np.array(test_predictions)
                    test_true_array = np.array(test_targets)

                    mse = mean_squared_error(test_true_array, test_pred_array)
                    mae = mean_absolute_error(test_true_array, test_pred_array)
                    rmse = np.sqrt(mse)

                    metrics = {'mse': float(mse), 'mae': float(mae), 'rmse': float(rmse)}
                else:
                    # 如果没有测试数据，使用训练数据评估
                    if len(train_target) > 0:
                        if train_target[-1].ndim == 3 and train_target[-1].shape[-1] == 1:
                            train_true = train_target[-1].squeeze(-1)
                        else:
                            train_true = train_target[-1]

                        if train_true.ndim > 1:
                            train_true = train_true.flatten()

                        train_pred = predictions.flatten()[:len(train_true)]

                        mse = mean_squared_error(train_true, train_pred)
                        mae = mean_absolute_error(train_true, train_pred)
                        rmse = np.sqrt(mse)

                        metrics = {'mse': float(mse), 'mae': float(mae), 'rmse': float(rmse)}
                    else:
                        metrics = {'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf')}

            except Exception as e:
                self.logger.error(f"客户端 {client_id} 传统模型训练失败: {e}")
                metrics = {'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf')}

        else:
            self.logger.warning(f"客户端 {client_id} 没有训练数据")
            metrics = {'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf')}

        self.logger.info(f"客户端 {client_id} 训练完成")
        self.logger.info(f"  测试MSE: {metrics['mse']:.6f}")
        self.logger.info(f"  测试MAE: {metrics['mae']:.6f}")
        self.logger.info(f"  测试RMSE: {metrics['rmse']:.6f}")

        return metrics

    def _evaluate_neural_model(self, test_loader) -> Dict[str, float]:
        """评估神经网络模型"""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch_data in test_loader:
                x_enc, y_true, x_mark, y_mark = batch_data
                x_enc = x_enc.to(self.device)
                y_true = y_true.to(self.device)
                x_mark = x_mark.to(self.device)
                y_mark = y_mark.to(self.device)

                # 前向传播
                if hasattr(self.args, 'label_len') and self.args.label_len > 0:
                    batch_size = x_enc.size(0)
                    x_dec = torch.zeros(batch_size, self.args.label_len + self.args.pred_len,
                                      x_enc.size(-1)).to(self.device)
                    x_dec[:, :self.args.label_len, :] = x_enc[:, -self.args.label_len:, :]
                    x_mark_dec = torch.cat([x_mark[:, -self.args.label_len:, :], y_mark], dim=1)

                    y_pred = self.model(x_enc, x_mark, x_dec, x_mark_dec)
                else:
                    y_pred = self.model(x_enc, x_mark, None, y_mark)

                # 计算指标
                mse_loss = nn.MSELoss()(y_pred, y_true)
                mae_loss = nn.L1Loss()(y_pred, y_true)

                total_loss += mse_loss.item() * x_enc.size(0)
                total_mae += mae_loss.item() * x_enc.size(0)
                num_samples += x_enc.size(0)

        return {
            'mse': total_loss / num_samples,
            'mae': total_mae / num_samples,
            'rmse': np.sqrt(total_loss / num_samples)
        }

    def train_distributed(self) -> Dict:
        """执行分布式训练"""
        if not all([self.federated_data, self.model]):
            raise ValueError("请先完成数据和模型的设置")

        self.logger.info("开始分布式训练")

        client_results = {}
        all_metrics = []

        # 为每个客户端独立训练模型
        for client_id, client_data in self.federated_data['clients'].items():
            try:
                if self.is_neural_model:
                    # 重置模型参数（每个客户端独立训练）
                    self._reset_model_parameters()
                    client_metrics = self.train_neural_model_client(client_id, client_data)
                else:
                    client_metrics = self.train_traditional_model_client(client_id, client_data)

                client_results[client_id] = client_metrics

                # 只添加有效的指标
                if client_metrics['mse'] != float('inf'):
                    all_metrics.append(client_metrics)

            except Exception as e:
                self.logger.error(f"客户端 {client_id} 训练失败: {e}")
                client_results[client_id] = {
                    'error': str(e),
                    'mse': float('inf'),
                    'mae': float('inf'),
                    'rmse': float('inf')
                }

            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # 计算平均指标
        if all_metrics:
            avg_metrics = {}
            for metric in ['mse', 'mae', 'rmse']:
                values = [m[metric] for m in all_metrics if m[metric] != float('inf')]
                if values:
                    avg_metrics[f'avg_{metric}'] = np.mean(values)
                    avg_metrics[f'std_{metric}'] = np.std(values)
                    avg_metrics[f'min_{metric}'] = np.min(values)
                    avg_metrics[f'max_{metric}'] = np.max(values)
                else:
                    avg_metrics[f'avg_{metric}'] = float('inf')
                    avg_metrics[f'std_{metric}'] = 0.0
                    avg_metrics[f'min_{metric}'] = float('inf')
                    avg_metrics[f'max_{metric}'] = float('inf')
        else:
            avg_metrics = {
                'avg_mse': float('inf'), 'std_mse': 0.0, 'min_mse': float('inf'), 'max_mse': float('inf'),
                'avg_mae': float('inf'), 'std_mae': 0.0, 'min_mae': float('inf'), 'max_mae': float('inf'),
                'avg_rmse': float('inf'), 'std_rmse': 0.0, 'min_rmse': float('inf'), 'max_rmse': float('inf')
            }

        # 输出统计信息
        successful_clients = len(all_metrics)
        total_clients = len(client_results)

        self.logger.info(f"\n=== 分布式训练完成 ===")
        self.logger.info(f"成功训练客户端: {successful_clients}/{total_clients}")

        if successful_clients > 0:
            self.logger.info(f"平均指标:")
            self.logger.info(f"  MSE: {avg_metrics['avg_mse']:.6f} ± {avg_metrics['std_mse']:.6f}")
            self.logger.info(f"  MAE: {avg_metrics['avg_mae']:.6f} ± {avg_metrics['std_mae']:.6f}")
            self.logger.info(f"  RMSE: {avg_metrics['avg_rmse']:.6f} ± {avg_metrics['std_rmse']:.6f}")

        results = {
            'client_results': client_results,
            'average_metrics': avg_metrics,
            'successful_clients': successful_clients,
            'total_clients': total_clients,
            'model_type': 'neural' if self.is_neural_model else 'traditional'
        }

        return results

    def _reset_model_parameters(self):
        """重置神经网络模型参数"""
        if self.is_neural_model:
            def init_weights(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if 'weight_ih' in name:
                            torch.nn.init.xavier_uniform_(param.data)
                        elif 'weight_hh' in name:
                            torch.nn.init.orthogonal_(param.data)
                        elif 'bias' in name:
                            param.data.fill_(0)
                            # 设置forget gate bias为1
                            n = param.size(0)
                            param.data[(n//4):(n//2)].fill_(1)

            self.model.apply(init_weights)

    def save_results(self, results: Dict, save_dir: str):
        """保存训练结果"""
        os.makedirs(save_dir, exist_ok=True)

        # 保存训练结果
        def convert_to_serializable(obj):
            """递归转换对象为可序列化格式"""
            if isinstance(obj, dict):
                # 转换字典，确保键是字符串
                return {str(k): convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif obj == float('inf'):
                return "inf"
            elif obj == float('-inf'):
                return "-inf"
            elif isinstance(obj, float) and np.isnan(obj):
                return "nan"
            elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32, np.float16)):
                return float(obj)
            else:
                return obj

        results_serializable = convert_to_serializable(results)

        with open(os.path.join(save_dir, 'training_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)

        # 保存配置
        config_dict = {
            'model_type': getattr(self.args, 'model_type', 'unknown'),
            'dataset': getattr(self.args, 'file_path', 'unknown'),
            'data_type': getattr(self.args, 'data_type', 'unknown'),
            'seq_len': self.args.seq_len,
            'pred_len': self.args.pred_len,
            'num_clients': getattr(self.args, 'num_clients', 0),
            'is_neural_model': self.is_neural_model,
            'device': self.device,
            'seed': self.args.seed
        }

        # 添加模型特定参数
        model_specific_params = {}
        for attr_name in dir(self.args):
            if not attr_name.startswith('_'):
                attr_value = getattr(self.args, attr_name)
                if isinstance(attr_value, (int, float, str, bool)):
                    model_specific_params[attr_name] = attr_value

        config_dict.update(model_specific_params)

        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        # 保存简化的汇总结果
        if results['successful_clients'] > 0:
            summary = {
                'experiment_name': getattr(self.args, 'experiment_name', 'unknown'),
                'model_type': getattr(self.args, 'model_type', 'unknown'),
                'dataset': getattr(self.args, 'file_path', 'unknown'),
                'data_type': getattr(self.args, 'data_type', 'unknown'),
                'successful_clients': results['successful_clients'],
                'total_clients': results['total_clients'],
                'success_rate': results['successful_clients'] / results['total_clients'],
                'final_metrics': {
                    'mse': results['average_metrics']['avg_mse'],
                    'mae': results['average_metrics']['avg_mae'],
                    'rmse': results['average_metrics']['avg_rmse']
                },
                'training_mode': 'distributed_classical'
            }

            with open(os.path.join(save_dir, 'summary.json'), 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"训练结果已保存到: {save_dir}")


def get_classical_model_class(model_type: str):
    """根据模型类型获取对应的模型类"""
    model_type = model_type.lower()

    if model_type == 'arima':
        from models.ARIMA import Model
        return Model
    elif model_type == 'lasso':
        from models.Lasso import Model
        return Model
    elif model_type == 'svr':
        from models.SVR import Model
        return Model
    elif model_type == 'prophet':
        from models.Prophet import Model
        return Model
    elif model_type == 'lstm':
        from models.LSTM import Model
        return Model
    elif model_type == 'tft':
        from models.TFT import Model
        return Model
    else:
        raise ValueError(f"不支持的经典模型类型: {model_type}")


def setup_classical_model_args(args, model_type: str):
    """为经典模型设置特定参数"""
    model_type = model_type.lower()

    # 通用参数
    args.task_name = 'long_term_forecast'
    args.enc_in = 1
    args.c_out = 1
    args.embed = 'timeF'
    args.freq = 'h'
    args.activation = 'gelu'

    # 模型特定参数
    if model_type in ['arima', 'lasso', 'svr', 'prophet']:
        # 传统机器学习模型
        args.epochs = 1  # 传统模型不需要多轮训练
    elif model_type == 'lstm':
        # LSTM特定参数
        args.d_model = getattr(args, 'lstm_hidden_size', 128)
        args.dropout = getattr(args, 'dropout', 0.1)
    elif model_type == 'tft':
        # TFT特定参数
        args.d_model = getattr(args, 'd_model', 128)
        args.n_heads = getattr(args, 'n_heads', 8)
        args.dropout = getattr(args, 'dropout', 0.1)

    return args