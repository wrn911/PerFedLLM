"""
联邦学习客户端实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List
from torch.utils.data import DataLoader


class FederatedClient:
    """联邦学习客户端"""

    def __init__(self, client_id: str, model: nn.Module, data_loader: DataLoader,
                 args, logger: logging.Logger, device: torch.device):
        self.client_id = client_id
        self.model = model
        self.data_loader = data_loader
        self.args = args
        self.logger = logger
        self.device = device

        # 优化器配置
        self.optimizer = self._create_optimizer()
        self.criterion = nn.MSELoss()

        # 可训练参数名称（用于梯度聚合）
        self.trainable_param_names = self._get_trainable_param_names()

    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器，仅优化可训练参数"""
        trainable_params = []

        # 收集可训练参数
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)

        # 使用AdamW优化器，适合LLM微调
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        return optimizer

    def _get_trainable_param_names(self) -> List[str]:
        """获取可训练参数名称列表"""
        trainable_names = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_names.append(name)
        return trainable_names

    def local_train(self, global_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        客户端局部训练

        Args:
            global_params: 全局模型参数

        Returns:
            gradients: 计算得到的梯度字典
        """
        # 1. 加载全局参数
        self._load_global_params(global_params)

        # 2. 记录初始参数（用于计算梯度）
        initial_params = {}
        for name in self.trainable_param_names:
            param = dict(self.model.named_parameters())[name]
            initial_params[name] = param.data.clone()

        # 3. 局部训练K步
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # 限制训练步数
        max_steps = min(self.args.local_ep, len(self.data_loader))

        for step, batch_data in enumerate(self.data_loader):
            if step >= max_steps:
                break

            self.optimizer.zero_grad()

            # 解包数据
            if len(batch_data) == 4:  # TimeLLM格式
                x_enc, y_true, x_mark, y_mark = batch_data
                x_enc = x_enc.to(self.device)
                y_true = y_true.to(self.device)
                x_mark = x_mark.to(self.device)
                y_mark = y_mark.to(self.device)

                # 前向传播
                y_pred = self.model(x_enc, x_mark, None, y_mark)
            else:  # 传统格式
                x_enc, y_true = batch_data
                x_enc = x_enc.to(self.device)
                y_true = y_true.to(self.device)

                # 创建虚拟时间标记
                batch_size, seq_len = x_enc.shape[:2]
                x_mark = torch.zeros(batch_size, seq_len, 4).to(self.device)
                y_mark = torch.zeros(batch_size, self.args.pred_len, 4).to(self.device)

                # 前向传播
                y_pred = self.model(x_enc.unsqueeze(-1), x_mark, None, y_mark)

            # 计算损失
            loss = self.criterion(y_pred, y_true.unsqueeze(-1) if len(y_true.shape) == 2 else y_true)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 参数更新
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # 4. 计算参数变化作为梯度
        gradients = {}
        current_params = dict(self.model.named_parameters())

        for name in self.trainable_param_names:
            param_diff = initial_params[name] - current_params[name].data
            gradients[name] = param_diff / self.args.lr  # 转换为梯度形式

        avg_loss = total_loss / max(num_batches, 1)
        self.logger.info(f"客户端 {self.client_id} 完成训练，平均损失: {avg_loss:.6f}")

        return gradients

    def _load_global_params(self, global_params: Dict[str, torch.Tensor]):
        """加载全局参数到本地模型"""
        model_state = self.model.state_dict()

        for name, param in global_params.items():
            if name in model_state:
                model_state[name].copy_(param)

    def evaluate(self) -> Dict[str, float]:
        """评估客户端模型性能"""
        self.model.eval()
        total_loss = 0.0
        total_mae = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch_data in self.data_loader:
                # 解包数据
                if len(batch_data) == 4:  # TimeLLM格式
                    x_enc, y_true, x_mark, y_mark = batch_data
                    x_enc = x_enc.to(self.device)
                    y_true = y_true.to(self.device)
                    x_mark = x_mark.to(self.device)
                    y_mark = y_mark.to(self.device)

                    y_pred = self.model(x_enc, x_mark, None, y_mark)
                else:  # 传统格式
                    x_enc, y_true = batch_data
                    x_enc = x_enc.to(self.device)
                    y_true = y_true.to(self.device)

                    batch_size, seq_len = x_enc.shape[:2]
                    x_mark = torch.zeros(batch_size, seq_len, 4).to(self.device)
                    y_mark = torch.zeros(batch_size, self.args.pred_len, 4).to(self.device)

                    y_pred = self.model(x_enc.unsqueeze(-1), x_mark, None, y_mark)

                # 计算指标
                y_true_eval = y_true.unsqueeze(-1) if len(y_true.shape) == 2 else y_true
                mse_loss = nn.MSELoss()(y_pred, y_true_eval)
                mae_loss = nn.L1Loss()(y_pred, y_true_eval)

                total_loss += mse_loss.item() * x_enc.size(0)
                total_mae += mae_loss.item() * x_enc.size(0)
                num_samples += x_enc.size(0)

        return {
            'mse': total_loss / num_samples,
            'mae': total_mae / num_samples,
            'rmse': np.sqrt(total_loss / num_samples)
        }

    def personalized_finetune_and_test(self, epochs: int = 5) -> Dict[str, float]:
        """个性化微调并测试（不保存模型，只返回测试指标）"""
        self.logger.info(f"客户端 {self.client_id} 开始个性化微调...")

        # 检查是否有测试数据
        if not hasattr(self, 'test_loader') or self.test_loader is None:
            # 使用训练数据进行测试（实际应用中应该有单独的测试集）
            self.logger.warning(f"客户端 {self.client_id} 没有测试集，使用训练集进行评估")
            test_loader = self.data_loader
        else:
            test_loader = self.test_loader

        self.model.train()
        best_loss = float('inf')

        # 完整的本地训练（可以使用更多epoch）
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            # 训练整个数据集，不限制步数
            for batch_data in self.data_loader:
                self.optimizer.zero_grad()

                # 解包数据
                if len(batch_data) == 4:  # TimeLLM格式
                    x_enc, y_true, x_mark, y_mark = batch_data
                    x_enc = x_enc.to(self.device)
                    y_true = y_true.to(self.device)
                    x_mark = x_mark.to(self.device)
                    y_mark = y_mark.to(self.device)

                    y_pred = self.model(x_enc, x_mark, None, y_mark)
                else:  # 传统格式
                    x_enc, y_true = batch_data
                    x_enc = x_enc.to(self.device)
                    y_true = y_true.to(self.device)

                    batch_size, seq_len = x_enc.shape[:2]
                    x_mark = torch.zeros(batch_size, seq_len, 4).to(self.device)
                    y_mark = torch.zeros(batch_size, self.args.pred_len, 4).to(self.device)

                    y_pred = self.model(x_enc.unsqueeze(-1), x_mark, None, y_mark)

                # 计算损失
                loss = self.criterion(y_pred, y_true.unsqueeze(-1) if len(y_true.shape) == 2 else y_true)

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss

            if epoch % max(1, epochs // 5) == 0:  # 打印进度
                self.logger.info(f"个性化微调 Epoch {epoch+1}/{epochs}, 训练损失: {avg_loss:.6f}")

        # 切换到评估模式并在测试集上评估
        self.model.eval()
        test_metrics = self._evaluate_on_test_set(test_loader)

        self.logger.info(f"客户端 {self.client_id} 个性化微调完成")
        self.logger.info(f"测试集指标: MSE={test_metrics['mse']:.6f}, MAE={test_metrics['mae']:.6f}, RMSE={test_metrics['rmse']:.6f}")

        return test_metrics

    def _evaluate_on_test_set(self, test_loader) -> Dict[str, float]:
        """在测试集上评估模型性能"""
        total_loss = 0.0
        total_mae = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch_data in test_loader:
                # 解包数据
                if len(batch_data) == 4:  # TimeLLM格式
                    x_enc, y_true, x_mark, y_mark = batch_data
                    x_enc = x_enc.to(self.device)
                    y_true = y_true.to(self.device)
                    x_mark = x_mark.to(self.device)
                    y_mark = y_mark.to(self.device)

                    y_pred = self.model(x_enc, x_mark, None, y_mark)
                else:  # 传统格式
                    x_enc, y_true = batch_data
                    x_enc = x_enc.to(self.device)
                    y_true = y_true.to(self.device)

                    batch_size, seq_len = x_enc.shape[:2]
                    x_mark = torch.zeros(batch_size, seq_len, 4).to(self.device)
                    y_mark = torch.zeros(batch_size, self.args.pred_len, 4).to(self.device)

                    y_pred = self.model(x_enc.unsqueeze(-1), x_mark, None, y_mark)

                # 计算指标
                y_true_eval = y_true.unsqueeze(-1) if len(y_true.shape) == 2 else y_true
                mse_loss = nn.MSELoss()(y_pred, y_true_eval)
                mae_loss = nn.L1Loss()(y_pred, y_true_eval)

                total_loss += mse_loss.item() * x_enc.size(0)
                total_mae += mae_loss.item() * x_enc.size(0)
                num_samples += x_enc.size(0)

        return {
            'mse': total_loss / num_samples,
            'mae': total_mae / num_samples,
            'rmse': np.sqrt(total_loss / num_samples)
        }