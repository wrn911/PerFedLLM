"""
联邦学习客户端实现 - 支持真实时间特征
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List
from torch.utils.data import DataLoader
import gc


class FederatedClient:
    """联邦学习客户端"""

    def __init__(self, client_id: str, shared_model: nn.Module, data_loader: DataLoader,
                 args, logger: logging.Logger, device: torch.device):
        self.client_id = client_id
        self.shared_model = shared_model  # 共享模型引用
        self.data_loader = data_loader
        self.args = args
        self.logger = logger
        self.device = device
        self.test_loader = None
        self.criterion = nn.MSELoss()

        # 获取可训练参数名称
        self.trainable_param_names = self._get_trainable_param_names()

        # 用于保存训练前的参数状态
        self._saved_state = None

    def _get_trainable_param_names(self) -> List[str]:
        """获取可训练参数名称列表"""
        trainable_names = []
        for name, param in self.shared_model.named_parameters():
            if param.requires_grad:
                trainable_names.append(name)
        return trainable_names

    def _save_model_state(self):
        """保存当前模型状态"""
        self._saved_state = {}
        for name, param in self.shared_model.named_parameters():
            if param.requires_grad:
                self._saved_state[name] = param.data.clone()

    def _restore_model_state(self):
        """恢复模型状态"""
        if self._saved_state is not None:
            for name, param in self.shared_model.named_parameters():
                if name in self._saved_state:
                    param.data.copy_(self._saved_state[name])
            # 清理保存的状态
            self._saved_state = None
            torch.cuda.empty_cache()

    def _load_global_params(self, global_params: Dict[str, torch.Tensor]):
        """加载全局参数到共享模型"""
        model_state = self.shared_model.state_dict()
        for name, param in global_params.items():
            if name in model_state:
                model_state[name].copy_(param)

    def local_train(self, global_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        客户端局部训练 - 支持真实时间特征
        """
        # 1. 保存当前模型状态（用于训练后恢复）
        self._save_model_state()

        try:
            # 2. 加载全局参数
            self._load_global_params(global_params)

            # 3. 记录训练前参数（用于计算梯度）
            initial_params = {}
            for name in self.trainable_param_names:
                param = dict(self.shared_model.named_parameters())[name]
                initial_params[name] = param.data.clone()

            # 4. 创建优化器
            trainable_params = [p for p in self.shared_model.parameters() if p.requires_grad]
            optimizer = optim.AdamW(
                trainable_params,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )

            # 在训练循环开始前设置上下文信息
            if hasattr(self.shared_model, 'set_context_info') and hasattr(self.data_loader.dataset, 'timestamps'):
                coordinates = getattr(self, 'coordinates', None)
                timestamps = self.data_loader.dataset.timestamps

                if timestamps and len(timestamps) > 0:
                    # 获取第一个和最后一个时间戳来确定整体时间范围
                    first_timestamp = timestamps[0]
                    last_timestamp = timestamps[-1]

                    start_time = first_timestamp['start_time']
                    end_time = last_timestamp['pred_end']

                    self.shared_model.set_context_info(
                        coordinates=coordinates,
                        start_timestamp=start_time,
                        end_timestamp=end_time
                    )

            # 5. 局部训练
            self.shared_model.train()
            total_loss = 0.0
            num_batches = 0
            max_steps = min(self.args.local_ep, len(self.data_loader))

            for step, batch_data in enumerate(self.data_loader):
                if step >= max_steps:
                    break

                # 每个batch设置当前时间范围
                if hasattr(self.shared_model, 'set_context_info') and hasattr(self.data_loader.dataset, 'timestamps'):
                    batch_start_idx = step * self.data_loader.batch_size
                    batch_end_idx = min(batch_start_idx + self.data_loader.batch_size,
                                        len(self.data_loader.dataset.timestamps))

                    if batch_start_idx < len(self.data_loader.dataset.timestamps):
                        batch_timestamps = self.data_loader.dataset.timestamps[batch_start_idx:batch_end_idx]
                        if batch_timestamps:
                            current_start = batch_timestamps[0]['start_time']
                            current_end = batch_timestamps[-1]['pred_end'] if len(batch_timestamps) > 0 else \
                            batch_timestamps[0]['pred_end']

                            self.shared_model.set_context_info(
                                coordinates=getattr(self, 'coordinates', None),
                                start_timestamp=current_start,
                                end_timestamp=current_end
                            )

                optimizer.zero_grad()

                # 解包数据（现在包含真实时间特征）
                x_enc, y_true, x_mark, y_mark = batch_data
                x_enc = x_enc.to(self.device)
                y_true = y_true.to(self.device)
                x_mark = x_mark.to(self.device)
                y_mark = y_mark.to(self.device)

                # 为TimeLLM准备decoder输入（如果需要label_len）
                if hasattr(self.args, 'label_len'):
                    batch_size = x_enc.size(0)
                    # 创建decoder输入：[batch, label_len + pred_len, features]
                    x_dec = torch.zeros(batch_size, self.args.label_len + self.args.pred_len, x_enc.size(-1)).to(self.device)
                    # 填入label部分（从encoder的最后label_len步）
                    x_dec[:, :self.args.label_len, :] = x_enc[:, -self.args.label_len:, :]

                    # 准备decoder时间标记
                    x_mark_dec = torch.cat([x_mark[:, -self.args.label_len:, :], y_mark], dim=1)

                    # 前向传播（标准TimeLLM格式）
                    y_pred = self.shared_model(x_enc, x_mark, x_dec, x_mark_dec)
                else:
                    # 简化版本（如果没有label_len配置）
                    y_pred = self.shared_model(x_enc, x_mark, None, y_mark)

                # 计算损失和反向传播
                loss = self.criterion(y_pred, y_true)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.shared_model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            # 6. 计算梯度（参数变化）
            gradients = {}
            current_params = dict(self.shared_model.named_parameters())
            for name in self.trainable_param_names:
                if name in current_params and name in initial_params:
                    param_diff = initial_params[name] - current_params[name].data
                    gradients[name] = param_diff / self.args.lr

            avg_loss = total_loss / max(num_batches, 1)
            self.logger.info(f"客户端 {self.client_id} 完成训练，平均损失: {avg_loss:.6f}")

            return gradients

        finally:
            # 7. 恢复模型状态（关键：确保不污染全局模型）
            self._restore_model_state()
            torch.cuda.empty_cache()

    def evaluate(self) -> Dict[str, float]:
        """评估客户端模型性能"""
        self.shared_model.eval()

        # 设置评估时的上下文信息
        if hasattr(self.shared_model, 'set_context_info') and hasattr(self.data_loader.dataset, 'timestamps'):
            coordinates = getattr(self, 'coordinates', None)
            timestamps = self.data_loader.dataset.timestamps

            if timestamps and len(timestamps) > 0:
                start_time = timestamps[0]['start_time']
                end_time = timestamps[-1]['pred_end']

                self.shared_model.set_context_info(
                    coordinates=coordinates,
                    start_timestamp=start_time,
                    end_timestamp=end_time
                )

        total_loss = 0.0
        total_mae = 0.0
        num_samples = 0

        with torch.no_grad():
            for step, batch_data in enumerate(self.data_loader):
                # 为每个batch设置具体的时间范围
                if hasattr(self.shared_model, 'set_context_info') and hasattr(self.data_loader.dataset, 'timestamps'):
                    batch_start_idx = step * self.data_loader.batch_size
                    if batch_start_idx < len(self.data_loader.dataset.timestamps):
                        batch_timestamps = self.data_loader.dataset.timestamps[
                                           batch_start_idx:batch_start_idx + batch_data[0].size(0)]
                        if batch_timestamps:
                            current_start = batch_timestamps[0]['start_time']
                            current_end = batch_timestamps[-1]['pred_end'] if len(batch_timestamps) > 0 else \
                            batch_timestamps[0]['pred_end']

                            self.shared_model.set_context_info(
                                coordinates=getattr(self, 'coordinates', None),
                                start_timestamp=current_start,
                                end_timestamp=current_end
                            )

                x_enc, y_true, x_mark, y_mark = batch_data
                x_enc = x_enc.to(self.device)
                y_true = y_true.to(self.device)
                x_mark = x_mark.to(self.device)
                y_mark = y_mark.to(self.device)

                # 使用与训练一致的前向传播逻辑
                if hasattr(self.args, 'label_len'):
                    batch_size = x_enc.size(0)
                    x_dec = torch.zeros(batch_size, self.args.label_len + self.args.pred_len, x_enc.size(-1)).to(self.device)
                    x_dec[:, :self.args.label_len, :] = x_enc[:, -self.args.label_len:, :]
                    x_mark_dec = torch.cat([x_mark[:, -self.args.label_len:, :], y_mark], dim=1)
                    y_pred = self.shared_model(x_enc, x_mark, x_dec, x_mark_dec)
                else:
                    y_pred = self.shared_model(x_enc, x_mark, None, y_mark)

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

    def personalized_finetune_and_test(self, epochs: int = 5) -> Dict[str, float]:
        """个性化微调并测试"""
        self.logger.info(f"客户端 {self.client_id} 开始个性化微调...")

        # 使用测试集或训练集
        test_loader = self.test_loader if hasattr(self, 'test_loader') and self.test_loader else self.data_loader

        # 创建优化器
        trainable_params = [p for p in self.shared_model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(
            trainable_params,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        try:
            self.shared_model.train()

            # 完整的本地训练
            for epoch in range(epochs):
                total_loss = 0.0
                num_batches = 0

                for batch_data in self.data_loader:
                    optimizer.zero_grad()

                    # 使用与训练时相同的数据处理逻辑
                    x_enc, y_true, x_mark, y_mark = batch_data
                    x_enc = x_enc.to(self.device)
                    y_true = y_true.to(self.device)
                    x_mark = x_mark.to(self.device)
                    y_mark = y_mark.to(self.device)

                    # 准备decoder输入（与训练时保持一致）
                    if hasattr(self.args, 'label_len'):
                        batch_size = x_enc.size(0)
                        x_dec = torch.zeros(batch_size, self.args.label_len + self.args.pred_len, x_enc.size(-1)).to(self.device)
                        x_dec[:, :self.args.label_len, :] = x_enc[:, -self.args.label_len:, :]
                        x_mark_dec = torch.cat([x_mark[:, -self.args.label_len:, :], y_mark], dim=1)
                        y_pred = self.shared_model(x_enc, x_mark, x_dec, x_mark_dec)
                    else:
                        y_pred = self.shared_model(x_enc, x_mark, None, y_mark)

                    loss = self.criterion(y_pred, y_true)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.shared_model.parameters(), max_norm=1.0)
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                if epoch % max(1, epochs // 5) == 0:
                    avg_loss = total_loss / max(num_batches, 1)
                    self.logger.info(f"个性化微调 Epoch {epoch+1}/{epochs}, 训练损失: {avg_loss:.6f}")

            # 测试评估
            self.shared_model.eval()
            test_metrics = self._evaluate_on_test_set(test_loader)

            self.logger.info(f"客户端 {self.client_id} 个性化微调完成")
            self.logger.info(f"测试指标: MSE={test_metrics['mse']:.6f}, MAE={test_metrics['mae']:.6f}, RMSE={test_metrics['rmse']:.6f}")

            return test_metrics

        finally:
            torch.cuda.empty_cache()

    def _evaluate_on_test_set(self, test_loader) -> Dict[str, float]:
        """在测试集上评估模型性能"""
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

                # 使用与训练一致的前向传播逻辑
                if hasattr(self.args, 'label_len'):
                    batch_size = x_enc.size(0)
                    x_dec = torch.zeros(batch_size, self.args.label_len + self.args.pred_len, x_enc.size(-1)).to(self.device)
                    x_dec[:, :self.args.label_len, :] = x_enc[:, -self.args.label_len:, :]
                    x_mark_dec = torch.cat([x_mark[:, -self.args.label_len:, :], y_mark], dim=1)
                    y_pred = self.shared_model(x_enc, x_mark, x_dec, x_mark_dec)
                else:
                    y_pred = self.shared_model(x_enc, x_mark, None, y_mark)

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