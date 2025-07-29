"""
联邦学习算法基类和具体实现
支持FedAvg, FedProx, Per-FedAvg等多种算法
"""

import torch
import torch.nn as nn
import copy
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class FederatedAlgorithmBase(ABC):
    """联邦学习算法基类"""

    def __init__(self, global_model: nn.Module, args, logger, device):
        self.global_model = global_model
        self.args = args
        self.logger = logger
        self.device = device
        self.trainable_param_names = self._get_trainable_param_names()

    def _get_trainable_param_names(self) -> List[str]:
        """获取可训练参数名称"""
        return [name for name, param in self.global_model.named_parameters() if param.requires_grad]

    def get_global_params(self) -> Dict[str, torch.Tensor]:
        """获取全局模型参数"""
        global_params = {}
        model_state = self.global_model.state_dict()
        for param_name in self.trainable_param_names:
            if param_name in model_state:
                global_params[param_name] = model_state[param_name].clone()
        return global_params

    @abstractmethod
    def client_update(self, client, global_params: Dict[str, torch.Tensor]):
        """客户端更新 - 由子类实现具体算法"""
        pass

    @abstractmethod
    def server_aggregate(self, client_updates: List):
        """服务器聚合 - 由子类实现具体算法"""
        pass


class FedAvgAlgorithm(FederatedAlgorithmBase):
    """FedAvg算法实现"""

    def client_update(self, client, global_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """FedAvg客户端更新：返回模型权重"""
        # 保存训练前状态
        client._save_model_state()

        try:
            # 加载全局参数
            client._load_global_params(global_params)

            # 本地训练
            trainable_params = [p for p in client.shared_model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )

            client.shared_model.train()
            for step, batch_data in enumerate(client.data_loader):
                if step >= self.args.local_ep:
                    break

                optimizer.zero_grad()
                x_enc, y_true, x_mark, y_mark = batch_data
                x_enc = x_enc.to(self.device)
                y_true = y_true.to(self.device)
                x_mark = x_mark.to(self.device)
                y_mark = y_mark.to(self.device)

                # 前向传播
                if hasattr(self.args, 'label_len'):
                    batch_size = x_enc.size(0)
                    x_dec = torch.zeros(batch_size, self.args.label_len + self.args.pred_len, x_enc.size(-1)).to(
                        self.device)
                    x_dec[:, :self.args.label_len, :] = x_enc[:, -self.args.label_len:, :]
                    x_mark_dec = torch.cat([x_mark[:, -self.args.label_len:, :], y_mark], dim=1)
                    y_pred = client.shared_model(x_enc, x_mark, x_dec, x_mark_dec)
                else:
                    y_pred = client.shared_model(x_enc, x_mark, None, y_mark)

                loss = client.criterion(y_pred, y_true)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(client.shared_model.parameters(), max_norm=1.0)
                optimizer.step()

            # 返回更新后的模型权重
            updated_params = {}
            current_state = client.shared_model.state_dict()
            for name in self.trainable_param_names:
                if name in current_state:
                    updated_params[name] = current_state[name].clone()

            return updated_params

        finally:
            # 恢复模型状态
            client._restore_model_state()

    def server_aggregate(self, client_updates: List[Dict[str, torch.Tensor]]) -> None:
        """FedAvg服务器聚合：直接平均模型权重"""
        # 等权重平均
        num_clients = len(client_updates)

        # 平均模型参数
        model_state = self.global_model.state_dict()
        for param_name in self.trainable_param_names:
            if param_name in model_state:
                param_sum = None
                valid_updates = 0

                for client_params in client_updates:
                    if param_name in client_params:
                        if param_sum is None:
                            param_sum = client_params[param_name].clone()
                        else:
                            param_sum += client_params[param_name]
                        valid_updates += 1

                if valid_updates > 0:
                    model_state[param_name] = param_sum / valid_updates

        self.global_model.load_state_dict(model_state)
        self.logger.info(f"FedAvg聚合完成，平均{len(client_updates)}个客户端的权重")


class FedProxAlgorithm(FederatedAlgorithmBase):
    """FedProx算法实现"""

    def __init__(self, global_model: nn.Module, args, logger, device, mu: float = 0.01):
        super().__init__(global_model, args, logger, device)
        self.mu = mu  # FedProx正则化参数

    def client_update(self, client, global_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """FedProx客户端更新：添加邻近项正则化"""
        # 保存训练前状态
        client._save_model_state()

        try:
            # 加载全局参数
            client._load_global_params(global_params)

            # 保存全局参数用于计算邻近项
            global_params_for_prox = copy.deepcopy(global_params)

            # 本地训练
            trainable_params = [p for p in client.shared_model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.args.lr,
                weight_decay=self.args.weight_decay
            )

            client.shared_model.train()
            for step, batch_data in enumerate(client.data_loader):
                if step >= self.args.local_ep:
                    break

                optimizer.zero_grad()
                x_enc, y_true, x_mark, y_mark = batch_data
                x_enc = x_enc.to(self.device)
                y_true = y_true.to(self.device)
                x_mark = x_mark.to(self.device)
                y_mark = y_mark.to(self.device)

                # 前向传播
                if hasattr(self.args, 'label_len'):
                    batch_size = x_enc.size(0)
                    x_dec = torch.zeros(batch_size, self.args.label_len + self.args.pred_len, x_enc.size(-1)).to(
                        self.device)
                    x_dec[:, :self.args.label_len, :] = x_enc[:, -self.args.label_len:, :]
                    x_mark_dec = torch.cat([x_mark[:, -self.args.label_len:, :], y_mark], dim=1)
                    y_pred = client.shared_model(x_enc, x_mark, x_dec, x_mark_dec)
                else:
                    y_pred = client.shared_model(x_enc, x_mark, None, y_mark)

                # 原始损失
                loss = client.criterion(y_pred, y_true)

                # 添加FedProx邻近项
                prox_term = 0.0
                current_params = dict(client.shared_model.named_parameters())
                for name in self.trainable_param_names:
                    if name in current_params and name in global_params_for_prox:
                        prox_term += torch.norm(
                            current_params[name] - global_params_for_prox[name].to(self.device)) ** 2

                total_loss = loss + (self.mu / 2) * prox_term
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(client.shared_model.parameters(), max_norm=1.0)
                optimizer.step()

            # 返回更新后的模型权重
            updated_params = {}
            current_state = client.shared_model.state_dict()
            for name in self.trainable_param_names:
                if name in current_state:
                    updated_params[name] = current_state[name].clone()

            return updated_params

        finally:
            # 恢复模型状态
            client._restore_model_state()

    def server_aggregate(self, client_updates: List[Dict[str, torch.Tensor]]) -> None:
        """FedProx服务器聚合：与FedAvg相同"""
        # FedProx的聚合策略与FedAvg相同，直接调用FedAvg的聚合方法
        fedavg_algorithm = FedAvgAlgorithm(self.global_model, self.args, self.logger, self.device)
        fedavg_algorithm.server_aggregate(client_updates)


class PerFedAvgAlgorithm(FederatedAlgorithmBase):
    """Per-FedAvg算法实现（梯度上传版本）"""

    def client_update(self, client, global_params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Per-FedAvg客户端更新：返回梯度"""
        return client.local_train(global_params)  # 直接使用现有的梯度计算逻辑

    def server_aggregate(self, client_gradients: List[Dict[str, torch.Tensor]]) -> None:
        """Per-FedAvg服务器聚合：聚合梯度并更新全局模型"""
        # 等权重平均梯度
        num_clients = len(client_gradients)

        # 聚合梯度
        aggregated_gradients = {}
        for param_name in self.trainable_param_names:
            if param_name in client_gradients[0]:
                grad_sum = None
                valid_grads = 0

                for client_grad in client_gradients:
                    if param_name in client_grad:
                        if grad_sum is None:
                            grad_sum = client_grad[param_name].clone()
                        else:
                            grad_sum += client_grad[param_name]
                        valid_grads += 1

                if valid_grads > 0:
                    aggregated_gradients[param_name] = grad_sum / valid_grads

        # 使用聚合梯度更新全局模型
        model_state = self.global_model.state_dict()
        for param_name, gradient in aggregated_gradients.items():
            if param_name in model_state:
                model_state[param_name] -= self.args.lr * gradient.to(self.device)

        self.global_model.load_state_dict(model_state)
        self.logger.info(f"Per-FedAvg聚合完成，平均{len(client_gradients)}个客户端的梯度")


def get_federated_algorithm(algorithm_name: str, global_model: nn.Module, args, logger, device):
    """工厂函数：根据算法名称创建对应的联邦算法实例"""
    algorithm_name = algorithm_name.lower()

    if algorithm_name == 'fedavg':
        return FedAvgAlgorithm(global_model, args, logger, device)
    elif algorithm_name == 'fedprox':
        mu = getattr(args, 'fedprox_mu', 0.01)
        return FedProxAlgorithm(global_model, args, logger, device, mu=mu)
    elif algorithm_name == 'perfedavg':
        return PerFedAvgAlgorithm(global_model, args, logger, device)
    else:
        raise ValueError(f"不支持的联邦算法: {algorithm_name}")