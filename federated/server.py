"""
联邦学习服务器实现
"""

import torch
import random
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
from .client import FederatedClient


class FederatedServer:
    """联邦学习服务器"""

    def __init__(self, global_model: torch.nn.Module, args, logger, device: torch.device):
        self.global_model = global_model
        self.args = args
        self.logger = logger
        self.device = device

        # 获取可训练参数名称
        self.trainable_param_names = self._get_trainable_param_names()

        # 训练历史
        self.training_history = {
            'round_losses': [],
            'round_metrics': [],
            'client_metrics': defaultdict(list)
        }

    def _get_trainable_param_names(self) -> List[str]:
        """获取可训练参数名称"""
        trainable_names = []
        for name, param in self.global_model.named_parameters():
            if param.requires_grad:
                trainable_names.append(name)
        return trainable_names

    def select_clients(self, all_clients: List[FederatedClient]) -> List[FederatedClient]:
        """随机选择参与训练的客户端"""
        num_selected = max(1, int(len(all_clients) * self.args.frac))
        selected_clients = random.sample(all_clients, num_selected)

        client_ids = [client.client_id for client in selected_clients]
        self.logger.info(f"选择客户端: {client_ids}")

        return selected_clients

    def aggregate_gradients(self, client_gradients: List[Dict[str, torch.Tensor]],
                          client_weights: Optional[List[float]] = None) -> Dict[str, torch.Tensor]:
        """
        FedAvg梯度聚合

        Args:
            client_gradients: 客户端梯度列表
            client_weights: 客户端权重（如果为None则等权重）

        Returns:
            aggregated_gradients: 聚合后的梯度
        """
        if client_weights is None:
            client_weights = [1.0 / len(client_gradients)] * len(client_gradients)

        # 确保权重和为1
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]

        aggregated_gradients = {}

        # 对每个参数进行加权平均
        for param_name in self.trainable_param_names:
            if param_name in client_gradients[0]:
                weighted_grads = []

                for i, client_grad in enumerate(client_gradients):
                    if param_name in client_grad:
                        weighted_grad = client_grad[param_name] * client_weights[i]
                        weighted_grads.append(weighted_grad)

                if weighted_grads:
                    aggregated_gradients[param_name] = torch.stack(weighted_grads).sum(dim=0)

        self.logger.info(f"聚合 {len(client_gradients)} 个客户端的梯度")
        return aggregated_gradients

    def update_global_model(self, aggregated_gradients: Dict[str, torch.Tensor]):
        """使用聚合梯度更新全局模型"""
        model_state = self.global_model.state_dict()

        for param_name, gradient in aggregated_gradients.items():
            if param_name in model_state:
                # 应用梯度更新：θ = θ - lr * ∇θ
                model_state[param_name] -= self.args.lr * gradient.to(self.device)

        self.global_model.load_state_dict(model_state)
        self.logger.info("全局模型参数已更新")

    def get_global_params(self) -> Dict[str, torch.Tensor]:
        """获取全局模型的可训练参数"""
        global_params = {}
        model_state = self.global_model.state_dict()

        for param_name in self.trainable_param_names:
            if param_name in model_state:
                global_params[param_name] = model_state[param_name].clone()

        return global_params

    def federated_train(self, clients: List[FederatedClient]) -> Dict[str, float]:
        """执行联邦训练"""
        self.logger.info(f"开始联邦训练，共 {self.args.epochs} 轮")

        for round_idx in range(self.args.epochs):
            self.logger.info(f"\n=== 联邦训练第 {round_idx + 1}/{self.args.epochs} 轮 ===")

            # 1. 选择客户端
            selected_clients = self.select_clients(clients)

            # 2. 获取当前全局参数
            global_params = self.get_global_params()

            # 3. 客户端局部训练
            client_gradients = []
            client_weights = []

            for client in selected_clients:
                try:
                    # 客户端训练并返回梯度
                    gradients = client.local_train(global_params)
                    client_gradients.append(gradients)

                    # 计算客户端权重（基于数据量）
                    data_size = len(client.data_loader.dataset)
                    client_weights.append(data_size)

                except Exception as e:
                    self.logger.error(f"客户端 {client.client_id} 训练失败: {e}")
                    continue

            if not client_gradients:
                self.logger.warning(f"第 {round_idx + 1} 轮没有客户端成功训练")
                continue

            # 4. 梯度聚合
            aggregated_gradients = self.aggregate_gradients(client_gradients, client_weights)

            # 5. 更新全局模型
            self.update_global_model(aggregated_gradients)

            # 6. 评估当前轮次
            if round_idx % self.args.eval_interval == 0:
                round_metrics = self._evaluate_round(selected_clients)
                self.training_history['round_metrics'].append(round_metrics)

                self.logger.info(f"第 {round_idx + 1} 轮评估结果:")
                for metric, value in round_metrics.items():
                    self.logger.info(f"  {metric}: {value:.6f}")

        self.logger.info("联邦训练完成!")
        return self.training_history['round_metrics'][-1] if self.training_history['round_metrics'] else {}

    def _evaluate_round(self, clients: List[FederatedClient]) -> Dict[str, float]:
        """评估当前轮次的性能"""
        all_metrics = defaultdict(list)

        # 为每个客户端分发当前全局参数
        global_params = self.get_global_params()

        for client in clients:
            # 加载全局参数
            client._load_global_params(global_params)

            # 评估客户端
            client_metrics = client.evaluate()

            for metric, value in client_metrics.items():
                all_metrics[metric].append(value)

            # 记录客户端历史
            self.training_history['client_metrics'][client.client_id].append(client_metrics)

        # 计算平均指标
        avg_metrics = {}
        for metric, values in all_metrics.items():
            avg_metrics[f'avg_{metric}'] = np.mean(values)
            avg_metrics[f'std_{metric}'] = np.std(values)

        return avg_metrics

    def personalized_phase(self, clients: List[FederatedClient]) -> Dict[str, Dict[str, float]]:
        """个性化阶段：每个客户端进行本地微调并测试"""
        self.logger.info("\n=== 开始个性化阶段 ===")

        # 分发最终的全局模型参数
        final_global_params = self.get_global_params()

        personalized_results = {}

        for client in clients:
            try:
                # 加载全局参数作为初始点
                client._load_global_params(final_global_params)

                # 个性化微调（完整的本地训练）
                client_results = client.personalized_finetune_and_test(
                    epochs=self.args.personalized_epochs
                )

                personalized_results[client.client_id] = client_results

            except Exception as e:
                self.logger.error(f"客户端 {client.client_id} 个性化微调失败: {e}")
                personalized_results[client.client_id] = {'error': str(e)}

        # 计算个性化后的平均性能
        valid_results = [r for r in personalized_results.values() if 'error' not in r]
        if valid_results:
            avg_personalized = {}
            for metric in valid_results[0].keys():
                values = [r[metric] for r in valid_results]
                avg_personalized[f'personalized_avg_{metric}'] = np.mean(values)
                avg_personalized[f'personalized_std_{metric}'] = np.std(values)

            self.logger.info("个性化阶段完成，平均指标:")
            for metric, value in avg_personalized.items():
                self.logger.info(f"  {metric}: {value:.6f}")

        return personalized_results