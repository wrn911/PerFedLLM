"""
更新的联邦学习服务器实现 - 支持多种联邦算法
"""

import torch
import random
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
from .client import FederatedClient
from .federated_algorithms import get_federated_algorithm


class FederatedServer:
    """支持多种联邦算法的联邦学习服务器"""

    def __init__(self, global_model: torch.nn.Module, args, logger, device: torch.device):
        self.global_model = global_model
        self.args = args
        self.logger = logger
        self.device = device

        # 获取联邦算法实例
        fed_algorithm_name = getattr(args, 'fed_algorithm', 'perfedavg')
        self.fed_algorithm = get_federated_algorithm(
            fed_algorithm_name, global_model, args, logger, device
        )

        self.logger.info(f"使用联邦算法: {fed_algorithm_name.upper()}")

        # 训练历史
        self.training_history = {
            'round_losses': [],
            'round_metrics': [],
            'client_metrics': defaultdict(list)
        }

    def select_clients(self, all_clients: List[FederatedClient]) -> List[FederatedClient]:
        """随机选择参与训练的客户端"""
        num_selected = max(1, int(len(all_clients) * self.args.frac))
        selected_clients = random.sample(all_clients, num_selected)

        client_ids = [client.client_id for client in selected_clients]
        self.logger.info(f"选择客户端: {client_ids}")

        return selected_clients

    def federated_train(self, clients: List[FederatedClient]) -> Dict[str, float]:
        """执行联邦训练"""
        self.logger.info(f"开始联邦训练，共 {self.args.epochs} 轮")

        # 如果设置epochs为0，跳过联邦训练
        if self.args.epochs == 0:
            self.logger.info("联邦训练轮数为0，跳过联邦训练阶段")
            return {}

        for round_idx in range(self.args.epochs):
            self.logger.info(f"\n=== 联邦训练第 {round_idx + 1}/{self.args.epochs} 轮 ===")

            # 1. 选择客户端
            selected_clients = self.select_clients(clients)

            # 2. 获取当前全局参数
            global_params = self.fed_algorithm.get_global_params()

            # 3. 客户端局部训练
            client_updates = []

            for client in selected_clients:
                try:
                    # 使用对应算法的客户端更新方法
                    client_update = self.fed_algorithm.client_update(client, global_params)
                    client_updates.append(client_update)

                except Exception as e:
                    self.logger.error(f"客户端 {client.client_id} 训练失败: {e}")
                    continue

            if not client_updates:
                self.logger.warning(f"第 {round_idx + 1} 轮没有客户端成功训练")
                continue

            # 4. 服务器聚合（使用等权重平均）
            self.fed_algorithm.server_aggregate(client_updates)

            # 5. 评估当前轮次
            if round_idx % self.args.eval_interval == 0:
                round_metrics = self._evaluate_round(selected_clients)
                self.training_history['round_metrics'].append(round_metrics)

                self.logger.info(f"第 {round_idx + 1} 轮评估结果:")
                for metric, value in round_metrics.items():
                    self.logger.info(f"  {metric}: {value:.6f}")

        self.logger.info("联邦训练完成!")
        return self.training_history['round_metrics'][-1] if self.training_history['round_metrics'] else {}

    def get_global_params(self) -> Dict[str, torch.Tensor]:
        """获取全局模型参数（委托给联邦算法对象）"""
        return self.fed_algorithm.get_global_params()

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