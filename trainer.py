"""
PerFedLLM训练器主类 - 简洁的显存优化版
"""

import torch
import torch.nn as nn
import copy
import random
import numpy as np
import logging
import os
from typing import Dict
from federated.client import FederatedClient
from federated.server import FederatedServer
from utils.communication_stats import print_communication_comparison


class PerFedLLMTrainerOptimized:
    """PerFedLLM训练器主类 - 简洁优化版"""

    def __init__(self, args, logger: logging.Logger):
        self.args = args
        self.logger = logger
        self.device = args.device

        # 设置随机种子
        self._set_seed()

        # 初始化组件
        self.federated_data = None
        self.global_model = None
        self.server = None
        self.clients = []

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
        """设置联邦数据"""
        self.federated_data = federated_data
        self.data_loader_factory = data_loader_factory
        self.logger.info(f"联邦数据设置完成，客户端数量: {len(federated_data['clients'])}")

    def setup_model(self, model_class, model_args):
        """设置全局模型"""
        self.global_model = model_class(model_args).to(self.device)

        # 打印模型信息
        total_params = sum(p.numel() for p in self.global_model.parameters())
        trainable_params = sum(p.numel() for p in self.global_model.parameters() if p.requires_grad)

        self.logger.info(f"模型设置完成:")
        self.logger.info(f"  总参数: {total_params:,}")
        self.logger.info(f"  可训练参数: {trainable_params:,}")
        self.logger.info(f"  可训练比例: {100 * trainable_params / total_params:.2f}%")

    def setup_clients(self):
        """设置联邦客户端 - 优化版（共享模型但安全隔离）"""
        self.clients = []

        self.logger.info("正在创建优化版客户端（安全共享模式）...")

        for client_id, client_data in self.federated_data['clients'].items():
            # 创建数据加载器
            data_loaders = self.data_loader_factory.create_data_loaders(
                client_data['sequences'],
                batch_size=self.args.local_bs
            )

            train_loader = data_loaders['train']
            test_loader = data_loaders.get('test', None)

            # 创建优化版客户端（共享模型但参数隔离）
            client = FederatedClient(
                client_id=str(client_id),
                shared_model=self.global_model,  # 共享模型引用
                data_loader=train_loader,
                args=self.args,
                logger=self.logger,
                device=self.device
            )
            client.coordinates = client_data['coordinates']

            # 添加测试数据加载器
            if test_loader is not None:
                client.test_loader = test_loader

            self.clients.append(client)

        self.logger.info(f"创建 {len(self.clients)} 个优化版联邦客户端")
        self.logger.info("显存优化: 客户端共享模型但参数安全隔离")

    def setup_server(self):
        """设置联邦服务器"""
        self.server = FederatedServer(
            global_model=self.global_model,
            args=self.args,
            logger=self.logger,
            device=self.device
        )
        self.logger.info("联邦服务器设置完成")

    def train(self) -> Dict:
        """执行完整的PerFedLLM训练流程"""
        if not all([self.global_model, self.server, self.clients]):
            raise ValueError("请先完成模型、服务器和客户端的设置")

        self.logger.info("开始PerFedLLM训练流程（显存优化版）")

        # 阶段1: 联邦训练
        federated_metrics = self.server.federated_train(self.clients)

        # 阶段2: 个性化微调（按批次处理）
        personalized_metrics = self._personalized_phase_batched()

        # 新增：打印通信成本分析
        if(self.args.calculate_communication):
            self._print_communication_analysis()

        # 整理结果
        results = {
            'federated_metrics': federated_metrics,
            'personalized_metrics': personalized_metrics,
            'training_history': self.server.training_history
        }

        self.logger.info("PerFedLLM训练流程完成")
        return results

    def _personalized_phase_batched(self) -> Dict[str, Dict[str, float]]:
        """批次处理的个性化阶段"""
        self.logger.info("开始个性化阶段（批次处理）")

        # 获取最终全局参数
        final_global_params = self.server.get_global_params()
        personalized_results = {}

        # 批次大小：一次处理几个客户端
        batch_size = getattr(self.args, 'client_batch_size', 1)
        self.logger.info(f"批次处理客户端，每批 {batch_size} 个")

        for i in range(0, len(self.clients), batch_size):
            batch_clients = self.clients[i:i + batch_size]
            batch_ids = [client.client_id for client in batch_clients]

            self.logger.info(f"处理批次 {i//batch_size + 1}: {batch_ids}")

            # 为当前批次的所有客户端加载全局参数
            for client in batch_clients:
                client._load_global_params(final_global_params)

            # 逐个处理当前批次的客户端
            for client in batch_clients:
                try:
                    client_results = client.personalized_finetune_and_test(
                        epochs=self.args.personalized_epochs
                    )
                    personalized_results[client.client_id] = client_results

                except Exception as e:
                    self.logger.error(f"客户端 {client.client_id} 个性化微调失败: {e}")
                    personalized_results[client.client_id] = {'error': str(e)}

                # 每个客户端完成后清理显存
                torch.cuda.empty_cache()

        # 计算平均性能统计
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

    def _print_communication_analysis(self):
        """打印通信成本分析"""
        # 获取当前算法的通信追踪器
        current_tracker = self.server.get_communication_tracker()
        current_algo = self.args.fed_algorithm

        # 创建追踪器字典用于对比
        trackers = {current_algo: current_tracker}

        # 打印当前算法的通信统计
        print_communication_comparison(trackers)

        # 打印算法特性分析
        self._print_algorithm_characteristics(current_algo)

    def _print_algorithm_characteristics(self, algo_name: str):
        """打印算法特性分析"""
        print(f"\nAlgorithm Characteristics - {algo_name.upper()}:")
        print("-" * 50)

        if algo_name == 'fedavg':
            print("• Upload Type: Complete model weights")
            print("• Download Type: Global model weights")
            print("• Communication Pattern: Symmetric (upload ≈ download)")
            print("• Memory Efficiency: Standard")
            print("• Convergence: Fast convergence, higher communication cost")

        elif algo_name == 'perfedavg':
            print("• Upload Type: Gradients")
            print("• Download Type: Global model weights")
            print("• Communication Pattern: Asymmetric (upload ≈ download in size)")
            print("• Memory Efficiency: Better gradient aggregation")
            print("• Convergence: Good convergence, moderate communication cost")

        elif algo_name == 'fedprox':
            print("• Upload Type: Complete model weights")
            print("• Download Type: Global model weights")
            print("• Communication Pattern: Symmetric (upload ≈ download)")
            print("• Memory Efficiency: Standard + proximal term overhead")
            print("• Convergence: Stable convergence, similar to FedAvg communication cost")