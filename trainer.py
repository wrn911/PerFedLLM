"""
PerFedLLM训练器主类
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


class PerFedLLMTrainer:
    """PerFedLLM训练器主类"""

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
        """设置联邦客户端"""
        self.clients = []

        for client_id, client_data in self.federated_data['clients'].items():
            # 创建客户端模型（深拷贝全局模型）
            client_model = copy.deepcopy(self.global_model)

            # 创建数据加载器
            data_loaders = self.data_loader_factory.create_data_loaders(
                client_data['sequences'],
                batch_size=self.args.local_bs
            )

            train_loader = data_loaders['train']
            test_loader = data_loaders.get('test', None)  # 可能没有独立测试集

            # 创建客户端
            client = FederatedClient(
                client_id=str(client_id),
                model=client_model,
                data_loader=train_loader,
                args=self.args,
                logger=self.logger,
                device=self.device
            )

            # 添加测试数据加载器（如果存在）
            if test_loader is not None:
                client.test_loader = test_loader

            self.clients.append(client)

        self.logger.info(f"创建 {len(self.clients)} 个联邦客户端")

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

        self.logger.info("开始PerFedLLM训练流程")

        # 阶段1: 联邦训练
        federated_metrics = self.server.federated_train(self.clients)

        # 阶段2: 个性化微调
        personalized_metrics = self.server.personalized_phase(self.clients)

        # 整理结果
        results = {
            'federated_metrics': federated_metrics,
            'personalized_metrics': personalized_metrics,
            'training_history': self.server.training_history
        }

        self.logger.info("PerFedLLM训练流程完成")
        return results