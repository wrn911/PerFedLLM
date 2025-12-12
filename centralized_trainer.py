"""
Centralized Trainer - 用于在整个数据集上进行集中式训练
"""

import torch
import torch.nn as nn
import numpy as np
import random
import logging
import os
import json
from typing import Dict
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error


# --- 辅助函数 ---
def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """计算评估指标"""
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    return {'mse': mse, 'mae': mae, 'rmse': rmse}


def convert_numpy_types(obj):
    """递归转换NumPy类型为Python原生类型"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj


class CentralizedTrainer:
    """集中式训练器"""

    def __init__(self, args, logger: logging.Logger):
        self.args = args
        self.logger = logger
        self.device = args.device if hasattr(args, 'device') else 'cpu'
        self._set_seed()
        self.model = None

    def _set_seed(self):
        """设置随机种子"""
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)

    def setup_data(self, federated_data: Dict, data_loader_factory):
        """设置并合并数据"""
        self.logger.info("Setting up centralized data...")

        train_datasets = []
        test_datasets = []

        for client_id, client_data in federated_data['clients'].items():
            loaders = data_loader_factory.create_data_loaders(
                client_data['sequences'],
                batch_size=self.args.local_bs
            )
            if 'train' in loaders:
                train_datasets.append(loaders['train'].dataset)
            if 'test' in loaders:
                test_datasets.append(loaders['test'].dataset)

        # 合并所有客户端的数据集
        concatenated_train_dataset = ConcatDataset(train_datasets)
        concatenated_test_dataset = ConcatDataset(test_datasets)

        self.train_loader = DataLoader(
            concatenated_train_dataset,
            batch_size=self.args.local_bs,
            shuffle=True,
            num_workers=0
        )
        self.test_loader = DataLoader(
            concatenated_test_dataset,
            batch_size=self.args.local_bs,
            shuffle=False,
            num_workers=0
        )
        self.logger.info(
            f"Data setup complete. Training samples: {len(self.train_loader.dataset)}, Test samples: {len(self.test_loader.dataset)}")

    def setup_model(self, model_class, model_args):
        """设置模型、优化器和损失函数"""
        self.model = model_class(model_args)
        
        # 仅当模型是PyTorch模型时，才将其移动到设备
        if hasattr(self.model, 'to'):
            self.model.to(self.device)

        # 为模型设置优化器和损失函数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=model_args.learning_rate) if hasattr(self.model, 'parameters') else None
        self.criterion = nn.MSELoss()

    def train(self) -> Dict:
        """执行完整的集中式训练流程"""
        if not all([self.model, self.train_loader, self.test_loader]):
            raise ValueError("Please complete model and data setup first.")

        self.logger.info("Starting centralized training process...")

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=getattr(self.args, 'weight_decay', 1e-5)
        )
        criterion = nn.MSELoss()

        for epoch in range(self.args.epochs):
            self.model.train()
            total_loss = 0
            for batch_data in self.train_loader:
                optimizer.zero_grad()
                x_enc, y_true, x_mark, y_mark = [d.to(self.device) for d in batch_data]

                # --- FIX START: 构造解码器输入 ---
                # 为需要decoder输入的模型准备数据
                batch_size = x_enc.size(0)
                x_dec = torch.zeros(batch_size, self.args.label_len + self.args.pred_len,
                                    x_enc.size(-1)).to(self.device)
                x_dec[:, :self.args.label_len, :] = x_enc[:, -self.args.label_len:, :]
                x_mark_dec = torch.cat([x_mark[:, -self.args.label_len:, :], y_mark], dim=1)
                y_pred = self.model(x_enc, x_mark, x_dec, x_mark_dec)
                # --- FIX END ---

                loss = criterion(y_pred, y_true)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            self.logger.info(f"Epoch {epoch + 1}/{self.args.epochs}, Average Training Loss: {avg_loss:.6f}")

        # 评估模型
        test_metrics = self.evaluate()

        results = {
            'final_train_loss': avg_loss,
            'test_metrics': test_metrics
        }
        self.logger.info("Centralized training process complete.")
        return results

    def evaluate(self) -> Dict[str, float]:
        """在测试集上评估模型"""
        self.model.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for batch_data in self.test_loader:
                x_enc, y_true, x_mark, y_mark = [d.to(self.device) for d in batch_data]

                # --- FIX START: 构造解码器输入 ---
                batch_size = x_enc.size(0)
                x_dec = torch.zeros(batch_size, self.args.label_len + self.args.pred_len,
                                    x_enc.size(-1)).to(self.device)
                x_dec[:, :self.args.label_len, :] = x_enc[:, -self.args.label_len:, :]
                x_mark_dec = torch.cat([x_mark[:, -self.args.label_len:, :], y_mark], dim=1)
                y_pred = self.model(x_enc, x_mark, x_dec, x_mark_dec)
                # --- FIX END ---

                predictions.append(y_pred.cpu().numpy())
                targets.append(y_true.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        metrics = calculate_metrics(predictions, targets)
        self.logger.info(
            f"Evaluation results: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}")
        return metrics

    def save_results(self, results: Dict, save_dir: str):
        """保存训练结果"""
        os.makedirs(save_dir, exist_ok=True)

        # 保存模型
        torch.save(self.model.state_dict(), os.path.join(save_dir, 'centralized_model.pth'))

        # 保存结果
        results_serializable = convert_numpy_types(results)
        with open(os.path.join(save_dir, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)

        # 保存配置
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(self.args), f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results and model saved to: {save_dir}")