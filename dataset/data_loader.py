# -*- coding: utf-8 -*-
"""
联邦学习数据加载器 - 包含真实时间特征
"""
import h5py
import pandas as pd
import numpy as np
import random
import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict
import logging


def extract_time_features(timestamp):
    """从pandas时间戳提取时间特征"""
    return [
        (timestamp.month - 1) / 11.0,  # 月份 [0,1]
        (timestamp.day - 1) / 30.0,  # 日期 [0,1]
        timestamp.hour / 23.0,  # 小时 [0,1]
        timestamp.weekday() / 6.0  # 星期 [0,1]
    ]


class FederatedDataLoader:
    """联邦学习数据加载器"""

    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger(__name__)

        # 设置随机种子
        self._set_seed()

    def _set_seed(self):
        """设置随机种子"""
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.seed)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        加载数据

        Returns:
            normalized_df: 标准化后的流量数据
            original_df: 原始流量数据
            metadata: 元数据信息
        """
        self.logger.info("开始加载数据...")

        # 读取HDF5文件
        path = os.getcwd()
        file_path = os.path.join(path, 'dataset', self.args.file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        with h5py.File(file_path, 'r') as f:
            self.logger.info(f"HDF5文件字段: {list(f.keys())}")

            idx = f['idx'][()]
            cell = f['cell'][()]
            lng = f['lng'][()]
            lat = f['lat'][()]
            data = f[self.args.data_type][()][:, cell - 1]

        # 构建DataFrame
        df = pd.DataFrame(data, index=pd.to_datetime(idx.ravel(), unit='s'), columns=cell)
        df.fillna(0, inplace=True)

        self.logger.info(f"原始数据形状: {df.shape}")
        self.logger.info(f"时间范围: {df.index.min()} 到 {df.index.max()}")

        # 随机选择基站
        cell_pool = list(cell)
        selected_cells = sorted(random.sample(cell_pool, self.args.num_clients))
        selected_cells_idx = np.where(np.isin(cell_pool, selected_cells))[0]

        # 获取选中基站的坐标
        cell_coordinates = {
            cell_id: {'lng': float(lng[idx]), 'lat': float(lat[idx])}
            for cell_id, idx in zip(selected_cells, selected_cells_idx)
        }

        # 筛选选中基站的数据
        df_selected = df[selected_cells].copy()

        self.logger.info(f"选中基站数量: {len(selected_cells)}")
        self.logger.info(f"选中基站ID前5个: {selected_cells[:5]}")

        # 数据标准化（基于训练集）- 你的理解是正确的
        train_data = df_selected.iloc[:-self.args.test_days * 24]
        mean = train_data.mean()
        std = train_data.std()
        std = std.replace(0, 1)  # 避免除零错误

        normalized_df = (df_selected - mean) / std

        # 保存标准化参数
        norm_params = {'mean': mean, 'std': std}

        # 元数据
        metadata = {
            'coordinates': cell_coordinates,
            'selected_cells': selected_cells,
            'norm_params': norm_params,
            'num_clients': len(selected_cells)
        }

        self.logger.info("数据加载完成")
        return normalized_df, df_selected, metadata

    def create_sequences_for_cell(self, cell_data: pd.Series) -> Dict:
        """
        为单个基站创建带时间特征的序列
        """
        X, y, x_marks, y_marks, timestamps = [], [], [], [], []  # 添加timestamps列表

        for i in range(self.args.seq_len, len(cell_data) - self.args.pred_len + 1):
            # 历史序列
            hist_seq = cell_data.iloc[i - self.args.seq_len:i].values
            # 预测目标
            pred_seq = cell_data.iloc[i:i + self.args.pred_len].values

            # 历史时间特征
            hist_timestamps = cell_data.iloc[i - self.args.seq_len:i].index
            hist_time_features = [extract_time_features(ts) for ts in hist_timestamps]

            # 预测时间特征
            pred_timestamps = cell_data.iloc[i:i + self.args.pred_len].index
            pred_time_features = [extract_time_features(ts) for ts in pred_timestamps]

            X.append(hist_seq)
            y.append(pred_seq)
            x_marks.append(hist_time_features)
            y_marks.append(pred_time_features)

            # 保存真实时间戳
            timestamps.append({
                'start_time': hist_timestamps[0],
                'end_time': hist_timestamps[-1],
                'pred_start': pred_timestamps[0],
                'pred_end': pred_timestamps[-1]
            })

        # 转换为numpy数组
        X = np.array(X)
        y = np.array(y)
        x_marks = np.array(x_marks)
        y_marks = np.array(y_marks)

        # 数据分割
        test_len = self.args.test_days * 24
        val_len = getattr(self.args, 'val_days', 0) * 24
        train_len = len(X) - test_len - val_len

        sequences = {
            'train': {
                'history': X[:train_len],
                'target': y[:train_len],
                'hist_marks': x_marks[:train_len],
                'pred_marks': y_marks[:train_len],
                'timestamps': timestamps[:train_len]  # 添加时间戳信息
            },
            'test': {
                'history': X[-test_len:],
                'target': y[-test_len:],
                'hist_marks': x_marks[-test_len:],
                'pred_marks': y_marks[-test_len:],
                'timestamps': timestamps[-test_len:]  # 添加时间戳信息
            }
        }

        if val_len > 0:
            sequences['val'] = {
                'history': X[train_len:train_len + val_len],
                'target': y[train_len:train_len + val_len],
                'hist_marks': x_marks[train_len:train_len + val_len],
                'pred_marks': y_marks[train_len:train_len + val_len],
                'timestamps': timestamps[train_len:train_len + val_len]  # 添加时间戳信息
            }

        return sequences

    def prepare_federated_data(self, normalized_df: pd.DataFrame, metadata: Dict) -> Dict:
        """
        为联邦学习准备数据
        """
        self.logger.info("准备联邦学习数据...")

        federated_data = {
            'clients': {},
            'metadata': metadata
        }

        # 为每个客户端（基站）准备时序数据
        for cell_id in metadata['selected_cells']:
            cell_data = normalized_df[cell_id]
            sequences = self.create_sequences_for_cell(cell_data)

            federated_data['clients'][cell_id] = {
                'sequences': sequences,
                'coordinates': metadata['coordinates'][cell_id],
                'data_stats': {
                    'train_samples': len(sequences['train']['history']),
                    'test_samples': len(sequences['test']['history']),
                    'val_samples': len(sequences['val']['history']) if 'val' in sequences else 0
                }
            }

        self.logger.info("联邦数据准备完成")
        return federated_data

    def create_data_loaders(self, sequences: Dict, batch_size: int = None) -> Dict:
        """
        创建PyTorch数据加载器
        """
        if batch_size is None:
            batch_size = self.args.local_bs

        data_loaders = {}

        for split in ['train', 'val', 'test']:
            if split in sequences:
                X = torch.FloatTensor(sequences[split]['history']).unsqueeze(-1)  # [N, seq_len, 1]
                y = torch.FloatTensor(sequences[split]['target']).unsqueeze(-1)  # [N, pred_len, 1]
                x_mark = torch.FloatTensor(sequences[split]['hist_marks'])  # [N, seq_len, 4]
                y_mark = torch.FloatTensor(sequences[split]['pred_marks'])  # [N, pred_len, 4]

                dataset = TensorDataset(X, y, x_mark, y_mark)

                # 为dataset添加时间戳信息
                dataset.timestamps = sequences[split]['timestamps']

                # 训练集需要shuffle，测试集不需要
                shuffle = (split == 'train')
                drop_last = (split == 'train')

                data_loaders[split] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=drop_last,
                    num_workers=0  # 避免多进程问题
                )

        return data_loaders


def get_federated_data(args):
    """
    获取联邦学习数据的便捷函数
    """
    # 创建数据加载器
    data_loader = FederatedDataLoader(args)

    # 加载数据
    normalized_df, original_df, metadata = data_loader.load_data()

    # 准备联邦数据
    federated_data = data_loader.prepare_federated_data(normalized_df, metadata)

    return federated_data, data_loader