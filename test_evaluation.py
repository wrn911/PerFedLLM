# PerFedLLM/test_evaluation.py

"""
模型测试集评估脚本 - 支持联邦模型和分布式基线模型的零样本评估
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import json
import logging
import glob
import pandas as pd
from typing import Dict, Tuple
from collections import defaultdict

# 导入必要的模块
from dataset.data_loader import get_federated_data, FederatedDataLoader
from models.SimpleTimeLLM import Model as SimpleTimeLLMModel
from models.TimeLLM import Model as TimeLLMModel
from models.TimeMixer import Model as TimeMixerModel
from models.DLinear import Model as DLinearModel
from models.TimesNet import Model as TimesNetModel
from models.Autoformer import Model as AutoformerModel
from models.Informer import Model as InformerModel
from models.LSTM import Model as LSTMModel
from models.TFT import Model as TFTModel


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_config_from_json(config_path: str) -> argparse.Namespace:
    """从JSON文件加载配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    # 转换为argparse.Namespace对象
    args = argparse.Namespace()
    for key, value in config_dict.items():
        setattr(args, key, value)

    return args


def get_model_class_and_args(args):
    """获取模型类和参数"""
    args.dec_in = 1
    args.c_out = 1
    args.e_layers = 2
    args.d_layers = 1
    args.embed = 'timeF'
    args.freq = 'h'
    args.activation = 'gelu'
    args.task_name = 'long_term_forecast'

    model_map = {
        'dlinear': DLinearModel,
        'autoformer': AutoformerModel,
        'timesnet': TimesNetModel,
        'informer': InformerModel,
        'timemixer': TimeMixerModel,
        'timellm': TimeLLMModel,
        'simpletimellm': SimpleTimeLLMModel,
        'lstm': LSTMModel,
        'tft': TFTModel
    }

    model_type = args.model_type.lower()

    if model_type in model_map:
        if model_type == 'dlinear':
            args.moving_avg = 25
        elif model_type == 'autoformer':
            args.factor = 3
            args.moving_avg = 25
        elif model_type == 'timesnet':
            args.top_k = 5
            args.num_kernels = 6
        elif model_type == 'informer':
            args.factor = 5
            args.distil = True
        elif model_type == 'timemixer':
            args.down_sampling_layers = 3
            args.down_sampling_window = 2
            args.down_sampling_method = 'avg'
            args.use_norm = 1
            args.channel_independence = 0
            args.decomp_method = 'moving_avg'
            args.moving_avg = 25

        return model_map[model_type], args
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


def load_model(model_path: str, model_class, args, device: torch.device):
    """加载单个训练好的模型"""
    logger = logging.getLogger(__name__)
    model = model_class(args).to(device)
    if os.path.exists(model_path):
        logger.info(f"Loading model from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("Model loaded successfully")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return model

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

def save_results(global_metrics: Dict, client_metrics: Dict, save_path: str):
    """保存评估结果到JSON文件"""
    logger = logging.getLogger(__name__)
    results = {
        'global_metrics': convert_numpy_types(global_metrics),
        'client_metrics': convert_numpy_types(client_metrics)
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to: {save_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Zero-Shot Cross-Domain Evaluation')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model (.pth file) OR directory of client models')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the config.json file from the training run')
    parser.add_argument('--data_file', type=str, required=True,
                        help='Data file for evaluation (e.g., trento.h5)')
    parser.add_argument('--data_type', type=str, choices=['net', 'call', 'sms'], required=True,
                        help='Traffic data type for evaluation')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Path to save evaluation results (JSON format)')
    parser.add_argument('--save_predictions_csv', type=str, default=None,
                        help='Path to save all predictions and ground truth to a CSV file. If not specified, CSV will not be saved.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0/cpu)')

    args = parser.parse_args()
    logger = setup_logging()
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    logger.info(f"Using device: {device}")

    try:
        # 1. 加载配置
        logger.info("Loading configuration...")
        config_args = load_config_from_json(args.config_path)

        # 2. 覆盖数据文件设置（如果指定）
        if args.data_file:
            config_args.file_path = args.data_file
            logger.info(f"Using data file: {args.data_file}")
        if args.data_type:
            config_args.data_type = args.data_type
            logger.info(f"Using data type: {args.data_type}")

        model_class, config_args = get_model_class_and_args(config_args)
        federated_data, data_loader_factory = get_federated_data(config_args)

        # 2. 判断评估模式并加载模型
        models = []
        is_distributed_ensemble = os.path.isdir(args.model_path)

        if is_distributed_ensemble:
            logger.info("Distributed Ensemble mode detected. Loading all client models...")
            model_files = glob.glob(os.path.join(args.model_path,  'client_models', 'client_*_model.pth'))
            if not model_files:
                raise FileNotFoundError(f"No model files found in directory: {args.model_path}")

            for model_file in model_files:
                m = model_class(config_args).to(device)
                m.load_state_dict(torch.load(model_file, map_location=device))
                m.eval()
                models.append(m)
            logger.info(f"Successfully loaded {len(models)} client models for ensemble evaluation.")
        else:
            logger.info("Federated model mode detected. Loading global model...")
            model = load_model(args.model_path, model_class, config_args, device)
            model.eval()
            models.append(model)
            logger.info("Global model loaded successfully.")

        # 3. 执行评估
        all_client_metrics = {}
        all_preds = []  # 新增：收集所有客户端的所有预测
        all_trues = []  # 新增：收集所有客户端的所有真实值
        logger.info(f"Starting evaluation on {len(federated_data['clients'])} clients from {args.data_file}...")

        for client_id, client_data in federated_data['clients'].items():
            test_loader = data_loader_factory.create_data_loaders(
                client_data['sequences'], batch_size=32
            ).get('test')

            if not test_loader:
                logger.warning(f"No test data for client {client_id}. Skipping.")
                continue

            client_predictions, client_targets = [], []
            with torch.no_grad():
                for batch_data in test_loader:
                    x_enc, y_true, x_mark, y_mark = [d.to(device) for d in batch_data]

                    batch_ensemble_preds = []
                    for model in models:
                        # Forward pass logic copied from classical_trainer
                        if hasattr(config_args, 'label_len') and config_args.label_len > 0:
                            batch_size = x_enc.size(0)
                            x_dec = torch.zeros(batch_size, config_args.label_len + config_args.pred_len, x_enc.size(-1)).to(device)
                            x_dec[:, :config_args.label_len, :] = x_enc[:, -config_args.label_len:, :]
                            x_mark_dec = torch.cat([x_mark[:, -config_args.label_len:, :], y_mark], dim=1)
                            y_pred = model(x_enc, x_mark, x_dec, x_mark_dec)
                        else:
                            y_pred = model(x_enc, x_mark, None, y_mark)

                        batch_ensemble_preds.append(y_pred.cpu().numpy())

                    # Average predictions across the ensemble
                    avg_pred = np.mean(batch_ensemble_preds, axis=0)
                    client_predictions.append(avg_pred)
                    client_targets.append(y_true.cpu().numpy())

                    # 收集所有客户端的所有预测和真实值
                    all_preds.append(avg_pred)
                    all_trues.append(y_true.cpu().numpy())

            if client_predictions:
                preds = np.concatenate(client_predictions, axis=0)
                trues = np.concatenate(client_targets, axis=0)
                mse = np.mean((preds - trues) ** 2)
                mae = np.mean(np.abs(preds - trues))
                all_client_metrics[str(client_id)] = {'mse': mse, 'mae': mae}

        # ==================== 新增代码：保存所有预测结果到CSV ====================
        if args.save_predictions_csv: # 只有当参数被指定时才保存CSV
            logger.info("Saving all collected predictions and ground truth to CSV file...")
            try:
                # 合并所有批次的所有客户端数据
                all_preds_np = np.concatenate(all_preds, axis=0)
                all_trues_np = np.concatenate(all_trues, axis=0)

                # 如果是3D数组 (num_samples, pred_len, 1)，则降为2D
                if len(all_preds_np.shape) == 3 and all_preds_np.shape[2] == 1:
                    all_preds_np = all_preds_np.squeeze(-1)
                    all_trues_np = all_trues_np.squeeze(-1)

                # 创建列名
                pred_len = all_preds_np.shape[1]
                true_cols = [f'true_{i + 1}' for i in range(pred_len)]
                pred_cols = [f'pred_{i + 1}' for i in range(pred_len)]

                # 创建DataFrame
                df_trues = pd.DataFrame(all_trues_np, columns=true_cols)
                df_preds = pd.DataFrame(all_preds_np, columns=pred_cols)
                df_results = pd.concat([df_trues, df_preds], axis=1)

                # 确保目标目录存在
                output_csv_filename = args.save_predictions_csv
                output_dir = os.path.dirname(output_csv_filename)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)

                # 保存到CSV
                df_results.to_csv(output_csv_filename, index=False)
                logger.info(f"Successfully saved all prediction results to {output_csv_filename}")

            except Exception as e:
                logger.error(f"Failed to save all prediction CSV file: {e}")
        else:
            logger.info("Skipping saving predictions CSV file as --save_predictions_csv was not specified.")
        # =======================================================================

        # 4. 汇总和报告结果
        if all_client_metrics:
            avg_mse = np.mean([m['mse'] for m in all_client_metrics.values()])
            avg_mae = np.mean([m['mae'] for m in all_client_metrics.values()])

            logger.info("=" * 50)
            logger.info(f"Zero-Shot Evaluation Summary for {config_args.model_type}")
            logger.info(f"Trained on: {config_args.experiment_name}")
            logger.info(f"Evaluated on: {args.data_file} ({args.data_type})")
            logger.info(f"Evaluation Mode: {'Ensemble' if is_distributed_ensemble else 'Global Model'}")
            logger.info(f"Number of models in ensemble: {len(models)}")
            logger.info("-" * 50)
            logger.info(f"Average MSE across all clients: {avg_mse:.6f}")
            logger.info(f"Average MAE across all clients: {avg_mae:.6f}")
            logger.info("=" * 50)

            if args.save_results:
                summary = {'Global_MSE': avg_mse, 'Global_MAE': avg_mae}
                save_results(summary, all_client_metrics, args.save_results)
        else:
            logger.warning("Evaluation finished without any results.")

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()