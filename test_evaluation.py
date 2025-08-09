"""
模型测试集评估脚本 - 支持详细数据保存
用于加载训练好的模型，在指定数据集的测试集上计算MSE和MAE指标，并保存详细的预测数据
"""

import torch
import numpy as np
import argparse
import os
import json
import logging
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
    """获取模型类和参数（与main.py中的getModel函数相同）"""
    args.dec_in = 1
    args.c_out = 1
    args.e_layers = 2
    args.d_layers = 1
    args.embed = 'timeF'
    args.freq = 'h'
    args.activation = 'gelu'
    args.task_name = 'long_term_forecast'

    if args.model_type == 'dLinear':
        args.moving_avg = 25
        model_class = DLinearModel
    elif args.model_type == 'autoformer':
        args.factor = 3
        args.moving_avg = 25
        model_class = AutoformerModel
    elif args.model_type == 'timesNet':
        args.top_k = 5
        args.num_kernels = 6
        model_class = TimesNetModel
    elif args.model_type == 'informer':
        args.factor = 5
        args.distil = True
        model_class = InformerModel
    elif args.model_type == 'timeMixer':
        args.down_sampling_layers = 3
        args.down_sampling_window = 2
        args.down_sampling_method = 'avg'
        args.use_norm = 1
        args.channel_independence = 0
        args.decomp_method = 'moving_avg'
        args.moving_avg = 25
        model_class = TimeMixerModel
    elif args.model_type == 'timellm':
        model_class = TimeLLMModel
    elif args.model_type == 'simpletimellm':
        model_class = SimpleTimeLLMModel
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    return model_class, args


def load_model(model_path: str, model_class, args, device: torch.device):
    """加载训练好的模型"""
    logger = logging.getLogger(__name__)

    # 创建模型
    model = model_class(args).to(device)

    # 加载权重
    if os.path.exists(model_path):
        logger.info(f"Loading {args.model_type} model from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"{args.model_type} model loaded successfully")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return model


def get_normalization_params(federated_data: Dict, client_id: str):
    """获取客户端的标准化参数"""
    norm_params = federated_data['metadata'].get('norm_params', None)
    if norm_params and 'mean' in norm_params and 'std' in norm_params:
        # 尝试不同的键类型匹配
        for key_type in [int(client_id), str(client_id), client_id]:
            if key_type in norm_params['mean'] and key_type in norm_params['std']:
                mean = float(norm_params['mean'][key_type])
                std = float(norm_params['std'][key_type])
                return {'mean': mean, 'std': std}
    return None


def denormalize_data(data: np.ndarray, norm_params: Dict) -> np.ndarray:
    """反标准化数据"""
    if norm_params:
        return data * norm_params['std'] + norm_params['mean']
    return data


def calculate_sample_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """计算单个样本的指标"""
    mse = float(np.mean((pred - true) ** 2))
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(mse))

    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse
    }


def evaluate_model_detailed(model, model_type: str, federated_data: Dict,
                            data_loader_factory, device: torch.device, save_detailed: str = None) -> Dict[str, float]:
    """在整个数据集的测试集上评估模型并保存详细数据"""
    logger = logging.getLogger(__name__)
    model.eval()

    all_predictions = []
    all_targets = []
    client_metrics = {}

    # 准备详细数据保存目录
    if save_detailed:
        os.makedirs(save_detailed, exist_ok=True)
        logger.info(f"将保存详细数据到: {save_detailed}")

    logger.info(f"Evaluating {model_type} model on {len(federated_data['clients'])} clients...")

    for client_id, client_data in federated_data['clients'].items():
        client_id_str = str(client_id)
        logger.info(f"Processing client {client_id_str}...")

        # 为每个客户端创建测试数据加载器
        data_loaders = data_loader_factory.create_data_loaders(
            client_data['sequences'],
            batch_size=32
        )

        if 'test' not in data_loaders:
            logger.warning(f"No test data for client {client_id_str}")
            continue

        test_loader = data_loaders['test']
        coordinates = client_data.get('coordinates', None)

        # 设置模型的上下文信息
        if hasattr(model, 'set_context_info') and coordinates:
            model.set_context_info(coordinates=coordinates)

        # 获取标准化参数
        norm_params = get_normalization_params(federated_data, client_id_str)

        client_predictions = []
        client_targets = []
        client_history = []

        # 用于保存详细数据的列表
        client_detailed_data = {
            'client_id': client_id_str,
            'coordinates': coordinates,
            'normalization_params': norm_params,
            'samples': [],
            'client_metrics': {}
        }

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_loader):
                x_enc, y_true, x_mark, y_mark = batch_data
                x_enc = x_enc.to(device)
                y_true = y_true.to(device)
                x_mark = x_mark.to(device)
                y_mark = y_mark.to(device)

                # 根据不同模型类型进行前向传播
                if model_type in ['timellm', 'simpletimellm']:
                    batch_size = x_enc.size(0)
                    if hasattr(model, 'args') and hasattr(model.args, 'label_len'):
                        x_dec = torch.zeros(batch_size, model.args.label_len + model.args.pred_len, x_enc.size(-1)).to(
                            device)
                        x_dec[:, :model.args.label_len, :] = x_enc[:, -model.args.label_len:, :]
                        x_mark_dec = torch.cat([x_mark[:, -model.args.label_len:, :], y_mark], dim=1)
                        y_pred = model(x_enc, x_mark, x_dec, x_mark_dec)
                    else:
                        y_pred = model(x_enc, x_mark, None, y_mark)

                elif model_type in ['autoformer', 'informer']:
                    batch_size = x_enc.size(0)
                    x_dec = torch.zeros(batch_size, model.args.label_len + model.args.pred_len, x_enc.size(-1)).to(
                        device)
                    x_dec[:, :model.args.label_len, :] = x_enc[:, -model.args.label_len:, :]
                    x_mark_dec = torch.cat([x_mark[:, -model.args.label_len:, :], y_mark], dim=1)
                    y_pred = model(x_enc, x_mark, x_dec, x_mark_dec)

                elif model_type in ['dLinear', 'timeMixer', 'timesNet']:
                    y_pred = model(x_enc, x_mark, None, y_mark)

                else:
                    try:
                        y_pred = model(x_enc, x_mark, None, y_mark)
                    except:
                        batch_size = x_enc.size(0)
                        x_dec = torch.zeros(batch_size, model.args.label_len + model.args.pred_len, x_enc.size(-1)).to(
                            device)
                        x_dec[:, :model.args.label_len, :] = x_enc[:, -model.args.label_len:, :]
                        x_mark_dec = torch.cat([x_mark[:, -model.args.label_len:, :], y_mark], dim=1)
                        y_pred = model(x_enc, x_mark, x_dec, x_mark_dec)

                # 转换为numpy并收集数据
                hist_np = x_enc.cpu().numpy()
                pred_np = y_pred.cpu().numpy()
                true_np = y_true.cpu().numpy()

                client_predictions.append(pred_np)
                client_targets.append(true_np)
                client_history.append(hist_np)

                # 如果需要保存详细数据，处理每个样本
                if save_detailed:
                    for sample_idx in range(pred_np.shape[0]):
                        hist_sample = hist_np[sample_idx].squeeze()  # [seq_len]
                        pred_sample = pred_np[sample_idx].squeeze()  # [pred_len]
                        true_sample = true_np[sample_idx].squeeze()  # [pred_len]

                        # 计算归一化后的指标
                        norm_metrics = calculate_sample_metrics(pred_sample, true_sample)

                        # 准备样本数据
                        sample_data = {
                            'sample_id': len(client_detailed_data['samples']),
                            'normalized': {
                                'history': hist_sample.tolist(),
                                'prediction': pred_sample.tolist(),
                                'ground_truth': true_sample.tolist(),
                                'metrics': norm_metrics
                            }
                        }

                        # 如果有标准化参数，计算反标准化后的数据和指标
                        if norm_params:
                            hist_denorm = denormalize_data(hist_sample, norm_params)
                            pred_denorm = denormalize_data(pred_sample, norm_params)
                            true_denorm = denormalize_data(true_sample, norm_params)

                            denorm_metrics = calculate_sample_metrics(pred_denorm, true_denorm)

                            sample_data['denormalized'] = {
                                'history': hist_denorm.tolist(),
                                'prediction': pred_denorm.tolist(),
                                'ground_truth': true_denorm.tolist(),
                                'metrics': denorm_metrics
                            }

                        client_detailed_data['samples'].append(sample_data)

        if client_predictions:
            # 合并当前客户端的所有预测
            client_pred = np.concatenate(client_predictions, axis=0)
            client_true = np.concatenate(client_targets, axis=0)

            # 计算客户端整体指标（归一化后）
            client_mse = np.mean((client_pred - client_true) ** 2)
            client_mae = np.mean(np.abs(client_pred - client_true))
            client_rmse = np.sqrt(client_mse)

            client_metrics[client_id_str] = {
                'MSE': client_mse,
                'MAE': client_mae,
                'RMSE': client_rmse,
                'samples': len(client_pred)
            }

            # 如果需要保存详细数据
            if save_detailed:
                # 计算客户端平均指标
                if client_detailed_data['samples']:
                    # 归一化后的平均指标
                    norm_mses = [s['normalized']['metrics']['mse'] for s in client_detailed_data['samples']]
                    norm_maes = [s['normalized']['metrics']['mae'] for s in client_detailed_data['samples']]
                    norm_rmses = [s['normalized']['metrics']['rmse'] for s in client_detailed_data['samples']]

                    client_detailed_data['client_metrics']['normalized'] = {
                        'avg_mse': float(np.mean(norm_mses)),
                        'avg_mae': float(np.mean(norm_maes)),
                        'avg_rmse': float(np.mean(norm_rmses)),
                        'std_mse': float(np.std(norm_mses)),
                        'std_mae': float(np.std(norm_maes)),
                        'std_rmse': float(np.std(norm_rmses))
                    }

                    # 如果有反标准化数据，计算反标准化后的平均指标
                    if norm_params and 'denormalized' in client_detailed_data['samples'][0]:
                        denorm_mses = [s['denormalized']['metrics']['mse'] for s in client_detailed_data['samples']]
                        denorm_maes = [s['denormalized']['metrics']['mae'] for s in client_detailed_data['samples']]
                        denorm_rmses = [s['denormalized']['metrics']['rmse'] for s in client_detailed_data['samples']]

                        client_detailed_data['client_metrics']['denormalized'] = {
                            'avg_mse': float(np.mean(denorm_mses)),
                            'avg_mae': float(np.mean(denorm_maes)),
                            'avg_rmse': float(np.mean(denorm_rmses)),
                            'std_mse': float(np.std(denorm_mses)),
                            'std_mae': float(np.std(denorm_maes)),
                            'std_rmse': float(np.std(denorm_rmses))
                        }

                # 保存客户端详细数据
                client_file = os.path.join(save_detailed, f'client_{client_id_str}.json')
                with open(client_file, 'w', encoding='utf-8') as f:
                    json.dump(client_detailed_data, f, indent=2, ensure_ascii=False)

                logger.info(f"客户端 {client_id_str} 详细数据已保存到 {client_file}")

            # 添加到全局列表
            all_predictions.extend(client_predictions)
            all_targets.extend(client_targets)

    # 计算全局指标
    if all_predictions:
        all_pred = np.concatenate(all_predictions, axis=0)
        all_true = np.concatenate(all_targets, axis=0)

        global_mse = np.mean((all_pred - all_true) ** 2)
        global_mae = np.mean(np.abs(all_pred - all_true))

        global_metrics = {
            'Model_type': model_type,
            'Global_MSE': global_mse,
            'Global_MAE': global_mae,
            'Global_RMSE': np.sqrt(global_mse),
            'Total_samples': len(all_pred),
            'Num_clients': len(client_metrics)
        }

        # 计算客户端平均指标
        if client_metrics:
            avg_mse = np.mean([metrics['MSE'] for metrics in client_metrics.values()])
            avg_mae = np.mean([metrics['MAE'] for metrics in client_metrics.values()])
            std_mse = np.std([metrics['MSE'] for metrics in client_metrics.values()])
            std_mae = np.std([metrics['MAE'] for metrics in client_metrics.values()])

            global_metrics.update({
                'Avg_client_MSE': avg_mse,
                'Avg_client_MAE': avg_mae,
                'Std_client_MSE': std_mse,
                'Std_client_MAE': std_mae
            })
    else:
        global_metrics = {
            'Model_type': model_type,
            'Global_MSE': float('inf'),
            'Global_MAE': float('inf'),
            'Global_RMSE': float('inf'),
            'Total_samples': 0,
            'Num_clients': 0
        }

    return global_metrics, client_metrics


def print_evaluation_results(global_metrics: Dict, client_metrics: Dict):
    """打印评估结果"""
    logger = logging.getLogger(__name__)

    model_type = global_metrics.get('Model_type', 'Unknown')

    logger.info("=" * 60)
    logger.info(f"EVALUATION RESULTS - {model_type.upper()}")
    logger.info("=" * 60)

    # 全局指标
    logger.info("Global Metrics:")
    logger.info(f"  Model Type: {model_type}")
    logger.info(f"  Total Samples: {global_metrics['Total_samples']:,}")
    logger.info(f"  Number of Clients: {global_metrics['Num_clients']}")
    logger.info(f"  Global MSE: {global_metrics['Global_MSE']:.6f}")
    logger.info(f"  Global MAE: {global_metrics['Global_MAE']:.6f}")
    logger.info(f"  Global RMSE: {global_metrics['Global_RMSE']:.6f}")

    # 客户端平均指标
    if 'Avg_client_MSE' in global_metrics:
        logger.info("\nClient Average Metrics:")
        logger.info(f"  Average MSE: {global_metrics['Avg_client_MSE']:.6f} ± {global_metrics['Std_client_MSE']:.6f}")
        logger.info(f"  Average MAE: {global_metrics['Avg_client_MAE']:.6f} ± {global_metrics['Std_client_MAE']:.6f}")

    # 每个客户端的详细指标
    logger.info("\nPer-Client Metrics:")
    for client_id, metrics in sorted(client_metrics.items(), key=lambda x: float(x[1]['MSE'])):
        logger.info(f"  Client {client_id}: MSE={metrics['MSE']:.6f}, MAE={metrics['MAE']:.6f}, "
                    f"RMSE={metrics['RMSE']:.6f}, Samples={metrics['samples']}")


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

    # 转换NumPy类型为Python原生类型
    results = {
        'global_metrics': convert_numpy_types(global_metrics),
        'client_metrics': convert_numpy_types(client_metrics)
    }

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {save_path}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Model Test Set Evaluation with Detailed Data Saving')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model (.pth file)')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the config.json file')
    parser.add_argument('--data_file', type=str, default=None,
                        help='Data file name (e.g., milano.h5). If not specified, use config file setting')
    parser.add_argument('--data_type', type=str, default=None,
                        choices=['net', 'call', 'sms'],
                        help='Traffic data type. If not specified, use config file setting')
    parser.add_argument('--save_results', type=str, default=None,
                        help='Path to save evaluation results (JSON format)')
    parser.add_argument('--save_detailed', type=str, default=None,
                        help='Directory to save detailed prediction data for each client')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0/cpu)')

    args = parser.parse_args()

    # 设置日志
    logger = setup_logging()

    # 设置设备
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

        # 3. 获取模型类和配置参数
        model_class, config_args = get_model_class_and_args(config_args)
        logger.info(f"Model type: {config_args.model_type}")

        # 4. 加载数据
        logger.info(f"Loading federated data from {config_args.file_path} (type: {config_args.data_type})...")
        federated_data, data_loader_factory = get_federated_data(config_args)
        logger.info(f"Loaded {len(federated_data['clients'])} clients")

        # 5. 加载模型
        logger.info(f"Loading {config_args.model_type} model...")
        model = load_model(args.model_path, model_class, config_args, device)

        # 6. 评估模型（包含详细数据保存）
        logger.info("Evaluating model on test set...")
        global_metrics, client_metrics = evaluate_model_detailed(
            model, config_args.model_type, federated_data, data_loader_factory, device, args.save_detailed
        )

        # 7. 打印结果
        print_evaluation_results(global_metrics, client_metrics)

        # 8. 保存结果（如果指定）
        if args.save_results:
            save_results(global_metrics, client_metrics, args.save_results)

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()