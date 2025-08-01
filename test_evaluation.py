# ============================================================================
# 使用示例：
#
# 1. 评估SimpleTimeLLM模型：
# python test_evaluation.py \
#     --model_path ./experiments/simpletimellm/global_model.pth \
#     --config_path ./experiments/simpletimellm/config.json \
#     --save_results ./results/simpletimellm_metrics.json \
#     --device cuda:0
#
# 2. 评估DLinear模型：
# python test_evaluation.py \
#     --model_path ./experiments/dlinear/global_model.pth \
#     --config_path ./experiments/dlinear/config.json \
#     --save_results ./results/dlinear_metrics.json \
#     --device cuda:0
#
# 3. 在不同数据集上评估TimeMixer模型：
# python test_evaluation.py \
#     --model_path ./experiments/timemixer/global_model.pth \
#     --config_path ./experiments/timemixer/config.json \
#     --data_file milano.h5 \
#     --data_type call \
#     --save_results ./results/timemixer_call_metrics.json \
#     --device cuda:0
#
# 4. 评估Autoformer模型：
# python test_evaluation.py \
#     --model_path ./experiments/autoformer/global_model.pth \
#     --config_path ./experiments/autoformer/config.json \
#     --save_results ./results/autoformer_metrics.json \
#     --device cuda:0
#
# 5. 批量比较多个模型（使用脚本）：
# #!/bin/bash
# models=("simpletimellm" "dlinear" "timemixer" "autoformer" "timesnet")
# for model in "${models[@]}"; do
#     echo "Evaluating $model..."
#     python test_evaluation.py \
#         --model_path ./experiments/$model/global_model.pth \
#         --config_path ./experiments/$model/config.json \
#         --save_results ./results/${model}_metrics.json \
#         --device cuda:0
# done
#
# 6. 创建模型比较脚本（compare_models.py）：
# import json
# import pandas as pd
#
# models = ["simpletimellm", "dlinear", "timemixer", "autoformer", "timesnet"]
# comparison_data = []
#
# for model in models:
#     try:
#         with open(f"./results/{model}_metrics.json", "r") as f:
#             results = json.load(f)
#             global_metrics = results["global_metrics"]
#             comparison_data.append({
#                 "Model": model,
#                 "MSE": global_metrics["Global_MSE"],
#                 "MAE": global_metrics["Global_MAE"],
#                 "RMSE": global_metrics["Global_RMSE"],
#                 "Clients": global_metrics["Num_clients"],
#                 "Samples": global_metrics["Total_samples"]
#             })
#     except FileNotFoundError:
#         print(f"Results file for {model} not found")
#
# df = pd.DataFrame(comparison_data)
# df = df.sort_values("MSE")  # 按MSE排序
# print("Model Performance Comparison:")
# print(df.to_string(index=False))
# df.to_csv("./results/model_comparison.csv", index=False)
#
# 支持的模型类型：
# - simpletimellm: SimpleTimeLLM
# - timellm: TimeLLM
# - dLinear: DLinear
# - autoformer: Autoformer
# - informer: Informer
# - timeMixer: TimeMixer
# - timesNet: TimesNet
#
# 参数说明：
# --model_path: 训练好的模型文件路径
# --config_path: 配置文件路径（会自动识别模型类型）
# --data_file: 数据文件名（可选，不指定则使用配置文件中的设置）
# --data_type: 流量类型 net/call/sms（可选，不指定则使用配置文件中的设置）
# --save_results: 结果保存路径（可选，JSON格式）
# --device: 使用的设备
# ============================================================================"""
"""
模型测试集评估脚本
用于加载训练好的模型，在指定数据集的测试集上计算MSE和MAE指标
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


def evaluate_model_on_dataset(model, model_type: str, federated_data: Dict,
                              data_loader_factory, device: torch.device) -> Dict[str, float]:
    """在整个数据集的测试集上评估模型"""
    logger = logging.getLogger(__name__)
    model.eval()

    all_predictions = []
    all_targets = []
    client_metrics = {}

    logger.info(f"Evaluating {model_type} model on {len(federated_data['clients'])} clients...")

    for client_id, client_data in federated_data['clients'].items():
        # 为每个客户端创建测试数据加载器
        data_loaders = data_loader_factory.create_data_loaders(
            client_data['sequences'],
            batch_size=32  # 使用较大的batch size加速评估
        )

        if 'test' not in data_loaders:
            logger.warning(f"No test data for client {client_id}")
            continue

        test_loader = data_loaders['test']
        coordinates = client_data.get('coordinates', None)

        # 设置模型的上下文信息（如果支持，主要是LLM模型）
        if hasattr(model, 'set_context_info') and coordinates:
            model.set_context_info(coordinates=coordinates)

        client_predictions = []
        client_targets = []

        with torch.no_grad():
            for batch_data in test_loader:
                x_enc, y_true, x_mark, y_mark = batch_data
                x_enc = x_enc.to(device)
                y_true = y_true.to(device)
                x_mark = x_mark.to(device)
                y_mark = y_mark.to(device)

                # 根据不同模型类型进行前向传播
                if model_type in ['timellm', 'simpletimellm']:
                    # LLM系列模型
                    batch_size = x_enc.size(0)
                    if hasattr(model, 'args') and hasattr(model.args, 'label_len'):
                        # 创建decoder输入
                        x_dec = torch.zeros(batch_size, model.args.label_len + model.args.pred_len, x_enc.size(-1)).to(
                            device)
                        x_dec[:, :model.args.label_len, :] = x_enc[:, -model.args.label_len:, :]
                        x_mark_dec = torch.cat([x_mark[:, -model.args.label_len:, :], y_mark], dim=1)
                        y_pred = model(x_enc, x_mark, x_dec, x_mark_dec)
                    else:
                        # 简化版本
                        y_pred = model(x_enc, x_mark, None, y_mark)

                elif model_type in ['autoformer', 'informer']:
                    # Transformer系列模型，需要decoder输入
                    batch_size = x_enc.size(0)
                    # 创建decoder输入
                    x_dec = torch.zeros(batch_size, model.args.label_len + model.args.pred_len, x_enc.size(-1)).to(
                        device)
                    x_dec[:, :model.args.label_len, :] = x_enc[:, -model.args.label_len:, :]
                    x_mark_dec = torch.cat([x_mark[:, -model.args.label_len:, :], y_mark], dim=1)
                    y_pred = model(x_enc, x_mark, x_dec, x_mark_dec)

                elif model_type in ['dLinear', 'timeMixer', 'timesNet']:
                    # 其他模型，通常只需要encoder输入
                    y_pred = model(x_enc, x_mark, None, y_mark)

                else:
                    # 默认处理方式
                    try:
                        y_pred = model(x_enc, x_mark, None, y_mark)
                    except:
                        # 如果失败，尝试带decoder的方式
                        batch_size = x_enc.size(0)
                        x_dec = torch.zeros(batch_size, model.args.label_len + model.args.pred_len, x_enc.size(-1)).to(
                            device)
                        x_dec[:, :model.args.label_len, :] = x_enc[:, -model.args.label_len:, :]
                        x_mark_dec = torch.cat([x_mark[:, -model.args.label_len:, :], y_mark], dim=1)
                        y_pred = model(x_enc, x_mark, x_dec, x_mark_dec)

                # 收集预测和真实值
                client_predictions.append(y_pred.cpu().numpy())
                client_targets.append(y_true.cpu().numpy())

        if client_predictions:
            # 合并当前客户端的所有预测
            client_pred = np.concatenate(client_predictions, axis=0)
            client_true = np.concatenate(client_targets, axis=0)

            # 计算客户端指标
            client_mse = np.mean((client_pred - client_true) ** 2)
            client_mae = np.mean(np.abs(client_pred - client_true))

            client_metrics[str(client_id)] = {
                'MSE': client_mse,
                'MAE': client_mae,
                'RMSE': np.sqrt(client_mse),
                'samples': len(client_pred)
            }

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


def denormalize_if_needed(federated_data: Dict, client_metrics: Dict) -> Dict:
    """如果存在标准化参数，反标准化指标"""
    norm_params = federated_data['metadata'].get('norm_params', None)
    if norm_params is None:
        return client_metrics

    logger = logging.getLogger(__name__)
    logger.info("Denormalizing metrics...")

    denorm_metrics = {}
    for client_id, metrics in client_metrics.items():
        if client_id in norm_params['mean'] and client_id in norm_params['std']:
            std_val = norm_params['std'][int(client_id)]
            # MSE需要乘以std的平方，MAE和RMSE乘以std
            denorm_metrics[client_id] = {
                'MSE': metrics['MSE'] * (std_val ** 2),
                'MAE': metrics['MAE'] * std_val,
                'RMSE': metrics['RMSE'] * std_val,
                'samples': metrics['samples']
            }
        else:
            denorm_metrics[client_id] = metrics

    return denorm_metrics


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
    elif hasattr(obj, 'item'):  # 处理单元素numpy数组
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
        # 尝试保存一个简化版本
        try:
            simplified_results = {
                'global_metrics': {
                    'Model_type': str(global_metrics.get('Model_type', 'Unknown')),
                    'Global_MSE': float(global_metrics.get('Global_MSE', 0)),
                    'Global_MAE': float(global_metrics.get('Global_MAE', 0)),
                    'Global_RMSE': float(global_metrics.get('Global_RMSE', 0)),
                    'Total_samples': int(global_metrics.get('Total_samples', 0)),
                    'Num_clients': int(global_metrics.get('Num_clients', 0))
                },
                'summary': 'Simplified results due to serialization issues'
            }

            simplified_path = save_path.replace('.json', '_simplified.json')
            with open(simplified_path, 'w', encoding='utf-8') as f:
                json.dump(simplified_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Simplified results saved to: {simplified_path}")
        except Exception as e2:
            logger.error(f"Failed to save even simplified results: {e2}")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='SimpleTimeLLM Model Test Set Evaluation')
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

        # 6. 评估模型
        logger.info("Evaluating model on test set...")
        global_metrics, client_metrics = evaluate_model_on_dataset(
            model, config_args.model_type, federated_data, data_loader_factory, device
        )

        # 6. 反标准化指标（如果需要）
        client_metrics = denormalize_if_needed(federated_data, client_metrics)

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

# ============================================================================
# 使用示例：
#
# 1. 使用配置文件中的数据设置：
# python test_evaluation.py \
#     --model_path ./experiments/perfedllm_optimized/global_model.pth \
#     --config_path ./experiments/perfedllm_optimized/config.json \
#     --save_results ./results/test_metrics.json \
#     --device cuda:0
#
# 2. 指定不同的数据文件和流量类型：
# python test_evaluation.py \
#     --model_path ./experiments/perfedllm_optimized/global_model.pth \
#     --config_path ./experiments/perfedllm_optimized/config.json \
#     --data_file milano.h5 \
#     --data_type call \
#     --save_results ./results/call_traffic_metrics.json \
#     --device cuda:0
#
# 3. 评估不同数据集：
# python test_evaluation.py \
#     --model_path ./experiments/perfedllm_optimized/global_model.pth \
#     --config_path ./experiments/perfedllm_optimized/config.json \
#     --data_file another_dataset.h5 \
#     --data_type net \
#     --save_results ./results/another_dataset_metrics.json \
#     --device cuda:0
#
# 参数说明：
# --model_path: 训练好的模型文件路径
# --config_path: 配置文件路径
# --data_file: 数据文件名（可选，不指定则使用配置文件中的设置）
# --data_type: 流量类型 net/call/sms（可选，不指定则使用配置文件中的设置）
# --save_results: 结果保存路径（可选，JSON格式）
# --device: 使用的设备
# ============================================================================