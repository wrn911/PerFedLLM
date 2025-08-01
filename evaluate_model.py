"""
SimpleTimeLLM模型加载和可视化脚本
用于加载训练好的全局模型，选择客户端，并绘制预测结果对比图
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os
import json
from typing import Dict, List, Tuple
import logging

# 导入必要的模块
from dataset.data_loader import get_federated_data, FederatedDataLoader
from models.SimpleTimeLLM import Model as SimpleTimeLLMModel


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


def get_model_args(args):
    """设置SimpleTimeLLM模型参数"""
    args.dec_in = 1
    args.c_out = 1
    args.e_layers = 2
    args.d_layers = 1
    args.embed = 'timeF'
    args.freq = 'h'
    args.activation = 'gelu'
    args.task_name = 'long_term_forecast'

    return args


def load_model(model_path: str, args, device: torch.device) -> SimpleTimeLLMModel:
    """加载训练好的SimpleTimeLLM模型"""
    logger = logging.getLogger(__name__)

    # 创建模型
    model = SimpleTimeLLMModel(args).to(device)

    # 加载权重
    if os.path.exists(model_path):
        logger.info(f"Loading model from: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("Model loaded successfully")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return model


def select_client_data(federated_data: Dict, client_id: str = None) -> Tuple[str, Dict]:
    """选择客户端数据"""
    clients = federated_data['clients']
    available_client_ids = list(clients.keys())

    if client_id is None:
        # 如果没有指定，选择第一个客户端
        selected_client_id = available_client_ids[0]
    else:
        # 尝试不同的类型匹配
        selected_client_id = None

        # 1. 直接字符串匹配
        if client_id in clients:
            selected_client_id = client_id
        # 2. 尝试转换为整数后匹配
        elif client_id.isdigit():
            client_id_int = int(client_id)
            if client_id_int in clients:
                selected_client_id = client_id_int
            # 3. 尝试整数转字符串匹配
            elif str(client_id_int) in clients:
                selected_client_id = str(client_id_int)

        # 4. 如果还是没找到，检查是否存在类型转换问题
        if selected_client_id is None:
            # 尝试将所有available client IDs转换为字符串进行比较
            str_available_ids = [str(cid) for cid in available_client_ids]
            if client_id in str_available_ids:
                # 找到对应的原始ID
                original_index = str_available_ids.index(client_id)
                selected_client_id = available_client_ids[original_index]

        # 5. 如果仍然没找到，选择第一个并显示错误信息
        if selected_client_id is None:
            print(f"Client {client_id} not found. Available clients: {available_client_ids}")
            print(f"Client ID types in data: {[type(cid).__name__ for cid in available_client_ids[:5]]}")
            selected_client_id = available_client_ids[0]
            print(f"Using first available client: {selected_client_id}")

    return str(selected_client_id), clients[selected_client_id]


def predict_on_test_set(model: SimpleTimeLLMModel, test_loader, device: torch.device,
                       coordinates: Dict = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在测试集上进行预测"""
    model.eval()

    all_history = []
    all_predictions = []
    all_targets = []

    # 设置模型的上下文信息（如果模型支持）
    if hasattr(model, 'set_context_info') and coordinates:
        model.set_context_info(coordinates=coordinates)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            x_enc, y_true, x_mark, y_mark = batch_data
            x_enc = x_enc.to(device)
            y_true = y_true.to(device)
            x_mark = x_mark.to(device)
            y_mark = y_mark.to(device)

            # 为SimpleTimeLLM准备decoder输入
            batch_size = x_enc.size(0)
            if hasattr(model, 'args') and hasattr(model.args, 'label_len'):
                # 创建decoder输入
                x_dec = torch.zeros(batch_size, model.args.label_len + model.args.pred_len, x_enc.size(-1)).to(device)
                x_dec[:, :model.args.label_len, :] = x_enc[:, -model.args.label_len:, :]
                x_mark_dec = torch.cat([x_mark[:, -model.args.label_len:, :], y_mark], dim=1)
                y_pred = model(x_enc, x_mark, x_dec, x_mark_dec)
            else:
                # 简化版本
                y_pred = model(x_enc, x_mark, None, y_mark)

            # 收集数据
            all_history.append(x_enc.cpu().numpy())
            all_predictions.append(y_pred.cpu().numpy())
            all_targets.append(y_true.cpu().numpy())

    # 合并所有批次的数据
    history = np.concatenate(all_history, axis=0)  # [N, seq_len, 1]
    predictions = np.concatenate(all_predictions, axis=0)  # [N, pred_len, 1]
    targets = np.concatenate(all_targets, axis=0)  # [N, pred_len, 1]

    return history, predictions, targets


def denormalize_data(data: np.ndarray, norm_params: Dict, client_id) -> np.ndarray:
    """反标准化数据"""
    if norm_params and 'mean' in norm_params and 'std' in norm_params:
        mean = norm_params['mean'][client_id]
        std = norm_params['std'][client_id]
        return data * std + mean
    return data


def plot_single_prediction(history: np.ndarray, predictions: np.ndarray, targets: np.ndarray,
                          client_id: str, coordinates: Dict = None, sample_idx: int = 0,
                          seq_len: int = 96, pred_len: int = 24, save_path: str = None):
    """绘制单个样本的预测结果对比图"""

    # 选择一个样本进行可视化
    if sample_idx >= len(history):
        sample_idx = 0

    hist_data = history[sample_idx, :, 0]  # [seq_len]
    pred_data = predictions[sample_idx, :, 0]  # [pred_len]
    true_data = targets[sample_idx, :, 0]  # [pred_len]

    # 创建时间轴
    hist_time = np.arange(seq_len)
    pred_time = np.arange(seq_len, seq_len + pred_len)

    # 设置图形
    plt.figure(figsize=(12, 6))

    # 绘制历史数据
    plt.plot(hist_time, hist_data, label='Historical Data', color='blue', linewidth=2)

    # 绘制真实值
    plt.plot(pred_time, true_data, label='Ground Truth', color='green', linewidth=2)

    # 绘制预测值
    plt.plot(pred_time, pred_data, label='Prediction', color='red', linewidth=2, linestyle='--')

    # 添加分割线
    plt.axvline(x=seq_len, color='gray', linestyle=':', alpha=0.7, label='Prediction Start')

    # 设置标题和标签
    location_info = ""
    if coordinates:
        lng = coordinates.get('lng', 0)
        lat = coordinates.get('lat', 0)
        location_info = f" (Lat: {lat:.4f}, Lng: {lng:.4f})"

    plt.title(f'Traffic Prediction Results - Client {client_id}{location_info} - Sample {sample_idx}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Traffic Volume', fontsize=12)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, alpha=0.3)

    # 移除MSE和MAE的显示，保持原有排版

    plt.tight_layout()

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


def plot_multiple_predictions(history: np.ndarray, predictions: np.ndarray, targets: np.ndarray,
                            client_id: str, coordinates: Dict = None, sample_indices: List[int] = None,
                            seq_len: int = 96, pred_len: int = 24, save_path: str = None,
                            max_samples: int = 6):
    """绘制多个样本的预测结果对比图"""

    # 确定要绘制的样本数量
    total_samples = len(history)

    if sample_indices is None:
        # 如果没有指定样本索引，均匀选择样本
        if max_samples >= total_samples:
            sample_indices = list(range(total_samples))
        else:
            step = total_samples // max_samples
            sample_indices = list(range(0, total_samples, step))[:max_samples]
    else:
        # 过滤掉超出范围的索引
        sample_indices = [idx for idx in sample_indices if 0 <= idx < total_samples]
        sample_indices = sample_indices[:max_samples]  # 限制最大样本数

    num_samples = len(sample_indices)
    if num_samples == 0:
        print("No valid samples to plot.")
        return

    # 计算子图布局
    if num_samples == 1:
        rows, cols = 1, 1
        figsize = (12, 6)
    elif num_samples == 2:
        rows, cols = 1, 2
        figsize = (20, 6)
    elif num_samples <= 4:
        rows, cols = 2, 2
        figsize = (20, 12)
    elif num_samples <= 6:
        rows, cols = 2, 3
        figsize = (24, 12)
    else:
        rows, cols = 3, 3
        figsize = (24, 18)

    # 创建子图
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if num_samples == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows * cols > 1 else [axes]

    # 地理位置信息
    location_info = ""
    if coordinates:
        lng = coordinates.get('lng', 0)
        lat = coordinates.get('lat', 0)
        location_info = f" (Lat: {lat:.4f}, Lng: {lng:.4f})"

    # 计算所有样本的整体指标
    all_mse = []
    all_mae = []

    # 绘制每个样本
    for i, sample_idx in enumerate(sample_indices):
        ax = axes[i]

        # 获取数据
        hist_data = history[sample_idx, :, 0]  # [seq_len]
        pred_data = predictions[sample_idx, :, 0]  # [pred_len]
        true_data = targets[sample_idx, :, 0]  # [pred_len]

        # 创建时间轴
        hist_time = np.arange(seq_len)
        pred_time = np.arange(seq_len, seq_len + pred_len)

        # 绘制数据
        ax.plot(hist_time, hist_data, label='Historical Data', color='blue', linewidth=1.5)
        ax.plot(pred_time, true_data, label='Ground Truth', color='green', linewidth=1.5)
        ax.plot(pred_time, pred_data, label='Prediction', color='red', linewidth=1.5, linestyle='--')

        # 添加分割线
        ax.axvline(x=seq_len, color='gray', linestyle=':', alpha=0.7)

        # 设置标题和标签
        ax.set_title(f'Sample {sample_idx}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Steps', fontsize=10)
        ax.set_ylabel('Traffic Volume', fontsize=10)
        ax.grid(True, alpha=0.3)

        # 计算并显示指标
        mse = np.mean((pred_data - true_data) ** 2)
        mae = np.mean(np.abs(pred_data - true_data))
        all_mse.append(mse)
        all_mae.append(mae)

        ax.text(0.02, 0.98, f'MSE: {mse:.4f}\nMAE: {mae:.4f}',
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7),
                verticalalignment='top')

        # 只在第一个子图显示图例
        if i == 0:
            ax.legend(fontsize=9, loc='upper right')

    # 隐藏多余的子图
    for i in range(num_samples, len(axes)):
        axes[i].set_visible(False)

    # 设置整体标题
    avg_mse = np.mean(all_mse)
    avg_mae = np.mean(all_mae)
    fig.suptitle(f'Traffic Prediction Results - Client {client_id}{location_info}\n'
                 f'Average MSE: {avg_mse:.6f}, Average MAE: {avg_mae:.6f}',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # 为suptitle留出空间

    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Multi-sample plot saved to: {save_path}")

    plt.show()


def plot_prediction_results(history: np.ndarray, predictions: np.ndarray, targets: np.ndarray,
                          client_id: str, coordinates: Dict = None, sample_indices: List[int] = None,
                          seq_len: int = 96, pred_len: int = 24, save_path: str = None,
                          plot_mode: str = 'single', max_samples: int = 6):
    """绘制预测结果对比图的统一接口"""

    if plot_mode == 'single':
        sample_idx = sample_indices[0] if sample_indices else 0
        plot_single_prediction(history, predictions, targets, client_id, coordinates,
                             sample_idx, seq_len, pred_len, save_path)
    elif plot_mode == 'multiple':
        plot_multiple_predictions(history, predictions, targets, client_id, coordinates,
                                sample_indices, seq_len, pred_len, save_path, max_samples)
    else:
        raise ValueError(f"Unknown plot_mode: {plot_mode}. Use 'single' or 'multiple'.")


def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """计算评估指标"""
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)

    # 计算MAPE (避免除零错误)
    mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100

    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='SimpleTimeLLM Model Evaluation and Visualization')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model (.pth file)')
    parser.add_argument('--config_path', type=str, required=True,
                       help='Path to the config.json file')
    parser.add_argument('--client_id', type=str, default=None,
                       help='Specific client ID to evaluate (if None, use first client)')
    parser.add_argument('--sample_indices', type=str, default='0',
                       help='Comma-separated indices of test samples to visualize (e.g., "0,5,10" or "0" for single)')
    parser.add_argument('--plot_mode', type=str, choices=['single', 'multiple'], default='single',
                       help='Plot mode: "single" for one sample, "multiple" for multiple samples')
    parser.add_argument('--max_samples', type=int, default=6,
                       help='Maximum number of samples to plot in multiple mode (default: 6)')
    parser.add_argument('--save_plot', type=str, default=None,
                       help='Path to save the plot (if None, only display)')
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
        config_args = get_model_args(config_args)

        # 2. 加载数据
        logger.info("Loading federated data...")
        federated_data, data_loader_factory = get_federated_data(config_args)

        # 3. 选择客户端
        client_id, client_data = select_client_data(federated_data, args.client_id)
        logger.info(f"Selected client: {client_id}")

        # 4. 创建测试数据加载器
        test_loader = data_loader_factory.create_data_loaders(
            client_data['sequences'],
            batch_size=1  # 使用batch_size=1便于可视化
        )['test']

        # 5. 加载模型
        logger.info("Loading SimpleTimeLLM model...")
        model = load_model(args.model_path, config_args, device)

        # 6. 进行预测
        logger.info("Making predictions on test set...")
        coordinates = client_data.get('coordinates', None)
        history, predictions, targets = predict_on_test_set(
            model, test_loader, device, coordinates
        )

        # 7. 反标准化数据（如果需要）
        norm_params = federated_data['metadata'].get('norm_params', None)
        if norm_params:
            history = denormalize_data(history, norm_params, int(client_id))
            predictions = denormalize_data(predictions, norm_params, int(client_id))
            targets = denormalize_data(targets, norm_params, int(client_id))

        # 8. 计算整体指标
        metrics = calculate_metrics(predictions, targets)
        logger.info("Overall Test Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.6f}")

        # 解析样本索引
        if args.sample_indices:
            try:
                sample_indices = [int(x.strip()) for x in args.sample_indices.split(',')]
            except ValueError:
                logger.error("Invalid sample_indices format. Use comma-separated integers like '0,5,10'")
                return
        else:
            sample_indices = [0]

        # 验证样本索引范围
        max_available = len(history)
        valid_indices = [idx for idx in sample_indices if 0 <= idx < max_available]
        if not valid_indices:
            logger.warning(f"No valid sample indices. Available range: 0-{max_available-1}")
            valid_indices = [0]
        elif len(valid_indices) < len(sample_indices):
            invalid_indices = [idx for idx in sample_indices if idx not in valid_indices]
            logger.warning(f"Invalid sample indices {invalid_indices} ignored. Available range: 0-{max_available-1}")

        sample_indices = valid_indices

        # 9. 绘制结果
        if args.plot_mode == 'single':
            logger.info(f"Plotting single sample result for sample {sample_indices[0]}...")
        else:
            logger.info(f"Plotting multiple samples: {sample_indices}")

        plot_prediction_results(
            history, predictions, targets,
            client_id=client_id,
            coordinates=coordinates,
            sample_indices=sample_indices,
            seq_len=config_args.seq_len,
            pred_len=config_args.pred_len,
            save_path=args.save_plot,
            plot_mode=args.plot_mode,
            max_samples=args.max_samples
        )

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
# 1. 绘制单个样本：
# python evaluate_model.py \
#     --model_path ./experiments/perfedllm_optimized/global_model.pth \
#     --config_path ./experiments/perfedllm_optimized/config.json \
#     --client_id 12345 \
#     --sample_indices "0" \
#     --plot_mode single \
#     --save_plot ./results/single_prediction.png \
#     --device cuda:0
#
# 2. 绘制多个指定样本：
# python evaluate_model.py \
#     --model_path ./experiments/perfedllm_optimized/global_model.pth \
#     --config_path ./experiments/perfedllm_optimized/config.json \
#     --client_id 12345 \
#     --sample_indices "0,5,10,15,20,25" \
#     --plot_mode multiple \
#     --max_samples 6 \
#     --save_plot ./results/multiple_predictions.png \
#     --device cuda:0
#
# 3. 自动选择多个样本（均匀分布）：
# python evaluate_model.py \
#     --model_path ./experiments/perfedllm_optimized/global_model.pth \
#     --config_path ./experiments/perfedllm_optimized/config.json \
#     --client_id 12345 \
#     --plot_mode multiple \
#     --max_samples 6 \
#     --save_plot ./results/auto_multiple_predictions.png \
#     --device cuda:0
#
# 参数说明：
# --model_path: 训练好的模型文件路径
# --config_path: 配置文件路径
# --client_id: 要评估的客户端ID（可选，默认选择第一个）
# --sample_indices: 要可视化的样本索引，用逗号分隔（如 "0,5,10"）
# --plot_mode: 绘图模式，"single"单个样本，"multiple"多个样本
# --max_samples: 多样本模式下的最大样本数（默认6个）
# --save_plot: 图片保存路径（可选，不指定则只显示）
# --device: 使用的设备
# ============================================================================