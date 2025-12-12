# PerFedLLM/train_classical_centralized.py

"""
传统统计模型真正的集中式训练脚本
"""
import argparse
import json
import logging
import os
import warnings
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

from dataset.data_loader import get_federated_data
from models.ARIMA import Model as ArimaModel
from models.Lasso import Model as LassoModel
from models.SVR import Model as SvrModel
from models.Prophet import Model as ProphetModel

warnings.filterwarnings('ignore')

def setup_logging(exp_name):
    """设置日志"""
    log_dir = os.path.join("experiments", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, "training.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("ClassicalCentralizedTrainer")

def get_model_class(model_type: str):
    """根据模型类型获取模型类"""
    model_map = {
        'arima': ArimaModel,
        'lasso': LassoModel,
        'svr': SvrModel,
        'prophet': ProphetModel,
    }
    model_class = model_map.get(model_type.lower())
    if model_class is None:
        raise ValueError(f"不支持的模型类型: {model_type}")
    return model_class

def aggregate_data(federated_data, data_loader_factory, logger):
    """聚合所有客户端的数据为一个长序列"""
    logger.info("开始聚合所有客户端的数据...")
    all_train_series = []
    all_test_series = []

    # 聚合训练数据
    # 为了保证时间连续性，我们找到最早的开始时间和最晚的结束时间
    # 但一个更简单的方法是直接拼接所有序列
    for client_id, client_data in federated_data['clients'].items():
        train_sequences = client_data['sequences']['train']
        history = train_sequences['history'] # (N, seq_len, 1)
        target = train_sequences['target']   # (N, pred_len, 1)
        
        # 将每个客户端的所有样本拼接成一个长序列
        client_full_series = np.concatenate([h.flatten() for h in history] + [t.flatten() for t in target])
        all_train_series.append(client_full_series)

    # 简单地将所有客户端的长序列拼接起来
    # 注意：这可能不完全符合时间序列的连续性，但对于这些模型是可行的训练方式
    train_series = np.concatenate(all_train_series)
    logger.info(f"聚合后的训练数据总长度: {len(train_series)}")

    # 聚合测试数据，测试数据通常是在训练数据之后
    # 我们选择一个有代表性的客户端的测试数据或者拼接所有测试数据
    # 这里我们选择拼接所有测试数据的 history 部分作为测试输入
    test_histories = []
    test_targets = []
    for client_id, client_data in federated_data['clients'].items():
        if 'test' in client_data['sequences']:
            test_sequences = client_data['sequences']['test']
            test_histories.extend([h.flatten() for h in test_sequences['history']])
            test_targets.extend([t.flatten() for t in test_sequences['target']])
            
    logger.info(f"聚合测试样本数: {len(test_histories)}")

    return train_series, test_histories, test_targets

def main():
    parser = argparse.ArgumentParser(description='真正集中式训练传统模型')
    parser.add_argument('--model_type', type=str, required=True, help='模型类型 (ARIMA, SVR, Lasso, Prophet)')
    parser.add_argument('--file_path', type=str, required=True, help='数据集文件路径 (e.g., milano.h5)')
    parser.add_argument('--data_type', type=str, required=True, help='数据类型 (e.g., net, call, sms)')
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度')
    parser.add_argument('--pred_len', type=int, default=24, help='预测序列长度')
    parser.add_argument('--experiment_name', type=str, default='classical_centralized_exp', help='实验名称')
    parser.add_argument('--label_len', type=int, default=48, help='label_len for data loader') # data_loader需要
    parser.add_argument('--num_clients', type=int, default=50, help='客户端数量 for data loader') # data_loader需要
    parser.add_argument('--seed', type=int, default=2024, help='随机种子')
    parser.add_argument('--test_days', type=int, default=7, help='用作测试集的天数 (data_loader需要)')
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='任务名称')

    # 添加通用的模型配置参数，以防模型内部需要
    parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--freq', type=str, default='h', help='freq for time features')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    
    args = parser.parse_args()

    logger = setup_logging(args.experiment_name)
    logger.info("实验配置: %s", args)
    
    # 1. 加载数据
    logger.info("=== 步骤1: 加载并聚合数据 ===")
    try:
        federated_data, data_loader_factory = get_federated_data(args)
        train_series, test_histories, test_targets = aggregate_data(federated_data, data_loader_factory, logger)
    except Exception as e:
        logger.error(f"数据加载或聚合失败: {e}")
        return

    # 2. 设置模型
    logger.info("=== 步骤2: 设置模型 ===")
    try:
        model_class = get_model_class(args.model_type)
        model = model_class(args)
        logger.info(f"模型 {args.model_type} 设置成功")
    except Exception as e:
        logger.error(f"模型设置失败: {e}")
        return

    # 3. 训练和评估模型
    logger.info("=== 步骤3 & 4: 在测试集上进行拟合和预测 ===")
    start_time = time.time()
    
    if not test_histories:
        logger.warning("没有可用的测试数据进行评估。")
        return
        
    predictions = []
    try:
        # 对于ARIMA这类模型，每次预测都是基于当前的history进行一次新的拟合
        for history in test_histories:
            # 确保history是 (seq_len, 1) 的形状
            history_reshaped = history.reshape(-1, 1)
            pred = model.fit_and_predict(history_reshaped, args.pred_len)
            predictions.append(pred.flatten())
        
        eval_time = time.time() - start_time
        logger.info(f"模型评估完成, 耗时: {eval_time:.2f} 秒")

    except Exception as e:
        logger.error(f"模型拟合或预测失败: {e}")
        import traceback
        traceback.print_exc()
        return

    predictions = np.array(predictions)
    test_targets = np.array(test_targets)

    # 确保预测和真实的形状匹配
    if predictions.shape != test_targets.shape:
        logger.warning(f"预测结果 shape {predictions.shape} 与真实值 shape {test_targets.shape} 不匹配。评估可能不准确。")
        # 尝试裁剪以匹配
        min_len = min(predictions.shape[1], test_targets.shape[1])
        predictions = predictions[:, :min_len]
        test_targets = test_targets[:, :min_len]

    mse = mean_squared_error(test_targets, predictions)
    mae = mean_absolute_error(test_targets, predictions)
    logger.info("--- 整体评估指标 ---")
    logger.info(f"  MSE: {mse:.6f}")
    logger.info(f"  MAE: {mae:.6f}")

    # 5. 保存结果
    logger.info("=== 步骤5: 保存结果 ===")
    save_dir = os.path.join("experiments", args.experiment_name)
    summary = {
        'experiment_name': args.experiment_name,
        'model_type': args.model_type,
        'dataset': args.file_path,
        'data_type': args.data_type,
        'final_metrics': {'mse': mse, 'mae': mae},
        'training_mode': 'classical_centralized'
    }
    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    logger.info(f"结果已保存到: {save_dir}")

if __name__ == "__main__":
    main()
