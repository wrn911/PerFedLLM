"""
PerFedLLM 主启动脚本 - 支持经典模型版
"""

import argparse
import logging
import os
import sys

from centralized_trainer import CentralizedTrainer
from dataset.data_loader import get_federated_data
from models.TimeLLM import Model as TimeLLMModel
from models.TimeMixer import Model as TimeMixerModel
from models.DLinear import Model as DLinearModel
from models.TimesNet import Model as TimesNetModel
from models.Autoformer import Model as AutoformerModel
from models.Informer import Model as InformerModel
from models.SimpleTimeLLM import Model as SimpleTimeLLMModel

# 导入经典模型
try:
    from models.ARIMA import Model as ARIMAModel
except ImportError:
    ARIMAModel = None
    print("警告: ARIMA模型导入失败，请检查statsmodels依赖")

try:
    from models.Lasso import Model as LassoModel
except ImportError:
    LassoModel = None
    print("警告: Lasso模型导入失败，请检查scikit-learn依赖")

try:
    from models.SVR import Model as SVRModel
except ImportError:
    SVRModel = None
    print("警告: SVR模型导入失败，请检查scikit-learn依赖")

try:
    from models.Prophet import Model as ProphetModel
except ImportError:
    ProphetModel = None
    print("警告: Prophet模型导入失败，请检查prophet依赖")

try:
    from models.LSTM import Model as LSTMModel
except ImportError:
    LSTMModel = None
    print("警告: LSTM模型导入失败")

try:
    from models.TFT import Model as TFTModel
except ImportError:
    TFTModel = None
    print("警告: TFT模型导入失败")

from trainer import PerFedLLMTrainerOptimized
from classical_trainer import ClassicalModelTrainer, setup_classical_model_args
import torch
import json


def setup_logging(level: str = 'INFO') -> logging.Logger:
    """设置日志系统"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger('PerFedLLM')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PerFedLLM: 基于LLM的个性化联邦学习及经典模型训练')

    # =============== 数据配置 ===============
    parser.add_argument('--dataset_dir', type=str, default='./dataset',
                       help='数据集目录')
    parser.add_argument('--file_path', type=str, default='milano.h5',
                       help='数据文件路径')
    parser.add_argument('--data_type', type=str, default='net',
                        help='流量类型 (net/call/sms)')
    parser.add_argument('--num_clients', type=int, default=50,
                       help='客户端数量')
    parser.add_argument('--seq_len', type=int, default=96,
                       help='输入序列长度')
    parser.add_argument('--label_len', type=int, default=48,
                       help='标签序列长度（用于TimeLLM decoder）')
    parser.add_argument('--pred_len', type=int, default=24,
                       help='预测序列长度')
    parser.add_argument('--test_days', type=int, default=7,
                       help='测试数据天数')
    parser.add_argument('--val_days', type=int, default=3,
                       help='验证数据天数')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Ratio of training samples to use per client (0.0-1.0, uses first N% of time series)')
    parser.add_argument('--sample_mode', type=str, default='continuous', choices=['continuous', 'random'],
                        help='Sampling mode: continuous (first N%) or random sampling')

    # =============== 模型配置 ===============
    parser.add_argument('--model_type', type=str, default='timellm',
                        choices=['timellm', 'simpletimellm', 'dLinear', 'timeMixer', 'autoformer',
                               'informer', 'timesNet', 'arima', 'lasso', 'svr', 'prophet', 'lstm', 'tft'],
                       help='模型类型')
    parser.add_argument('--llm_model', type=str, default='Qwen3',
                       choices=['GPT2', 'BERT', 'LLAMA', 'Qwen3', 'DeepSeek'],
                       help='LLM模型选择')
    parser.add_argument('--llm_layers', type=int, default=6,
                       help='LLM层数')
    parser.add_argument('--llm_dim', type=int, default=1024,
                       help='LLM隐藏维度')
    parser.add_argument('--d_model', type=int, default=64,
                       help='模型维度')
    parser.add_argument('--d_ff', type=int, default=256,
                       help='前馈网络维度')
    parser.add_argument('--patch_len', type=int, default=16,
                       help='补丁长度')
    parser.add_argument('--stride', type=int, default=8,
                       help='步长')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout率')
    parser.add_argument('--enc_in', type=int, default=1,
                       help='编码器输入维度')
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                       help='任务名称')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='注意力头数')
    parser.add_argument('--use_prompt', action='store_true', help='启用提示词')
    parser.add_argument('--no_prompt', dest='use_prompt', action='store_false', help='禁用提示词')
    parser.set_defaults(use_prompt=True)

    # =============== 经典模型特定参数 ===============
    # ARIMA参数
    parser.add_argument('--arima_p', type=int, default=2,
                       help='ARIMA自回归项数')
    parser.add_argument('--arima_d', type=int, default=1,
                       help='ARIMA差分次数')
    parser.add_argument('--arima_q', type=int, default=1,
                       help='ARIMA移动平均项数')

    # Lasso参数
    parser.add_argument('--lasso_alpha', type=float, default=0.01,
                       help='Lasso正则化强度')
    parser.add_argument('--lasso_max_iter', type=int, default=1000,
                       help='Lasso最大迭代次数')

    # SVR参数
    parser.add_argument('--svr_kernel', type=str, default='rbf',
                       help='SVR核函数类型')
    parser.add_argument('--svr_C', type=float, default=1.0,
                       help='SVR正则化参数')
    parser.add_argument('--svr_gamma', type=str, default='scale',
                       help='SVR核函数参数')
    parser.add_argument('--svr_epsilon', type=float, default=0.1,
                       help='SVR epsilon-tube参数')

    # Prophet参数
    parser.add_argument('--prophet_yearly_seasonality', action='store_true',
                       help='Prophet年度季节性')
    parser.add_argument('--prophet_weekly_seasonality', action='store_true', default=True,
                       help='Prophet周季节性')
    parser.add_argument('--prophet_daily_seasonality', action='store_true', default=True,
                       help='Prophet日季节性')
    parser.add_argument('--prophet_seasonality_mode', type=str, default='additive',
                       help='Prophet季节性模式')

    # LSTM参数
    parser.add_argument('--lstm_hidden_size', type=int, default=128,
                       help='LSTM隐藏层大小')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                       help='LSTM层数')
    parser.add_argument('--lstm_bidirectional', action='store_true',
                       help='LSTM是否双向')

    # TFT参数
    parser.add_argument('--tft_lstm_hidden', type=int, default=128,
                       help='TFT中LSTM隐藏层大小')
    parser.add_argument('--tft_num_quantiles', type=int, default=3,
                       help='TFT分位数数量')

    # =============== LoRA配置 ===============
    parser.add_argument('--use_lora', action='store_true', default=False,
                       help='是否使用LoRA')
    parser.add_argument('--lora_rank', type=int, default=16,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout')

    # =============== 联邦学习配置 ===============
    parser.add_argument('--frac', type=float, default=0.3,
                       help='每轮参与的客户端比例')
    parser.add_argument('--local_ep', type=int, default=10,
                       help='客户端本地训练步数')
    parser.add_argument('--local_bs', type=int, default=32,
                       help='本地批处理大小')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--epochs', type=int, default=20,
                       help='联邦训练轮数或经典模型训练轮数')
    parser.add_argument('--eval_interval', type=int, default=2,
                       help='评估间隔')
    parser.add_argument('--personalized_epochs', type=int, default=5,
                       help='个性化微调轮数')

    # =============== 显存优化配置 ===============
    parser.add_argument('--client_batch_size', type=int, default=1,
                       help='个性化阶段客户端批处理大小')

    # =============== 训练模式配置 ===============
    parser.add_argument('--training_mode', type=str, default='auto',
                       choices=['auto', 'federated', 'distributed', 'centralized'],
                       help='训练模式: auto(自动选择), federated(联邦学习), distributed(分布式), centralized(集中式)')

    # =============== 其他配置 ===============
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--prompt_domain', action='store_true',
                       help='是否使用域提示')
    parser.add_argument('--content', type=str, default="The dataset records the wireless traffic of a certain base station",
                       help='数据集描述')
    parser.add_argument('--save_dir', type=str, default='./experiments',
                       help='结果保存目录')
    parser.add_argument('--experiment_name', type=str, default='perfedllm_optimized',
                       help='实验名称')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备 (cpu/cuda/auto)')
    parser.add_argument('--save_detailed', type=str, default=None,
                        help='为每一个客户端保存预测值真实值')

    # =============== 联邦算法配置 ===============
    parser.add_argument('--fed_algorithm', type=str, default='perfedavg',
                       choices=['fedavg', 'fedprox', 'perfedavg'],
                       help='选择联邦学习算法')
    parser.add_argument('--fedprox_mu', type=float, default=0.01,
                       help='FedProx正则化参数')
    parser.add_argument('--calculate_communication', action='store_true', default=False,
                       help='是否计算通信量')

    return parser.parse_args()

def getModel(args):
    """获取模型"""
    args.dec_in = 1
    args.c_out = 1
    args.e_layers = 2
    args.d_layers = 1
    args.embed = 'timeF'
    args.freq = 'h'
    args.activation = 'gelu'

    model_type = args.model_type.lower()

    # 经典模型
    if model_type == 'arima':
        if ARIMAModel is None:
            raise ImportError("ARIMA模型不可用，请安装statsmodels: pip install statsmodels")
        args = setup_classical_model_args(args, 'arima')
        return ARIMAModel, args

    elif model_type == 'lasso':
        if LassoModel is None:
            raise ImportError("Lasso模型不可用，请安装scikit-learn: pip install scikit-learn")
        args = setup_classical_model_args(args, 'lasso')
        return LassoModel, args

    elif model_type == 'svr':
        if SVRModel is None:
            raise ImportError("SVR模型不可用，请安装scikit-learn: pip install scikit-learn")
        args = setup_classical_model_args(args, 'svr')
        return SVRModel, args

    elif model_type == 'prophet':
        if ProphetModel is None:
            raise ImportError("Prophet模型不可用，请安装prophet: pip install prophet")
        args = setup_classical_model_args(args, 'prophet')
        return ProphetModel, args

    elif model_type == 'lstm':
        if LSTMModel is None:
            raise ImportError("LSTM模型不可用")
        args = setup_classical_model_args(args, 'lstm')
        return LSTMModel, args

    elif model_type == 'tft':
        if TFTModel is None:
            raise ImportError("TFT模型不可用")
        args = setup_classical_model_args(args, 'tft')
        return TFTModel, args

    # 现有的深度学习模型
    elif args.model_type == 'dLinear':
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
        raise ValueError(f"Unknown model: {args.model_type}")

    return model_class, args


def is_classical_model(model_type: str) -> bool:
    """判断是否为经典模型"""
    classical_models = ['arima', 'lasso', 'svr', 'prophet', 'lstm', 'tft']
    return model_type.lower() in classical_models


def determine_training_mode(args) -> str:
    """确定训练模式"""
    if args.training_mode != 'auto':
        return args.training_mode

    # 自动选择训练模式
    if is_classical_model(args.model_type):
        return 'distributed'  # 经典模型默认使用分布式训练
    else:
        return 'federated'    # 深度学习模型默认使用联邦学习


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 设置日志
    logger = setup_logging(args.log_level)

    logger.info("=== PerFedLLM & 经典模型训练开始 ===")
    logger.info(f"实验配置:")
    logger.info(f"  模型类型: {args.model_type}")
    logger.info(f"  数据集: {args.file_path}")
    logger.info(f"  数据类型: {args.data_type}")
    logger.info(f"  客户端数量: {args.num_clients}")
    logger.info(f"  序列长度: seq_len={args.seq_len}, label_len={args.label_len}, pred_len={args.pred_len}")
    logger.info(f"  设备: {args.device}")

    # 确定训练模式
    training_mode = determine_training_mode(args)
    logger.info(f"  训练模式: {training_mode}")

    try:
        # 1. 加载联邦数据
        logger.info("=== 步骤1: 加载数据 ===")
        federated_data, data_loader_factory = get_federated_data(args)

        # 2. 获取模型
        logger.info("=== 步骤2: 设置模型 ===")
        model_class, args = getModel(args)

        # 3. 根据训练模式选择训练器
        if training_mode == 'distributed':
            logger.info("=== 步骤3: 创建分布式训练器 ===")
            trainer = ClassicalModelTrainer(args, logger)
            trainer.setup_data(federated_data, data_loader_factory)
            trainer.setup_model(model_class, args)
            logger.info("=== 步骤4: 开始分布式训练 ===")
            results = trainer.train_distributed()

        elif training_mode == 'centralized':
            logger.info("=== 步骤3: 创建集中式训练器 ===")
            trainer = CentralizedTrainer(args, logger)
            trainer.setup_data(federated_data, data_loader_factory)
            trainer.setup_model(model_class, args)
            results = trainer.train()

        else:  # federated mode
            logger.info("=== 步骤3: 创建联邦学习训练器 ===")
            trainer = PerFedLLMTrainerOptimized(args, logger)
            logger.info("=== 步骤4: 设置数据 ===")
            trainer.setup_data(federated_data, data_loader_factory)
            logger.info("=== 步骤5: 设置模型 ===")
            trainer.setup_model(model_class, args)
            logger.info("=== 步骤6: 设置客户端和服务器 ===")
            trainer.setup_clients()
            trainer.setup_server()
            logger.info("=== 步骤7: 开始联邦训练 ===")
            results = trainer.train()

        # 8. 保存结果
        logger.info("=== 步骤8: 保存结果 ===")
        save_dir = os.path.join(args.save_dir, args.experiment_name)
        os.makedirs(save_dir, exist_ok=True)

        if training_mode == 'distributed':
            # 分布式训练结果保存
            trainer.save_results(results, save_dir)

            # 输出统计信息
            if results['successful_clients'] > 0:
                logger.info("=== 分布式训练完成 ===")
                logger.info(f"成功训练客户端: {results['successful_clients']}/{results['total_clients']}")
                logger.info(f"平均指标:")
                logger.info(f"  MSE: {results['average_metrics']['avg_mse']:.6f}")
                logger.info(f"  MAE: {results['average_metrics']['avg_mae']:.6f}")
                logger.info(f"  RMSE: {results['average_metrics']['avg_rmse']:.6f}")
            else:
                logger.warning("没有客户端成功完成训练")

        elif training_mode == 'centralized':
            trainer.save_results(results, save_dir)
            logger.info("=== 集中式训练完成 ===")
            logger.info(f"测试指标: MSE={results['test_metrics']['mse']:.6f}, MAE={results['test_metrics']['mae']:.6f}")

        else:
            # 联邦学习结果保存
            # 保存全局模型
            if hasattr(trainer, 'global_model'):
                torch.save(trainer.global_model.state_dict(),
                          os.path.join(save_dir, 'global_model.pth'))

                # 保存训练曲线数据为CSV格式
                training_history = results.get('training_history', {})
                if training_history.get('round_metrics'):
                    import pandas as pd

                    # 准备CSV数据
                    csv_data = []
                    round_metrics = training_history['round_metrics']
                    round_losses = training_history.get('round_losses', [])

                    for i, metrics in enumerate(round_metrics):
                        row = {
                            'round': i + 1,
                            'avg_mse': metrics.get('avg_mse', None),
                            'avg_mae': metrics.get('avg_mae', None),
                            'avg_rmse': metrics.get('avg_rmse', None),
                            'std_mse': metrics.get('std_mse', None),
                            'std_mae': metrics.get('std_mae', None),
                            'std_rmse': metrics.get('std_rmse', None),
                        }

                        # 如果有对应的训练损失数据，也加入
                        if i < len(round_losses):
                            row['training_loss'] = round_losses[i]

                        csv_data.append(row)

                    # 保存CSV文件
                    if csv_data:
                        df = pd.DataFrame(csv_data)
                        csv_path = os.path.join(save_dir, 'training_curves.csv')
                        df.to_csv(csv_path, index=False)
                        logger.info(f"训练曲线数据已保存到: {csv_path}")

            # 保存训练结果
            def convert_tensors(obj):
                """递归转换tensor为列表"""
                if isinstance(obj, dict):
                    return {k: convert_tensors(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_tensors(item) for item in obj]
                elif isinstance(obj, torch.Tensor):
                    return obj.cpu().numpy().tolist()
                elif hasattr(obj, 'item'):
                    return obj.item()
                else:
                    return obj

            results_serializable = convert_tensors(results)

            with open(os.path.join(save_dir, 'training_results.json'), 'w', encoding='utf-8') as f:
                json.dump(results_serializable, f, indent=2, ensure_ascii=False)

            # 保存配置
            with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
                json.dump(vars(args), f, indent=2, ensure_ascii=False)

            # 输出联邦学习结果
            logger.info("=== 联邦训练完成 ===")

            # 联邦训练结果
            if results.get('federated_metrics'):
                logger.info("联邦训练最终指标:")
                for metric, value in results['federated_metrics'].items():
                    logger.info(f"  {metric}: {value:.6f}")

            # 个性化结果统计
            personalized_results = results.get('personalized_metrics', {})
            valid_results = [r for r in personalized_results.values() if 'error' not in r]

            if valid_results:
                logger.info(f"\n=== 个性化测试结果统计 ===")
                logger.info(f"成功完成个性化的客户端: {len(valid_results)}/{len(personalized_results)}")

                # 计算平均指标
                import numpy as np
                avg_metrics = {}
                for metric in valid_results[0].keys():
                    values = [r[metric] for r in valid_results]
                    avg_metrics[metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }

                logger.info("各客户端个性化测试指标:")
                for client_id, metrics in personalized_results.items():
                    if 'error' not in metrics:
                        logger.info(f"  客户端{client_id}: MSE={metrics['mse']:.6f}, MAE={metrics['mae']:.6f}, RMSE={metrics['rmse']:.6f}")

                logger.info("\n最终平均指标:")
                for metric, stats in avg_metrics.items():
                    logger.info(f"  {metric.upper()}:")
                    logger.info(f"    平均值: {stats['mean']:.6f} ± {stats['std']:.6f}")
                    logger.info(f"    范围: [{stats['min']:.6f}, {stats['max']:.6f}]")

                # 保存简化的汇总结果
                summary_results = {
                    'experiment_name': args.experiment_name,
                    'model_type': args.model_type,
                    'training_mode': training_mode,
                    'num_clients': args.num_clients,
                    'successful_clients': len(valid_results),
                    'sequence_config': {
                        'seq_len': args.seq_len,
                        'label_len': args.label_len,
                        'pred_len': args.pred_len
                    },
                    'average_metrics': {k: v['mean'] for k, v in avg_metrics.items()},
                    'std_metrics': {k: v['std'] for k, v in avg_metrics.items()},
                    'federated_final_metrics': results.get('federated_metrics', {}),
                    'personalized_improvement': {}
                }

                # 计算个性化相对于联邦的改进
                fed_metrics = results.get('federated_metrics', {})
                for metric in ['mse', 'mae', 'rmse']:
                    fed_key = f'avg_{metric}'
                    if fed_key in fed_metrics and metric in avg_metrics:
                        fed_value = fed_metrics[fed_key]
                        pers_value = avg_metrics[metric]['mean']
                        improvement = (fed_value - pers_value) / fed_value * 100
                        summary_results['personalized_improvement'][metric] = improvement
                        logger.info(f"  {metric.upper()} 个性化改进: {improvement:+.2f}%")

                # 保存汇总结果
                with open(os.path.join(save_dir, 'summary_results.json'), 'w', encoding='utf-8') as f:
                    json.dump(summary_results, f, indent=2, ensure_ascii=False)

            else:
                logger.warning("没有客户端成功完成个性化微调")

        logger.info(f"\n所有结果已保存到: {save_dir}")
        logger.info(f"训练模式: {training_mode}")

    except Exception as e:
        logger.error(f"训练过程出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # 最终清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)