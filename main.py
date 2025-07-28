"""
PerFedLLM 主启动脚本
"""

import argparse
import logging
import os
import sys
from dataset.data_loader import get_federated_data
from models.TimeLLM import Model as TimeLLMModel
from trainer import PerFedLLMTrainer
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
    parser = argparse.ArgumentParser(description='PerFedLLM: 基于LLM的个性化联邦学习')

    # =============== 数据配置 ===============
    parser.add_argument('--dataset_dir', type=str, default='./dataset',
                       help='数据集目录')
    parser.add_argument('--file_path', type=str, default='milano.h5',
                       help='数据文件路径')
    parser.add_argument('--data_type', type=str, default='net',
                        help='流量类型 (net/call/sms)')
    parser.add_argument('--num_clients', type=int, default=10,
                       help='客户端数量')
    parser.add_argument('--seq_len', type=int, default=96,
                       help='输入序列长度')
    parser.add_argument('--pred_len', type=int, default=24,
                       help='预测序列长度')
    parser.add_argument('--test_days', type=int, default=7,
                       help='测试数据天数')
    parser.add_argument('--val_days', type=int, default=3,
                       help='验证数据天数')

    # =============== 模型配置 ===============
    parser.add_argument('--model_type', type=str, default='timellm',
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

    # =============== LoRA配置 ===============
    parser.add_argument('--use_lora', action='store_true', default=True,
                       help='是否使用LoRA')
    parser.add_argument('--lora_rank', type=int, default=8,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                       help='LoRA dropout')

    # =============== 联邦学习配置 ===============
    parser.add_argument('--frac', type=float, default=0.6,
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
                       help='联邦训练轮数')
    parser.add_argument('--eval_interval', type=int, default=2,
                       help='评估间隔')
    parser.add_argument('--personalized_epochs', type=int, default=5,
                       help='个性化微调轮数')

    # =============== 数据增强配置 ===============
    parser.add_argument('--enable_augmentation', action='store_true',
                       help='是否启用数据增强')
    parser.add_argument('--mixup_prob', type=float, default=0.2,
                       help='Mixup概率')
    parser.add_argument('--jittering_prob', type=float, default=0.15,
                       help='抖动概率')
    parser.add_argument('--scaling_prob', type=float, default=0.1,
                       help='缩放概率')
    parser.add_argument('--augmentation_ratio', type=float, default=0.3,
                       help='数据增强比例')
    parser.add_argument('--similarity_threshold', type=float, default=0.6,
                       help='相似性阈值')
    parser.add_argument('--candidate_pool_size', type=int, default=5,
                       help='候选池大小')
    parser.add_argument('--augmentation_lambda_min', type=float, default=0.6,
                       help='增强lambda最小值')
    parser.add_argument('--augmentation_lambda_max', type=float, default=0.8,
                       help='增强lambda最大值')
    parser.add_argument('--enable_regularization_constraints', action='store_true', default=True,
                       help='启用正则化约束')
    parser.add_argument('--max_deviation_ratio', type=float, default=0.3,
                       help='最大偏离比例')
    parser.add_argument('--min_correlation_threshold', type=float, default=0.5,
                       help='最小相关性阈值')
    parser.add_argument('--constraint_correction_weight', type=float, default=0.3,
                       help='约束修正权重')

    # =============== 其他配置 ===============
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--prompt_domain', action='store_true',
                       help='是否使用域提示')
    parser.add_argument('--content', type=str, default="The dataset records the wireless traffic of a certain base station",
                       help='数据集描述')
    parser.add_argument('--save_dir', type=str, default='./experiments',
                       help='结果保存目录')
    parser.add_argument('--experiment_name', type=str, default='perfedllm_default',
                       help='实验名称')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备 (cpu/cuda/auto)')

    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 设置日志
    logger = setup_logging(args.log_level)

    logger.info("=== PerFedLLM 训练开始 ===")
    logger.info(f"实验配置:")
    logger.info(f"  LLM模型: {args.llm_model}")
    logger.info(f"  使用LoRA: {args.use_lora}")
    logger.info(f"  客户端数量: {args.num_clients}")
    logger.info(f"  联邦轮数: {args.epochs}")
    logger.info(f"  个性化轮数: {args.personalized_epochs}")
    logger.info(f"  设备: {args.device}")

    try:
        # 1. 加载联邦数据
        logger.info("=== 步骤1: 加载联邦数据 ===")
        federated_data, data_loader_factory = get_federated_data(args)

        # 2. 创建训练器
        logger.info("=== 步骤2: 创建训练器 ===")
        trainer = PerFedLLMTrainer(args, logger)

        # 3. 设置数据
        logger.info("=== 步骤3: 设置数据 ===")
        trainer.setup_data(federated_data, data_loader_factory)

        # 4. 设置模型
        logger.info("=== 步骤4: 设置模型 ===")
        trainer.setup_model(TimeLLMModel, args)

        # 5. 设置客户端和服务器
        logger.info("=== 步骤5: 设置客户端和服务器 ===")
        trainer.setup_clients()
        trainer.setup_server()

        # 6. 开始训练
        logger.info("=== 步骤6: 开始训练 ===")
        results = trainer.train()

        # 7. 保存结果（只保存全局模型和测试指标）
        logger.info("=== 步骤7: 保存结果 ===")
        save_dir = os.path.join(args.save_dir, args.experiment_name)
        os.makedirs(save_dir, exist_ok=True)

        # 只保存全局模型（节省存储空间）
        torch.save(trainer.global_model.state_dict(),
                  os.path.join(save_dir, 'global_model.pth'))

        # 保存训练结果（重点是个性化测试指标）
        def convert_tensors(obj):
            """递归转换tensor为列表"""
            if isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(item) for item in obj]
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            else:
                return obj

        results_serializable = convert_tensors(results)

        with open(os.path.join(save_dir, 'training_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)

        # 保存配置
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, indent=2, ensure_ascii=False)

        # 8. 输出最终结果
        logger.info("=== 训练完成 ===")

        # 联邦训练结果
        if results['federated_metrics']:
            logger.info("联邦训练最终指标:")
            for metric, value in results['federated_metrics'].items():
                logger.info(f"  {metric}: {value:.6f}")

        # 个性化结果统计
        personalized_results = results['personalized_metrics']
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
                'llm_model': args.llm_model,
                'use_lora': args.use_lora,
                'num_clients': args.num_clients,
                'successful_clients': len(valid_results),
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
        logger.info("注意: 为节省存储空间，只保存了全局模型，客户端个性化模型仅用于测试后即删除")

    except Exception as e:
        logger.error(f"训练过程出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
