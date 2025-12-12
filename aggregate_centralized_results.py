# PerFedLLM/aggregate_centralized_results.py

"""
集中式训练结果聚合脚本
用于扫描所有集中式训练实验的结果，并将其汇总到一个CSV文件中。
"""

import os
import json
import pandas as pd
import glob
import argparse
import numpy as np

def find_result_files(experiments_dir: str):
    """
    在实验目录下查找所有 summary.json 和 results.json 文件。
    """
    summary_files = glob.glob(os.path.join(experiments_dir, '*', 'summary.json'))
    results_files = glob.glob(os.path.join(experiments_dir, '*', 'results.json'))
    return summary_files + results_files

def process_result_file(json_path: str):
    """
    读取并处理单个结果文件 (summary.json 或 results.json)，提取所需信息。
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        filename = os.path.basename(json_path)
        result_row = {}

        if filename == 'summary.json':
            # 处理由 train_classical_centralized.py 生成的 summary.json
            training_mode = data.get('training_mode', '')
            if 'centralized' not in training_mode: # 确保只收集集中式训练
                return None
            
            metrics = data.get('final_metrics', {})
            if not metrics:
                return None

            dataset_full_path = data.get('dataset', 'unknown')
            dataset_name = os.path.splitext(os.path.basename(dataset_full_path))[0]
            
            result_row = {
                'model': data.get('model_type', 'N/A'),
                'dataset': dataset_name,
                'data_type': data.get('data_type', 'N/A'),
                'training_mode': training_mode,
                'mse': metrics.get('mse'),
                'mae': metrics.get('mae'),
            }
        elif filename == 'results.json':
            # 处理由 centralized_trainer.py 生成的 results.json
            metrics = data.get('test_metrics', {})
            if not metrics:
                return None

            # 从实验文件夹名称中推断模型、数据集和数据类型
            # 假设文件夹命名格式为: Model_DatasetParts_DataType_...
            exp_folder_name = os.path.basename(os.path.dirname(json_path))
            parts = exp_folder_name.split('_')
            
            if len(parts) >= 3:
                model_type = parts[0]
                
                # 识别 data_type (通常是 'net', 'call', 'sms', 'traffic')
                known_data_types = ['net', 'call', 'sms', 'traffic']
                inferred_data_type = None
                data_type_index = -1

                # 从第三个部分开始向后查找已知数据类型
                for i in range(2, len(parts)):
                    if parts[i].lower() in known_data_types:
                        inferred_data_type = parts[i]
                        data_type_index = i
                        break
                
                if inferred_data_type is None:
                    print(f"警告: 无法从文件夹名 '{exp_folder_name}' 推断数据类型。跳过文件 {json_path}")
                    return None
                
                data_type = inferred_data_type
                # dataset_name 是 model_type 和 data_type 之间的所有部分
                dataset_name = "_".join(parts[1:data_type_index])

                # 特别处理TimeLLM系列模型，以包含LLM的名称
                if model_type.lower() in ['timellm', 'simpletimellm'] and len(parts) > data_type_index + 1:
                    # 假设LLM名称是data_type_index之后，'centralized'之前的最后一个部分
                    llm_candidate_parts = parts[data_type_index+1:]
                    # 查找'centralized'的位置
                    centralized_idx = -1
                    for i, part in enumerate(llm_candidate_parts):
                        if part == 'centralized':
                            centralized_idx = i
                            break
                    
                    if centralized_idx != -1 and centralized_idx + 1 < len(llm_candidate_parts):
                         llm_name = llm_candidate_parts[centralized_idx + 1] # centralized_GPT2
                    elif centralized_idx == -1 and len(llm_candidate_parts) > 0:
                         llm_name = llm_candidate_parts[-1] # 如果没有centralized，就取最后一个

                    if 'llm_name' in locals() and llm_name.upper() in ['GPT2', 'BERT', 'DEEPSEEK', 'QWEN3']:
                         model_type = f"{model_type}-{llm_name}"

                training_mode = 'centralized'
            else:
                print(f"警告: 文件夹名 '{exp_folder_name}' 格式不正确，无法推断模型/数据集/数据类型。跳过文件 {json_path}")
                return None

            result_row = {
                'model': model_type,
                'dataset': dataset_name,
                'data_type': data_type,
                'training_mode': 'centralized', # 直接标记为集中式
                'mse': metrics.get('mse'),
                'mae': metrics.get('mae'),
            }
        else:
            return None # 忽略其他文件

        # 统一计算RMSE
        mse_val = result_row.get('mse')
        result_row['rmse'] = np.sqrt(mse_val) if mse_val is not None and mse_val != 'inf' else None
        
        return result_row
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"警告: 处理文件 {json_path} 时出错: {e}")
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='聚合集中式训练的实验结果并保存为CSV')
    parser.add_argument('--experiments_dir', type=str, default='./experiments',
                        help='包含所有实验结果的父目录')
    parser.add_argument('--output_csv', type=str, default='./centralized_training_results.csv',
                        help='输出的CSV文件路径')
    
    args = parser.parse_args()

    print(f"正在扫描目录: {args.experiments_dir}")
    
    result_files = find_result_files(args.experiments_dir)
    
    if not result_files:
        print(f"错误: 在 '{args.experiments_dir}' 中没有找到任何 summary.json 或 results.json 文件。")
        return

    print(f"找到了 {len(result_files)} 个结果文件。正在处理...")

    all_results = []
    for file_path in result_files:
        result = process_result_file(file_path)
        if result:
            all_results.append(result)

    if not all_results:
        print("错误: 没有找到任何有效的集中式训练结果。")
        return

    # 创建DataFrame
    df = pd.DataFrame(all_results)
    
    # 定义列顺序，使输出更美观
    columns = ['model', 'dataset', 'data_type', 'training_mode', 'mse', 'mae', 'rmse']
    # 过滤掉数据中可能不存在的列
    existing_columns = [col for col in columns if col in df.columns]
    df = df[existing_columns]

    # 按模型和数据集排序
    df.sort_values(by=['model', 'dataset', 'data_type'], inplace=True)
    
    # 保存到CSV
    try:
        df.to_csv(args.output_csv, index=False, float_format='%.6f')
        print(f"聚合结果已成功保存到: {args.output_csv}")
        print("\n--- 结果预览 ---")
        print(df.to_string())
        print("-----------------")
    except Exception as e:
        print(f"错误: 保存CSV文件失败: {e}")

if __name__ == "__main__":
    main()
