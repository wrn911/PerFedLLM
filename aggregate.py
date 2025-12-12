import os
import json


def collect_statistical_results():
    """
    遍历 'experiments' 文件夹，为每个实验统一从 'average_metrics'
    和 'std_metrics' 对象中提取 mse 和 mae 的均值与标准差。
    """
    all_results = []
    base_path = './experiments'

    # 检查 'experiments' 目录是否存在
    if not os.path.isdir(base_path):
        print(f"错误: 在当前目录 '{os.getcwd()}' 中未找到 '{base_path}' 文件夹。")
        return None

    print(f"正在从 '{os.path.abspath(base_path)}' 目录收集统计数据...")

    known_federated_methods = ['fedavg', 'fedprox', 'perfedavg']

    # 遍历所有实验文件夹
    for dirname in os.listdir(base_path):
        full_path = os.path.join(base_path, dirname)

        if os.path.exists(os.path.join(full_path, 'summary_results.json')):
            summary_file = os.path.join(full_path, 'summary_results.json')
            print(f"正在处理文件夹: {full_path}")
            if os.path.exists(summary_file):
                try:
                    # 读取 JSON 文件内容
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 解析元数据（模型、城市、数据集、训练方法）
                    parts = dirname.split('_')
                    model_name = parts[0]
                    city = parts[1]
                    dataset_type = parts[2]

                    training_method = ''  # 分布式训练默认为空字符串
                    if len(parts) > 3 and parts[-1].lower() in known_federated_methods:
                        training_method = parts[-1].lower()

                    # 提取均值
                    avg_metrics = data.get('average_metrics', {})
                    mse_mean = avg_metrics.get('mse')
                    mae_mean = avg_metrics.get('mae')

                    # 提取标准差
                    std_metrics = data.get('std_metrics', {})
                    mse_std = std_metrics.get('mse')
                    mae_std = std_metrics.get('mae')

                    # 检查是否所有四个值都成功提取
                    if all(v is not None for v in [mse_mean, mae_mean, mse_std, mae_std]):
                        all_results.append({
                            'model': model_name,
                            'city': city,
                            'dataset': dataset_type,
                            'training_method': training_method,
                            'mse_mean': mse_mean,
                            'mae_mean': mae_mean,
                            'mse_std': mse_std,
                            'mae_std': mae_std
                        })
                    else:
                        print(
                            f"警告: 在文件 '{summary_file}' 的 'average_metrics' 或 'std_metrics' 中未能找到完整的 mse/mae 指标。")

                except (json.JSONDecodeError, IndexError) as e:
                    print(f"处理文件 '{summary_file}' 时发生错误: {e}。已跳过。")
        
        elif os.path.exists(os.path.join(full_path, 'summary.json')):
            # 尝试读取新格式的 summary.json
            summary_file = os.path.join(full_path, 'summary.json')
            print(f"正在处理文件夹: {full_path}")

            if os.path.exists(summary_file):
                try:
                    with open(summary_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    # 优先从 experiment_name 解析
                    experiment_name = data.get("experiment_name", "")
                    if not experiment_name:
                        print(f"警告: 文件 '{summary_file}' 中缺少 'experiment_name'，跳过。")
                        continue

                    parts = experiment_name.split('_')
                    if len(parts) < 3:
                        print(f"警告: experiment_name 格式错误: {experiment_name}")
                        continue

                    model_name = parts[0].lower()
                    city = parts[1].lower()
                    dataset_type = parts[2].lower()

                    # 判断是否有联邦学习方法后缀
                    training_method = ''
                    if len(parts) > 3 and parts[-1].lower() in known_federated_methods:
                        training_method = parts[-1].lower()

                    # 提取 final_metrics 中的 mse 和 mae
                    final_metrics = data.get('final_metrics', {})
                    mse_mean = final_metrics.get('mse')
                    mae_mean = final_metrics.get('mae')

                    # 只需要均值，不需要标准差
                    if mse_mean is not None and mae_mean is not None:
                        all_results.append({
                            'model': model_name,
                            'city': city,
                            'dataset': dataset_type,
                            'training_method': training_method,
                            'mse_mean': mse_mean,
                            'mae_mean': mae_mean,
                            # 标准差设置为 0，因为新格式中没有 std_metrics
                            'mse_std': 0,
                            'mae_std': 0
                        })
                    else:
                        print(f"警告: 在文件 '{summary_file}' 的 'final_metrics' 中未能找到 mse 或 mae 指标。")

                except (json.JSONDecodeError, Exception) as e:
                    print(f"处理文件 '{summary_file}' 时发生错误: {e}。已跳过。")
        else:
            print("未找到支持的 summary 文件。")

    return all_results


def main():
    """
    主函数，运行脚本并保存最终的统计结果。
    """
    statistical_results = collect_statistical_results()

    if statistical_results is None:
        return

    if not statistical_results:
        print("\n未能收集到任何有效的实验数据。请检查 'experiments' 文件夹的结构和文件内容。")
        return

    # 定义新的输出文件名
    output_filename = 'statistical_results.json'
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(statistical_results, f, indent=4, ensure_ascii=False)
        print(f"\n✅ 成功生成统计结果文件: '{output_filename}'，共包含 {len(statistical_results)} 条记录。")
    except Exception as e:
        print(f"保存 JSON 文件时出错: {e}")


if __name__ == '__main__':
    main()