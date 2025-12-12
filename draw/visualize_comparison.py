import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


def plot_comparison(file1_path: str, name1: str, file2_path: str, name2: str,
                    sample_id: int, data_type: str, save_path: Optional[str] = None):
    """
    加载两个模型的预测结果JSON文件，并绘制指定样本的对比图。

    Args:
        file1_path (str): 第一个结果文件的路径。
        name1 (str): 第一个模型的名称 (用于图例)。
        file2_path (str): 第二个结果文件的路径。
        name2 (str): 第二个模型的名称 (用于图例)。
        sample_id (int): 要可视化的样本ID。
        data_type (str): 'normalized' 或 'denormalized'，指定要使用的数据类型。
        save_path (Optional[str]): 图表的保存路径。如果提供，图表将被保存。
    """
    try:
        # 加载两个JSON文件
        with open(file1_path, 'r', encoding='utf-8') as f:
            data1 = json.load(f)
        with open(file2_path, 'r', encoding='utf-8') as f:
            data2 = json.load(f)
    except FileNotFoundError as e:
        print(f"错误: 文件未找到 - {e.filename}")
        return
    except json.JSONDecodeError:
        print("错误: 文件不是有效的JSON格式。")
        return

    # 根据sample_id查找对应的样本数据
    sample1 = next((s for s in data1.get('samples', []) if s.get('sample_id') == sample_id), None)
    sample2 = next((s for s in data2.get('samples', []) if s.get('sample_id') == sample_id), None)

    if not sample1 or not sample2:
        print(f"错误: 在一个或两个文件中都找不到 Sample ID: {sample_id}")
        return

    # 提取绘图所需数据
    history = sample1[data_type]['history']
    ground_truth = sample1[data_type]['ground_truth']
    prediction1 = sample1[data_type]['prediction']
    prediction2 = sample2[data_type]['prediction']

    # 准备X轴坐标
    len_history = len(history)
    len_pred = len(ground_truth)
    x_history = range(len_history)
    x_pred = range(len_history, len_history + len_pred)

    # ========== 美化设置开始 ==========

    # 设置专业的图表样式
    plt.style.use('default')  # 使用默认样式，然后自定义
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'legend.framealpha': 0.95,
        'legend.fancybox': True,
        'legend.shadow': True
    })

    # 创建图形，使用黄金比例
    fig, ax = plt.subplots(figsize=(14, 8.5))

    # 专业配色方案
    colors = {
        'history': '#95A5A6',  # 优雅灰色
        'ground_truth': '#2E86AB',  # 专业蓝色
        'prediction1': '#F18F01',  # 暖橙色
        'prediction2': '#C73E1D'  # 深红色
    }

    # 绘制历史值 - 使用更细的线和透明度
    ax.plot(x_history, history, label='Historical Data',
            color=colors['history'], linewidth=6, alpha=0.8,
            zorder=1)

    # 绘制真实值 - 使用填充标记
    ax.plot(x_pred, ground_truth, label='Ground Truth',
            color=colors['ground_truth'],linewidth=6, markeredgewidth=2,
            zorder=4)

    # 绘制第一个模型的预测值 - 使用虚线
    ax.plot(x_pred, prediction1, label=f'{name1}',
            color=colors['prediction1'], linewidth=6,alpha=0.9,
            zorder=3)

    # 绘制第二个模型的预测值 - 使用点划线
    ax.plot(x_pred, prediction2, label=f'{name2}',
            color=colors['prediction2'], linewidth=6, alpha=0.9,
            zorder=2)

    # 添加预测区域的背景阴影
    y_min, y_max = ax.get_ylim()
    ax.axvspan(len_history, len_history + len_pred,
               alpha=0.1, color=colors['prediction1'], zorder=0)

    # 添加分界线
    ax.axvline(x=len_history, color='black', linestyle=':',
               linewidth=1.5, alpha=0.6)

    # 美化坐标轴
    ax.set_xlabel('Time Steps', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel(f'Traffic Volume ({data_type.capitalize()})',
                  fontsize=14, fontweight='bold', labelpad=10)

    # 美化标题
    ax.set_title(f'Traffic Prediction Comparison\nSample ID: {sample_id}',
                 fontsize=16, fontweight='bold', pad=20)

    # 设置图例 - 更专业的样式
    legend = ax.legend(loc='upper right', fontsize=30, frameon=True,
                      fancybox=True, shadow=True, framealpha=0.95,
                      edgecolor='black', facecolor='white')
    legend.get_frame().set_linewidth(1.2)

    # 美化网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # 设置坐标轴范围，留出适当边距
    x_margin = (max(x_pred) - min(x_history)) * 0.02
    ax.set_xlim(min(x_history) - x_margin, max(x_pred) + x_margin)

    y_range = max(max(history), max(ground_truth), max(prediction1), max(prediction2)) - \
              min(min(history), min(ground_truth), min(prediction1), min(prediction2))
    y_margin = y_range * 0.1
    ax.set_ylim(min(min(history), min(ground_truth), min(prediction1), min(prediction2)) - y_margin,
                max(max(history), max(ground_truth), max(prediction1), max(prediction2)) + y_margin)

    # 添加文本注释
    # ax.text(0.02, 0.98, f'Historical Length: {len_history}\nPrediction Length: {len_pred}',
    #         transform=ax.transAxes, fontsize=10, verticalalignment='top',
    #         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

    # 调整布局
    plt.tight_layout()

    # ========== 美化设置结束 ==========

    # 如果用户指定了保存路径，则保存图表
    if save_path:
        try:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"图表已成功保存到: {save_path}")
        except Exception as e:
            print(f"错误: 无法保存图表到 {save_path}。原因: {e}")

    # 显示图表
    # plt.show()


if __name__ == "__main__":
    # 使用argparse设置命令行参数解析
    parser = argparse.ArgumentParser(description="Visualize and compare prediction results from two models.")

    parser.add_argument('--file1', type=str, required=True, help='Path to the first JSON result file.')
    parser.add_argument('--name1', type=str, required=True,
                        help='Name of the model for the first file (e.g., TimeLLM).')

    parser.add_argument('--file2', type=str, required=True, help='Path to the second JSON result file.')
    parser.add_argument('--name2', type=str, required=True,
                        help='Name of the model for the second file (e.g., DLinear).')

    parser.add_argument('--sample_id', type=int, required=True, help='The ID of the sample to visualize.')

    parser.add_argument('--data_type', type=str, default='normalized', choices=['normalized', 'denormalized'],
                        help="Data type to plot: 'normalized' or 'denormalized'. Default is 'normalized'.")

    parser.add_argument('--output', type=str, default=None,
                        help="Optional path to save the plot image (e.g., 'comparison.png').")

    args = parser.parse_args()

    # 调用主函数
    plot_comparison(args.file1, args.name1, args.file2, args.name2,
                    args.sample_id, args.data_type, args.output)