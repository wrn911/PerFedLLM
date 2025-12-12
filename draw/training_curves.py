import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置科学期刊风格
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8

# 读取数据文件
try:
    # TimeLLM+FedAvg数据
    timellm_call = pd.read_csv('timellm_milano_call_fedavg_training_curves.csv')
    timellm_net = pd.read_csv('timellm_milano_net_fedavg_training_curves.csv')
    timellm_sms = pd.read_csv('timellm_milano_sms_fedavg_training_curves.csv')

    # FedLLM-WTP数据
    fedllm_call = pd.read_csv('FedLLM-WTP_milano_call_training_curves.csv')
    fedllm_net = pd.read_csv('FedLLM-WTP_milano_net_training_curves.csv')
    fedllm_sms = pd.read_csv('FedLLM-WTP_milano_sms_training_curves.csv')

    print("数据读取成功!")
    print(f"TimeLLM数据轮数: {len(timellm_call)}, FedLLM-WTP数据轮数: {len(fedllm_call)}")

except FileNotFoundError as e:
    print(f"文件未找到: {e}")


# 创建2x3子图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# fig.suptitle('Training Convergence Comparison: FedLLM-WTP vs TimeLLM+FedAvg',
#              fontsize=14, fontweight='bold', y=0.95)

# 数据和标题
data_pairs = [
    (timellm_call, fedllm_call, 'Call'),
    (timellm_net, fedllm_net, 'Net'),
    (timellm_sms, fedllm_sms, 'SMS')
]

colors = ['#FF6B6B', '#4ECDC4']  # 红色为TimeLLM+FedAvg, 青色为FedLLM-WTP
linestyles = ['-', '--']
markers = ['o', 's']

# 绘制MSE图 (第一行)
for i, (timellm_data, fedllm_data, task_type) in enumerate(data_pairs):
    ax = axes[0, i]

    # TimeLLM+FedAvg
    ax.errorbar(timellm_data['round'], timellm_data['avg_mse'],
                yerr=timellm_data['std_mse'],
                color=colors[0], linestyle=linestyles[0], marker=markers[0],
                markersize=6, linewidth=2, capsize=4, capthick=1,
                label='TimeLLM+FedAvg', alpha=0.8)

    # FedLLM-WTP
    ax.errorbar(fedllm_data['round'], fedllm_data['avg_mse'],
                yerr=fedllm_data['std_mse'],
                color=colors[1], linestyle=linestyles[1], marker=markers[1],
                markersize=6, linewidth=2, capsize=4, capthick=1,
                label='FedLLM-WTP (Ours)', alpha=0.8)

    ax.set_title(f'({chr(97 + i)}) {task_type} - MSE', fontsize=12, fontweight='bold')
    ax.set_xlabel('Communication Round', fontsize=11)
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=15, frameon=True, fancybox=True,
              markerscale=1.2, handlelength=2.5)

    # 设置x轴刻度
    max_round = max(timellm_data['round'].max(), fedllm_data['round'].max())
    ax.set_xlim(0.5, max_round + 0.5)
    ax.set_xticks(range(1, max_round + 1))

# 绘制MAE图 (第二行)
for i, (timellm_data, fedllm_data, task_type) in enumerate(data_pairs):
    ax = axes[1, i]

    # TimeLLM+FedAvg
    ax.errorbar(timellm_data['round'], timellm_data['avg_mae'],
                yerr=timellm_data['std_mae'],
                color=colors[0], linestyle=linestyles[0], marker=markers[0],
                markersize=6, linewidth=2, capsize=4, capthick=1,
                label='TimeLLM+FedAvg', alpha=0.8)

    # FedLLM-WTP
    ax.errorbar(fedllm_data['round'], fedllm_data['avg_mae'],
                yerr=fedllm_data['std_mae'],
                color=colors[1], linestyle=linestyles[1], marker=markers[1],
                markersize=6, linewidth=2, capsize=4, capthick=1,
                label='FedLLM-WTP (Ours)', alpha=0.8)

    ax.set_title(f'({chr(100 + i)}) {task_type} - MAE', fontsize=12, fontweight='bold')
    ax.set_xlabel('Communication Round', fontsize=11)
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=15, frameon=True, fancybox=True,
              markerscale=1.2, handlelength=2.5)

    # 设置x轴刻度
    max_round = max(timellm_data['round'].max(), fedllm_data['round'].max())
    ax.set_xlim(0.5, max_round + 0.5)
    ax.set_xticks(range(1, max_round + 1))

# 调整子图间距
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.show()

# 打印最终性能对比
print("\n=== Final Performance Comparison ===")
for i, (timellm_data, fedllm_data, task_type) in enumerate(data_pairs):
    timellm_final_mse = timellm_data['avg_mse'].iloc[-1]
    fedllm_final_mse = fedllm_data['avg_mse'].iloc[-1]
    timellm_final_mae = timellm_data['avg_mae'].iloc[-1]
    fedllm_final_mae = fedllm_data['avg_mae'].iloc[-1]

    mse_improvement = (timellm_final_mse - fedllm_final_mse) / timellm_final_mse * 100
    mae_improvement = (timellm_final_mae - fedllm_final_mae) / timellm_final_mae * 100

    print(f"{task_type}:")
    print(
        f"  MSE: TimeLLM+FedAvg={timellm_final_mse:.4f}, FedLLM-WTP={fedllm_final_mse:.4f} (改进:{mse_improvement:.1f}%)")
    print(
        f"  MAE: TimeLLM+FedAvg={timellm_final_mae:.4f}, FedLLM-WTP={fedllm_final_mae:.4f} (改进:{mae_improvement:.1f}%)")