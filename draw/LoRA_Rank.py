import matplotlib.pyplot as plt
import numpy as np

# 设置专业的图表样式
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 11,
    'axes.linewidth': 1.0,
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

# 数据定义
ranks = [4, 8, 16, 32, 64]

# BERT数据
bert_data = {
    'call_mse': [0.166091, 0.15561859, 0.15687256, 0.16774761, 0.18083181],
    'net_mse': [0.248336688, 0.243337878, 0.23820321, 0.240012275, 0.236507764],
    'sms_mse': [0.883622032, 0.854489993, 0.845466181, 0.889479903, 0.942660233],
    'call_mae': [0.295211, 0.281006529, 0.280857064, 0.291808138, 0.305629714],
    'net_mae': [0.350890064, 0.346854553, 0.343444448, 0.345894869, 0.339759757],
    'sms_mae': [0.594577117, 0.582300654, 0.575495183, 0.596604948, 0.616371412]
}

# GPT2数据
gpt2_data = {
    'call_mse': [0.214593, 0.164496, 0.191675, 0.198231, 0.216138],
    'net_mse': [0.256018, 0.255502, 0.254046, 0.255713, 0.254556],
    'sms_mse': [0.883411, 0.836539, 0.83988, 0.85338, 0.83939],
    'call_mae': [0.335721, 0.294603, 0.317698, 0.318938, 0.340034],
    'net_mae': [0.35786, 0.359258, 0.354846, 0.360006, 0.359478],
    'sms_mae': [0.595578, 0.579872, 0.576532, 0.582442, 0.580904]
}

# 颜色和样式设置
colors = ['#4ECDC4', '#FF6B6B']  # 青色为BERT，红色为GPT2
markers = ['o', 's']  # 圆形和方形
linestyles = ['-', '--']  # 实线和虚线

# 创建2x3子图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('LoRA Rank Analysis: Performance Comparison\nFedLLM-WTP with BERT vs GPT-2',
             fontsize=16, fontweight='bold', y=0.95)

# 任务名称
tasks = ['Call', 'Net', 'SMS']

# 绘制MSE图 (第一行)
for i, task in enumerate(tasks):
    ax = axes[0, i]
    task_lower = task.lower()

    # BERT线条
    ax.plot(ranks, bert_data[f'{task_lower}_mse'],
            color=colors[0], marker=markers[0], linestyle=linestyles[0],
            linewidth=2.5, markersize=8, markerfacecolor='white',
            markeredgecolor=colors[0], markeredgewidth=2,
            label='FedLLM-WTP (BERT)', alpha=0.9)

    # GPT2线条
    ax.plot(ranks, gpt2_data[f'{task_lower}_mse'],
            color=colors[1], marker=markers[1], linestyle=linestyles[1],
            linewidth=2.5, markersize=8, markerfacecolor='white',
            markeredgecolor=colors[1], markeredgewidth=2,
            label='FedLLM-WTP (GPT-2)', alpha=0.9)

    # 标记最优点
    bert_best_idx = np.argmin(bert_data[f'{task_lower}_mse'])
    gpt2_best_idx = np.argmin(gpt2_data[f'{task_lower}_mse'])

    ax.scatter(ranks[bert_best_idx], bert_data[f'{task_lower}_mse'][bert_best_idx],
               color=colors[0], s=120, marker='*', zorder=5,
               edgecolors='black', linewidth=1)
    ax.scatter(ranks[gpt2_best_idx], gpt2_data[f'{task_lower}_mse'][gpt2_best_idx],
               color=colors[1], s=120, marker='*', zorder=5,
               edgecolors='black', linewidth=1)

    ax.set_title(f'({chr(97 + i)}) {task} - MSE', fontsize=12, fontweight='bold')
    ax.set_xlabel('LoRA Rank', fontsize=11)
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ranks)

    if i == 0:  # 只在第一个子图显示图例
        ax.legend(fontsize=10, loc='upper right')

# 绘制MAE图 (第二行)
for i, task in enumerate(tasks):
    ax = axes[1, i]
    task_lower = task.lower()

    # BERT线条
    ax.plot(ranks, bert_data[f'{task_lower}_mae'],
            color=colors[0], marker=markers[0], linestyle=linestyles[0],
            linewidth=2.5, markersize=8, markerfacecolor='white',
            markeredgecolor=colors[0], markeredgewidth=2,
            label='FedLLM-WTP (BERT)', alpha=0.9)

    # GPT2线条
    ax.plot(ranks, gpt2_data[f'{task_lower}_mae'],
            color=colors[1], marker=markers[1], linestyle=linestyles[1],
            linewidth=2.5, markersize=8, markerfacecolor='white',
            markeredgecolor=colors[1], markeredgewidth=2,
            label='FedLLM-WTP (GPT-2)', alpha=0.9)

    # 标记最优点
    bert_best_idx = np.argmin(bert_data[f'{task_lower}_mae'])
    gpt2_best_idx = np.argmin(gpt2_data[f'{task_lower}_mae'])

    ax.scatter(ranks[bert_best_idx], bert_data[f'{task_lower}_mae'][bert_best_idx],
               color=colors[0], s=120, marker='*', zorder=5,
               edgecolors='black', linewidth=1)
    ax.scatter(ranks[gpt2_best_idx], gpt2_data[f'{task_lower}_mae'][gpt2_best_idx],
               color=colors[1], s=120, marker='*', zorder=5,
               edgecolors='black', linewidth=1)

    ax.set_title(f'({chr(100 + i)}) {task} - MAE', fontsize=12, fontweight='bold')
    ax.set_xlabel('LoRA Rank', fontsize=11)
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(ranks)

    if i == 0:  # 只在第一个子图显示图例
        ax.legend(fontsize=10, loc='upper right')

# 调整子图间距
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

# 打印最优Rank分析
print("=== 最优LoRA Rank分析 ===")
for task in tasks:
    task_lower = task.lower()
    print(f"\n{task}任务:")

    # MSE最优
    bert_best_mse_idx = np.argmin(bert_data[f'{task_lower}_mse'])
    gpt2_best_mse_idx = np.argmin(gpt2_data[f'{task_lower}_mse'])
    print(
        f"  MSE最优: BERT (rank={ranks[bert_best_mse_idx]}, MSE={bert_data[f'{task_lower}_mse'][bert_best_mse_idx]:.6f}), "
        f"GPT-2 (rank={ranks[gpt2_best_mse_idx]}, MSE={gpt2_data[f'{task_lower}_mse'][gpt2_best_mse_idx]:.6f})")

    # MAE最优
    bert_best_mae_idx = np.argmin(bert_data[f'{task_lower}_mae'])
    gpt2_best_mae_idx = np.argmin(gpt2_data[f'{task_lower}_mae'])
    print(
        f"  MAE最优: BERT (rank={ranks[bert_best_mae_idx]}, MAE={bert_data[f'{task_lower}_mae'][bert_best_mae_idx]:.6f}), "
        f"GPT-2 (rank={ranks[gpt2_best_mae_idx]}, MAE={gpt2_data[f'{task_lower}_mae'][gpt2_best_mae_idx]:.6f})")

# 计算整体最优Rank
print(f"\n=== 整体分析 ===")
# 计算平均性能
bert_avg_mse = np.mean([bert_data['call_mse'], bert_data['net_mse'], bert_data['sms_mse']], axis=0)
gpt2_avg_mse = np.mean([gpt2_data['call_mse'], gpt2_data['net_mse'], gpt2_data['sms_mse']], axis=0)
bert_avg_mae = np.mean([bert_data['call_mae'], bert_data['net_mae'], bert_data['sms_mae']], axis=0)
gpt2_avg_mae = np.mean([gpt2_data['call_mae'], gpt2_data['net_mae'], gpt2_data['sms_mae']], axis=0)

bert_best_avg_mse_idx = np.argmin(bert_avg_mse)
gpt2_best_avg_mse_idx = np.argmin(gpt2_avg_mse)
bert_best_avg_mae_idx = np.argmin(bert_avg_mae)
gpt2_best_avg_mae_idx = np.argmin(gpt2_avg_mae)

print(f"平均MSE最优: BERT (rank={ranks[bert_best_avg_mse_idx]}), GPT-2 (rank={ranks[gpt2_best_avg_mse_idx]})")
print(f"平均MAE最优: BERT (rank={ranks[bert_best_avg_mae_idx]}), GPT-2 (rank={ranks[gpt2_best_avg_mae_idx]})")