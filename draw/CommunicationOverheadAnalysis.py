import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体和科学期刊风格
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8

# 数据
methods = ['TimeLLM(GPT-2)\n+FedAvg', 'FedLLM-WTP\n(GPT-2)',
           'TimeLLM(BERT)\n+FedAvg', 'FedLLM-WTP\n(BERT)']
upload_mb = [210.26, 3.57, 137.17, 5.64]
download_mb = [210.26, 3.57, 137.17, 5.64]  # 上传下载相同
model_size_mb = [524.62, 316.10, 390.72, 261.16]

# 颜色设置 - 使用对比色突出我们的方法
colors = ['#FF6B6B', '#4ECDC4', '#FF6B6B', '#4ECDC4']  # 红色表示baseline，青色表示我们的方法

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# 子图1: 通信开销 (因为上传下载相同，显示单向通信量)
bars1 = ax1.bar(methods, upload_mb, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax1.set_ylabel('Communication Overhead (MB)', fontsize=11, fontweight='bold')
ax1.set_title('(a) Communication Overhead', fontsize=12, fontweight='bold')
ax1.set_ylim(0, max(upload_mb) * 1.1)

# 在柱状图上添加数值标签
for i, (bar, val) in enumerate(zip(bars1, upload_mb)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(upload_mb)*0.01,
             f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 子图2: 模型大小
bars2 = ax2.bar(methods, model_size_mb, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax2.set_ylabel('Model Size (MB)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Model Size', fontsize=12, fontweight='bold')
ax2.set_ylim(0, max(model_size_mb) * 1.1)

# 在柱状图上添加数值标签
for i, (bar, val) in enumerate(zip(bars2, model_size_mb)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(model_size_mb)*0.01,
             f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 设置x轴标签角度
for ax in [ax1, ax2]:
    ax.tick_params(axis='x', rotation=15, labelsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)

# 调整布局
plt.tight_layout()

# 添加图例说明我们的方法和baseline的区别
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#FF6B6B', alpha=0.8, label='Baseline Methods'),
                   Patch(facecolor='#4ECDC4', alpha=0.8, label='FedLLM-WTP (Ours)')]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
           ncol=2, fontsize=10)

plt.subplots_adjust(bottom=0.15)  # 为图例留出空间
plt.show()

# 计算并打印减少百分比
gpt2_comm_reduction = (210.26 - 3.57) / 210.26 * 100
bert_comm_reduction = (137.17 - 5.64) / 137.17 * 100
gpt2_size_reduction = (524.62 - 316.10) / 524.62 * 100
bert_size_reduction = (390.72 - 261.16) / 390.72 * 100

print(f"通信开销减少: GPT-2: {gpt2_comm_reduction:.1f}%, BERT: {bert_comm_reduction:.1f}%")
print(f"模型大小减少: GPT-2: {gpt2_size_reduction:.1f}%, BERT: {bert_size_reduction:.1f}%")