# utils/communication_stats.py
import torch
import sys
from typing import Dict, List


class CommunicationTracker:
    """通信成本追踪器"""

    def __init__(self):
        self.upload_bytes = 0
        self.download_bytes = 0
        self.upload_params = 0
        self.round_count = 0

    def record_upload(self, tensors: Dict[str, torch.Tensor]):
        """记录上传数据"""
        size_bytes = self._calculate_tensor_size(tensors)
        param_count = sum(t.numel() for t in tensors.values())

        self.upload_bytes += size_bytes
        self.upload_params += param_count

    def record_download(self, tensors: Dict[str, torch.Tensor]):
        """记录下载数据"""
        size_bytes = self._calculate_tensor_size(tensors)
        self.download_bytes += size_bytes

    def new_round(self):
        """新一轮开始"""
        self.round_count += 1

    def _calculate_tensor_size(self, tensors: Dict[str, torch.Tensor]) -> int:
        """计算张量字典的字节大小"""
        total_bytes = 0
        for tensor in tensors.values():
            # 每个float32参数占4字节
            total_bytes += tensor.numel() * 4
        return total_bytes

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_upload_bytes': self.upload_bytes,
            'total_download_bytes': self.download_bytes,
            'total_bytes': self.upload_bytes + self.download_bytes,
            'upload_params': self.upload_params,
            'rounds': self.round_count,
            'avg_upload_per_round': self.upload_bytes / max(1, self.round_count),
            'avg_download_per_round': self.download_bytes / max(1, self.round_count)
        }


def format_bytes(bytes_count: int) -> str:
    """格式化字节数为可读格式"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.2f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.2f} TB"


def print_communication_comparison(trackers: Dict[str, CommunicationTracker]):
    """打印通信成本对比"""
    print("\n" + "=" * 80)
    print(" " * 25 + "COMMUNICATION COST ANALYSIS")
    print("=" * 80)

    # 收集所有算法的统计信息
    all_stats = {}
    for algo_name, tracker in trackers.items():
        all_stats[algo_name] = tracker.get_stats()

    # 表头
    print(f"{'Algorithm':<12} {'Total Upload':<12} {'Total Download':<14} {'Total Comm':<12} {'Efficiency':<10}")
    print("-" * 80)

    # 基准算法（通常是FedAvg）
    baseline_algo = 'fedavg' if 'fedavg' in all_stats else list(all_stats.keys())[0]
    baseline_total = all_stats[baseline_algo]['total_bytes']

    # 打印每个算法的统计
    for algo_name, stats in sorted(all_stats.items()):
        upload_str = format_bytes(stats['total_upload_bytes'])
        download_str = format_bytes(stats['total_download_bytes'])
        total_str = format_bytes(stats['total_bytes'])

        # 计算相对于基准的效率
        efficiency = (baseline_total - stats['total_bytes']) / baseline_total * 100
        efficiency_str = f"{efficiency:+.1f}%" if algo_name != baseline_algo else "baseline"

        print(f"{algo_name:<12} {upload_str:<12} {download_str:<14} {total_str:<12} {efficiency_str:<10}")

    print("-" * 80)

    # 详细对比分析
    print("\nDetailed Analysis:")
    for algo_name, stats in all_stats.items():
        print(f"\n{algo_name.upper()}:")
        print(f"  • Upload per round: {format_bytes(stats['avg_upload_per_round'])}")
        print(f"  • Download per round: {format_bytes(stats['avg_download_per_round'])}")
        print(f"  • Parameters uploaded: {stats['upload_params']:,}")
        print(f"  • Communication rounds: {stats['rounds']}")

        if algo_name != baseline_algo:
            savings = (baseline_total - stats['total_bytes']) / baseline_total * 100
            saved_bytes = baseline_total - stats['total_bytes']
            print(f"  • Savings vs {baseline_algo}: {format_bytes(saved_bytes)} ({savings:.1f}%)")

    print("\n" + "=" * 80)