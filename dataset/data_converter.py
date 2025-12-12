import pandas as pd
import numpy as np
import h5py
import os
from datetime import datetime, timedelta
import argparse
import logging


class DataConverter:
    """数据转换工具：将CSV数据转换为HDF5格式"""

    def __init__(self, bs_geo_file, traffic_file, output_file):
        self.bs_geo_file = bs_geo_file
        self.traffic_file = traffic_file
        self.output_file = output_file
        self.logger = self._setup_logging()

    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def load_bs_geo_data(self):
        """加载基站地理信息数据"""
        self.logger.info(f"加载基站地理信息数据: {self.bs_geo_file}")

        df = pd.read_csv(self.bs_geo_file)
        self.logger.info(f"基站数据形状: {df.shape}")
        self.logger.info(f"基站数据列: {df.columns.tolist()}")

        # 提取基站ID、经纬度
        bs_data = {}
        for bs_id, row in df.iterrows():
            bs_data[bs_id] = {
                'name': row['address'],
                'lng': row['lon'],
                'lat': row['lat'],
            }

        self.logger.info(f"成功加载 {len(bs_data)} 个基站信息")
        return bs_data

    def load_traffic_data(self):
        """加载流量数据"""
        self.logger.info(f"加载流量数据: {self.traffic_file}")

        # 读取CSV文件
        df = pd.read_csv(self.traffic_file, encoding='utf-8')

        # 确保 'time' 列存在
        if 'time' not in df.columns:
            raise ValueError("CSV 文件缺少 'time' 列")

        # 解析时间列
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)

        # 转换数据类型为float
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        self.logger.info(f"流量数据形状: {df.shape}")
        self.logger.info(f"时间范围: {df.index.min()} 到 {df.index.max()}")
        self.logger.info(f"流量数据列数: {len(df.columns)}")

        return df

    def match_bs_with_traffic(self, bs_data, traffic_df):
        """匹配基站地理信息与流量数据"""
        self.logger.info("开始匹配基站地理信息与流量数据...")

        # 获取流量数据中的基站名称（列名）
        traffic_bs_names = traffic_df.columns.tolist()
        self.logger.info(f"流量数据中的基站名称数量: {len(traffic_bs_names)}")

        # 创建映射：基站名称 -> 基站ID
        name_to_id = {}
        for bs_id, info in bs_data.items():
            name_to_id[info['name']] = bs_id

        # 匹配基站
        matched_bs = []
        unmatched_bs = []

        for bs_name in traffic_bs_names:
            if bs_name in name_to_id:
                bs_id = name_to_id[bs_name]
                matched_bs.append({
                    'id': bs_id,
                    'name': bs_name,
                    'lng': bs_data[bs_id]['lng'],
                    'lat': bs_data[bs_id]['lat']
                })
            else:
                unmatched_bs.append(bs_name)

        self.logger.info(f"成功匹配 {len(matched_bs)} 个基站")
        if unmatched_bs:
            self.logger.warning(f"未匹配的基站数量: {len(unmatched_bs)}")
            self.logger.warning(f"前10个未匹配基站: {unmatched_bs[:10]}")

        return matched_bs, unmatched_bs

    def create_h5_dataset(self, matched_bs, traffic_df, unmatched_bs):
        """创建HDF5数据集"""
        self.logger.info(f"创建HDF5文件: {self.output_file}")

        # 提取匹配的基站数据
        bs_ids = [bs['id'] for bs in matched_bs]
        bs_names = [bs['name'] for bs in matched_bs]
        lngs = [bs['lng'] for bs in matched_bs]
        lats = [bs['lat'] for bs in matched_bs]

        # 提取匹配基站对应的流量数据
        matched_traffic_data = traffic_df[bs_names].values
        timestamps = (traffic_df.index.astype('int64') // 10 ** 9).astype('int64')  # 转换为Unix时间戳

        self.logger.info(f"时间戳数量: {len(timestamps)}")
        self.logger.info(f"基站数量: {len(bs_ids)}")
        self.logger.info(f"流量数据形状: {matched_traffic_data.shape}")

        # 创建HDF5文件
        with h5py.File(self.output_file, 'w') as f:
            # 保存基本数据
            f.create_dataset('idx', data=timestamps)  # 时间戳
            f.create_dataset('cell', data=bs_ids)  # 基站ID
            f.create_dataset('lng', data=lngs)  # 经度
            f.create_dataset('lat', data=lats)  # 纬度
            f.create_dataset('traffic', data=matched_traffic_data)  # 流量数据

            # 保存元数据
            f.attrs['creation_date'] = datetime.now().isoformat()
            f.attrs['num_timestamps'] = len(timestamps)
            f.attrs['num_cells'] = len(bs_ids)
            f.attrs['time_range_start'] = traffic_df.index.min().isoformat()
            f.attrs['time_range_end'] = traffic_df.index.max().isoformat()
            f.attrs['unmatched_stations'] = str(unmatched_bs)

            # 保存基站名称（作为属性）
            bs_names_encoded = [name.encode('utf-8') for name in bs_names]
            f.create_dataset('cell_names', data=bs_names_encoded)

        self.logger.info("HDF5文件创建完成")

        # 验证文件内容
        self._verify_h5_file()

    def _verify_h5_file(self):
        """验证生成的HDF5文件"""
        self.logger.info("验证HDF5文件...")

        with h5py.File(self.output_file, 'r') as f:
            self.logger.info(f"HDF5文件字段: {list(f.keys())}")
            self.logger.info(f"时间戳形状: {f['idx'][:].shape}")
            self.logger.info(f"基站ID形状: {f['cell'][:].shape}")
            self.logger.info(f"经度形状: {f['lng'][:].shape}")
            self.logger.info(f"纬度形状: {f['lat'][:].shape}")
            self.logger.info(f"流量数据形状: {f['traffic'][:].shape}")

            # 打印一些基本信息
            self.logger.info(f"时间范围: {f.attrs['time_range_start']} 到 {f.attrs['time_range_end']}")
            self.logger.info(f"基站数量: {f.attrs['num_cells']}")
            self.logger.info(f"时间点数量: {f.attrs['num_timestamps']}")

    def convert(self):
        """执行完整的数据转换流程"""
        self.logger.info("开始数据转换流程...")

        try:
            # 1. 加载基站地理信息
            bs_data = self.load_bs_geo_data()

            # 2. 加载流量数据
            traffic_df = self.load_traffic_data()

            # 3. 匹配基站
            matched_bs, unmatched_bs = self.match_bs_with_traffic(bs_data, traffic_df)

            if not matched_bs:
                raise ValueError("没有找到匹配的基站，无法创建数据集")

            # 4. 创建HDF5文件
            self.create_h5_dataset(matched_bs, traffic_df, unmatched_bs)

            self.logger.info("数据转换完成！")

        except Exception as e:
            self.logger.error(f"数据转换失败: {str(e)}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='将CSV数据转换为HDF5格式')
    parser.add_argument('--bs_geo', type=str, required=True, help='基站地理信息CSV文件路径')
    parser.add_argument('--traffic', type=str, required=True, help='流量数据CSV文件路径')
    parser.add_argument('--output', type=str, default='converted_data.h5', help='输出HDF5文件路径')

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.bs_geo):
        print(f"错误: 基站地理信息文件不存在: {args.bs_geo}")
        return

    if not os.path.exists(args.traffic):
        print(f"错误: 流量数据文件不存在: {args.traffic}")
        return

    # 执行转换
    converter = DataConverter(args.bs_geo, args.traffic, args.output)
    converter.convert()


# 使用示例
def example_usage():
    """使用示例"""
    bs_geo_file = "bs_geo.csv"
    traffic_file = "zte4gdown.csv"
    output_file = "federated_traffic_data.h5"

    converter = DataConverter(bs_geo_file, traffic_file, output_file)
    converter.convert()


if __name__ == "__main__":
    # 可以直接运行示例，或使用命令行参数
    if len(os.sys.argv) == 1:
        # 如果没有命令行参数，运行示例
        print("使用示例模式...")
        example_usage()
    else:
        main()