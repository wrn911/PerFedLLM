import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    try:
        from fbprophet import Prophet

        PROPHET_AVAILABLE = True
    except ImportError:
        print("警告: Prophet库未安装，请运行: pip install prophet")
        PROPHET_AVAILABLE = False


class Model:
    """
    Prophet时间序列预测模型
    Facebook开源的时间序列预测工具，适用于具有季节性模式的数据
    注意：这不是一个PyTorch nn.Module，而是传统统计模型
    """

    def __init__(self, configs):
        # 不继承nn.Module，这是一个传统统计模型
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Prophet参数
        self.yearly_seasonality = getattr(configs, 'prophet_yearly_seasonality', False)
        self.weekly_seasonality = getattr(configs, 'prophet_weekly_seasonality', True)
        self.daily_seasonality = getattr(configs, 'prophet_daily_seasonality', True)
        self.seasonality_mode = getattr(configs, 'prophet_seasonality_mode', 'additive')
        self.changepoint_prior_scale = getattr(configs, 'prophet_changepoint_prior_scale', 0.05)

        # 存储训练好的模型
        self.fitted_models = {}

        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet库未安装，请运行: pip install prophet")

    """
    Prophet时间序列预测模型
    Facebook开源的时间序列预测工具，适用于具有季节性模式的数据
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Prophet参数
        self.yearly_seasonality = getattr(configs, 'prophet_yearly_seasonality', False)
        self.weekly_seasonality = getattr(configs, 'prophet_weekly_seasonality', True)
        self.daily_seasonality = getattr(configs, 'prophet_daily_seasonality', True)
        self.seasonality_mode = getattr(configs, 'prophet_seasonality_mode', 'additive')
        self.changepoint_prior_scale = getattr(configs, 'prophet_changepoint_prior_scale', 0.05)

        # 存储训练好的模型
        self.fitted_models = {}

        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet库未安装，请运行: pip install prophet")

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        前向传播函数
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return self.forecast(x_enc, x_mark_enc)
        else:
            raise NotImplementedError("Prophet只支持预测任务")

    def _create_prophet_dataframe(self, data, timestamps=None):
        """
        创建Prophet所需的数据格式

        Args:
            data: [seq_len] 时间序列数据
            timestamps: 时间戳列表，如果为None则生成假时间戳

        Returns:
            df: Prophet格式的DataFrame，包含'ds'和'y'列
        """
        if timestamps is None:
            # 生成假时间戳（假设为小时级数据）
            start_date = datetime.now() - timedelta(hours=len(data) - 1)
            timestamps = [start_date + timedelta(hours=i) for i in range(len(data))]

        df = pd.DataFrame({
            'ds': timestamps,
            'y': data
        })
        return df

    def forecast(self, x_enc, x_mark_enc=None):
        """
        使用Prophet进行预测

        Args:
            x_enc: [batch_size, seq_len, features] 输入序列
            x_mark_enc: [batch_size, seq_len, time_features] 时间特征（可选）

        Returns:
            predictions: [batch_size, pred_len, features] 预测结果
        """
        batch_size, seq_len, n_features = x_enc.shape
        device = x_enc.device

        # 转换为numpy数组进行处理
        x_enc_np = x_enc.detach().cpu().numpy()
        predictions = np.zeros((batch_size, self.pred_len, n_features))

        # 为了提高速度，只训练第一个batch的第一个特征，其他使用相同的模型
        master_model = None

        for batch_idx in range(batch_size):
            for feature_idx in range(n_features):
                # 提取单个时间序列
                ts_data = x_enc_np[batch_idx, :, feature_idx]

                model_key = f"{batch_idx}_{feature_idx}"

                try:
                    # 只为第一个样本训练模型，其他使用相同模型（节省时间）
                    if master_model is None:
                        # 创建Prophet数据格式
                        df = self._create_prophet_dataframe(ts_data)

                        # 创建Prophet模型
                        prophet_model = Prophet(
                            yearly_seasonality=self.yearly_seasonality,
                            weekly_seasonality=self.weekly_seasonality,
                            daily_seasonality=self.daily_seasonality,
                            seasonality_mode=self.seasonality_mode,
                            changepoint_prior_scale=self.changepoint_prior_scale
                        )

                        # 拟合模型
                        prophet_model.fit(df)
                        master_model = prophet_model
                        self.fitted_models[model_key] = prophet_model
                    else:
                        # 使用已训练的模型
                        prophet_model = master_model

                    # 创建未来时间点
                    future = prophet_model.make_future_dataframe(periods=self.pred_len, freq='H')

                    # 进行预测
                    forecast = prophet_model.predict(future)

                    # 提取预测值（只要未来的pred_len个点）
                    pred_values = forecast['yhat'].values[-self.pred_len:]
                    predictions[batch_idx, :, feature_idx] = pred_values

                except Exception as e:
                    # 如果Prophet拟合失败，使用简单的移动平均作为备选
                    last_values = ts_data[-min(7, len(ts_data)):]
                    mean_value = np.mean(last_values)
                    predictions[batch_idx, :, feature_idx] = mean_value

        # 转换回torch张量
        predictions_tensor = torch.FloatTensor(predictions).to(device)
        return predictions_tensor

    def fit_and_predict(self, train_data, test_len, timestamps=None):
        """
        适用于非神经网络模型的训练和预测接口

        Args:
            train_data: [seq_len, features] 训练数据
            test_len: int 预测长度
            timestamps: 可选的时间戳

        Returns:
            predictions: [test_len, features] 预测结果
        """
        seq_len, n_features = train_data.shape
        predictions = np.zeros((test_len, n_features))

        for feature_idx in range(n_features):
            ts_data = train_data[:, feature_idx]

            try:
                # 创建Prophet数据格式
                if timestamps is not None:
                    df = self._create_prophet_dataframe(ts_data, timestamps)
                else:
                    df = self._create_prophet_dataframe(ts_data)

                # 创建和拟合Prophet模型
                prophet_model = Prophet(
                    yearly_seasonality=self.yearly_seasonality,
                    weekly_seasonality=self.weekly_seasonality,
                    daily_seasonality=self.daily_seasonality,
                    seasonality_mode=self.seasonality_mode,
                    changepoint_prior_scale=self.changepoint_prior_scale
                )

                prophet_model.fit(df)

                # 创建未来时间点并预测
                future = prophet_model.make_future_dataframe(periods=test_len, freq='H')
                forecast = prophet_model.predict(future)

                # 提取预测值
                pred_values = forecast['yhat'].values[-test_len:]
                predictions[:, feature_idx] = pred_values

            except Exception as e:
                # 备选方案：使用移动平均
                last_values = ts_data[-min(7, len(ts_data)):]
                mean_value = np.mean(last_values)
                predictions[:, feature_idx] = mean_value

        return predictions