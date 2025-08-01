import torch
import torch.nn as nn
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')


class Model:
    """
    ARIMA (AutoRegressive Integrated Moving Average) 模型
    适用于单变量时间序列预测
    注意：这不是一个PyTorch nn.Module，而是传统统计模型
    """

    def __init__(self, configs):
        # 不继承nn.Module，这是一个传统统计模型
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # ARIMA参数 (p, d, q)
        self.p = getattr(configs, 'arima_p', 1)  # 自回归项数
        self.d = getattr(configs, 'arima_d', 1)  # 差分次数
        self.q = getattr(configs, 'arima_q', 1)  # 移动平均项数

        # 存储已训练的模型
        self.fitted_models = {}

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        前向传播函数（兼容PyTorch接口）
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return self.forecast(x_enc)
        else:
            raise NotImplementedError("ARIMA只支持预测任务")

    def forecast(self, x_enc):
        """
        使用ARIMA进行预测

        Args:
            x_enc: [batch_size, seq_len, features] 输入序列

        Returns:
            predictions: [batch_size, pred_len, features] 预测结果
        """
        batch_size, seq_len, n_features = x_enc.shape
        device = x_enc.device

        # 转换为numpy数组进行处理
        x_enc_np = x_enc.detach().cpu().numpy()
        predictions = np.zeros((batch_size, self.pred_len, n_features))

        for batch_idx in range(batch_size):
            for feature_idx in range(n_features):
                # 提取单个时间序列
                ts_data = x_enc_np[batch_idx, :, feature_idx]

                try:
                    # 创建并拟合ARIMA模型
                    model_key = f"{batch_idx}_{feature_idx}"

                    # 如果是训练阶段，重新拟合模型
                    if model_key not in self.fitted_models:
                        arima_model = ARIMA(ts_data, order=(self.p, self.d, self.q))
                        fitted_model = arima_model.fit()
                        self.fitted_models[model_key] = fitted_model
                    else:
                        fitted_model = self.fitted_models[model_key]

                    # 进行预测
                    forecast = fitted_model.forecast(steps=self.pred_len)
                    predictions[batch_idx, :, feature_idx] = forecast

                except Exception as e:
                    # 如果ARIMA拟合失败，使用简单的移动平均作为备选
                    last_values = ts_data[-min(5, len(ts_data)):]
                    mean_value = np.mean(last_values)
                    predictions[batch_idx, :, feature_idx] = mean_value

        # 转换回torch张量
        predictions_tensor = torch.FloatTensor(predictions).to(device)
        return predictions_tensor

    def fit_and_predict(self, train_data, test_len):
        """
        适用于非神经网络模型的训练和预测接口

        Args:
            train_data: [seq_len, features] 训练数据
            test_len: int 预测长度

        Returns:
            predictions: [test_len, features] 预测结果
        """
        seq_len, n_features = train_data.shape
        predictions = np.zeros((test_len, n_features))

        for feature_idx in range(n_features):
            ts_data = train_data[:, feature_idx]

            try:
                # 拟合ARIMA模型
                arima_model = ARIMA(ts_data, order=(self.p, self.d, self.q))
                fitted_model = arima_model.fit()

                # 进行预测
                forecast = fitted_model.forecast(steps=test_len)
                predictions[:, feature_idx] = forecast

            except Exception as e:
                # 备选方案：使用移动平均
                last_values = ts_data[-min(5, len(ts_data)):]
                mean_value = np.mean(last_values)
                predictions[:, feature_idx] = mean_value

        return predictions

    """
    ARIMA (AutoRegressive Integrated Moving Average) 模型
    适用于单变量时间序列预测
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # ARIMA参数 (p, d, q)
        self.p = getattr(configs, 'arima_p', 1)  # 自回归项数
        self.d = getattr(configs, 'arima_d', 1)  # 差分次数
        self.q = getattr(configs, 'arima_q', 1)  # 移动平均项数

        # 存储已训练的模型
        self.fitted_models = {}

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        前向传播函数
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return self.forecast(x_enc)
        else:
            raise NotImplementedError("ARIMA只支持预测任务")

    def forecast(self, x_enc):
        """
        使用ARIMA进行预测

        Args:
            x_enc: [batch_size, seq_len, features] 输入序列

        Returns:
            predictions: [batch_size, pred_len, features] 预测结果
        """
        batch_size, seq_len, n_features = x_enc.shape
        device = x_enc.device

        # 转换为numpy数组进行处理
        x_enc_np = x_enc.detach().cpu().numpy()
        predictions = np.zeros((batch_size, self.pred_len, n_features))

        for batch_idx in range(batch_size):
            for feature_idx in range(n_features):
                # 提取单个时间序列
                ts_data = x_enc_np[batch_idx, :, feature_idx]

                try:
                    # 创建并拟合ARIMA模型
                    model_key = f"{batch_idx}_{feature_idx}"

                    # 如果是训练阶段，重新拟合模型
                    if self.training or model_key not in self.fitted_models:
                        arima_model = ARIMA(ts_data, order=(self.p, self.d, self.q))
                        fitted_model = arima_model.fit()
                        self.fitted_models[model_key] = fitted_model
                    else:
                        fitted_model = self.fitted_models[model_key]

                    # 进行预测
                    forecast = fitted_model.forecast(steps=self.pred_len)
                    predictions[batch_idx, :, feature_idx] = forecast

                except Exception as e:
                    # 如果ARIMA拟合失败，使用简单的移动平均作为备选
                    last_values = ts_data[-min(5, len(ts_data)):]
                    mean_value = np.mean(last_values)
                    predictions[batch_idx, :, feature_idx] = mean_value

        # 转换回torch张量
        predictions_tensor = torch.FloatTensor(predictions).to(device)
        return predictions_tensor

    def fit_and_predict(self, train_data, test_len):
        """
        适用于非神经网络模型的训练和预测接口

        Args:
            train_data: [seq_len, features] 训练数据
            test_len: int 预测长度

        Returns:
            predictions: [test_len, features] 预测结果
        """
        seq_len, n_features = train_data.shape
        predictions = np.zeros((test_len, n_features))

        for feature_idx in range(n_features):
            ts_data = train_data[:, feature_idx]

            try:
                # 拟合ARIMA模型
                arima_model = ARIMA(ts_data, order=(self.p, self.d, self.q))
                fitted_model = arima_model.fit()

                # 进行预测
                forecast = fitted_model.forecast(steps=test_len)
                predictions[:, feature_idx] = forecast

            except Exception as e:
                # 备选方案：使用移动平均
                last_values = ts_data[-min(5, len(ts_data)):]
                mean_value = np.mean(last_values)
                predictions[:, feature_idx] = mean_value

        return predictions