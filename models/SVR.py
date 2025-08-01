import torch
import torch.nn as nn
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')


class Model:
    """
    SVR (Support Vector Regression) 模型用于时间序列预测
    使用滑动窗口方法将时间序列转换为回归问题
    注意：这不是一个PyTorch nn.Module，而是传统机器学习模型
    """

    def __init__(self, configs):
        # 不继承nn.Module，这是一个传统机器学习模型
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # SVR参数
        self.kernel = getattr(configs, 'svr_kernel', 'rbf')  # 核函数类型
        self.C = getattr(configs, 'svr_C', 1.0)  # 正则化参数
        self.gamma = getattr(configs, 'svr_gamma', 'scale')  # 核函数参数
        self.epsilon = getattr(configs, 'svr_epsilon', 0.1)  # epsilon-tube

        # 存储训练好的模型和标准化器
        self.fitted_models = {}
        self.scalers = {}

    """
    SVR (Support Vector Regression) 模型用于时间序列预测
    使用滑动窗口方法将时间序列转换为回归问题
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # SVR参数
        self.kernel = getattr(configs, 'svr_kernel', 'rbf')  # 核函数类型
        self.C = getattr(configs, 'svr_C', 1.0)  # 正则化参数
        self.gamma = getattr(configs, 'svr_gamma', 'scale')  # 核函数参数
        self.epsilon = getattr(configs, 'svr_epsilon', 0.1)  # epsilon-tube

        # 存储训练好的模型和标准化器
        self.fitted_models = {}
        self.scalers = {}

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        前向传播函数
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return self.forecast(x_enc)
        else:
            raise NotImplementedError("SVR只支持预测任务")

    def _create_sequences(self, data, seq_len, pred_len):
        """
        创建用于回归的序列

        Args:
            data: [time_steps, features] 时间序列数据
            seq_len: 输入序列长度
            pred_len: 预测序列长度

        Returns:
            X: [samples, seq_len * features] 输入特征
            y: [samples, pred_len * features] 目标值
        """
        time_steps, n_features = data.shape
        X, y = [], []

        for i in range(seq_len, time_steps - pred_len + 1):
            # 输入特征：展平的历史窗口
            x_seq = data[i - seq_len:i].flatten()
            # 目标值：未来pred_len步的值
            y_seq = data[i:i + pred_len].flatten()

            X.append(x_seq)
            y.append(y_seq)

        return np.array(X), np.array(y)

    def forecast(self, x_enc):
        """
        使用SVR进行预测

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
            # 提取当前批次的数据
            batch_data = x_enc_np[batch_idx]  # [seq_len, features]

            model_key = f"batch_{batch_idx}"

            # 如果是训练阶段或模型不存在，重新训练
            if self.training or model_key not in self.fitted_models:
                try:
                    # 创建回归数据
                    if seq_len >= 2 * self.pred_len:
                        # 使用适当的窗口大小
                        window_size = min(seq_len // 3, 15)  # SVR对于大特征空间可能较慢
                        target_size = min(self.pred_len, seq_len // 5)

                        X, y = self._create_sequences(batch_data, window_size, target_size)

                        if len(X) > 0 and len(X) >= 3:  # 至少需要3个样本
                            # 标准化特征
                            scaler_X = StandardScaler()
                            scaler_y = StandardScaler()

                            X_scaled = scaler_X.fit_transform(X)
                            y_scaled = scaler_y.fit_transform(y)

                            # 创建SVR模型
                            # 对于多输出问题，使用MultiOutputRegressor
                            svr_base = SVR(
                                kernel=self.kernel,
                                C=self.C,
                                gamma=self.gamma,
                                epsilon=self.epsilon,
                                cache_size=200  # 限制缓存大小
                            )

                            svr_model = MultiOutputRegressor(svr_base, n_jobs=1)  # 避免并行以节省内存
                            svr_model.fit(X_scaled, y_scaled)

                            # 存储模型和标准化器
                            self.fitted_models[model_key] = svr_model
                            self.scalers[model_key] = (scaler_X, scaler_y)

                            # 使用最新的数据进行预测
                            latest_seq = batch_data[-window_size:].flatten().reshape(1, -1)
                            latest_seq_scaled = scaler_X.transform(latest_seq)
                            pred_scaled = svr_model.predict(latest_seq_scaled)
                            pred = scaler_y.inverse_transform(pred_scaled)

                            # 重塑预测结果
                            pred_reshaped = pred.reshape(target_size, n_features)

                            # 如果预测长度不足，使用最后的值进行扩展
                            if target_size < self.pred_len:
                                last_pred = pred_reshaped[-1:].repeat(self.pred_len - target_size, axis=0)
                                pred_reshaped = np.vstack([pred_reshaped, last_pred])

                            predictions[batch_idx] = pred_reshaped[:self.pred_len]
                        else:
                            # 备选方案：使用移动平均
                            for f in range(n_features):
                                last_values = batch_data[-min(5, seq_len):, f]
                                predictions[batch_idx, :, f] = np.mean(last_values)
                    else:
                        # 数据不足，使用简单预测
                        for f in range(n_features):
                            last_values = batch_data[-min(3, seq_len):, f]
                            predictions[batch_idx, :, f] = np.mean(last_values)

                except Exception as e:
                    # 出错时使用备选方案
                    for f in range(n_features):
                        last_values = batch_data[-min(5, seq_len):, f]
                        predictions[batch_idx, :, f] = np.mean(last_values)

            else:
                # 使用已训练的模型进行预测
                try:
                    svr_model = self.fitted_models[model_key]
                    scaler_X, scaler_y = self.scalers[model_key]

                    # 获取正确的窗口大小
                    expected_features = scaler_X.n_features_in_
                    window_size = expected_features // n_features

                    latest_seq = batch_data[-window_size:].flatten().reshape(1, -1)
                    latest_seq_scaled = scaler_X.transform(latest_seq)
                    pred_scaled = svr_model.predict(latest_seq_scaled)
                    pred = scaler_y.inverse_transform(pred_scaled)

                    # 重塑并处理预测结果
                    target_size = pred.shape[1] // n_features
                    pred_reshaped = pred.reshape(target_size, n_features)

                    if target_size < self.pred_len:
                        last_pred = pred_reshaped[-1:].repeat(self.pred_len - target_size, axis=0)
                        pred_reshaped = np.vstack([pred_reshaped, last_pred])

                    predictions[batch_idx] = pred_reshaped[:self.pred_len]

                except Exception as e:
                    # 备选方案
                    for f in range(n_features):
                        last_values = batch_data[-min(5, seq_len):, f]
                        predictions[batch_idx, :, f] = np.mean(last_values)

        # 转换回torch张量
        predictions_tensor = torch.FloatTensor(predictions).to(device)
        return predictions_tensor

    def fit_and_predict(self, train_data, test_len):
        """
        适用于独立训练的接口

        Args:
            train_data: [seq_len, features] 训练数据
            test_len: int 预测长度

        Returns:
            predictions: [test_len, features] 预测结果
        """
        seq_len, n_features = train_data.shape

        try:
            # 创建训练样本 - SVR对大数据量敏感，使用较小窗口
            window_size = min(seq_len // 4, 12)
            pred_size = min(test_len, seq_len // 6)

            X, y = self._create_sequences(train_data, window_size, pred_size)

            if len(X) >= 3:  # SVR至少需要3个样本
                # 标准化
                scaler_X = StandardScaler()
                scaler_y = StandardScaler()

                X_scaled = scaler_X.fit_transform(X)
                y_scaled = scaler_y.fit_transform(y)

                # 训练SVR模型
                svr_base = SVR(
                    kernel=self.kernel,
                    C=self.C,
                    gamma=self.gamma,
                    epsilon=self.epsilon,
                    cache_size=200
                )

                svr_model = MultiOutputRegressor(svr_base, n_jobs=1)
                svr_model.fit(X_scaled, y_scaled)

                # 预测
                latest_seq = train_data[-window_size:].flatten().reshape(1, -1)
                latest_seq_scaled = scaler_X.transform(latest_seq)
                pred_scaled = svr_model.predict(latest_seq_scaled)
                pred = scaler_y.inverse_transform(pred_scaled)

                predictions = pred.reshape(pred_size, n_features)

                # 如果预测长度不够，用最后的值填充
                if pred_size < test_len:
                    last_pred = predictions[-1:].repeat(test_len - pred_size, axis=0)
                    predictions = np.vstack([predictions, last_pred])

                return predictions[:test_len]

        except Exception as e:
            pass

        # 备选方案：移动平均
        predictions = np.zeros((test_len, n_features))
        for f in range(n_features):
            last_values = train_data[-min(10, seq_len):, f]
            predictions[:, f] = np.mean(last_values)

        return predictions