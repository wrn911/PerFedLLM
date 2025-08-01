import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    """
    LSTM (Long Short-Term Memory) 模型用于时间序列预测
    经典的循环神经网络模型，擅长处理长序列依赖
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = getattr(configs, 'c_out', configs.enc_in)

        # LSTM参数
        self.hidden_size = getattr(configs, 'lstm_hidden_size', 128)
        self.num_layers = getattr(configs, 'lstm_num_layers', 2)
        self.dropout = getattr(configs, 'dropout', 0.1)
        self.bidirectional = getattr(configs, 'lstm_bidirectional', False)

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=self.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )

        # 输出层
        lstm_output_size = self.hidden_size * (2 if self.bidirectional else 1)

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # 预测任务：从最后的隐藏状态预测未来序列
            self.projection = nn.Sequential(
                nn.Linear(lstm_output_size, lstm_output_size // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(lstm_output_size // 2, self.pred_len * self.c_out)
            )
        elif self.task_name == 'classification':
            self.projection = nn.Sequential(
                nn.Linear(lstm_output_size, lstm_output_size // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(lstm_output_size // 2, configs.num_class)
            )
        else:
            # 其他任务（如异常检测、插值）
            self.projection = nn.Linear(lstm_output_size, self.c_out)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化LSTM权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # 设置forget gate bias为1（LSTM的常见技巧）
                n = param.size(0)
                param.data[(n // 4):(n // 2)].fill_(1)

        # 初始化投影层权重
        for module in self.projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        前向传播函数

        Args:
            x_enc: [batch_size, seq_len, features] 输入序列
            其他参数：为了兼容性保留，但在LSTM中不使用

        Returns:
            根据任务类型返回相应形状的输出
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return self.forecast(x_enc)
        elif self.task_name == 'classification':
            return self.classification(x_enc)
        elif self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        elif self.task_name == 'imputation':
            return self.imputation(x_enc, mask)
        else:
            raise NotImplementedError(f"任务 {self.task_name} 未实现")

    def forecast(self, x_enc):
        """
        时间序列预测

        Args:
            x_enc: [batch_size, seq_len, features] 输入序列

        Returns:
            predictions: [batch_size, pred_len, features] 预测结果
        """
        batch_size = x_enc.size(0)

        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x_enc)

        # 使用最后一个时间步的输出
        if self.bidirectional:
            # 双向LSTM：拼接前向和后向的最后输出
            last_output = lstm_out[:, -1, :]
        else:
            # 单向LSTM：使用最后一个输出
            last_output = lstm_out[:, -1, :]

        # 通过投影层得到预测
        predictions = self.projection(last_output)

        # 重塑为 [batch_size, pred_len, features]
        predictions = predictions.view(batch_size, self.pred_len, self.c_out)

        return predictions

    def classification(self, x_enc):
        """
        分类任务

        Args:
            x_enc: [batch_size, seq_len, features] 输入序列

        Returns:
            logits: [batch_size, num_classes] 分类概率
        """
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x_enc)

        # 使用最后一个时间步的输出进行分类
        last_output = lstm_out[:, -1, :]

        # 通过投影层得到分类结果
        logits = self.projection(last_output)

        return logits

    def anomaly_detection(self, x_enc):
        """
        异常检测任务

        Args:
            x_enc: [batch_size, seq_len, features] 输入序列

        Returns:
            reconstructions: [batch_size, seq_len, features] 重构结果
        """
        batch_size, seq_len, _ = x_enc.size()

        # LSTM前向传播
        lstm_out, _ = self.lstm(x_enc)

        # 对每个时间步应用投影层
        reconstructions = self.projection(lstm_out)

        return reconstructions

    def imputation(self, x_enc, mask=None):
        """
        数据插值任务

        Args:
            x_enc: [batch_size, seq_len, features] 输入序列（含缺失值）
            mask: [batch_size, seq_len, features] 掩码，1表示观测值，0表示缺失值

        Returns:
            imputed: [batch_size, seq_len, features] 插值后的结果
        """
        # LSTM前向传播
        lstm_out, _ = self.lstm(x_enc)

        # 通过投影层得到插值结果
        imputed = self.projection(lstm_out)

        # 如果有掩码，只替换缺失值位置
        if mask is not None:
            imputed = x_enc * mask + imputed * (1 - mask)

        return imputed

    def predict_step_by_step(self, x_enc, pred_len=None):
        """
        逐步预测方法（用于长期预测的替代方案）

        Args:
            x_enc: [batch_size, seq_len, features] 输入序列
            pred_len: 预测长度，如果为None则使用self.pred_len

        Returns:
            predictions: [batch_size, pred_len, features] 预测结果
        """
        if pred_len is None:
            pred_len = self.pred_len

        batch_size, seq_len, n_features = x_enc.size()
        device = x_enc.device

        # 初始化预测结果
        predictions = []

        # 当前输入序列
        current_seq = x_enc.clone()

        for step in range(pred_len):
            # LSTM前向传播
            lstm_out, _ = self.lstm(current_seq)

            # 使用最后时间步的输出预测下一步
            last_output = lstm_out[:, -1:, :]  # 保持时间维度

            # 简单线性投影到输出维度
            if not hasattr(self, 'step_projection'):
                self.step_projection = nn.Linear(
                    last_output.size(-1), n_features
                ).to(device)

            next_pred = self.step_projection(last_output)
            predictions.append(next_pred)

            # 更新输入序列：移除最旧的，添加最新预测的
            current_seq = torch.cat([current_seq[:, 1:, :], next_pred], dim=1)

        # 拼接所有预测
        predictions = torch.cat(predictions, dim=1)

        return predictions