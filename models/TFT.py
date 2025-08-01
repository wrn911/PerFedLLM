import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Embed import DataEmbedding


class GatedLinearUnit(nn.Module):
    """门控线性单元"""

    def __init__(self, input_size, output_size, dropout=0.1):
        super(GatedLinearUnit, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size)
        self.linear2 = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear1(x) * torch.sigmoid(self.linear2(x))


class GateAddNorm(nn.Module):
    """门控残差连接与层归一化"""

    def __init__(self, input_size, dropout=0.1):
        super(GateAddNorm, self).__init__()
        self.glu = GatedLinearUnit(input_size, input_size, dropout)
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        gated_x = self.glu(x)
        return self.layer_norm(gated_x + residual)


class VariableSelectionNetwork(nn.Module):
    """变量选择网络"""

    def __init__(self, input_size, num_inputs, hidden_size, dropout=0.1):
        super(VariableSelectionNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_inputs = num_inputs

        # 单变量处理器
        self.single_variable_grns = nn.ModuleList([
            GatedLinearUnit(input_size, hidden_size, dropout)
            for _ in range(num_inputs)
        ])

        # 变量选择权重网络
        self.variable_selection = nn.Sequential(
            GatedLinearUnit(num_inputs * hidden_size, num_inputs, dropout),
            nn.Softmax(dim=-1)
        )

    def forward(self, variables):
        """
        Args:
            variables: List of [batch_size, seq_len, input_size] tensors
        Returns:
            selected_variables: [batch_size, seq_len, hidden_size]
            weights: [batch_size, seq_len, num_inputs]
        """
        batch_size, seq_len = variables[0].shape[:2]

        # 处理每个变量
        processed_vars = []
        for i, var in enumerate(variables):
            processed = self.single_variable_grns[i](var)
            processed_vars.append(processed)

        # 拼接所有处理后的变量
        concatenated = torch.cat(processed_vars, dim=-1)

        # 计算变量选择权重
        weights = self.variable_selection(concatenated)

        # 加权组合
        selected = torch.zeros(batch_size, seq_len, self.hidden_size).to(variables[0].device)
        for i, processed in enumerate(processed_vars):
            selected += weights[..., i:i + 1] * processed

        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """可解释的多头注意力机制"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super(InterpretableMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)

        # 线性变换
        Q = self.w_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        output = self.w_o(attended)

        # 返回平均注意力权重用于可解释性
        avg_attention = attention_weights.mean(dim=1)  # 平均所有头的注意力

        return output, avg_attention


class Model(nn.Module):
    """
    TFT (Temporal Fusion Transformer) 模型
    Google Research开发的时间序列预测模型，结合了LSTM和Transformer
    具有很强的可解释性
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = getattr(configs, 'label_len', configs.pred_len // 2)

        # 模型参数
        self.d_model = getattr(configs, 'd_model', 128)
        self.n_heads = getattr(configs, 'n_heads', 8)
        self.dropout = getattr(configs, 'dropout', 0.1)
        self.enc_in = configs.enc_in
        self.c_out = getattr(configs, 'c_out', configs.enc_in)

        # TFT特定参数
        self.lstm_hidden_size = getattr(configs, 'tft_lstm_hidden', self.d_model)
        self.num_quantiles = getattr(configs, 'tft_num_quantiles', 3)  # 分位数预测

        # 输入嵌入
        self.enc_embedding = DataEmbedding(
            self.enc_in, self.d_model, 'timeF', 'h', self.dropout
        )

        # 变量选择网络
        self.encoder_variable_selection = VariableSelectionNetwork(
            self.d_model, 1, self.d_model, self.dropout
        )

        self.decoder_variable_selection = VariableSelectionNetwork(
            self.d_model, 1, self.d_model, self.dropout
        )

        # LSTM编码器 - 修复维度问题
        self.encoder_lstm = nn.LSTM(
            self.d_model, self.d_model,  # 保持输入输出维度一致
            num_layers=1, batch_first=True, dropout=self.dropout
        )

        # LSTM解码器 - 修复维度问题
        self.decoder_lstm = nn.LSTM(
            self.d_model, self.d_model,  # 保持输入输出维度一致
            num_layers=1, batch_first=True, dropout=self.dropout
        )

        # 门控残差连接 - 修复维度问题
        self.encoder_gate_add_norm = GateAddNorm(self.d_model, self.dropout)
        self.decoder_gate_add_norm = GateAddNorm(self.d_model, self.dropout)

        # 可解释的多头注意力 - 修复维度问题
        self.self_attention = InterpretableMultiHeadAttention(
            self.d_model, self.n_heads, self.dropout
        )

        # 位置前馈网络 - 修复维度问题
        self.position_wise_ff = nn.Sequential(
            GatedLinearUnit(self.d_model, self.d_model, self.dropout),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model)
        )

        # 最终输出层
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = nn.Linear(
                self.d_model, self.c_out * self.num_quantiles
            )
        else:
            self.output_projection = nn.Linear(self.d_model, self.c_out)

        # 分位数损失权重
        self.register_buffer('quantile_levels',
                             torch.tensor([0.1, 0.5, 0.9]))  # 10%, 50%, 90%分位数

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        """
        前向传播函数
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            raise NotImplementedError(f"任务 {self.task_name} 未实现")

    def forecast(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        """
        时间序列预测

        Args:
            x_enc: [batch_size, seq_len, features] 编码器输入
            x_mark_enc: [batch_size, seq_len, time_features] 编码器时间特征
            x_dec: [batch_size, label_len + pred_len, features] 解码器输入
            x_mark_dec: [batch_size, label_len + pred_len, time_features] 解码器时间特征

        Returns:
            predictions: [batch_size, pred_len, features] 预测结果
        """
        batch_size = x_enc.size(0)

        # 编码器部分
        # 嵌入输入
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # 变量选择
        enc_selected, enc_weights = self.encoder_variable_selection([enc_out])

        # LSTM编码
        lstm_enc_out, (enc_hidden, enc_cell) = self.encoder_lstm(enc_selected)

        # 门控残差连接
        enc_out = self.encoder_gate_add_norm(lstm_enc_out, enc_selected)

        # 解码器部分
        if x_dec is None:
            # 如果没有提供解码器输入，创建零输入
            x_dec = torch.zeros(batch_size, self.pred_len, x_enc.size(-1)).to(x_enc.device)
            if x_mark_dec is None:
                x_mark_dec = torch.zeros(batch_size, self.pred_len, x_mark_enc.size(-1)).to(x_enc.device)

        # 取解码器的前pred_len部分
        dec_input = x_dec[:, -self.pred_len:, :]
        dec_mark = x_mark_dec[:, -self.pred_len:, :] if x_mark_dec is not None else None

        # 解码器嵌入
        if dec_mark is not None:
            dec_out = self.enc_embedding(dec_input, dec_mark)
        else:
            dec_out = self.enc_embedding(dec_input, None)

        # 解码器变量选择
        dec_selected, dec_weights = self.decoder_variable_selection([dec_out])

        # LSTM解码（使用编码器的最终状态作为初始状态）
        lstm_dec_out, _ = self.decoder_lstm(dec_selected, (enc_hidden, enc_cell))

        # 门控残差连接
        dec_out = self.decoder_gate_add_norm(lstm_dec_out, dec_selected)

        # 自注意力机制
        attended_out, attention_weights = self.self_attention(dec_out, dec_out, dec_out)

        # 位置前馈网络
        ff_out = self.position_wise_ff(attended_out)

        # 最终输出投影
        output = self.output_projection(ff_out)

        if self.num_quantiles > 1:
            # 重塑为分位数输出 [batch_size, pred_len, features, num_quantiles]
            output = output.view(batch_size, self.pred_len, self.c_out, self.num_quantiles)
            # 返回中位数预测 (50%分位数)
            return output[:, :, :, 1]  # 假设中位数是第二个分位数
        else:
            return output

    def predict_quantiles(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        """
        分位数预测（TFT的核心特性之一）

        Returns:
            quantile_predictions: [batch_size, pred_len, features, num_quantiles]
        """
        batch_size = x_enc.size(0)

        # 与forecast相同的前向传播
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_selected, _ = self.encoder_variable_selection([enc_out])
        lstm_enc_out, (enc_hidden, enc_cell) = self.encoder_lstm(enc_selected)
        enc_out = self.encoder_gate_add_norm(lstm_enc_out, enc_selected)

        if x_dec is None:
            x_dec = torch.zeros(batch_size, self.pred_len, x_enc.size(-1)).to(x_enc.device)
            if x_mark_dec is None:
                x_mark_dec = torch.zeros(batch_size, self.pred_len, x_mark_enc.size(-1)).to(x_enc.device)

        dec_input = x_dec[:, -self.pred_len:, :]
        dec_mark = x_mark_dec[:, -self.pred_len:, :] if x_mark_dec is not None else None

        if dec_mark is not None:
            dec_out = self.enc_embedding(dec_input, dec_mark)
        else:
            dec_out = self.enc_embedding(dec_input, None)

        dec_selected, _ = self.decoder_variable_selection([dec_out])
        lstm_dec_out, _ = self.decoder_lstm(dec_selected, (enc_hidden, enc_cell))
        dec_out = self.decoder_gate_add_norm(lstm_dec_out, dec_selected)

        attended_out, _ = self.self_attention(dec_out, dec_out, dec_out)
        ff_out = self.position_wise_ff(attended_out)
        output = self.output_projection(ff_out)

        # 返回所有分位数预测
        if self.num_quantiles > 1:
            return output.view(batch_size, self.pred_len, self.c_out, self.num_quantiles)
        else:
            return output.unsqueeze(-1)

    def quantile_loss(self, predictions, targets, quantiles):
        """
        计算分位数损失

        Args:
            predictions: [batch_size, pred_len, features, num_quantiles]
            targets: [batch_size, pred_len, features]
            quantiles: [num_quantiles] 分位数水平

        Returns:
            loss: 分位数损失
        """
        targets = targets.unsqueeze(-1)  # [batch_size, pred_len, features, 1]

        errors = targets - predictions  # [batch_size, pred_len, features, num_quantiles]

        # 分位数损失计算
        loss = torch.max(
            quantiles * errors,
            (quantiles - 1) * errors
        )

        return loss.mean()

    def get_attention_weights(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        """
        获取注意力权重用于可解释性分析

        Returns:
            attention_weights: [batch_size, n_heads, pred_len, pred_len]
            variable_weights: 变量重要性权重
        """
        batch_size = x_enc.size(0)

        # 前向传播直到注意力层
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_selected, enc_var_weights = self.encoder_variable_selection([enc_out])
        lstm_enc_out, (enc_hidden, enc_cell) = self.encoder_lstm(enc_selected)
        enc_out = self.encoder_gate_add_norm(lstm_enc_out, enc_selected)

        if x_dec is None:
            x_dec = torch.zeros(batch_size, self.pred_len, x_enc.size(-1)).to(x_enc.device)
            if x_mark_dec is None:
                x_mark_dec = torch.zeros(batch_size, self.pred_len, x_mark_enc.size(-1)).to(x_enc.device)

        dec_input = x_dec[:, -self.pred_len:, :]
        dec_mark = x_mark_dec[:, -self.pred_len:, :] if x_mark_dec is not None else None

        if dec_mark is not None:
            dec_out = self.enc_embedding(dec_input, dec_mark)
        else:
            dec_out = self.enc_embedding(dec_input, None)

        dec_selected, dec_var_weights = self.decoder_variable_selection([dec_out])
        lstm_dec_out, _ = self.decoder_lstm(dec_selected, (enc_hidden, enc_cell))
        dec_out = self.decoder_gate_add_norm(lstm_dec_out, dec_selected)

        # 获取注意力权重
        _, attention_weights = self.self_attention(dec_out, dec_out, dec_out)

        return {
            'attention_weights': attention_weights,
            'encoder_variable_weights': enc_var_weights,
            'decoder_variable_weights': dec_var_weights
        }

    def interpret_predictions(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        """
        解释预测结果（TFT的重要特性）

        Returns:
            dict: 包含预测结果和解释信息
        """
        # 获取预测结果
        predictions = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # 获取注意力权重和变量重要性
        interpretation = self.get_attention_weights(x_enc, x_mark_enc, x_dec, x_mark_dec)

        # 分位数预测
        quantile_predictions = self.predict_quantiles(x_enc, x_mark_enc, x_dec, x_mark_dec)

        return {
            'predictions': predictions,
            'quantile_predictions': quantile_predictions,
            'attention_weights': interpretation['attention_weights'],
            'encoder_variable_importance': interpretation['encoder_variable_weights'],
            'decoder_variable_importance': interpretation['decoder_variable_weights']
        }