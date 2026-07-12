import torch
import torch.nn as nn
import torch.nn.functional as F
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space
import math


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制，用于智能体间通信和特征增强
    """
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear projections in batch
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        
        output = self.out(context)
        return output


class ResidualBlock(nn.Module):
    """
    残差块，增强特征传递和梯度流动
    """
    def __init__(self, hidden_size, use_orthogonal=True):
        super(ResidualBlock, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        
        self.fc1 = init_(nn.Linear(hidden_size, hidden_size))
        self.fc2 = init_(nn.Linear(hidden_size, hidden_size))
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.ln2(self.fc2(x))
        return F.relu(x + residual)


class CuriosityModule(nn.Module):
    """
    基于ICM (Intrinsic Curiosity Module) 的内在奖励模块
    鼓励智能体探索新状态
    """
    def __init__(self, obs_dim, action_dim, hidden_size=128, use_orthogonal=True):
        super(CuriosityModule, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))
        
        # Forward model: predicts next state feature
        self.forward_model = nn.Sequential(
            init_(nn.Linear(hidden_size + action_dim, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size))
        )
        
        # Inverse model: predicts action from state transitions
        self.inverse_model = nn.Sequential(
            init_(nn.Linear(hidden_size * 2, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, action_dim))
        )
        
        # Feature encoding
        self.feature_encoder = nn.Sequential(
            init_(nn.Linear(obs_dim, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size))
        )
        
    def forward(self, obs, next_obs, action):
        """
        计算内在奖励（预测误差）
        """
        # Encode observations
        phi_obs = self.feature_encoder(obs)
        phi_next_obs = self.feature_encoder(next_obs)
        
        # Forward model prediction
        pred_phi_next = self.forward_model(torch.cat([phi_obs, action], dim=-1))
        
        # Intrinsic reward is the prediction error
        intrinsic_reward = F.mse_loss(pred_phi_next, phi_next_obs.detach(), reduction='none').mean(dim=-1, keepdim=True)
        
        # Inverse model for auxiliary loss
        pred_action = self.inverse_model(torch.cat([phi_obs, phi_next_obs], dim=-1))
        
        return intrinsic_reward, pred_action


class R_Actor_Advanced(nn.Module):
    """
    改进的Actor网络，包含:
    1. 多头注意力机制
    2. 残差连接
    3. 更好的特征提取
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor_Advanced, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        # 是否使用注意力机制
        self._use_attention = getattr(args, 'use_attention', True)
        # 是否使用残差连接
        self._use_residual = getattr(args, 'use_residual', True)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        # 添加注意力机制
        if self._use_attention:
            self.attention = MultiHeadAttention(self.hidden_size, num_heads=4)
        
        # 添加残差块（支持通过 args 调整数量以实现更深网络）
        if self._use_residual:
            residual_N = getattr(args, 'adv_residual_blocks', 2)
            self.residual_blocks = nn.ModuleList([
                ResidualBlock(self.hidden_size, self._use_orthogonal)
                for _ in range(residual_N)
            ])

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        前向传播，包含注意力和残差增强
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        # Base feature extraction
        actor_features = self.base(obs)
        
        # Apply attention mechanism
        if self._use_attention:
            # Reshape for attention (batch, seq_len=1, hidden)
            attn_input = actor_features.unsqueeze(1)
            attn_output = self.attention(attn_input)
            actor_features = actor_features + attn_output.squeeze(1)  # Residual connection
        
        # Apply residual blocks
        if self._use_residual:
            for residual_block in self.residual_blocks:
                actor_features = residual_block(actor_features)

        # RNN processing
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # Action distribution
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        评估给定动作的对数概率和熵
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        # Base feature extraction
        actor_features = self.base(obs)
        
        # Apply attention mechanism
        if self._use_attention:
            attn_input = actor_features.unsqueeze(1)
            attn_output = self.attention(attn_input)
            actor_features = actor_features + attn_output.squeeze(1)
        
        # Apply residual blocks
        if self._use_residual:
            for residual_block in self.residual_blocks:
                actor_features = residual_block(actor_features)

        # RNN processing
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        # Evaluate actions
        action_log_probs, dist_entropy = self.act.evaluate_actions(
            actor_features,
            action, 
            available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None
        )

        return action_log_probs, dist_entropy


class R_Critic_Advanced(nn.Module):
    """
    改进的Critic网络，包含注意力机制和残差连接
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic_Advanced, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        # 是否使用注意力和残差
        self._use_attention = getattr(args, 'use_attention', True)
        self._use_residual = getattr(args, 'use_residual', True)
        
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)
        
        # 添加注意力机制
        if self._use_attention:
            self.attention = MultiHeadAttention(self.hidden_size, num_heads=4)
        
        # 添加残差块（支持通过 args 调整数量以实现更深网络）
        if self._use_residual:
            residual_N = getattr(args, 'adv_residual_blocks', 2)
            self.residual_blocks = nn.ModuleList([
                ResidualBlock(self.hidden_size, self._use_orthogonal)
                for _ in range(residual_N)
            ])

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        前向传播，计算价值函数
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        # Base feature extraction
        critic_features = self.base(cent_obs)
        
        # Apply attention mechanism
        if self._use_attention:
            attn_input = critic_features.unsqueeze(1)
            attn_output = self.attention(attn_input)
            critic_features = critic_features + attn_output.squeeze(1)
        
        # Apply residual blocks
        if self._use_residual:
            for residual_block in self.residual_blocks:
                critic_features = residual_block(critic_features)
        
        # RNN processing
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        
        # Value output
        values = self.v_out(critic_features)

        return values, rnn_states
