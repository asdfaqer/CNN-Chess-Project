import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation block to adaptively recalibrate channel-wise feature responses.
    """
    def __init__(self, channel: int, reduction: int = 16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResidualBlock(nn.Module):
    """
    Standard Residual Block with optional Squeeze-and-Excitation.
    """
    def __init__(self, hidden_dim: int, use_se: bool = True):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(hidden_dim)
        
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(hidden_dim)
        
        self.use_se = use_se
        if self.use_se:
            self.se = SqueezeExcitation(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.norm1(self.conv1(x)), inplace=True)
        out = self.norm2(self.conv2(out))
        
        if self.use_se:
            out = self.se(out)
            
        out += residual
        return F.relu(out, inplace=True)

class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell for capturing spatial-temporal dependencies.
    """
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int, bias: bool):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2, kernel_size // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor: torch.Tensor, cur_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size: int, image_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width = image_size
        device = self.conv.weight.device
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))

class ChessRCCN(nn.Module):
    """
    Recurrent Convolutional Neural Network for Chess MOVE prediction.
    """
    def __init__(self, num_res_blocks: int = 10, hidden_dim: int = 64, use_se: bool = True, use_lstm: bool = True, input_channels: int = 17):
        super(ChessRCCN, self).__init__()
        self.use_lstm = use_lstm
        self.input_channels = input_channels
        
        self.conv_input = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim, use_se=use_se) for _ in range(num_res_blocks)])
        
        if self.use_lstm:
            self.conv_lstm = ConvLSTMCell(input_dim=hidden_dim, hidden_dim=hidden_dim, kernel_size=3, bias=True)
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 73, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        if not self.use_lstm:
            # Optimize for CNN_ONLY mode: Process all moves in parallel
            # Shape x: [batch, seq_len, 17, 8, 8] or [batch, 17, 8, 8]
            if x.dim() == 4:
                # [batch, 17, 8, 8]
                ft = F.relu(self.conv_input(x))
                for block in self.res_blocks:
                    ft = block(ft)
                policy = self.policy_head(ft).view(x.size(0), -1)
                return policy, None
            else:
                # [batch, seq_len, C, 8, 8] -> flatten to [batch * seq_len, C, 8, 8]
                b, s, c, h, w = x.size()
                x_flat = x.view(-1, c, h, w)
                ft = F.relu(self.conv_input(x_flat))
                for block in self.res_blocks:
                    ft = block(ft)
                policy = self.policy_head(ft).view(b, s, -1)
                return policy, None

        # Sequential mode for LSTM
        b, seq_len, c, h, w = x.size()
        
        if hidden_state is None:
            hidden_state = self.conv_lstm.init_hidden(b, (h, w))
            
        outputs = []
        
        for t in range(seq_len):
            xt = x[:, t, :, :, :]
            
            ft = F.relu(self.conv_input(xt))
            for block in self.res_blocks:
                ft = block(ft)
            
            h_next, c_next = self.conv_lstm(ft, hidden_state)
            policy_input = h_next
            
            # Use .flip() instead of torch.flip() for potential speedup
            h_next = h_next.flip(2)
            c_next = c_next.flip(2)
            hidden_state = (h_next, c_next)
            
            policy = self.policy_head(policy_input).view(b, -1)
            outputs.append(policy)
            
        outputs = torch.stack(outputs, dim=1)
        return outputs, hidden_state
