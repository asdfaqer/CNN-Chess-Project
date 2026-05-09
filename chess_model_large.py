# chess_cnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlockSE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction=16)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out) 
        out = out + identity
        return F.relu(out)

class ChessCNN(nn.Module):
    def __init__(self, num_policy_outputs=4672):
        super().__init__()
        
        self.board_size = 8
        self.in_channels = 19
        self.num_channels = 64
        self.num_res_blocks = 10
        head_fc_size = 32
        head_conv_channels = 2
        
        fc1_input_size = head_conv_channels * self.board_size * self.board_size

        # 1. Input
        self.conv_in = nn.Conv2d(self.in_channels, self.num_channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(self.num_channels)

        # 2. Residual Tower
        self.res_stack = nn.ModuleList([ResidualBlockSE(self.num_channels, self.num_channels) for _ in range(self.num_res_blocks)])
        
        self.flatten = nn.Flatten()
        
        # 3. Value Head
        self.value_conv = nn.Conv2d(self.num_channels, head_conv_channels, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(head_conv_channels)
        self.value_fc1 = nn.Linear(fc1_input_size, head_fc_size)
        self.value_fc2 = nn.Linear(head_fc_size, 1)

        # 4. Policy Head (AlphaZero Style)
        # We output 73 planes. 
        # When flattened, this becomes 73 * 8 * 8 = 4672 outputs
        self.policy_conv = nn.Conv2d(self.num_channels, 73, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(73)
        # No final Linear layer needed for policy if we map directly to the conv outputs!
        
    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        
        for block in self.res_stack:
            x = block(x)
            
        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = self.flatten(v) 
        v = F.relu(self.value_fc1(v))
        value_out = torch.sigmoid(self.value_fc2(v))
        
        # Policy
        # Output shape: (Batch, 73, 8, 8)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        # Flatten to (Batch, 4672) to match CrossEntropyLoss expectation
        policy_logits = self.flatten(p) 
        
        return value_out, policy_logits