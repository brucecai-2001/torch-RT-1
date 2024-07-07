import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenLearner(nn.Module):
    """TokenLearner module Version 1.1, using slightly different conv. layers.
    """

    def __init__(self, num_tokens, input_dim, bottleneck_dim=64, dropout_rate=0.0):
        super(TokenLearner, self).__init__()
        self.num_tokens = num_tokens
        self.bottleneck_dim = bottleneck_dim
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate

        self.layers = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(bottleneck_dim, num_tokens)
        )

    def forward(self, inputs, inference=True):
        """Applies learnable tokenization to the 2D inputs.

        Args:
          inputs: Inputs of shape `[bs*t, c, h, w]`.
          inference: Whether we are in the inference mode (e.g., inference time) or not.

        Returns:
          Output of shape `[bs*t, n_token, c]`.
        """
        if inputs.dim() == 4:
            bs_t, c, h, w = inputs.shape
            inputs = inputs.view(bs_t, -1, c) # Shape: [bs*t, h*w, c]

        attention_weights = self.layers(inputs)
        attention_weights = attention_weights.view(bs_t, self.num_tokens, -1)  # Shape: [bs*t, n_token, h*w]

        if not inference:
            attention_weights = F.dropout(attention_weights, p=self.dropout_rate, training=True)

        attention_weights = F.softmax(attention_weights, dim=-1)

        # Perform the einsum operation equivalent in PyTorch using bmm (batch matrix multiplication)
        feat = torch.bmm(attention_weights, inputs) # [bs*t, n_token, h*w] @ [bs, h*w, c]
        return feat # [bs*t, n_token, c]


# 测试TokenLearner模块
# tl = TokenLearner(num_tokens=8, input_dim=512, bottleneck_dim=512)
# input = torch.randn((1, 512, 9, 9))
# output = tl(input)
# print("Output shape:", output.shape)