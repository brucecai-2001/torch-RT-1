import torch
import torch.nn as nn
import torch.nn.functional as F


class FilmConditioningLayer(nn.Module):
    """
    FiLM层是一种条件层, 用于调节神经网络的特征图, 使其能够根据条件信息(conditioning)进行特征的线性变换。
    条件信息可能是与图像相关的问题文本。这里是instructions的embedding
    """
    def __init__(self, num_channels: int, embed_dim=512):
        super(FilmConditioningLayer, self).__init__()

        # 用于学习FiLM层的加性（additive）变换参数。
        # 这个层将条件信息映射到与输入特征图相同数量的通道上，初始化为0。
        # num_channel 指定了卷积层输出的特征图的通道数
        self._projection_add = nn.Linear(embed_dim, num_channels)
        self._projection_add.weight.data.zero_()
        self._projection_add.bias.data.zero_()

        # 用于学习FiLM层的乘性（multiplicative）变换参数，也是初始化为0。
        self._projection_mult = nn.Linear(embed_dim, num_channels)
        self._projection_mult.weight.data.zero_()
        self._projection_mult.bias.data.zero_()

    def forward(self, conv_feature, conditioning):
        """
        result shape = conv_feature shape
        """
        projected_cond_add = self._projection_add(conditioning)
        projected_cond_add = projected_cond_add.unsqueeze(2).unsqueeze(3) # 1 x 24 x 1 x 1

        projected_cond_mult = self._projection_mult(conditioning)
        projected_cond_mult = projected_cond_mult.unsqueeze(2).unsqueeze(3)  # 1 x 24 x 1 x 1

        # 应用FiLM变换
        result = (1 + projected_cond_mult) * conv_feature + projected_cond_add
        return result
    

# f = FilmConditioningLayer(num_channels=24)
# frame = torch.randn(1, 24, 150, 150)
# con = torch.randn(1,512)
# o = f(frame, con)
# print(o.shape)