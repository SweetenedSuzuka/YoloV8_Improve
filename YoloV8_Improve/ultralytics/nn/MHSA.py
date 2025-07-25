import torch
import torch.nn as nn


class MHSA(nn.Module):
    """
    多头自注意力机制（Multi-Head Self-Attention, MHSA）实现

    参数：
    - n_dims: 输入特征的通道数（即维度大小）
    - width: 输入特征图的宽度
    - height: 输入特征图的高度
    - heads: 注意力头的数量
    - pos_emb: 是否启用相对位置编码
    """

    def __init__(self, n_dims, width=14, height=14, heads=4, pos_emb=False):
        super(MHSA, self).__init__()

        self.heads = heads

        # 初始化查询（query）、键（key）、值（value）的卷积映射
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)

        self.pos = pos_emb  # 是否使用位置编码

        # 如果使用相对位置编码，则初始化对应的相对位置权重参数
        if self.pos:
            # 高度方向的相对位置权重参数：[1, heads, 维度/头数, 1, 高度]
            self.rel_h_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, 1, int(height)]),
                                             requires_grad=True)
            # 宽度方向的相对位置权重参数：[1, heads, 维度/头数, 宽度, 1]
            self.rel_w_weight = nn.Parameter(torch.randn([1, heads, (n_dims) // heads, int(width), 1]),
                                             requires_grad=True)

        # 初始化 softmax 激活函数用于注意力得分的归一化
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        前向传播：

        参数：
        - x: 输入特征图，形状为 [batch_size, channels, width, height]

        返回：
        - out: 处理后的特征图，形状与输入相同
        """
        # 获取批次大小、通道数、宽度和高度
        n_batch, C, width, height = x.size()

        # 计算查询、键和值，并将它们拆分成多头结构
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)  # [B, heads, C/heads, W*H]
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)  # [B, heads, C/heads, W*H]
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)  # [B, heads, C/heads, W*H]

        # 计算内容之间的注意力权重（自注意力）：Q * K^T
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)  # [B, heads, W*H, W*H]
        c1, c2, c3, c4 = content_content.size()

        if self.pos:
            # 如果启用了相对位置编码，则计算位置权重的贡献
            # 合并相对位置编码的高度和宽度权重
            content_position = (self.rel_h_weight + self.rel_w_weight).view(1, self.heads, C // self.heads, -1).permute(
                0, 1, 3, 2)  # [1, heads, W*H, C/heads] # 1,4,1024,64

            # 将位置编码映射到查询空间
            content_position = torch.matmul(content_position, q)  # [B, heads, W*H, C/heads] # ([1, 4, 1024, 256])

            # 维度对齐
            content_position = content_position if (
                    content_content.shape == content_position.shape) else content_position[:, :, :c3, ]

            # 断言内容权重与位置权重的形状相等
            assert (content_content.shape == content_position.shape)

            # 总的能量矩阵为内容注意力 + 位置编码
            energy = content_content + content_position
        else:
            # 如果未使用位置编码，仅使用内容的自注意力
            energy = content_content

        # 通过 softmax 计算注意力权重
        attention = self.softmax(energy)

        # 加权求和并调整输出形状
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))  # [B, heads, C/heads, W*H] # 1,4,1024,64
        out = out.view(n_batch, C, width, height)

        return out
