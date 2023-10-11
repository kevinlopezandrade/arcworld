import math

import torch.nn
from torch import nn
from torch.nn import (
    Linear,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from torch.nn.init import xavier_uniform_


class Embedding2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            11, 11, kernel_size=1, stride=1, padding=0, padding_mode="zeros"
        )
        self.conv2 = nn.Conv2d(
            11, 15, kernel_size=3, stride=1, padding=1, padding_mode="zeros"
        )
        self.conv3 = nn.Conv2d(
            11, 20, kernel_size=5, stride=1, padding=2, padding_mode="zeros"
        )
        self.conv4 = nn.Conv2d(
            11, 33, kernel_size=11, stride=1, padding=5, padding_mode="zeros"
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return torch.concatenate((x4, x3, x2, x1), dim=1)


class Output2d(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.conv = nn.Conv2d(
            d_model, 11, kernel_size=1, stride=1, padding=0, padding_mode="zeros"
        )

    def forward(self, x):
        return self.conv(x)


class PositionalEncoding2d(nn.Module):
    def __init__(self, d_model, h: int = 30, w: int = 30):
        super().__init__()
        pe = positionalencoding2d(d_model, h, w)
        self.register_buffer("pe", pe)

    def forward(self, seq):
        return seq + self.pe


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dimension (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = (
        torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[1:d_model:2, :, :] = (
        torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    )
    pe[d_model::2, :, :] = (
        torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )
    pe[d_model + 1 :: 2, :, :] = (
        torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    )

    return pe


class PixelTransformer(nn.Module):
    def __init__(self, h: int = 30, w: int = 30):
        super().__init__()
        self.d_model = 80
        num_encoder_heads = 4
        num_decoder_heads = 4
        num_encoder_layers = 8
        num_decoder_layers = 8
        dim_feedforward = 64
        self.embedding2d = Embedding2d()
        self.pos_encoding = PositionalEncoding2d(
            self.d_model, h=h, w=w
        )  # add pixel pos + inp/outp
        encoder_layer = TransformerEncoderLayer(
            self.d_model, num_encoder_heads, dim_feedforward, norm_first=True
        )
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, enable_nested_tensor=False
        )

        self.linear = Linear(3 * self.d_model, self.d_model)

        decoder_layer = TransformerDecoderLayer(
            self.d_model, num_decoder_heads, dim_feedforward, norm_first=True
        )

        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.final = Output2d(self.d_model)

        self.register_buffer(
            "inp_out_channel",
            torch.concatenate(
                (
                    torch.zeros((1, 1, 1, h, w)),
                    torch.ones((1, 1, 1, h, w)),
                ),
                dim=0,
            ),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, tgt):
        """
        Args:
            src: Tensor with shape [B, N_examples * 2, C, H, W]
            tgt: Tensor with shape [B, C, H, W]
        """
        b, s, c, h, w = src.shape
        src = src.permute(1, 0, 2, 3, 4)

        n_train_examples = s // 2

        src = self.embedding2d(src.reshape(s * b, c, h, w))

        tgt = self.embedding2d(tgt)
        # Last channel otput will be all 0 denoting the input.
        tgt = torch.concatenate(
            (tgt, self.inp_out_channel[0].expand(b, -1, -1, -1)), dim=1
        )
        tgt = self.pos_encoding(tgt)

        src = src.view(s, b, self.d_model - 1, h, w)
        total_memory = None
        for i in range(n_train_examples):
            inp = i * 2
            src_subset = src[
                inp : inp + 2
            ]  # 2 (input output pairs) x batch_size x embedding size x height x width
            src_subset = torch.concatenate(
                (src_subset, self.inp_out_channel.expand(-1, b, -1, -1, -1)), dim=2
            )
            src_subset = self.pos_encoding(src_subset)
            src_subset = (
                src_subset.permute(0, 4, 3, 1, 2)
                .contiguous()
                .view(h * w * 2, b, self.d_model)
            )

            # Encoders in Pytorch expect by default the batch at the second dimension.
            memory = self.encoder(src_subset)  # 1800 len seq, 2 batch size, 80 d model

            if total_memory is None:
                total_memory = memory
            else:
                total_memory = torch.concatenate(
                    (total_memory, memory), dim=2
                )  # dim 2 is the embedding dimension

        tgt = tgt.permute(3, 2, 0, 1).contiguous().view(h * w, b, self.d_model)

        output = self.decoder(tgt, self.linear(total_memory))
        output = output.view(h, w, b, self.d_model).permute(2, 3, 0, 1)

        return self.final(output)
