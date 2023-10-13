import math
import warnings

import torch.nn
from torch import nn
from torch.nn import (
    Linear,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from torch.nn.init import xavier_uniform_


class EmbeddingLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            11, 11, kernel_size=1, stride=1, padding=0, padding_mode="zeros"
        )
        self.conv2 = nn.Conv2d(
            11, 15, kernel_size=1, stride=1, padding=0, padding_mode="zeros"
        )
        self.conv3 = nn.Conv2d(
            11, 20, kernel_size=1, stride=1, padding=0, padding_mode="zeros"
        )
        self.conv4 = nn.Conv2d(
            11, 33, kernel_size=1, stride=1, padding=0, padding_mode="zeros"
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return torch.concatenate((x4, x3, x2, x1), dim=1)


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


class PositionalEncoding1d(nn.Module):
    def __init__(self, d_model, h: int = 30, w: int = 30):
        super().__init__()
        pe = positionalencoding1d(
            d_model=d_model, length=h * w
        )  # reshaping inside function.
        pe = pe.view(d_model, h, w)
        self.register_buffer("pe", pe)

    def forward(self, seq):
        return seq + self.pe


class PositionalEncoding2d(nn.Module):
    def __init__(self, d_model, h: int = 30, w: int = 30):
        super().__init__()
        pe = positionalencoding2d(d_model, h, w)
        self.register_buffer("pe", pe)

    def forward(self, seq):
        return seq + self.pe


def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError(
            "Cannot use sin/cos positional encoding with "
            "odd dim (got dim={:d})".format(d_model)
        )
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp(
        (
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
    )
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe


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


class PixelTransformerModified(nn.Module):
    def __init__(self, h: int = 30, w: int = 30, pos_encoding="2D", embedding="conv"):
        super().__init__()
        self.d_model = 81
        num_encoder_heads = 4
        num_decoder_heads = 4
        num_encoder_layers = 8
        num_decoder_layers = 8
        dim_feedforward = 64
        if embedding == "conv":
            self.embedding = Embedding2d()
        elif embedding == "linear":
            self.embedding = EmbeddingLinear()
        else:
            warnings.warn(
                "Wrong embedding type entered. Please use 'conv' or 'linear'. \
                    Using default conv pos encoding..."
            )
        if pos_encoding == "2D":
            self.pos_encoding = PositionalEncoding2d(
                self.d_model, h=h, w=w
            )  # add pixel pos + inp/outp
        elif pos_encoding == "1D":
            self.pos_encoding = PositionalEncoding1d(
                self.d_model, h=h, w=w
            )  # add pixel pos + inp/outp
        else:
            warnings.warn(
                "Wrong positional encoding type entered. Please use '1D' or '2D'. \
                    Using default 2D pos encoding..."
            )
            self.pos_encoding = PositionalEncoding2d(self.d_model, h=h, w=w)

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

        self.program_len = 50
        self.program = torch.nn.parameter.Parameter(torch.empty((self.program_len, 1, self.d_model - 2)))

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

        self.register_buffer("not_program", torch.zeros((2, 1, 1, h, w)))
        self.register_buffer("program_padding",
                             torch.concatenate(
                                 [
                             torch.zeros((self.program_len, 1,1)),
                            torch.ones(self.program_len,1,1)

                                         ], dim = 2
                             )
                             )


        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def concat_zeros(self, x, b):
        return torch.concatenate(
            (x, self.inp_out_channel[0].expand(b, -1, -1, -1)), dim=1
        )

    def concat_ones(self, x, b):
        return torch.concatenate(
            (x, self.inp_out_channel[1].expand(b, -1, -1, -1)), dim=1
        )

    def forward(self, src, tgt):
        """
        Args:
            src: Tensor with shape [B, N_examples * 2, C, H, W]
            tgt: Tensor with shape [B, C, H, W]
        """
        b, s, c, h, w = src.shape
        src = src.permute(1, 0, 2, 3, 4)

        n_train_examples = s // 2

        src = self.embedding(src.reshape(s * b, c, h, w))

        empty_canvas = torch.zeros_like(tgt)
        empty_canvas = self.embedding(empty_canvas)

        tgt = self.embedding(tgt)
        # second to last channel otput will be all 0 denoting the input.
        tgt = self.concat_zeros(tgt, b)
        # second to last channel will all be 1 denoting output
        empty_canvas = self.concat_ones(empty_canvas, b)

        # last channel will all be 0 denoting its not a programm
        tgt = self.concat_zeros(tgt, b)
        empty_canvas = self.concat_zeros(empty_canvas, b)

        tgt = self.pos_encoding(tgt)
        empty_canvas = self.pos_encoding(empty_canvas)

        # second to last channel will be all zeroes because its not input
        # last channel all 1 because its program
        program = self.program.expand(-1,b,-1)
        program = torch.concatenate([program, self.program_padding.expand(-1,b,-1)], dim=2)
        program += positionalencoding1d(d_model=self.d_model, length=self.program_len)


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

            # add zeroes to indicate its not a program
            src_subset = torch.concatenate(
                (src_subset, self.not_program.expand(-1, b, -1, -1, -1)), dim=2
            )
            src_subset = self.pos_encoding(src_subset)
            src_subset = (
                src_subset.permute(0, 4, 3, 1, 2)
                .contiguous()
                .view(h * w * 2, b, self.d_model)
            )

            src_subset = torch.concatenate([program.clone(), src_subset])

            # Encoders in Pytorch expect by default the batch at the second dimension.
            memory = self.encoder(src_subset)  # 1800 len seq, 2 batch size, 80 d model
            memory = memory[:self.program_len]

            if total_memory is None:
                total_memory = memory
            else:
                total_memory = torch.concatenate(
                    (total_memory, memory), dim=2
                )  # dim 2 is the embedding dimension

        tgt = tgt.permute(3, 2, 0, 1).contiguous().view(h * w, b, self.d_model)
        empty_canvas = (
            empty_canvas.permute(3, 2, 0, 1).contiguous().view(h * w, b, self.d_model)
        )

        len_output = empty_canvas.shape[0]

        tgt = torch.concatenate([empty_canvas, tgt])

        output = self.decoder(tgt, self.linear(total_memory))
        output = output.view(h, w, b, self.d_model).permute(2, 3, 0, 1)
        output = output[:len_output]
        return self.final(output)


class PixelTransformer(nn.Module):
    def __init__(self, h: int = 30, w: int = 30, pos_encoding="2D", embedding="conv"):
        super().__init__()
        self.d_model = 80
        num_encoder_heads = 4
        num_decoder_heads = 4
        num_encoder_layers = 8
        num_decoder_layers = 8
        dim_feedforward = 64

        if embedding == "conv":
            self.embedding = Embedding2d()
        elif embedding == "linear":
            self.embedding = EmbeddingLinear()
        else:
            warnings.warn(
                "Wrong embedding type entered. Please use 'conv' or 'linear'. \
                    Using default conv pos encoding..."
            )
        if pos_encoding == "2D":
            self.pos_encoding = PositionalEncoding2d(
                self.d_model, h=h, w=w
            )  # add pixel pos + inp/outp
        elif pos_encoding == "1D":
            self.pos_encoding = PositionalEncoding1d(
                self.d_model, h=h, w=w
            )  # add pixel pos + inp/outp
        else:
            warnings.warn(
                "Wrong positional encoding type entered. Please use '1D' or '2D'. \
                    Using default 2D pos encoding..."
            )
            self.pos_encoding = PositionalEncoding2d(self.d_model, h=h, w=w)
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

        src = self.embedding(src.reshape(s * b, c, h, w))

        tgt = self.embedding(tgt)
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
