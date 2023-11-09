import math
import random
import warnings

import torch.nn
from torch import Tensor, nn
from torch.nn import (
    Linear,
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerEncoderLayer,
)
from torch.nn.init import xavier_uniform_


class EmbeddingLinear(nn.Module):
    def __init__(self, embedding_scaling=1, new_architecture=False):
        super().__init__()
        self.embedding_scaling = embedding_scaling
        self.new_architecture = new_architecture
        self.conv1 = nn.Conv2d(
            11,
            11 * self.embedding_scaling,
            kernel_size=1,
            stride=1,
            padding=0,
            padding_mode="zeros",
        )
        self.conv2 = nn.Conv2d(
            11,
            15 * self.embedding_scaling,
            kernel_size=1,
            stride=1,
            padding=0,
            padding_mode="zeros",
        )
        self.conv3 = nn.Conv2d(
            11,
            20 * self.embedding_scaling,
            kernel_size=1,
            stride=1,
            padding=0,
            padding_mode="zeros",
        )
        if not self.new_architecture:
            self.conv4 = nn.Conv2d(
                11,
                33 * self.embedding_scaling + self.embedding_scaling - 1,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode="zeros",
            )
        else:
            self.conv4 = nn.Conv2d(
                11,
                33 * self.embedding_scaling + self.embedding_scaling - 2,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode="zeros",
            )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        return torch.concatenate((x4, x3, x2, x1), dim=1)


class Embedding2d(nn.Module):
    def __init__(self, embedding_scaling=1, new_architecture=False):
        super().__init__()
        self.embedding_scaling = embedding_scaling
        self.new_architecture = new_architecture
        self.conv1 = nn.Conv2d(
            11,
            11 * self.embedding_scaling,
            kernel_size=1,
            stride=1,
            padding=0,
            padding_mode="zeros",
        )
        self.conv2 = nn.Conv2d(
            11,
            15 * self.embedding_scaling,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="zeros",
        )
        self.conv3 = nn.Conv2d(
            11,
            20 * self.embedding_scaling,
            kernel_size=5,
            stride=1,
            padding=2,
            padding_mode="zeros",
        )
        if not self.new_architecture:
            self.conv4 = nn.Conv2d(
                11,
                33 * self.embedding_scaling + self.embedding_scaling - 1,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode="zeros",
            )
        else:
            self.conv4 = nn.Conv2d(
                11,
                33 * self.embedding_scaling + self.embedding_scaling - 2,
                kernel_size=1,
                stride=1,
                padding=0,
                padding_mode="zeros",
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
    def __init__(self, d_model, image_size=None, seq_len=None):
        super().__init__()
        if image_size:
            self.is_seq = False
            pe = positionalencoding1d(
                d_model=d_model, length=image_size * image_size
            )  # reshaping inside function.
            pe = pe.view(image_size, image_size, d_model)
            pe = pe.permute(2, 0, 1)
        elif seq_len:
            self.is_seq = True
            pe = positionalencoding1d(
                d_model=d_model, length=seq_len
            )  # reshaping inside function.
            pe = pe.unsqueeze(1)
        else:
            raise ValueError
        self.register_buffer("pe", pe)

    def forward(self, seq):
        if self.is_seq:
            return seq + self.pe.expand(-1, seq.shape[1], -1)
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


class PixelTransformer(nn.Module):
    def __init__(
        self,
        h: int = 30,
        w: int = 30,
        pos_encoding: str = "2D",
        embedding: str = "conv",
        embedding_scaling: int = 1,
        max_input_otput_pairs: int = 4,
    ):
        super().__init__()
        self.embedding_scaling = int(embedding_scaling)
        self.d_model = 80 * self.embedding_scaling
        self.max_input_otput_pairs = max_input_otput_pairs
        num_encoder_heads = 4
        num_decoder_heads = 4
        num_encoder_layers = 8
        num_decoder_layers = 8
        dim_feedforward = 64

        if embedding == "conv":
            self.embedding = Embedding2d(
                embedding_scaling=self.embedding_scaling, new_architecture=False
            )
        elif embedding == "linear":
            self.embedding = EmbeddingLinear(
                embedding_scaling=self.embedding_scaling, new_architecture=False
            )
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
                self.d_model, image_size=h
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

        self.linear = Linear(
            (self.max_input_otput_pairs - 1) * self.d_model, self.d_model
        )

        decoder_layer = TransformerDecoderLayer(
            self.d_model, num_decoder_heads, dim_feedforward, norm_first=True
        )

        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
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

    def forward(self, src: Tensor, tgt: Tensor):
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

        num_missing_samples = (self.max_input_otput_pairs - 1) - s // 2

        assert total_memory is not None

        if num_missing_samples > 0:
            # Repeat some context vectors
            samples_to_repeat = random.choices(
                range(n_train_examples), k=num_missing_samples
            )

            for i in samples_to_repeat:
                total_memory = torch.concatenate(
                    (
                        total_memory,
                        total_memory[:, :, i * self.d_model : (i + 1) * self.d_model],
                    ),
                    dim=2,
                )

        tgt = tgt.permute(3, 2, 0, 1).contiguous().view(h * w, b, self.d_model)

        output = self.decoder(tgt, self.linear(total_memory))
        output = output.view(h, w, b, self.d_model).permute(2, 3, 0, 1)

        return self.final(output)
