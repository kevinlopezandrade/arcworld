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

from arcworld.training.models.pixeltransformer import (
    Embedding2d,
    EmbeddingLinear,
    Output2d,
    PositionalEncoding1d,
    PositionalEncoding2d,
)


class PixelTransformerModified(nn.Module):
    def __init__(
        self,
        h: int = 30,
        w: int = 30,
        pos_encoding: str = "2D",
        embedding: str = "conv",
        embedding_scaling: int = 1,
        max_input_output_pairs: int = 4,
    ):
        super().__init__()
        self.embedding_scaling = int(embedding_scaling)
        self.d_model = 80 * self.embedding_scaling
        self.max_input_output_pairs = max_input_output_pairs
        num_encoder_heads = 4
        num_decoder_heads = 4
        num_encoder_layers = 8
        num_decoder_layers = 8
        dim_feedforward = 64
        if embedding == "conv":
            self.embedding = Embedding2d(
                embedding_scaling=self.embedding_scaling, new_architecture=True
            )
        elif embedding == "linear":
            self.embedding = EmbeddingLinear(
                embedding_scaling=self.embedding_scaling, new_architecture=True
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
            (self.max_input_output_pairs - 1) * self.d_model, self.d_model
        )

        decoder_layer = TransformerDecoderLayer(
            self.d_model, num_decoder_heads, dim_feedforward, norm_first=True
        )

        self.program_len = 50

        self.pos_encoding_program = PositionalEncoding1d(
            self.d_model, seq_len=self.program_len
        )
        self.program = torch.nn.parameter.Parameter(
            torch.empty((self.program_len, 1, self.d_model - 2))
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

        self.register_buffer("not_program", torch.zeros((2, 1, 1, h, w)))
        self.register_buffer(
            "program_padding",
            torch.concatenate(
                [
                    torch.zeros((self.program_len, 1, 1)),
                    torch.ones(self.program_len, 1, 1),
                ],
                dim=2,
            ),
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
        program = self.program.expand(-1, b, -1)
        program = torch.concatenate(
            [program, self.program_padding.expand(-1, b, -1)], dim=2
        )
        program = self.pos_encoding_program(program)

        src = src.view(s, b, self.d_model - 2, h, w)
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
            memory = memory[: self.program_len]

            if total_memory is None:
                total_memory = memory
            else:
                total_memory = torch.concatenate(
                    (total_memory, memory), dim=2
                )  # dim 2 is the embedding dimension

        num_missing_samples = (self.max_input_output_pairs - 1) - s // 2

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
        empty_canvas = (
            empty_canvas.permute(3, 2, 0, 1).contiguous().view(h * w, b, self.d_model)
        )

        len_output = empty_canvas.shape[0]

        tgt = torch.concatenate([empty_canvas, tgt])

        output = self.decoder(tgt, self.linear(total_memory))
        output = output[:len_output]
        output = output.view(h, w, b, self.d_model).permute(2, 3, 0, 1)
        return self.final(output)
