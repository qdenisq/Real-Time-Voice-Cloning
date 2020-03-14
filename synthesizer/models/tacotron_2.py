
import torch
from torch import nn
from torch.nn import functional as F
from synthesizer.utils.symbols import symbols
from synthesizer.infolog import log
from synthesizer.models.helpers import TacoTrainingHelper, TacoTestHelper
# from synthesizer.models.modules2 import *
from tensorflow.contrib.seq2seq import dynamic_decode
# from synthesizer.models.architecture_wrappers import TacotronEncoderCell, TacotronDecoderCell
from synthesizer.models.custom_decoder import CustomDecoder
from synthesizer.models.attention import LocationSensitiveAttention
from math import sqrt

import numpy as np


def split_func(x, split_pos):
    rst = []
    start = 0
    # x will be a numpy array with the contents of the placeholder below
    for i in range(split_pos.shape[0]):
        rst.append(x[:, start:start + split_pos[i]])
        start += split_pos[i]
    return rst


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Encoder(torch.nn.Module):
    def __init__(self, hparams):
        super(Encoder, self).__init__()
        self._hparams = hparams
        dims = [hparams.character_embedding_dim] + hparams.conv_dim

        self.conv_layers = nn.ModuleList()
        for i in range(len(dims) - 2):
            self.conv_layers.append(nn.Sequential(
                    ConvNorm(dims[i],
                             dims[i+1],
                             kernel_size=hparams.kernel_size,
                             stride=1,
                             padding=int((hparams.kernel_size - 1) / 2),
                             dilation=1,
                             w_init_gain='relu'),
                    nn.BatchNorm1d(hparams.encoder_embedding_dim)
            ))

        lstm_dim = hparams.lstm_dim if not hparams.lstm_bidirectional else hparams.lstm_dim // 2
        self.lstm = nn.LSTM(dims[-1], lstm_dim, 1, batch_first=True, bidirectional=hparams.lstm_bidirectional)

    def forward(self, x, input_lengths):
        for c in self.conv_layers:
            x = F.dropout(F.relu(c(x)), self._hparams.dropout, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for c in self.conv_layers:
            x = F.dropout(F.relu(c(x)), self._hparams.dropout, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs



class Decoder(torch.nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()


class Prenet(torch.nn.Module):
    def __init__(self, hparams):
        super(Prenet, self).__init__()


class Postnet(torch.nn.Module):
    def __init__(self, hparams):
        super(Postnet, self).__init__()


class Attention(torch.nn.Module):
    def __init__(self, hparams):
        super(Attention, self).__init__()







class Tacotron2(torch.nn.Module):
    """
    Tacotron-2 PyTorch
    """

    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self._hparams = hparams
        self.character_embedding = torch.nn.Embedding(hparams)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.character_embedding.weight.data.uniform_(-val, val)

        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.character_embedding(text_inputs).transpose(1,2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        #
        # log("Initialized Tacotron model. Dimensions (? = dynamic shape): ")
        # log("  Train mode:               {}".format(is_training))
        # log("  Eval mode:                {}".format(is_evaluating))
        # log("  GTA mode:                 {}".format(gta))
        # log("  Synthesis mode:           {}".format(not (is_training or is_evaluating)))
        # log("  Input:                    {}".format(inputs.shape))
        # for i in range(hp.tacotron_num_gpus + hp.tacotron_gpu_start_idx):
        #     log("  device:                   {}".format(i))
        #     log("  embedding:                {}".format(tower_embedded_inputs[i].shape))
        #     log("  enc conv out:             {}".format(tower_enc_conv_output_shape[i]))
        #     log("  encoder out (cond):       {}".format(tower_encoder_cond_outputs[i].shape))
        #     log("  decoder out:              {}".format(self.tower_decoder_output[i].shape))
        #     log("  residual out:             {}".format(tower_residual[i].shape))
        #     log("  projected residual out:   {}".format(tower_projected_residual[i].shape))
        #     log("  mel out:                  {}".format(self.tower_mel_outputs[i].shape))
        #     if post_condition:
        #         log("  linear out:               {}".format(self.tower_linear_outputs[i].shape))
        #     log("  <stop_token> out:         {}".format(self.tower_stop_token_prediction[i].shape))
        #
        #     # 1_000_000 is causing syntax problems for some people?! Python please :)
        #     log("  Tacotron Parameters       {:.3f} Million.".format(
        #         np.sum([np.prod(v.get_shape().as_list()) for v in self.all_vars]) / 1000000))