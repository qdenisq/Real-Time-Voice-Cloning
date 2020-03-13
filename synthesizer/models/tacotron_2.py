import torch
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


class Encoder(torch.nn.Module):
    def __init__(self, hparams):
        super(Encoder, self).__init__()


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

    def forward(self, *input: Any, **kwargs: Any) -> T_co:


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