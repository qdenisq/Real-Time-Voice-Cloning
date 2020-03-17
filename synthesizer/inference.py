# from synthesizer.tacotron2 import Tacotron2
import torch
from synthesizer.tacotron2.model import Tacotron2
from synthesizer.tacotron2.train import load_checkpoint
from synthesizer.tacotron2.hparams import create_hparams
from synthesizer.hparams import hparams
from synthesizer.tacotron2.data_utils import text_to_sequence
from multiprocess.pool import Pool  # You're free to use either one
#from multiprocessing import Pool   # 
from synthesizer import audio
from pathlib import Path
from typing import Union, List
import tensorflow as tf
import numpy as np
import numba.cuda
import librosa


class Synthesizer:
    # sample_rate = hparams.sample_rate
    hparams = create_hparams()
    
    def __init__(self, checkpoint_path: Path, verbose=True, low_mem=False):
        """
        Creates a synthesizer ready for inference. The actual model isn't loaded in memory until
        needed or until load() is called.
        
        :param checkpoints_dir: path to the directory containing the checkpoint file as well as the
        weight files (.data, .index and .meta files)
        :param verbose: if False, only tensorflow's output will be printed TODO: suppress them too
        :param low_mem: if True, the model will be loaded in a separate process and its resources 
        will be released after each usage. Adds a large overhead, only recommended if your GPU 
        memory is low (<= 2gb)
        """
        self.verbose = verbose
        self._low_mem = low_mem
        
        # Prepare the model
        self._model = None  # type: Tacotron2
        self.checkpoint_path = checkpoint_path

        # checkpoint_state = tf.train.get_checkpoint_state(checkpoints_dir)
        # if checkpoint_state is None:
        #     raise Exception("Could not find any synthesizer weights under %s" % checkpoints_dir)
        # self.checkpoint_fpath = checkpoint_state.model_checkpoint_path
        # if verbose:
        #     model_name = checkpoints_dir.parent.name.replace("logs-", "")
        #     step = int(self.checkpoint_fpath[self.checkpoint_fpath.rfind('-') + 1:])
        #     print("Found synthesizer \"%s\" trained to step %d" % (model_name, step))
     
    def is_loaded(self):
        """
        Whether the model is loaded in GPU memory.
        """
        return self._model is not None
    
    def load(self):
        """
        Effectively loads the model to GPU memory given the weights file that was passed in the
        constructor.
        """
        if self._low_mem:
            raise Exception("Cannot load the synthesizer permanently in low mem mode")

        optimizer=None
        self._model = Tacotron2(Synthesizer.hparams)
        learning_rate = Synthesizer.hparams.learning_rate
        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate,
                                     weight_decay=Synthesizer.hparams.weight_decay)
        self._model, optimizer, _learning_rate, iteration = load_checkpoint(
            self.checkpoint_path, self._model, optimizer)

        # tf.reset_default_graph()
        # self._model = Tacotron2(self.checkpoint_fpath, hparams)
            
    def synthesize_spectrograms(self, texts: List[str],
                                embeddings: Union[np.ndarray, List[np.ndarray]],
                                return_alignments=False):
        """
        Synthesizes mel spectrograms from texts and speaker embeddings.

        :param texts: a list of N text prompts to be synthesized
        :param embeddings: a numpy array or list of speaker embeddings of shape (N, 256) 
        :param return_alignments: if True, a matrix representing the alignments between the 
        characters
        and each decoder output step will be returned for each spectrogram
        :return: a list of N melspectrograms as numpy arrays of shape (80, Mi), where Mi is the 
        sequence length of spectrogram i, and possibly the alignments.
        """
        if not self._low_mem:
            # Usual inference mode: load the model on the first request and keep it loaded.
            if not self.is_loaded():
                self.load()
            # specs, alignments = self._model.my_synthesize(embeddings, texts)

            input_lengths, ids_sorted_decreasing = torch.sort(
                torch.LongTensor([len(x) for x in texts]),
                dim=0, descending=True)
            max_input_len = input_lengths[0]

            text_padded = torch.LongTensor(len(texts), max_input_len)
            text_padded.zero_()
            for i in range(len(ids_sorted_decreasing)):
                text = texts[ids_sorted_decreasing[i]]
                text_norm = torch.IntTensor(text_to_sequence(text, Synthesizer.hparams.text_cleaners))
                text_padded[i, :len(text_norm)] = text_norm

            embeddings = torch.from_numpy(embeddings)
            text_padded = text_padded
            _, specs, gates, alignments = self._model.inference(text_padded, embeddings)

            specs = specs.detach().cpu().numpy()
            gates = gates.detach().cpu().numpy()
            alignments = alignments.detach().cpu().numpy()


        else:
            # Low memory inference mode: load the model upon every request. The model has to be 
            # loaded in a separate process to be able to release GPU memory (a simple workaround 
            # to tensorflow's intricacies)
            specs, alignments = Pool(1).starmap(Synthesizer._one_shot_synthesize_spectrograms, 
                                                [(self.checkpoint_fpath, embeddings, texts)])[0]
    
        return (specs, alignments) if return_alignments else specs

    @staticmethod
    def _one_shot_synthesize_spectrograms(checkpoint_fpath, embeddings, texts):
        # Load the model and forward the inputs
        # tf.reset_default_graph()
        # model = Tacotron2(checkpoint_fpath, hparams)
        # specs, alignments = model.my_synthesize(embeddings, texts)

        model = None
        learning_rate = Synthesizer.hparams.learning_rate
        model = Tacotron2(Synthesizer.hparams)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=Synthesizer.hparams.weight_decay)
        model, optimizer, _learning_rate, iteration = load_checkpoint(checkpoint_fpath, model, optimizer)
        specs, alignments = model.inference(texts, embeddings)

        # Detach the outputs (not doing so will cause the process to hang)
        specs, alignments = [spec.copy() for spec in specs], alignments.copy()
        
        # Close cuda for this process
        model.session.close()
        numba.cuda.select_device(0)
        numba.cuda.close()
        
        return specs, alignments

    @staticmethod
    def load_preprocess_wav(fpath):
        """
        Loads and preprocesses an audio file under the same conditions the audio files were used to
        train the synthesizer. 
        """
        wav = librosa.load(fpath, hparams.sample_rate)[0]
        if hparams.rescale:
            wav = wav / np.abs(wav).max() * hparams.rescaling_max
        return wav

    @staticmethod
    def make_spectrogram(fpath_or_wav: Union[str, Path, np.ndarray]):
        """
        Creates a mel spectrogram from an audio file in the same manner as the mel spectrograms that 
        were fed to the synthesizer when training.
        """
        if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
            wav = Synthesizer.load_preprocess_wav(fpath_or_wav)
        else:
            wav = fpath_or_wav
        
        mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
        return mel_spectrogram
    
    @staticmethod
    def griffin_lim(mel):
        """
        Inverts a mel spectrogram using Griffin-Lim. The mel spectrogram is expected to have been built
        with the same parameters present in hparams.py.
        """
        return audio.inv_mel_spectrogram(mel, hparams)
    