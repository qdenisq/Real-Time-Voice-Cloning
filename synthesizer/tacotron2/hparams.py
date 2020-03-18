import tensorflow as tf
from synthesizer.tacotron2.text import symbols


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = tf.contrib.training.HParams(
        ################################
        # Experiment Parameters        #
        ################################
        epochs=500,
        iters_per_checkpoint=500,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=True,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight'],

        ################################
        # Data Parameters             #
        ################################
        load_mel_from_disk=True,
        training_dir='C:\Study\Real-Time-Voice-Cloning\data\SV2TTS\synthesizer',
        test_size=0.05,
        # training_files='filelists/ljs_audio_text_train_filelist.txt',
        # validation_files='filelists/ljs_audio_text_val_filelist.txt',
        text_cleaners=['english_cleaners'],

        ################################
        # Audio Parameters             #
        ################################
        # max_wav_value=32768.0,
        # sampling_rate=22050,
        # filter_length=1024,
        # hop_length=256,
        # win_length=1024,
        # n_mel_channels=80,
        # mel_fmin=0.0,
        # mel_fmax=8000.0,

        max_wav_value=32768.0,
        sampling_rate=16000,
        filter_length=800,
        hop_length=200,
        win_length=800,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        # n_fft=800,  # Extra window size is filled with 0 paddings to match this parameter
        # hop_size=200,  # For 16000Hz, 200 = 12.5 ms (0.0125 * sample_rate)
        # win_size=800,  # For 16000Hz, 800 = 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
        # sample_rate=16000,  # 16000Hz (corresponding to librispeech) (sox --i <filename>)
        # max_mel_frames=900,
        ################################
        # Model Parameters             #
        ################################
        n_symbols=len(symbols),
        symbols_embedding_dim=512,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=512,

        # Decoder parameters
        n_frames_per_step=1,  # currently only 1 is supported
        decoder_rnn_dim=1024, #was 1024
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=1024, # was 1024
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=512,
        postnet_kernel_size=5,
        postnet_n_convolutions=3, # was 5

        ################################
        # Optimization Hyperparameters #
        ################################
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=6,
        mask_padding=True,  # set model's padded outputs to padded values

        #################################
        # Speaker Embedding
        #################################
        speaker_embedding_size = 256
    )

    if hparams_string:
        tf.logging.info('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        tf.logging.info('Final parsed hparams: %s', hparams.values())

    return hparams
