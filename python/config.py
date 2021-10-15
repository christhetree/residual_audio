import logging
import os

import torch as tr

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))


OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../out'))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
RAW_AUDIO_DIR = os.path.join(DATA_DIR, 'raw_audio')
PROC_AUDIO_DIR = os.path.join(DATA_DIR, 'proc_audio')

SR = 32000  # I can hear only up to 16kHz :(
BUFFER_SIZE_SECONDS = 3.00796875  # Makes STFT timesteps a nice number
N_SAMPLES = int(SR * BUFFER_SIZE_SECONDS)
STEP_SIZE_SECONDS = 0.05  # Hop length for chopping up raw audio
PEAK_VALUE = 0.99  # Max absolute value of training data waveforms
EPS = 1e-7
N_FFT = 1024
HOP_LENGTH = N_FFT // 4

GPU = None
CPU_BATCH_SIZE = 8
GPU_BATCH_SIZE = 64
BATCH_SIZE = CPU_BATCH_SIZE
NUM_WORKERS = 0

if tr.cuda.is_available():
    GPU = [0]
    BATCH_SIZE = GPU_BATCH_SIZE
    NUM_WORKERS = 4
