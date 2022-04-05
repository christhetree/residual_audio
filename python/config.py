import logging
import os

import torch as tr

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get('LOGLEVEL', 'INFO'))

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../out'))
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
RAW_AUDIO_DIR = os.path.join(DATA_DIR, 'raw_audio_in')

SR = 44100
# SR = 48000
EPS = 1e-8

N_FFT = 2048
HOP_LEN = N_FFT // 4
CENTER = True
MODEL_IO_N_FRAMES = 16

N_BINS = (N_FFT // 2) + 1
AUDIO_CHUNKS_PT_DIR = os.path.join(DATA_DIR, f'audio_chunks_pt'
                                             f'__sr_{SR}'
                                             f'__n_fft_{N_FFT}'
                                             f'__center_{CENTER}'
                                             f'__n_frames_{MODEL_IO_N_FRAMES}')

GPU_IDX = None
CPU_BATCH_SIZE = 128
GPU_BATCH_SIZE = 512
BATCH_SIZE = CPU_BATCH_SIZE
NUM_WORKERS = 0

if tr.cuda.is_available():
    GPU_IDX = 1
    BATCH_SIZE = GPU_BATCH_SIZE
    NUM_WORKERS = 0

GPU = None
if GPU_IDX is not None:
    GPU = [GPU_IDX]
