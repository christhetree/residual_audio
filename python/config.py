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
EPS = 1e-8

N_FFT = 2048
# N_FFT = 1024
HOP_LEN = N_FFT // 4

AUDIO_CHUNKS_PT_DIR = os.path.join(DATA_DIR, f'audio_chunks_pt__{N_FFT}')

GPU_IDX = None
CPU_BATCH_SIZE = 8
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
