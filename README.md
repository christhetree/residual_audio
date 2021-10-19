# residual_audio

Repo for compressing audio using PyTorch and obtaining the residual between the original and the reconstructed audio.

### Setup:

1. Install requirements in `requirements.txt` (repo is developed using Python 3.9), `requirements_all.txt` is a more verbose backup in case the former doesn't work
2. Ensure the `python` folder is appended to your `PYTHON_PATH`
3. Use `python/data_preparation.py` to process a folder of .wav files into training data.
4. Train and evaluate a model using `python/experiments.py`
5. Change global settings in `python/config.py`
