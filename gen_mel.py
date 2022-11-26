import argparse
from utils.dsp import *
import numpy
from utils import hparams as hp
from utils.display import save_spectrogram
parser = argparse.ArgumentParser(description='Generating mel spectogram')
parser.add_argument('--path', '-p', help='path to wav file to transform')
parser.add_argument('--save_path', '-s', help='path to save mel')
parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
args = parser.parse_args()
hp.configure(args.hp_file)
y = load_wav(args.path)
peak = np.abs(y).max()
if hp.peak_norm or peak > 1.0:
    y /= peak
mel = melspectrogram(y)
mel = mel.astype(np.float32)

save_spectrogram(mel, args.save_path)
np.save(args.save_path + ".npy", mel, allow_pickle=False)