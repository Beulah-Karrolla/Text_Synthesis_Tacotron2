import sys
import os
import IPython
from IPython.display import Audio

inputs = [
    "Jean-Paul Marat fait deux voyages en Angleterre au temps de la Révolution.|00-fr|fr",
    "Jean-Paul Marat fait deux voyages en Angleterre au temps de la Révolution.|00-fr|de*0.1:fr*0.9",
    "Jean-Paul Marat fait deux voyages en Angleterre au temps de la Révolution.|00-fr|de*0.2:fr*0.8",
    "Jean-Paul Marat fait deux voyages en Angleterre au temps de la Révolution.|00-fr|de*0.3:fr*0.7",
    "Jean-Paul Marat fait deux voyages en Angleterre au temps de la Révolution.|00-fr|de*0.4:fr*0.6",
    "Jean-Paul Marat fait deux voyages en Angleterre au temps de la Révolution.|00-fr|de*0.5:fr*0.5",
    "Jean-Paul Marat fait deux voyages en Angleterre au temps de la Révolution.|00-fr|de*0.6:fr*0.4",
    "Jean-Paul Marat fait deux voyages en Angleterre au temps de la Révolution.|00-fr|de*0.7:fr*0.3",
    "Jean-Paul Marat fait deux voyages en Angleterre au temps de la Révolution.|00-fr|de*0.8:fr*0.2",
    "Jean-Paul Marat fait deux voyages en Angleterre au temps de la Révolution.|00-fr|de*0.9:fr*0.1",
    "Jean-Paul Marat fait deux voyages en Angleterre au temps de la Révolution.|00-fr|de",
]
tacotron_dir = "Multilingual_Text_to_Speech"
wavernn_dir = "WaveRNN"
tacotron_chpt = "generated_switching.pyt"
wavernn_chpt = "wavernn_weight.pyt"


os.chdir(os.path.join(os.path.expanduser("~"),"TTS_project", tacotron_dir))
if "utils" in sys.modules: del sys.modules["utils"]

from Multilingual_Text_to_Speech.synthesize import synthesize
from Multilingual_Text_to_Speech.utils import build_model

model = build_model(os.path.join(os.path.expanduser("~"),"TTS_project", "checkpoints", tacotron_chpt))
model.eval()

spectrograms = [synthesize(model, "|" + i) for i in inputs]


print(spectrograms)


os.chdir(os.path.join(os.path.expanduser("~"),"TTS_project", wavernn_dir))
if "utils" in sys.modules: del sys.modules["utils"]

from WaveRNN.wavernn.models.fatchord_version import WaveRNN
from WaveRNN.wavernn.utils import hparams as hp
from WaveRNN.scripts.gen_wavernn import generate
import torch

hp.configure('hparams.py')
model = WaveRNN(rnn_dims=hp.voc_rnn_dims, fc_dims=hp.voc_fc_dims, bits=hp.bits, pad=hp.voc_pad, upsample_factors=hp.voc_upsample_factors, 
                feat_dims=hp.num_mels, compute_dims=hp.voc_compute_dims, res_out_dims=hp.voc_res_out_dims, res_blocks=hp.voc_res_blocks, 
                hop_length=hp.hop_length, sample_rate=hp.sample_rate, mode=hp.voc_mode).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model.load(os.path.join(os.path.expanduser("~"),"TTS_project", "checkpoints", wavernn_chpt))

waveforms = [generate(model, s, hp.voc_gen_batched, hp.voc_target, hp.voc_overlap) for s in spectrograms]
from scipy.io.wavfile import write
from IPython.display import Audio
  

for idx, w in enumerate(waveforms):
  print(inputs[idx])
  #IPython.display.display(IPython.display.Audio(data=w, rate=hp.sample_rate))
  IPython.display.Audio(data=w, rate=hp.sample_rate)
  Audio(w, rate=hp.sample_rate)
  #write("audio.wav", hp.sample_rate, w)