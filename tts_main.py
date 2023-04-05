import sys
import os
import IPython
from IPython.display import Audio

inputs = [
          "Las hojas con los cantos afilados cortan los objetos que se sitúan entre ellas.|spanish",
          "C'est l'un des plus beaux palais de Mistretta dont le nom dérive d'une ancienne famille seigneuriale de la ville.|french",
          "Nach seiner Rückkehr wurde er Vorstand der Abteilung für Landesaufnahme im sächsischen Generalstab.|german",
          "Αυτές οι κριτικές αναζητήσεις εμφανίζονται στην ελληνική φωτογραφία μόλις στα τέλη της δεκαετίας.|greek",
          "Soms is dit inderdaad een papieren bonnetje, maar vaak ook een plastic muntje of een metalen munt.|dutch",
          "Päälaivojen korkeus antoi tilaa klerestorioikkunoille sivulaivojen yläpuolelle.|finnish",  
          "Háborús felek között általánosan elismerten a megadás vagy a fegyverszünet jeleként ismert a legtöbbek számára.|hungarian",
          "jìsuànjī dàxué zhǔyào xuékē shì kēxué hé jìzhúbù， xuéshēng kěyǐ huòqǔ jìsuànjīkēxué hé jìzhú de běnkē xuéwèi。|chinese",
          "yokuasa、 saheiji ha riyuu wo tsuke te jibun ha mou ippaku suru mune wo nakama ni tsuge、 mina wo kaeshi te shimau。|japanese",
          "Из города к церкви по склону холма ведут роскошно декорированная лестница в стиле необарокко.|russian",         
]
tacotron_dir = "Multilingual_Text_to_Speech"
wavernn_dir = "WaveRNN"
tacotron_chpt = "generated_training.pyt"
wavernn_chpt = "wavernn_weight.pyt"


os.chdir(os.path.join(os.path.expanduser("~"),"TTS_project", tacotron_dir))
if "utils" in sys.modules: del sys.modules["utils"]

from Multilingual_Text_to_Speech.synthesize import synthesize
from Multilingual_Text_to_Speech.utils import build_model

model = build_model(os.path.join(os.path.expanduser("~"),"TTS_project", "checkpoints", tacotron_chpt))
model.eval()

spectrograms = []
for i in inputs:
  tokens = i.split("|")
  s = synthesize(model, "|" + tokens[0] + "|" + tokens[1] + "|" + tokens[1])
  spectrograms.append(s)

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
  IPython.display.display(Audio(data=w, rate=hp.sample_rate))
  Audio(w, rate=hp.sample_rate)
  #write("audio.wav", hp.sample_rate, w)