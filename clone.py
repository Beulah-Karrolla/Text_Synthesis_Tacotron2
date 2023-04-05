os.chdir(os.path.expanduser("~"))
    
tacotron_dir = "Multilingual_Text_to_Speech"
if not os.path.exists(tacotron_dir):
  git clone https://github.com/Tomiinek/Multilingual_Text_to_Speech # $tacotron_dir

wavernn_dir = "WaveRNN"
if not os.path.exists(wavernn_dir):
  ! git clone https://github.com/Tomiinek/$wavernn_dir