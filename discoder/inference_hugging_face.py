import torch
from IPython.display import Audio, display
from discoder.models import DisCoder
from discoder import meldataset, utils

device = "cpu"
audio_path = "input_dir/favorite crime- by Olivia Rodrigo (cover by Sally Kim).wav"

# load pretrained DisCoder model
discoder = DisCoder.from_pretrained("disco-eth/discoder").eval().to(device)

# load and pad 44.1 kHz audio file
audio_org, _ = meldataset.load_wav(full_path=audio_path, sr_target=discoder.config["sample_rate"], resample=True, normalize=True)
audio = torch.tensor(audio_org).unsqueeze(dim=0).to(device)
to_pad = discoder.config["segment_size"] - (audio.shape[-1] % discoder.config["segment_size"])
audio = torch.nn.functional.pad(audio, pad=(0, to_pad), mode="constant", value=0)

# create mel spectrogram
mel = utils.get_mel_spectrogram_from_config(audio, discoder.config)  # [B, 128, frames]

# reconstruct audio
with torch.no_grad():
    wav_recon = discoder(mel)  # [B, 1, time]
    wav_recon = wav_recon[..., 0:-to_pad]  # remove padding

display(Audio(audio_org, rate=discoder.config["sample_rate"]))
display(Audio(wav_recon.squeeze().cpu().numpy(), rate=discoder.config["sample_rate"]))