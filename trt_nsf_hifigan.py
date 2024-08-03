from trt_inference import load_engine, TRT_Inference

import torch
import torch.nn as nn
import numpy as np

from time import time
import parselmouth
import torchaudio
import librosa
from glob import glob
import soundfile as sf
from librosa.filters import mel as librosa_mel_fn

import onnxruntime as ort

# https://github.com/fishaudio/fish-diffusion/blob/main/tests/test_nsf_hifigan.py
def repeat_expand(content, target_len, mode="nearest"):
    ndim = content.ndim

    if content.ndim == 1:
        content = content[None, None]
    elif content.ndim == 2:
        content = content[None]

    assert content.ndim == 3

    is_np = isinstance(content, np.ndarray)
    if is_np:
        content = torch.from_numpy(content)

    results = torch.nn.functional.interpolate(content, size=target_len, mode=mode)

    if is_np:
        results = results.numpy()

    if ndim == 1:
        return results[0, 0]
    elif ndim == 2:
        return results[0]
    
def interpolate(x, xp, fp, left=None, right=None):
    i = torch.clip(torch.searchsorted(xp, x, right=True), 1, len(xp) - 1)
    interped = (fp[i - 1] * (xp[i] - x) + fp[i] * (x - xp[i - 1])) / (xp[i] - xp[i - 1])

    if left is None:
        left = fp[0]

    interped = torch.where(x < xp[0], left, interped)

    if right is None:
        right = fp[-1]

    interped = torch.where(x > xp[-1], right, interped)

    return interped
    
class BasePitchExtractor(nn.Module):
    def __init__(self, hop_length=512, f0_min=50.0, f0_max=1100.0, keep_zeros=True):
        super().__init__()

        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.keep_zeros = keep_zeros

    def post_process(self, x, sampling_rate, f0, pad_to):
        if isinstance(f0, np.ndarray):
            f0 = torch.from_numpy(f0).float().to(x.device)

        if pad_to is None:
            return f0

        f0 = repeat_expand(f0, pad_to)

        if self.keep_zeros:
            return f0

        nzindex = torch.nonzero(f0).squeeze()
        f0 = torch.index_select(f0, dim=0, index=nzindex)
        time_org = self.hop_length / sampling_rate * nzindex
        time_frame = (torch.arange(pad_to, device=x.device) * self.hop_length / sampling_rate)

        if f0.shape[0] <= 0:
            return torch.zeros(pad_to, dtype=torch.float, device=x.device)

        if f0.shape[0] == 1:
            return torch.ones(pad_to, dtype=torch.float, device=x.device) * f0[0]

        return interpolate(time_frame, time_org, f0, left=f0[0], right=f0[-1])
    
class ParselMouthPitchExtractor(BasePitchExtractor):
    def __call__(self, x, sampling_rate=44100, pad_to=None):
        assert x.ndim == 2, f"Expected 2D tensor, got {x.ndim}D tensor."
        assert x.shape[0] == 1, f"Expected 1 channel, got {x.shape[0]} channels."

        time_step = self.hop_length / sampling_rate

        f0 = (parselmouth.Sound(x[0].cpu().numpy(), sampling_rate).to_pitch_ac(time_step=time_step, voicing_threshold=0.6, pitch_floor=self.f0_min, pitch_ceiling=self.f0_max).selected_array["frequency"])

        if pad_to is not None:
            total_pad = pad_to - f0.shape[0]
            f0 = np.pad(f0, (total_pad // 2, total_pad - total_pad // 2), "constant")

        return self.post_process(x, sampling_rate, f0, pad_to)

class PitchAdjustableMelSpectrogram:
    def __init__(self, sample_rate=44100, n_fft=2048, win_length=2048, hop_length=512, f_min=40, f_max=16000, n_mels=128, center=False):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_size = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels
        self.center = center

        self.mel_basis = {}
        self.hann_window = {}

    def __call__(self, y, key_shift=0, speed=1.0):
        factor = 2 ** (key_shift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_size_new = int(np.round(self.win_size * factor))
        hop_length = int(np.round(self.hop_length * speed))

        mel_basis_key = f"{self.f_max}_{y.device}"
        if mel_basis_key not in self.mel_basis:
            mel = librosa_mel_fn(sr=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels, fmin=self.f_min, fmax=self.f_max)
            self.mel_basis[mel_basis_key] = torch.from_numpy(mel).float().to(y.device)

        hann_window_key = f"{key_shift}_{y.device}"
        if hann_window_key not in self.hann_window:
            self.hann_window[hann_window_key] = torch.hann_window(win_size_new, device=y.device)

        y = torch.nn.functional.pad(y.unsqueeze(1), (int((win_size_new - hop_length) / 2), int((win_size_new - hop_length) / 2)), mode="reflect")
        y = y.squeeze(1)

        spec = torch.stft(y, n_fft_new, hop_length=hop_length, win_length=win_size_new, window=self.hann_window[hann_window_key],
                          center=self.center, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

        if key_shift != 0:
            size = self.n_fft // 2 + 1
            resize = spec.size(1)
            if resize < size:
                spec = torch.nn.functional.pad(spec, (0, 0, 0, size - resize))

            spec = spec[:, :size, :] * self.win_size / win_size_new

        spec = torch.matmul(self.mel_basis[mel_basis_key], spec)

        return spec

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def wav2spec(wav_torch, sr=44100, key_shift=0, speed=1.0, mel_transform=None, use_natural_log=False):
    if sr != 44100:
        _wav_torch = librosa.resample(wav_torch.cpu().numpy(), orig_sr=sr, target_sr=44100)
        wav_torch = torch.from_numpy(_wav_torch).to(wav_torch.device)

    mel_torch = mel_transform(wav_torch, key_shift=key_shift, speed=speed)[0]
    mel_torch = dynamic_range_compression(mel_torch)

    if use_natural_log is False:
        mel_torch = 0.434294 * mel_torch

    return mel_torch

def main(wav, nsf, hifigan, shapes, precision, max_workspace_size):
    time_start = time()
    engine = load_engine(hifigan, shapes, precision, max_workspace_size)

    trt_inference = TRT_Inference(engine)
    time_end = time()
    print(f"Hifigan engine loaded in {time_end - time_start:.2f} seconds")

    time_start = time()
    nsf_session = ort.InferenceSession(nsf) # 别用CUDA！
    time_end = time()
    print(f"NSF session    loaded in {time_end - time_start:.2f} seconds")

    filelist = glob(f"{wav}/*.wav", recursive=True)
    for wav in filelist:
        print(f"Processing {wav}")
        audio, _ = torchaudio.load(wav)

        time_start = time()
        mel_transform = PitchAdjustableMelSpectrogram()
        mel = wav2spec(audio, mel_transform=mel_transform)
        time_end = time()
        print(f"Mel computed in {time_end - time_start:.3f} seconds")

        time_start = time()
        f0 = ParselMouthPitchExtractor(f0_min=40.0, f0_max=2000.0, keep_zeros=False)(audio, 44100, pad_to=mel.shape[-1])
        time_end = time()
        print(f"F0  computed in {time_end - time_start:.3f} seconds")

        if mel.shape[1] > shapes["mel"][2][1]:
            raise ValueError(f"mel length {mel.shape[1]} exceeds maximum length {shapes['mel'][2][1]}")

        c = mel[None]
        c = c.permute(0, 2, 1)
        f0 = f0[None].to(c.dtype)

        nsf_inputs = {"f0": f0.numpy()}
        time_start = time()
        nsf_outputs = nsf_session.run(None, nsf_inputs)
        time_end = time()
        print(f"NSF     inference done in {time_end - time_start:.3f} seconds")
        Tanh_output_0 = nsf_outputs[0]

        hifigan_inputs = {"mel": c.numpy(), "/generator/m_source/l_tanh/Tanh_output_0": Tanh_output_0}
        time_start = time()
        outputs = trt_inference.infer(hifigan_inputs)
        time_end = time()
        print(f"Hifigan inference done in {time_end - time_start:.3f} seconds")
        output = outputs['waveform'].flatten()

        sf.write("output.wav", output, 44100)

if __name__ == '__main__':
    input_wav = "wav"
    input_nsf = "nsf.onnx"
    input_hifigan = "hifigan.onnx"
    max_workspace_size = 6 << 30  # 6 GB
    precision = "fp16"

    shapes_nsf = { # (min, opt, max)
    "f0": ((1, 128), (1, 1500), (1, 3000))
    }

    shapes_hifigan = { # (min, opt, max)
    "mel": ((1, 2656, 128), (1, 2656, 128), (1, 2656, 128)),
    "/generator/m_source/l_tanh/Tanh_output_0": ((1, 1359872, 1), (1, 1359872, 1), (1, 1359872, 1))
    }

    main(input_wav, input_nsf, input_hifigan, shapes_hifigan, precision, max_workspace_size)