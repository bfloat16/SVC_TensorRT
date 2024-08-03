from trt_inference import load_engine, TRT_Inference

import numpy as np

from time import time
import librosa
from glob import glob

def main(wav, contentvec, shapes, precision, max_workspace_size):
    time_start = time()
    engine = load_engine(contentvec, shapes, precision, max_workspace_size)

    trt_inference = TRT_Inference(engine)
    time_end = time()
    print(f"Contentvec engine loaded in {time_end - time_start:.3f} seconds")

    filelist = glob(f"{wav}/*.wav", recursive=True)
    for wav in filelist:
        print(f"Processing {wav}")
        audio, sr = librosa.load(wav, sr=16000)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        audio = audio.astype(np.float32)
        audio = np.expand_dims(audio, axis=0)
        audio = np.expand_dims(audio, axis=0)

        contentvec_inputs = {"input": audio}
        time_start = time()
        outputs = trt_inference.infer(contentvec_inputs)
        time_end = time()
        print(f"Contentvec inference done in {time_end - time_start:.3f} seconds")
        output = outputs['output']
        np.save("output.npy", output)

if __name__ == '__main__':
    input_wav = "wav"
    input_contentvec = "cvec.onnx"
    max_workspace_size = 6 << 30  # 6 GB
    precision = "fp16"

    shapes_contentvec = { # (min, opt, max)
    "input": ((1, 1, 256), (1, 1, 240128), (1, 1, 320000))
    }

    main(input_wav, input_contentvec, shapes_contentvec, precision, max_workspace_size)