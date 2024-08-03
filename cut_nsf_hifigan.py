import onnx

input_model_path = 'nsf_hifigan.onnx'
nsf_model_path = 'nsf.onnx'
hifigan_model_path = 'hifigan.onnx'


nsf_input = ["f0"]
nsf_output = ["/generator/m_source/l_tanh/Tanh_output_0"]

hifigan_input = ["mel", "/generator/m_source/l_tanh/Tanh_output_0"]
hifigan_output = ["waveform"]

onnx.utils.extract_model(input_model_path, nsf_model_path, nsf_input, nsf_output, check_model=True)
onnx.utils.extract_model(input_model_path, hifigan_model_path, hifigan_input, hifigan_output, check_model=True)