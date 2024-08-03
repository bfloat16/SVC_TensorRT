import os
import sys

# Recommended use vsmlrt-cuda for TensorRT inference
# https://github.com/AmusementClub/vs-mlrt/releases
cuda_runtime = r"D:\AI\SVC_TensorRT"
os.environ["PATH"] = os.pathsep.join([f"{cuda_runtime}/bin", os.environ.get("PATH", "")])
os.environ['CUDA_PATH'] = cuda_runtime

import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit # This is needed for initializing CUDA driver

from collections import OrderedDict
import hashlib

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def calculate_hash(*args) -> str:
    hash_object = hashlib.sha1()
    for arg in args:
        hash_object.update(str(arg).encode('utf-8'))
    return hash_object.hexdigest()

def build_engine(onnx_model_path, shapes, precision, max_workspace_size):
    # Check precision
    if precision not in ['fp32', 'fp16']:
        print(f"Unsupported precision: {precision}")
        return False

    # Load the ONNX model
    if not os.path.exists(onnx_model_path):
        print(f"ONNX model file does not exist: {onnx_model_path}")
        return False
    
    # Create a TRT logger and builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_model_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print(f"Failed to parse ONNX model: {onnx_model_path}")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    # Create builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
    
    if precision == 'fp16' and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Create optimization profile
    profile = builder.create_optimization_profile()
    
    for input_name, (min_shape, opt_shape, max_shape) in shapes.items():
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    
    config.add_optimization_profile(profile)

    # Calculate current environment hash
    cuda_version = trt.__version__
    device_name = cuda.Device(0).name()
    current_hash = calculate_hash(onnx_model_path, shapes, precision, max_workspace_size, cuda_version, device_name)

    # Check if engine already exists and is up to date
    engine_path = os.path.splitext(onnx_model_path)[0] + ".engine"
    hash_path = os.path.splitext(onnx_model_path)[0] + ".hash"
    
    if os.path.exists(engine_path) and os.path.exists(hash_path):
        with open(hash_path, 'r') as hash_file:
            saved_hash = hash_file.read()
        if saved_hash == current_hash:
            print("Engine is up to date")
            return True
        else:
            print("Engine is outdated, rebuilding...")
    else:
        print("Engine does not exist, building...")

    # Build the engine
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Failed to build TensorRT engine")
        return False

    # Serialize the engine and save it
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)

    # Save the current hash
    with open(hash_path, 'w') as hash_file:
        hash_file.write(current_hash)

    return True

def load_engine(onnx_model_path, shapes, precision, max_workspace_size):
    if not build_engine(onnx_model_path, shapes, precision, max_workspace_size):
        sys.exit(1)
    
    engine_path = os.path.splitext(onnx_model_path)[0] + ".engine"
    runtime = trt.Runtime(TRT_LOGGER)
    
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    return engine

def get_input_tensor_names(engine):
    input_tensor_names = []
    for binding in engine:
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            input_tensor_names.append(binding)
    return input_tensor_names

def get_output_tensor_names(engine):
    output_tensor_names = []
    for binding in engine:
        if engine.get_tensor_mode(binding) == trt.TensorIOMode.OUTPUT:
            output_tensor_names.append(binding)
    return output_tensor_names

class OutputAllocator(trt.IOutputAllocator):
    def __init__(self):
        super().__init__()
        self.buffers = {}
        self.shapes = {}

    def reallocate_output(self, tensor_name, memory, size, alignment):
        if tensor_name in self.buffers:
            del self.buffers[tensor_name]
        
        address = cuda.mem_alloc(size)
        self.buffers[tensor_name] = address
        return int(address)
        
    def notify_shape(self, tensor_name, shape):
        self.shapes[tensor_name] = tuple(shape)

class TRT_Inference:
    def __init__(self, engine):
        self.engine = engine
        self.output_allocator = OutputAllocator()
        # create execution context
        self.context = engine.create_execution_context()
        # get input and output tensor names
        self.input_tensor_names = get_input_tensor_names(engine)
        self.output_tensor_names = get_output_tensor_names(engine)
        # create stream
        self.stream = cuda.Stream()
        # Create a CUDA events
        self.start_event = cuda.Event()
        self.end_event = cuda.Event()
        
    def infer(self, inputs):
        """
        1. create execution context
        2. set input shapes
        3. allocate memory
        4. copy input data to device
        5. run inference on device
        6. copy output data to host and reshape
        """
        if isinstance(inputs, np.ndarray):
            inputs = [inputs]
        if isinstance(inputs, dict):
            inputs = [inp if name in self.input_tensor_names else None for (name, inp) in inputs.items()]
        if isinstance(inputs, list):
            for name, arr in zip(self.input_tensor_names, inputs):
                self.context.set_input_shape(name, arr.shape)

        buffers_host = []
        buffers_device = []
        # copy input data to device
        for name, arr in zip(self.input_tensor_names, inputs):
            host = cuda.pagelocked_empty(arr.shape, dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
            device = cuda.mem_alloc(arr.nbytes)
            
            host[:] = arr
            cuda.memcpy_htod_async(device, host, self.stream)
            buffers_host.append(host)
            buffers_device.append(device)
        # set input tensor address
        for name, buffer in zip(self.input_tensor_names, buffers_device):
            self.context.set_tensor_address(name, int(buffer))
        # set output tensor allocator
        for name in self.output_tensor_names:
            self.context.set_tensor_address(name, 0) # set nullptr
            self.context.set_output_allocator(name, self.output_allocator)
        # The do_inference function will return a list of outputs
        
        # Record the start event
        self.start_event.record(self.stream)
        # Run inference.
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        # Record the end event
        self.end_event.record(self.stream)

        # self.memory.copy_to_host()
        
        output_buffers = OrderedDict()
        for name in self.output_tensor_names:
            arr = cuda.pagelocked_empty(self.output_allocator.shapes[name], dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
            cuda.memcpy_dtoh_async(arr, self.output_allocator.buffers[name], stream=self.stream)
            output_buffers[name] = arr
        
        # Synchronize the stream
        self.stream.synchronize()
        
        return output_buffers