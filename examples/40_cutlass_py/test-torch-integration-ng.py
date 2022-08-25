"""
srun -p train --pty --cpus-per-task=96 -t 15:00:00 --gpus-per-node=8 --exclusive bash
conda activate torch_nightly_cuda

# First time
pip install cuda-python

export CUDA_VER_SHORT=114
export CUDA_VER=11.4

module unload cuda nccl nccl_efa
module load cuda/${CUDA_VER}
module load nccl/2.12.7-cuda.${CUDA_VER}
module load nccl_efa/1.2.0-nccl.2.12.7-cuda.${CUDA_VER}

export CUDA_HOME=/usr/local/cuda-${CUDA_VER}
export PATH=${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}:${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${CUDA_HOME}/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export CFLAGS=-I${CUDA_HOME}/include  # This helps CuPy find the right NCCL version
export LDFLAGS=-L${CUDA_HOME}/lib  # This helps CuPy find the right NCCL version
export CUDA_TOOLKIT_PATH=${CUDA_HOME}
export CUDNN_INSTALL_PATH=${CUDA_HOME}

export CUDACXX=${CUDA_HOME}/bin/nvcc
cd /fsx/users/${USER}/cutlass/
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=80               # compiles for NVIDIA Ampere GPU architecture

# Then
cd /fsx/users/${USER}/cutlass/examples/40_cutlass_py
export PYTHONPATH=/fsx/users/${USER}/cutlass/tools/library/scripts:$PYTHONPATH
python test-torch-integration.py

# For debugging:
CUDA_LAUNCH_BLOCKING=1 python examples/40_cutlass_py/test-torch-integration.py
"""

import torch
import numpy as np
from types import SimpleNamespace
from ctypes import c_void_p
from cuda import cuda
import library
import manifest as cutlass_manifest
import generator
import rt

cuda_ver = "11.4"
cuda_arch = "80"  # assuming A100

manifest = cutlass_manifest.Manifest()
generator.GenerateSM50_Simt(manifest, cuda_ver)

print(manifest.operations_by_name)
operation = manifest.operations_by_name['cutlass_simt_sgemm_128x128_8x2_nn_align1']

# TODO: can pass custom functor, e.g. `output_op = LinearCombinationReluFunctor()` if we have more pointwise ops to fuse
gemm = rt.Gemm(operation)

err, = cuda.cuInit(0)

if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError("CUDA Error %s" % str(err))

err, device = cuda.cuDeviceGet(0)

if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError("CUDA Error %s" % str(err))

err, context = cuda.cuCtxCreate(0, device)

if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError("CUDA Error %s" % str(err))

architectures = [int(cuda_arch),]
include_paths = [
  f'../../../cutlass/include',
  f'../../../cutlass/tools/util/include',
  f'/usr/local/cuda-{cuda_ver}/include',
  f'/usr/local/cuda-{cuda_ver}/targets/x86_64-linux/include',
]

compilation_options = rt.CompilationOptions(architectures, include_paths)

module = rt.Module('module.cu', [gemm], compilation_options)

# Formula: D = alpha * (A @ B) + beta * C
M, N, K = (3, 4, 5)
alpha = 1.0
beta = 1.0

tensor_A_torch = torch.arange(M*K, device='cuda', dtype=torch.float32).view(M, K)  # A
tensor_B_torch = torch.arange(K*N, device='cuda', dtype=torch.float32).view(K, N)  # B
tensor_C_torch = torch.arange(M*N, device='cuda', dtype=torch.float32).view(M, N)  # C, full-size, no broadcast
tensor_D_torch = torch.empty(M, N, device='cuda', dtype=torch.float32)  # D

pt_result = tensor_A_torch @ tensor_B_torch + tensor_C_torch

print(f"tensor_A_torch: {tensor_A_torch}")
print(f"tensor_B_torch: {tensor_B_torch}")
print(f"tensor_C_torch: {tensor_C_torch}")

arguments = rt.GemmArguments()
arguments.problem_size = rt.GemmCoord(M, N, K)
arguments.A = rt.TensorRef(tensor_A_torch.data_ptr(), tensor_A_torch.stride()[0])
arguments.B = rt.TensorRef(tensor_B_torch.data_ptr(), tensor_B_torch.stride()[0])
arguments.C = rt.TensorRef(tensor_C_torch.data_ptr(), tensor_C_torch.stride()[0])
arguments.D = rt.TensorRef(tensor_D_torch.data_ptr(), tensor_D_torch.stride()[0])
# TODO(yf225): can pass custom functor arguments: `arguments.output_op = LinearCombinationReluFunctorArguments(...)` if we have more pointwise ops to fuse
arguments.output_op.alpha = alpha
arguments.output_op.beta = beta

host_workspace = bytearray(gemm.get_host_workspace_size(arguments))
device_workspace = None

launch_config = gemm.plan(arguments)

byte_count = gemm.initialize(host_workspace, device_workspace, launch_config, arguments)

err = gemm.run(host_workspace, device_workspace, launch_config)

if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError('CUDA Error %s' % str(err))

print(f"tensor_D_torch: {tensor_D_torch}")
print(f"PyTorch result: {pt_result}")

assert torch.allclose(tensor_D_torch, pt_result)
