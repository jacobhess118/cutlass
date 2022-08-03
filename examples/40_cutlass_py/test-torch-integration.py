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

operation = manifest.operations_by_name['cutlass_simt_sgemm_128x128_8x2_nn_align1']

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
tensor_C_torch = torch.arange(M*N, device='cuda', dtype=torch.float32).view(M, N)  # C
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
