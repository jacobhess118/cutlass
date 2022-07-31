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

# manifest = cutlass_manifest.Manifest(args=SimpleNamespace(**dict(
#   operations="all",
#   build_dir=".",
#   curr_build_dir=".",
#   generator_target="library",
#   architectures=cuda_arch,
#   kernels="",
#   ignore_kernels="",
#   filter_by_cc="True",
#   cuda_version=cuda_ver,
#   kernel_filter_file=None,
#   selected_kernel_list=None,
#   interface_dir=None,
# )))
# generator.GenerateSM80_Simt_f32(manifest, None)
manifest = cutlass_manifest.Manifest()
generator.GenerateSM50_Simt(manifest, cuda_ver)

# operation = manifest.operations_by_name['cutlass_simt_sgemm_256x128_8x5_nt_align1']
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

M, N, K = (3, 4, 5)

# Formula: D = alpha * (A @ B) + beta * C

tensor_A_torch = torch.randn(M, K, device='cuda', dtype=torch.float32)  # A
tensor_B_torch = torch.randn(K, N, device='cuda', dtype=torch.float32)  # B
tensor_C_torch = torch.randn(M, N, device='cuda', dtype=torch.float32)  # C
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

host_workspace = bytearray(gemm.get_host_workspace_size(arguments))
device_workspace = None

launch_config = gemm.plan(arguments)

byte_count = gemm.initialize(host_workspace, device_workspace, launch_config, arguments)

err = gemm.run(host_workspace, device_workspace, launch_config)

if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError('CUDA Error %s' % str(err))

#
# Debug reporting of byte array contents
#

# def PrintBytearray(host_workspace):
#   uint_str = None
#   prefix = None
#   print("uint32_t host_workspace[] = {")
#   for idx, byte in enumerate(host_workspace):
#     if not (idx % 4):
#       if uint_str is not None:
#         print(prefix, uint_str, ",")
#       prefix = "/* offset: %d B */    0x" % idx
#       uint_str = ""
#     uint_str = "{:02x}".format(byte) + uint_str
#   print("};")

# PrintBytearray(host_workspace)

torch.cuda.synchronize()
print(f"tensor_D_torch: {tensor_D_torch}")
print(f"PyTorch result: {pt_result}")

assert torch.allclose(tensor_D_torch, pt_result)
