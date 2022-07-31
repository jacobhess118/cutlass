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

manifest = cutlass_manifest.Manifest(args=SimpleNamespace(**dict(
  operations="all",
  build_dir=".",
  curr_build_dir=".",
  generator_target="library",
  architectures=cuda_arch,
  kernels="",
  ignore_kernels="",
  filter_by_cc="True",
  cuda_version=cuda_ver,
  kernel_filter_file=None,
  selected_kernel_list=None,
  interface_dir=None,
)))
generator.GenerateSM80_Simt_f32(manifest, None)

operation = manifest.operations_by_name['cutlass_simt_sgemm_256x128_8x5_nt_align1']

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

M, N, K = (128, 512, 256)

tensor_A_torch = torch.empty(M, K, device='cuda', dtype=torch.float32)  # A
tensor_B_torch = torch.empty(K, N, device='cuda', dtype=torch.float32)  # B
tensor_C_torch = torch.empty(M, N, device='cuda', dtype=torch.float32)  # C
# tensor_D_torch = tensor_C_torch
tensor_D_torch = torch.empty(M, N, device='cuda', dtype=torch.float32)  # D

# ======
# tensor_A = np.ndarray(M * K, dtype=np.float32)
# tensor_B = np.ndarray(N * K, dtype=np.float32)
# tensor_C = np.ndarray(M * N, dtype=np.float32)
# tensor_D = np.ndarray(M * N, dtype=np.float32)

# err, tensor_A_d = cuda.cuMemAlloc(tensor_A.size * tensor_A.itemsize)
# if err != cuda.CUresult.CUDA_SUCCESS:
#   raise RuntimeError("CUDA Error %s" % str(err))

# err, tensor_B_d = cuda.cuMemAlloc(tensor_B.size * tensor_B.itemsize)
# if err != cuda.CUresult.CUDA_SUCCESS:
#   raise RuntimeError("CUDA Error %s" % str(err))

# err, tensor_C_d = cuda.cuMemAlloc(tensor_C.size * tensor_C.itemsize)
# if err != cuda.CUresult.CUDA_SUCCESS:
#   raise RuntimeError("CUDA Error %s" % str(err))

# err, tensor_D_d = cuda.cuMemAlloc(tensor_D.size * tensor_D.itemsize)
# if err != cuda.CUresult.CUDA_SUCCESS:
#   raise RuntimeError("CUDA Error %s" % str(err))

# err, stream = cuda.cuStreamCreate(0)
# if err != cuda.CUresult.CUDA_SUCCESS:
#   raise RuntimeError("CUDA Error %s" % str(err))

# tensors = [
#   (tensor_A_d, tensor_A),
#   (tensor_B_d, tensor_B),
#   (tensor_C_d, tensor_C),
#   (tensor_D_d, tensor_D)
# ]

# for tensor_device, tensor_host in tensors:
#   bytes = tensor_host.size * tensor_host.itemsize
#   print("Tensor has dimensions: %s (%d bytes)" % (str(tensor_host.size), tensor_host.itemsize))
#   err, = cuda.cuMemcpyHtoDAsync(tensor_device, tensor_host, bytes, stream)
#   print("updating tensor in device memory ", hex(int(tensor_device)))
#   if err != cuda.CUresult.CUDA_SUCCESS:
#     raise RuntimeError('CUDA Error %s' % str(err))
# ======

arguments = rt.GemmArguments()
arguments.problem_size = rt.GemmCoord(M, N, K)
arguments.A = rt.TensorRef(cuda.CUdeviceptr(tensor_A_torch.data_ptr()), tensor_A_torch.stride()[0])
arguments.B = rt.TensorRef(cuda.CUdeviceptr(tensor_B_torch.data_ptr()), tensor_B_torch.stride()[0])
arguments.C = rt.TensorRef(cuda.CUdeviceptr(tensor_C_torch.data_ptr()), tensor_C_torch.stride()[0])
arguments.D = rt.TensorRef(cuda.CUdeviceptr(tensor_D_torch.data_ptr()), tensor_D_torch.stride()[0])

# arguments.A = rt.TensorRef(tensor_A_d, M)
# arguments.B = rt.TensorRef(tensor_B_d, N)
# arguments.C = rt.TensorRef(tensor_C_d, M)
# arguments.D = rt.TensorRef(tensor_D_d, M)

host_workspace = bytearray(gemm.get_host_workspace_size(arguments))
device_workspace = None

launch_config = gemm.plan(arguments)

byte_count = gemm.initialize(host_workspace, device_workspace, launch_config, arguments)

err = gemm.run(host_workspace, device_workspace, launch_config)

if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError('CUDA Error %s' % str(err))
