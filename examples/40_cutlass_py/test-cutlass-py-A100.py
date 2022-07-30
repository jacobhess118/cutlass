"""
export CUDA_VER=11.4
python test-cutlass-py-A100.py ${CUDA_VER}
"""


# System modules
import numpy as np
import os.path
import sys
import ctypes
from types import SimpleNamespace

# CUDA Python modules
from cuda import cuda
from cuda import nvrtc

# CUTLASS modules
import library
import manifest as cutlass_manifest
import generator
import rt

import torch


cuda_ver = sys.argv[1]
cuda_arch = "80"  # assuming A100

#
# Construct an SGEMM
#

manifest = cutlass_manifest.Manifest(args=SimpleNamespace(**dict(
  operations="all",  # Specifies the operation to generate (gemm, all)
  build_dir=".",  # CUTLASS top-level build directory
  curr_build_dir=".",  # CUTLASS current build directory. cmake files will be emitted in this directory
  generator_target="library",  # Target of CUTLASS Library Generator
  architectures=cuda_arch,  # Target compute architectures, can be 53;60;61;70;75;80
  kernels="",  # Comma delimited list to filter kernels by name
  ignore_kernels="",  # Comma delimited list of kernels to exclude from build
  filter_by_cc="True",  # If enabled, kernels whose comupte capability range is not satisfied by the build target are excluded
  cuda_version=cuda_ver,  # Semantic version string of CUDA Toolkit
  kernel_filter_file=None,  # Full path of filter file
  selected_kernel_list=None,  # Specify the output log file containing all enabled kernels in this build
  interface_dir=None,  # Interface header to kernels
)))
generator.GenerateSM80_Simt_f32(manifest, None)

#
# Look up the GEMM operation
#

# # List all operations available
# print(f"manifest.operations_by_name: {manifest.operations_by_name}")

operation = manifest.operations_by_name['cutlass_simt_sgemm_256x128_8x5_nt_align1']

#
# Construct a runtime GEMM operation
#
gemm = rt.Gemm(operation)

#
# Initialize context
#
err, = cuda.cuInit(0)

if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError("CUDA Error %s" % str(err))

err, device = cuda.cuDeviceGet(0)

if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError("CUDA Error %s" % str(err))

err, context = cuda.cuCtxCreate(0, device)

if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError("CUDA Error %s" % str(err))

#
# Construct a module
#

architectures = [int(cuda_arch),]
include_paths = [
  '../../include',
  '../../tools/util/include',
  f'/usr/local/cuda-{cuda_ver}/include',
  f'/usr/local/cuda-{cuda_ver}/targets/x86_64-linux/include',
]

compilation_options = rt.CompilationOptions(architectures, include_paths)

module = rt.Module('module.cu', [gemm], compilation_options)

#
# Setup a workspace
#

M, N, K = (128, 128, 128)

tensor_A = torch.empty(M * K, dtype=torch.float)
tensor_B = torch.empty(N * K, dtype=torch.float)
tensor_C = torch.empty(M * N, dtype=torch.float)
tensor_D = torch.empty(M * N, dtype=torch.float)

err, tensor_A_d = cuda.cuMemAlloc(tensor_A.size * tensor_A.itemsize)
if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError("CUDA Error %s" % str(err))

err, tensor_B_d = cuda.cuMemAlloc(tensor_B.size * tensor_B.itemsize)
if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError("CUDA Error %s" % str(err))

err, tensor_C_d = cuda.cuMemAlloc(tensor_C.size * tensor_C.itemsize)
if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError("CUDA Error %s" % str(err))

err, tensor_D_d = cuda.cuMemAlloc(tensor_D.size * tensor_D.itemsize)
if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError("CUDA Error %s" % str(err))

err, stream = cuda.cuStreamCreate(0)
if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError("CUDA Error %s" % str(err))

tensors = [
  (tensor_A_d, tensor_A),
  (tensor_B_d, tensor_B),
  (tensor_C_d, tensor_C),
  (tensor_D_d, tensor_D)
]

for tensor_device, tensor_host in tensors:
  bytes = tensor_host.size * tensor_host.itemsize
  print("Tensor has dimensions: %s (%d bytes)" % (str(tensor_host.size), tensor_host.itemsize))
  err, = cuda.cuMemcpyHtoDAsync(tensor_device, tensor_host, bytes, stream)
  print("updating tensor in device memory ", hex(int(tensor_device)))
  if err != cuda.CUresult.CUDA_SUCCESS:
    raise RuntimeError('CUDA Error %s' % str(err))

#
# Initialize a host buffer
#

arguments = rt.GemmArguments()

arguments.problem_size = rt.GemmCoord(M, N, K)

arguments.A = rt.TensorRef(tensor_A_d, M)
arguments.B = rt.TensorRef(tensor_B_d, N)
arguments.C = rt.TensorRef(tensor_C_d, M)
arguments.D = rt.TensorRef(tensor_D_d, M)

host_workspace = bytearray(gemm.get_host_workspace_size(arguments))
device_workspace = None

launch_config = gemm.plan(arguments)

byte_count = gemm.initialize(host_workspace, device_workspace, launch_config, arguments)

#
# Launch the kernel
#

err = gemm.run(host_workspace, device_workspace, launch_config)

if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError('CUDA Error %s' % str(err))

#
# Verify results
#
err, = cuda.cuStreamSynchronize(stream)

if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError("CUDA Error %s" % str(err))


#
# Debug reporting of byte array contents
#

def PrintBytearray(host_workspace):
  uint_str = None
  prefix = None
  print("uint32_t host_workspace[] = {")
  for idx, byte in enumerate(host_workspace):
    if not (idx % 4):
      if uint_str is not None:
        print(prefix, uint_str, ",")
      prefix = "/* offset: %d B */    0x" % idx
      uint_str = ""
    uint_str = "{:02x}".format(byte) + uint_str
  print("};")
