import torch

# System modules
from types import SimpleNamespace

# CUDA Python modules
from cuda import cuda

# CUTLASS modules
import library
import manifest as cutlass_manifest
import generator
import rt

cuda_ver = "11.4"
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

operation = manifest.operations_by_name['cutlass_simt_sgemm_256x128_8x5_nn_align1']  # TODO(yf225): should it be nn or nt?

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
  f'../../../cutlass/include',
  f'../../../cutlass/tools/util/include',
  f'/usr/local/cuda-{cuda_ver}/include',
  f'/usr/local/cuda-{cuda_ver}/targets/x86_64-linux/include',
]

compilation_options = rt.CompilationOptions(architectures, include_paths)

module = rt.Module('module.cu', [gemm], compilation_options)

#
# Setup a workspace
#

arguments = rt.GemmArguments()

arguments.problem_size = rt.GemmCoord(M, N, K)

tensor_A_torch = torch.empty(3, 4, device='cuda'),  # a
tensor_B_torch = torch.empty(4, 5, device='cuda'),  # b
tensor_C_torch = torch.empty(3, 5, device='cuda'),  # c
tensor_D_torch = tensor_C_torch

arguments.A = rt.TensorRef(cuda.CUdeviceptr(tensor_A_torch.data_ptr()), M)
arguments.B = rt.TensorRef(cuda.CUdeviceptr(tensor_B_torch.data_ptr()), N)
arguments.C = rt.TensorRef(cuda.CUdeviceptr(tensor_C_torch.data_ptr()), M)
arguments.D = rt.TensorRef(cuda.CUdeviceptr(tensor_D_torch.data_ptr()), M)

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
# err, = cuda.cuStreamSynchronize(stream)

if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError("CUDA Error %s" % str(err))
