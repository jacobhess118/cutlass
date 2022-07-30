from types import SimpleNamespace
from cuda import cuda
import library
import manifest as cutlass_manifest
import generator
import rt
import cupy

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

operation = manifest.operations_by_name['cutlass_simt_sgemm_256x128_8x5_nn_align1']

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

M, N, K = (300, 500, 400)

tensor_A_cupy = cupy.ndarray(M * K, dtype=cupy.float32)
tensor_B_cupy = cupy.ndarray(N * K, dtype=cupy.float32)
tensor_C_cupy = cupy.ndarray(M * N, dtype=cupy.float32)
tensor_D_cupy = cupy.ndarray(M * N, dtype=cupy.float32)

print(f"tensor_A_cupy: {tensor_A_cupy}")
print(f"tensor_A_cupy.device: {tensor_A_cupy.device}")
print(f"tensor_A_cupy.data.ptr: {tensor_A_cupy.data.ptr}")

arguments = rt.GemmArguments()
arguments.problem_size = rt.GemmCoord(M, N, K)
arguments.A = rt.TensorRef(cuda.CUdeviceptr(tensor_A_cupy.data.ptr), M)
arguments.B = rt.TensorRef(cuda.CUdeviceptr(tensor_B_cupy.data.ptr), N)
arguments.C = rt.TensorRef(cuda.CUdeviceptr(tensor_C_cupy.data.ptr), M)
arguments.D = rt.TensorRef(cuda.CUdeviceptr(tensor_D_cupy.data.ptr), M)

host_workspace = bytearray(gemm.get_host_workspace_size(arguments))
device_workspace = None

launch_config = gemm.plan(arguments)

byte_count = gemm.initialize(host_workspace, device_workspace, launch_config, arguments)

err = gemm.run(host_workspace, device_workspace, launch_config)

if err != cuda.CUresult.CUDA_SUCCESS:
  raise RuntimeError('CUDA Error %s' % str(err))
