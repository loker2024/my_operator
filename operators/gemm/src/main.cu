// ============================================================================
// main.cu —— GEMM 测试入口：先运行 cuBLAS 对照，再注册被测内核；固定只运行一个正常场景。
// 后续版本只需在同名 .cuh/.cu 中实现 GemmKernel 签名，再向 kKernels 追加一项（名字、
// 内核地址、启动配置）即可复用 test_gemm_kernel；模板内核须在此处显式指定模板实参。
// 构建：cmake --build build --target gemm && ./build/operators/gemm/gemm。
// ============================================================================

#include <cstddef>
#include <cstdio>

#include "include/sgemm_v0.cuh"
#include "include/sgemm_v1.cuh"
#include "include/test.cuh"

namespace {

// sgemm_v0 的启动配置（每线程一个输出，见 include/sgemm_v0.cuh）：block 16×16 铺 M×N。
constexpr int kBlockX = 16;
constexpr int kBlockY = 16;
// sgemm_v1.cuh 的原始线性线程映射：BLOCKSIZE 是输出 tile 边长，需由 32×32 个线程覆盖。
constexpr int kSgemmV1BlockSize = 32;
constexpr int kSgemmV1Threads = kSgemmV1BlockSize * kSgemmV1BlockSize;

// 启动配置 —— 物理 block、grid 覆盖的输出 tile 与动态共享内存字节数，以各版本 .cuh 的
// 启动约束为准。v1 的物理 block 为 1024×1，但每 block 覆盖 32×32 个输出元素。
struct LaunchConfig {
	int block_x;
	int block_y;
	int tile_rows;
	int tile_cols;
	std::size_t smem_bytes;
};

struct KernelEntry {
	const char* name;
	GemmKernel kernel;
	LaunchConfig launch;
};

const KernelEntry kKernels[] = {
    {"sgemm_v0 (one thread per output element, global-memory baseline)",
     reinterpret_cast<GemmKernel>(sgemm_v0),
     {kBlockX, kBlockY, kBlockX, kBlockY, 0}},
    {"sgemm_v1 (BLOCKSIZE=32, original global-memory implementation)",
     reinterpret_cast<GemmKernel>(sgemm_v1<kSgemmV1BlockSize>),
     {kSgemmV1Threads, 1, kSgemmV1BlockSize, kSgemmV1BlockSize, 0}},
};

struct Scenario {
	const char* label;
	int M;
	int N;
	int K;
};

// 固定的正常测试场景：每个内核和 cuBLAS 对照只运行一次。
const Scenario kNormalScenarios[] = {
    {"normal: 512x512x512", 512, 512, 512},
};

int DivUp(int value, int divisor) {
	return (value + divisor - 1) / divisor;
}

}  // namespace

int main() {
	std::printf("==== GEMM test: correctness (tolerance 1e-3) + performance ====\n");
	std::printf("kernels under test = %zu\n", sizeof(kKernels) / sizeof(kKernels[0]));
	std::printf("scenario: 512x512x512\n");
	std::printf("sampling: 1 warmup + 100 iterations\n\n");

	bool all_ok = true;
	int passed = 0;
	int total = 0;
	for (const Scenario& scenario : kNormalScenarios) {
		char name[192];
		std::snprintf(name, sizeof(name), "cublasSgemm (CUBLAS_PEDANTIC_MATH) | %s",
		              scenario.label);
		const bool ok = test_cublas_sgemm(name, scenario.M, scenario.N, scenario.K);
		all_ok = all_ok && ok;
		passed += ok ? 1 : 0;
		++total;
		std::printf("\n");
	}
	for (const KernelEntry& kernel : kKernels) {
		for (const Scenario& scenario : kNormalScenarios) {
			char name[192];
			std::snprintf(name, sizeof(name), "%s | %s", kernel.name, scenario.label);
			const bool ok =
			    test_gemm_kernel(kernel.kernel, name, scenario.M, scenario.N, scenario.K,
			                     DivUp(scenario.M, kernel.launch.tile_rows),
			                     DivUp(scenario.N, kernel.launch.tile_cols), kernel.launch.block_x,
			                     kernel.launch.block_y, kernel.launch.smem_bytes);
			all_ok = all_ok && ok;
			passed += ok ? 1 : 0;
			++total;
			std::printf("\n");
		}
	}

	std::printf("==== Result: %d/%d items PASS%s ====\n", passed, total,
	            all_ok ? "" : ", FAIL present");
	return all_ok ? 0 : 1;
}
