// ============================================================================
// main.cu —— GEMM 测试入口：注册被测内核并固定运行两组正常场景。
// 后续版本只需在同名 .cuh/.cu 中实现 GemmKernel 签名，再向 kKernels 添加一项即可复用
// test_gemm_kernel。构建：cmake --build build --target gemm && ./build/operators/gemm/gemm。
// ============================================================================

#include <cstddef>
#include <cstdio>

#include "include/sgemm_v0.cuh"
#include "include/test.cuh"

namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

struct KernelEntry {
	const char* name;
	GemmKernel kernel;
};

const KernelEntry kKernels[] = {
    {"sgemm_v0 (one thread per output element, global-memory baseline)", sgemm_v0},
};

struct Scenario {
	const char* label;
	int M;
	int N;
	int K;
};

const Scenario kNormalScenarios[] = {
    {"normal: 512x512x512", 512, 512, 512},
    {"normal: 513x511x509 (unaligned)", 513, 511, 509},
};

int DivUp(int value, int divisor) {
	return (value + divisor - 1) / divisor;
}

}  // namespace

int main() {
	std::printf("==== GEMM test: correctness (tolerance 1e-3) + performance ====\n");
	std::printf("block = (%d, %d); kernels under test = %zu\n", kBlockX, kBlockY,
	            sizeof(kKernels) / sizeof(kKernels[0]));
	std::printf("scenarios: 512x512x512 and 513x511x509; sampling: 1 warmup + 100 iterations\n\n");

	bool all_ok = true;
	int passed = 0;
	int total = 0;
	for (const KernelEntry& kernel : kKernels) {
		for (const Scenario& scenario : kNormalScenarios) {
			char name[192];
			std::snprintf(name, sizeof(name), "%s | %s", kernel.name, scenario.label);
			const bool ok = test_gemm_kernel(kernel.kernel, name, scenario.M, scenario.N,
			                                 scenario.K, DivUp(scenario.M, kBlockX),
			                                 DivUp(scenario.N, kBlockY), kBlockX, kBlockY);
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
