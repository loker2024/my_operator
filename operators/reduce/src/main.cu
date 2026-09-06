// ============================================================================
// main.cu —— 一维归约算子的执行入口：运行测试（基准）
//
// 职责
//   本文件只做“执行”：把被测的归约内核（reduce_v0，未来 v1/v2…）依次注册
//   给 test_reduce_kernel（声明见 test.cuh，实现见 test.cu），完成全部测试并
//   以退出码汇报结果（0 = 全部通过）。
//
// 新增内核版本的接入方式（可复用测试的关键）：
//   只需在 main() 里照抄一行 test_reduce_kernel(...) 并更换第一个参数，
//   例如 reduce_v1、reduce_v2；无需改动任何测试代码。
//
// 各文件职责总览
//   reduce.cuh / reduce.cu   算子接口声明与实现（被测试对象）
//   test.cuh  / test.cu      可复用测试驱动的声明与实现
//   main.cu                  执行入口（本文件）
//
// 构建与运行（仓库根目录）：
//   CMake：  cmake -S . -B build && cmake --build build --target reduce
//            ./build/operators/reduce/reduce
//   或直接 nvcc（无需触碰工程构建目录）：
//   nvcc -std=c++17 -arch=native -I common/include \
//        operators/reduce/src/reduce.cu operators/reduce/src/test.cu \
//        operators/reduce/src/main.cu -o /tmp/reduce_run && /tmp/reduce_run
// ============================================================================

#include <cstdio>  // printf

#include "test.cuh"    // test_reduce_kernel 声明（内部经 reduce.cuh 引入算子接口）
#include "reduce.cuh"  // reduce_v0 等被测内核的声明（与 test.cuh 同源，显式写清依赖）

int main() {
  const int block = 256;

  // 场景 1：标准长度（n 恰好是 block 的整数倍）。
  const int n1 = 1 << 20;
  // 场景 2：尾部非对齐（n2 不是 block 整数倍），覆盖越界线程补 0 的路径。
  const int n2 = (1 << 20) + 1000;

  std::printf("==== Reduce 测试：正确性(容差 1e-3) + 性能 ====\n");
  bool all_ok = true;
  all_ok = test_reduce_kernel(reduce_v0, "reduce_v0", n1, (n1 + block - 1) / block,
                              block) &&
           all_ok;
  all_ok = test_reduce_kernel(reduce_v0, "reduce_v0(尾部非对齐)", n2,
                              (n2 + block - 1) / block, block) &&
           all_ok;
  // 可选：严格基准模式（耗时较长），例如
  //   all_ok = test_reduce_kernel(reduce_v0, "reduce_v0(strict)", n1,
  //                               (n1 + block - 1) / block, block,
  //                               /*strict_benchmark=*/true) && all_ok;

  std::printf("==== 结果：%s ====\n", all_ok ? "全部 PASS" : "存在 FAIL");
  return all_ok ? 0 : 1;
}
