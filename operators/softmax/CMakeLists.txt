# Softmax 算子目标（已由顶层 CMakeLists 注册）。
# 规则：src/main.cu 尚未创建时 configure 自动跳过；创建后自动构建。
# src/ 下新增任意 .cu 文件（CONFIGURE_DEPENDS 跟踪）都会自动编入该目标，
# 因此内核文件拆分自由，无需修改本文件。
if(NOT CUDAToolkit_FOUND)
  message(STATUS "operators/softmax: 跳过（未找到 CUDA）")
  return()
endif()

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu)
  message(STATUS "operators/softmax: src/main.cu 未创建，跳过（参考 README.md）")
  return()
endif()

file(GLOB SOFTMAX_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
add_executable(softmax ${SOFTMAX_SOURCES})
target_link_libraries(softmax PRIVATE operator_common)

# ---- cuDNN 对照参考（-DSOFTMAX_WITH_CUDNN，默认开启）--------------------------
# cuDNN 不随 CUDA Toolkit 安装，需单独获取（apt / tar 包 / pip 的 nvidia-cudnn-cu12）。
# 开启后 src/main.cu 会把厂商库 cudnnSoftmaxForward 当作参考实现跑同一套场景与判据
# （实现见 src/softmax.cu 的 softmax_cudnn）。本机已装 cuDNN 故默认开启；未装时用
# -DSOFTMAX_WITH_CUDNN=OFF 关闭，否则下面的探测会直接报错。
# 注意 option() 的默认值只在首次 configure 时写进缓存：缓存里已是 OFF 时改这里无效，
# 须显式传 -DSOFTMAX_WITH_CUDNN=ON 重新 configure。
option(SOFTMAX_WITH_CUDNN "把 cuDNN 的 cudnnSoftmaxForward 作为 softmax 对照参考" OFF)

if(SOFTMAX_WITH_CUDNN)
  # 候选根目录（按优先级）：-DCUDNN_ROOT=... > 环境变量 CUDNN_ROOT > conda 环境 >
  # 系统安装 > pip 版 nvidia-cudnn-cu12（PyTorch 的依赖，落在 site-packages 下，
  # 头文件与库齐全但与框架隔离，不在 ldconfig 视界内）。
  set(_cudnn_roots "")
  if(CUDNN_ROOT)
    list(APPEND _cudnn_roots ${CUDNN_ROOT})
  endif()
  if(DEFINED ENV{CUDNN_ROOT})
    list(APPEND _cudnn_roots $ENV{CUDNN_ROOT})
  endif()
  if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND _cudnn_roots $ENV{CONDA_PREFIX})
  endif()
  list(APPEND _cudnn_roots /usr/local/cuda /usr)
  file(GLOB _cudnn_pip_roots
    "$ENV{HOME}/miniconda3/envs/*/lib/python*/site-packages/nvidia/cudnn"
    "$ENV{HOME}/miniforge3/envs/*/lib/python*/site-packages/nvidia/cudnn"
    "$ENV{HOME}/anaconda3/envs/*/lib/python*/site-packages/nvidia/cudnn"
    "$ENV{HOME}/.local/lib/python*/site-packages/nvidia/cudnn")
  list(APPEND _cudnn_roots ${_cudnn_pip_roots})

  find_path(CUDNN_INCLUDE_DIR NAMES cudnn.h HINTS ${_cudnn_roots} PATH_SUFFIXES include)
  find_library(CUDNN_LIBRARY NAMES cudnn HINTS ${_cudnn_roots} PATH_SUFFIXES lib lib64)
  if(NOT CUDNN_LIBRARY)
    # 兜底：pip / conda 版只提供带 SONAME 版本号的 libcudnn.so.9，没有 libcudnn.so
    # 软链，find_library 匹配不到 → 直接按文件名 glob（排序后取第一个，得到 .so 或
    # 版本号最小的那个，两者链接效果相同）。
    foreach(_root IN LISTS _cudnn_roots)
      foreach(_sub lib lib64)
        file(GLOB _cudnn_candidates "${_root}/${_sub}/libcudnn.so*")
        if(_cudnn_candidates)
          list(SORT _cudnn_candidates)
          list(GET _cudnn_candidates 0 CUDNN_LIBRARY)
          break()
        endif()
      endforeach()
      if(CUDNN_LIBRARY)
        break()
      endif()
    endforeach()
  endif()

  if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
    message(FATAL_ERROR
      "SOFTMAX_WITH_CUDNN=ON 但未找到 cuDNN: include=${CUDNN_INCLUDE_DIR}, "
      "lib=${CUDNN_LIBRARY}。用 -DCUDNN_ROOT=<cuDNN 根目录> 指定，"
      "或改为 -DSOFTMAX_WITH_CUDNN=OFF。")
  endif()

  get_filename_component(_cudnn_lib_dir ${CUDNN_LIBRARY} DIRECTORY)
  message(STATUS "operators/softmax: cuDNN 对照参考已启用 (${CUDNN_LIBRARY})")
  target_include_directories(softmax PRIVATE ${CUDNN_INCLUDE_DIR})
  target_link_libraries(softmax PRIVATE ${CUDNN_LIBRARY})
  target_compile_definitions(softmax PRIVATE SOFTMAX_WITH_CUDNN=1)
  # cuDNN 所在目录（尤其 pip / conda 路径）不在 ldconfig 视界内，写 rpath 免去运行时
  # 手动设置 LD_LIBRARY_PATH。
  target_link_options(softmax PRIVATE "-Wl,-rpath,${_cudnn_lib_dir}")
endif()
