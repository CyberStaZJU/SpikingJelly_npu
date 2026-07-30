# Deterministic multi-operator replacement for msopgen's single-op skeleton.
if ("${CMAKE_BUILD_TYPE}x" STREQUAL "Debugx")
    add_ops_compile_options(ALL OPTIONS -g -O0 --cce-ignore-always-inline=true)
endif()

add_kernel_compile(AsPyFedSNNDecayLifForward ${CMAKE_CURRENT_SOURCE_DIR}/as_py_fed_snn_decay_lif_forward.cpp)
add_kernel_compile(AsPyFedSNNDecayLifBackward ${CMAKE_CURRENT_SOURCE_DIR}/as_py_fed_snn_decay_lif_backward.cpp)
add_kernel_compile(AsPyIfForward ${CMAKE_CURRENT_SOURCE_DIR}/as_py_if_forward.cpp)
add_kernel_compile(AsPyIfBackward ${CMAKE_CURRENT_SOURCE_DIR}/as_py_if_backward.cpp)
add_kernel_compile(AsPyLifForward ${CMAKE_CURRENT_SOURCE_DIR}/as_py_lif_forward.cpp)
add_kernel_compile(AsPyLifBackward ${CMAKE_CURRENT_SOURCE_DIR}/as_py_lif_backward.cpp)
add_kernel_compile(AsPyKlifForward ${CMAKE_CURRENT_SOURCE_DIR}/as_py_klif_forward.cpp)
add_kernel_compile(AsPyKlifBackward ${CMAKE_CURRENT_SOURCE_DIR}/as_py_klif_backward.cpp)
add_kernel_compile(AsPyPlifForward ${CMAKE_CURRENT_SOURCE_DIR}/as_py_plif_forward.cpp)
add_kernel_compile(AsPyPlifBackward ${CMAKE_CURRENT_SOURCE_DIR}/as_py_plif_backward.cpp)

if (ENABLE_TEST AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/testcases)
    add_subdirectory(testcases)
endif()
