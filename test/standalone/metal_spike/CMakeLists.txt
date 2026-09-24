cmake_minimum_required(VERSION 3.20)
project(metal_spike_air_rewrite CXX)

# Out-of-tree build against a prebuilt LLVM (find_package(LLVM)), because this tool is
# built against the host LLVM, not the tree Zig vendors. Configure it with the build
# directory OUTSIDE the repository, e.g.
#
#   env PATH=/usr/bin:/bin cmake -B /tmp/metal-spike/air-rewrite-build \
#       -S test/standalone/metal_spike -G Ninja \
#       -DLLVM_DIR=/home/autark/src/zig/zigpp-bootstrap/out/host/lib/cmake/llvm \
#       -DCMAKE_CXX_FLAGS=-DNDEBUG && \
#   cmake --build /tmp/metal-spike/air-rewrite-build
#
# -DNDEBUG is required: the LLVM archives here are a release build (assertions off), and
# without NDEBUG header-only code refers to symbols the archives do not define (for
# instance Value::dump()).
find_package(LLVM REQUIRED CONFIG)
message(STATUS "air-rewrite: building against LLVM ${LLVM_PACKAGE_VERSION}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${LLVM_CMAKE_DIR})
include(AddLLVM)

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")
add_definitions(${LLVM_DEFINITIONS_LIST})

# Everything air-rewrite needs: bitcode IO, IR, the middle-end pipeline it runs
# (PassBuilder plus the passes behind default<O3>, GlobalDCE and StripDeadPrototypes), and
# the nvptx backend for the stand-in target machine that gives the pipeline its cost model
# (see makeStandInTargetMachine in air-rewrite.cpp).
llvm_map_components_to_libnames(LLVM_LIBS
  aggressiveinstcombine
  analysis
  asmprinter
  bitreader
  bitwriter
  codegen
  codegentypes
  core
  globalisel
  instcombine
  instrumentation
  ipo
  mc
  nvptxcodegen
  nvptxdesc
  nvptxinfo
  passes
  scalaropts
  selectiondag
  support
  target
  targetparser
  transformutils
  vectorize)

add_executable(air-rewrite air-rewrite.cpp)
target_link_libraries(air-rewrite PRIVATE ${LLVM_LIBS})
