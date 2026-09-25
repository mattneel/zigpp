#pragma once
// The in-tree downgrader de-statics a cl::opt global in
// lib/Bitcode/Writer/BitcodeWriter.cpp (global namespace) so the legacy writers
// can reuse it. When building out-of-tree against a prebuilt libLLVM that
// symbol stays `static` (internal), so we declare it here -- matching the
// writers' own global-scope `extern cl::opt<...>` declaration -- and define our
// own copy in common/legacy_opts.cpp. This header is force-included into every
// translation unit (see CMakeLists.txt) so even the writers that use the global
// without re-declaring it (e.g. BitcodeWriter50.cpp) compile.
#include "llvm/Support/CommandLine.h"
extern llvm::cl::opt<unsigned> IndexThreshold;
