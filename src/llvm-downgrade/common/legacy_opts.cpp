#include "llvm/Support/CommandLine.h"
// Defined at global scope to match the legacy writers' `extern cl::opt<...>`
// declaration. The option name is deliberately distinct from libLLVM's own
// "bitcode-mdindex-threshold" so that registering ours does not collide with
// the copy libLLVM still registers internally.
llvm::cl::opt<unsigned> IndexThreshold("legacy-bitcode-mdindex-threshold",
                                       llvm::cl::Hidden, llvm::cl::init(25));
