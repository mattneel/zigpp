#pragma once

#include "llvm/ADT/Twine.h"
#include <string>

namespace llvmdg {

struct DowngradeError {
  std::string Message;
};

// Throw directly from the writers: LLVM may be built without unwind cleanups.
[[noreturn]] inline void reportError(const llvm::Twine &Message) {
  throw DowngradeError{Message.str()};
}

} // namespace llvmdg
