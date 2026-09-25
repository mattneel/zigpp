//===- llvm-downgrade.cpp - C API over the legacy bitcode writers ---------===//

#include "llvm-downgrade.h"
#include "DowngradeError.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

struct LLVMDGOpaqueMemoryBuffer {
  std::string Data;
};

using namespace llvm;

namespace {

std::mutex APILock;

char *makeMessage(StringRef Message, StringRef Prefix) {
  size_t Size = Prefix.size() + Message.size();
  char *P = static_cast<char *>(std::malloc(Size + 1));
  if (P) {
    if (!Prefix.empty())
      std::memcpy(P, Prefix.data(), Prefix.size());
    std::memcpy(P + Prefix.size(), Message.data(), Message.size());
    P[Size] = '\0';
  }
  return P;
}

struct Target {
  const char *Name;
  unsigned Major, Minor;
  void (*Write)(Module &M, raw_ostream &OS);
};

constexpr Target Targets[] = {
    {"5.0", 5, 0,
     [](Module &M, raw_ostream &OS) {
       BitcodeWriter50::prepareModule(M);
       WriteBitcode50ToFile(M, OS);
     }},
    {"7.0", 7, 0,
     [](Module &M, raw_ostream &OS) {
       BitcodeWriter70::prepareModule(M);
       WriteBitcode70ToFile(M, OS);
     }},
#ifdef LLVMDG_HAS_140
    {"14.0", 14, 0,
     [](Module &M, raw_ostream &OS) {
       BitcodeWriter140::prepareModule(M);
       WriteBitcode140ToFile(M, OS);
     }},
#endif
#ifdef LLVMDG_HAS_150
    {"15.0", 15, 0,
     [](Module &M, raw_ostream &OS) {
       BitcodeWriter150::prepareModule(M);
       WriteBitcode150ToFile(M, OS);
     }},
#endif
#ifdef LLVMDG_HAS_180
    {"18.0", 18, 0,
     [](Module &M, raw_ostream &OS) {
       BitcodeWriter180::prepareModule(M);
       WriteBitcode180ToFile(M, OS);
     }},
#endif
};

} // namespace

extern "C" {

void LLVMDGGetLLVMVersion(unsigned *Major, unsigned *Minor, unsigned *Patch) {
  if (Major)
    *Major = LLVM_VERSION_MAJOR;
  if (Minor)
    *Minor = LLVM_VERSION_MINOR;
  if (Patch)
    *Patch = LLVM_VERSION_PATCH;
}

void LLVMDGDisposeMessage(char *Message) { std::free(Message); }

const char *LLVMDGGetBufferStart(LLVMDGMemoryBufferRef Buffer) {
  return Buffer->Data.data();
}
size_t LLVMDGGetBufferSize(LLVMDGMemoryBufferRef Buffer) {
  return Buffer->Data.size();
}
void LLVMDGDisposeMemoryBuffer(LLVMDGMemoryBufferRef Buffer) { delete Buffer; }

const char *const *LLVMDGGetTargets(size_t *Count) {
  static constexpr auto Names = [] {
    std::array<const char *, std::size(Targets)> R{};
    for (size_t I = 0; I < R.size(); ++I)
      R[I] = Targets[I].Name;
    return R;
  }();
  if (Count)
    *Count = Names.size();
  return Names.data();
}

int LLVMDGDowngrade(const char *Bitcode, size_t Length, unsigned Major,
                    unsigned Minor, LLVMDGMemoryBufferRef *OutBitcode,
                    char **OutMessage) {
  if (OutMessage)
    *OutMessage = nullptr;
  if (OutBitcode)
    *OutBitcode = nullptr;

  auto fail = [&](StringRef Message, StringRef Prefix = {}) {
    if (OutMessage)
      *OutMessage = makeMessage(Message, Prefix);
    return 1;
  };

  if (!Bitcode || !OutBitcode)
    return fail("input and output buffer pointers must not be null");

  const Target *T = nullptr;
  for (const Target &Candidate : Targets)
    if (Candidate.Major == Major && Candidate.Minor == Minor)
      T = &Candidate;
  if (!T)
    return fail("unsupported bitcode version");

  try {
    std::lock_guard<std::mutex> Guard(APILock);
    LLVMContext Context;
    // The IR parser requires a null-terminated buffer: copy.
    SMDiagnostic ParseError;
    auto Buffer =
        MemoryBuffer::getMemBufferCopy(StringRef(Bitcode, Length), "<input>");
    auto M = parseIR(Buffer->getMemBufferRef(), ParseError, Context);
    if (!M) {
      std::string Message;
      raw_string_ostream OS(Message);
      ParseError.print(nullptr, OS, false);
      if (!Message.empty() && Message.back() == '\n')
        Message.pop_back();
      return fail(Message);
    }

    SmallVector<char, 0> Output;
    raw_svector_ostream OS(Output);
    T->Write(*M, OS);

    auto Out = std::make_unique<LLVMDGOpaqueMemoryBuffer>();
    Out->Data.assign(Output.begin(), Output.end());
    *OutBitcode = Out.release();
    return 0;
  } catch (const llvmdg::DowngradeError &E) {
    return fail(E.Message, "LLVM ERROR: ");
  } catch (const std::exception &E) {
    return fail(E.what());
  } catch (...) {
    return fail("unexpected exception while downgrading");
  }
}

} // extern "C"
