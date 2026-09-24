//===- ModuleRewriter180.cpp - Rewrite IR for LLVM 18 ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements BitcodeWriter180::prepareModule, which lowers the
// current module to a form the LLVM 18 bitcode writer can emit.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ModRef.h"
using namespace llvm;

// LLVM made some pointer-typed intrinsics overloaded on the pointer type
// after 18, mangling their names with a pointer suffix (va_start/va_end/
// va_copy in 19, thread.pointer in 21; e.g. llvm.va_start.p0). LLVM 18 only
// knows the unmangled names; rename them back. (stacksave/stackrestore were
// already mangled in 17, so they keep their host names.)
static bool renameLegacyIntrinsics(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    if (!F.isIntrinsic())
      continue;
    StringRef Name;
    switch (F.getIntrinsicID()) {
    case Intrinsic::vastart:        Name = "llvm.va_start";       break;
    case Intrinsic::vaend:          Name = "llvm.va_end";         break;
    case Intrinsic::vacopy:         Name = "llvm.va_copy";        break;
    case Intrinsic::thread_pointer: Name = "llvm.thread.pointer"; break;
    default: continue;
    }
    if (F.getName() != Name) {
      F.setName(Name);
      Changed = true;
    }
  }
  return Changed;
}

// Remove llvm.lifetime.start/end markers: LLVM 22 dropped their size
// argument, so the modern form cannot be expressed against any legacy
// signature (old readers upgrade the call by name and crash on the missing
// argument). They are pure optimization hints, so dropping them is safe.
static bool dropLifetimeIntrinsics(Module &M) {
  bool Changed = false;
  for (Function &F : llvm::make_early_inc_range(M)) {
    if (F.getIntrinsicID() != Intrinsic::lifetime_start &&
        F.getIntrinsicID() != Intrinsic::lifetime_end)
      continue;
    for (User *U : llvm::make_early_inc_range(F.users()))
      if (auto *CI = dyn_cast<CallInst>(U))
        CI->eraseFromParent();
    if (F.use_empty())
      F.eraseFromParent();
    Changed = true;
  }
  return Changed;
}

// Remove attributes whose representation postdates LLVM 18 and has no LLVM 18
// encoding: the `range`/`initializes` ConstantRange(-list) attributes (LLVM
// 19). These are neither enum, int, string nor type attributes. They are pure
// optimization hints, so dropping them is semantically safe.
//
// NOTE: `memory(...)` is native to LLVM 18, and `captures(...)` is an *int*
// attribute the writer (encodeAttribute180) lowers to the legacy `nocapture`
// enum attribute, so both are left in place; the writer's counting mode keeps
// the enumerated attribute groups consistent with what it emits.
static AttributeList stripUnsupportedAttrs(LLVMContext &C, AttributeList AL,
                                           bool &Changed) {
  for (unsigned Index : AL.indexes()) {
    for (Attribute A : AL.getAttributes(Index)) {
      if (!A.isEnumAttribute() && !A.isIntAttribute() &&
          !A.isStringAttribute() && !A.isTypeAttribute()) {
        AL = AL.removeAttributeAtIndex(C, Index, A.getKindAsEnum());
        Changed = true;
      }
    }
  }
  return AL;
}

bool BitcodeWriter180::prepareModule(Module &M) {
  // LLVM 18 is opaque-pointer-only and supports target extension types,
  // memory(...) and DIAssignID natively, so only attributes that postdate it
  // and the vararg intrinsic names need rewriting.
  bool Changed = false;
  LLVMContext &C = M.getContext();
  for (Function &F : M) {
    F.setAttributes(stripUnsupportedAttrs(C, F.getAttributes(), Changed));
    for (Instruction &I : instructions(F))
      if (auto *CB = dyn_cast<CallBase>(&I))
        CB->setAttributes(stripUnsupportedAttrs(C, CB->getAttributes(), Changed));
  }

  Changed |= renameLegacyIntrinsics(M);
  Changed |= dropLifetimeIntrinsics(M);
  return Changed;
}
