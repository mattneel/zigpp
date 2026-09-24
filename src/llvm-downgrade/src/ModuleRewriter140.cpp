//===- ModuleRewriter140.cpp - Rewrite IR for LLVM 14 ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements BitcodeWriter140::prepareModule, which lowers the
// current module to a form the LLVM 14 bitcode writer can emit.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitcodeWriter.h"
#include "PointerRewriter.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
using namespace llvm;

// Remove attributes whose representation postdates LLVM 14 and has no LLVM 14
// encoding: the `range`/`initializes` ConstantRange(-list) attributes (LLVM
// 18/19) and `captures` (LLVM 21). These are neither enum, int, string nor type
// attributes. They are pure optimization hints (and Metal's AIR loader rejects
// them anyway), so dropping them is semantically safe.
//
// NOTE: `memory(...)` is an *int* attribute and is intentionally left in place;
// the writer (writeAttributeGroupTable) decomposes it into the legacy
// argmemonly/readonly/... enum attributes the LLVM 14 reader understands. We
// must strip the others *here*, before the ValueEnumerator runs, so that the
// enumerated attribute groups stay consistent with what the writer emits.
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

bool BitcodeWriter140::prepareModule(Module &M) {
  // Unlike the 5.0/7.0 targets, LLVM 14 natively supports freeze, fneg and the
  // bfloat type, so none of those need to be rewritten.
  bool Changed = false;
  LLVMContext &C = M.getContext();
  for (Function &F : M) {
    F.setAttributes(stripUnsupportedAttrs(C, F.getAttributes(), Changed));
    for (Instruction &I : instructions(F))
      if (auto *CB = dyn_cast<CallBase>(&I))
        CB->setAttributes(stripUnsupportedAttrs(C, CB->getAttributes(), Changed));
  }

  // Lower intrinsics to their legacy (LLVM 14) names and signatures.
  Changed |= PointerRewriter::prepareIntrinsics(M, 14);

  // Downgrade module flags whose behavior postdates LLVM 14.
  Changed |= PointerRewriter::downgradeModuleFlags(M);

  // Lower opaque pointers to typed ones, which the AIR loader requires.
  PointerRewriter PR(M);
  Changed |= PR.run();
  return Changed;
}
