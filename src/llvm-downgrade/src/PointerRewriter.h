//===-- Bitcode/LegacyWriter/PointerRewriter.h - Rewrite pointers -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class supports writing opaque pointers in typed IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_BITCODE_LEGACYWRITER_POINTERREWRITER_H
#define LLVM_LIB_BITCODE_LEGACYWRITER_POINTERREWRITER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

class Constant;
class Value;
class Module;
class TypedPointerType;

using PointerTypeMap = DenseMap<const Value *, TypedPointerType *>;

class PointerRewriter {
public:
  PointerRewriter(Module &M) : M(M) {}

  bool run();

  static PointerTypeMap buildPointerMap(const Module &M);
  static bool requiresPointerRewriting(const Constant *C);

  // Lower intrinsics to their legacy form: drop lifetime markers (whose
  // signature changed incompatibly in LLVM 22), restore the typed-pointer
  // name mangling (llvm.memcpy.p0i8.p0i8.i64, llvm.stacksave, ...), and, for
  // a 5.0 target, re-insert the explicit align argument of the memory
  // intrinsics. TargetMajor is the LLVM major of the target bitcode format.
  static bool prepareIntrinsics(Module &M, unsigned TargetMajor);

  // Abort on called intrinsics with pointer-typed parameters that have no
  // typed legacy signature: their {}*-typed arguments would be rejected by
  // the legacy verifiers ("Intrinsic has incorrect argument type!"). Also
  // abort on intrinsics whose typed signature differs in the target LLVM,
  // like the AMDGPU pointer intrinsics before the LLVM 7 address-space
  // remapping. Used by the 5.0/7.0 targets; TargetMajor is the LLVM major of
  // the target bitcode format.
  static void checkIntrinsics(Module &M, unsigned TargetMajor);

  // Downgrade llvm.module.flags whose behavior postdates the typed targets:
  // the Min behavior (LLVM 15) on the "PIC Level" flag becomes Max, the
  // behavior clang emitted for that flag before LLVM 15. Any other flag with
  // a behavior newer than Max aborts.
  static bool downgradeModuleFlags(Module &M);

  // Return the typed pointer types in `PointerMap` in a deterministic module
  // order. Iterating the DenseMap directly orders types by `Value *` address,
  // which makes the emitted type table — and thus the whole bitcode — depend on
  // allocation addresses and so non-reproducible across runs. May repeat types;
  // the ValueEnumerator dedups them.
  static SmallVector<TypedPointerType *, 16>
  orderedPointerTypes(const Module &M, const PointerTypeMap &PointerMap);

private:
  Module &M;
};


} // End llvm namespace

#endif
