//===- PointerRewriter.cpp - Rewrite opaque pointers for typed IR ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PointerRewriter class.
//
//===----------------------------------------------------------------------===//

// Old LLVM versions do not support opaque pointers, so we need to emit typed
// instructions when writing the bitcode. This is hard, as the element type
// information is lost. We deal with this by surrounding all known typed pointer
// uses and definitions with bitcasts to a custom opaque pointer type. Since we
// cannot represent typed pointers in IR (it is illegal to cast to
// TypedPointerTypes), these casts are emitted by the bitcode writer. However,
// to make that easier, we already emit no-op bitcasts here so that the
// ValueEnumerator reserves instruction IDs correctly.
//
// To expose the element type information to the bitcode writer, we provide a
// pointer map that maps values to their typed pointer types.
//
// All this is similar to LLVM's PointerTypeAnalysis pass for DXIL. That pass
// tries to infer the element type of opaque pointers by looking at the uses of
// a pointer, and subsequently the DXIL module writer tries to keep values typed
// for much longer time. This turns out to be fragile, breaking / requiring
// special handling for many more instructions (like select or phi), while also
// not correctly handling multiple (but differently typed) uses of the same
// opaque pointer. To avoid that complexity, we simply emit a bitcast
// surrounding every use or definition of a typed value, keeping pointers opaque
// for the rest of the function.
//
// We also support front-ends customizing element type information, i.e., to
// indicate that operands to certain function calls need to be typed, the
// analysis supports !arg_eltypes metadata on function declarations, containing
// pairs of operand indices and null values representing the element type of the
// operand. This is very useful for custom intrinsics whose type information
// cannot be inferred from the IR.

#include "DowngradeError.h"
#include "PointerRewriter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalIFunc.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/IR/TypedPointerType.h"
using namespace llvm;

static bool requiresPointerRewriting(const Constant *C,
                                     SmallPtrSetImpl<const Constant *> &Seen) {
  if (!Seen.insert(C).second)
    return false;

  if (isa<ConstantPointerNull, UndefValue, PoisonValue>(C))
    return false;

  if (C->getType()->isPtrOrPtrVectorTy() &&
      (isa<GlobalValue, BlockAddress, ConstantExpr>(C)))
    return true;

  for (const Value *Op : C->operands())
    if (const auto *OpC = dyn_cast<Constant>(Op))
      if (requiresPointerRewriting(OpC, Seen))
        return true;
  return false;
}

bool PointerRewriter::requiresPointerRewriting(const Constant *C) {
  SmallPtrSet<const Constant *, 8> Seen;
  return ::requiresPointerRewriting(C, Seen);
}

// Materialize function-local constants whose pointer operands need typed
// reconstruction. LLVM's utility handles nested aggregates, PHI placement, and
// shared constants without violating dominance.
static void collectPointerConstants(Constant *C,
                                    SmallPtrSetImpl<Constant *> &Seen,
                                    SmallVectorImpl<Constant *> &Worklist) {
  if (!Seen.insert(C).second || !PointerRewriter::requiresPointerRewriting(C))
    return;

  if (!isa<ConstantExpr, ConstantAggregate>(C))
    return;

  Worklist.push_back(C);
  for (Value *Op : C->operands())
    if (auto *OpC = dyn_cast<Constant>(Op))
      collectPointerConstants(OpC, Seen, Worklist);
}

static bool demotePointerConstants(Module &M) {
  SmallVector<Constant *, 8> Worklist;
  SmallPtrSet<Constant *, 8> Seen;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        for (const Use &Op : I.operands())
          if (auto *C = dyn_cast<Constant>(Op))
            collectPointerConstants(C, Seen, Worklist);
  if (Worklist.empty())
    return false;

  return convertUsersOfConstantsToInstructions(Worklist, nullptr, true, true);
}

// determine the typed function type based on known intrinsic signatures,
// !arg_eltypes metadata, and pointee-type parameter attributes
static FunctionType *getTypedFunctionType(const Function *F) {
  auto &Ctx = F->getContext();
  auto *FTy = F->getFunctionType();

  // handle known intrinsics
  if (F->isIntrinsic()) {
    switch (F->getIntrinsicID()) {
    case Intrinsic::vastart:
      // void @llvm.va_start(i8* <arglist>)
      return FunctionType::get(
          Type::getVoidTy(Ctx),
          {TypedPointerType::get(
              Type::getInt8Ty(Ctx),
              FTy->getParamType(0)->getPointerAddressSpace())},
          false);
    case Intrinsic::vaend:
      // void @llvm.va_end(i8* <arglist>)
      return FunctionType::get(
          Type::getVoidTy(Ctx),
          {TypedPointerType::get(
              Type::getInt8Ty(Ctx),
              FTy->getParamType(0)->getPointerAddressSpace())},
          false);
    case Intrinsic::vacopy:
      // void @llvm.va_copy(i8* <destarglist>, i8* <srcarglist>)
      return FunctionType::get(
          Type::getVoidTy(Ctx),
          {TypedPointerType::get(
               Type::getInt8Ty(Ctx),
               FTy->getParamType(0)->getPointerAddressSpace()),
           TypedPointerType::get(
               Type::getInt8Ty(Ctx),
               FTy->getParamType(1)->getPointerAddressSpace())},
          false);
    case Intrinsic::gcroot:
      // void @llvm.gcroot(i8** %ptrloc, i8* %metadata)
      return FunctionType::get(
          Type::getVoidTy(Ctx),
          {TypedPointerType::get(
               TypedPointerType::get(
                   Type::getInt8Ty(Ctx),
                   FTy->getParamType(0)->getPointerAddressSpace()),
               FTy->getParamType(0)->getPointerAddressSpace()),
           TypedPointerType::get(
               Type::getInt8Ty(Ctx),
               FTy->getParamType(1)->getPointerAddressSpace())},
          false);
    case Intrinsic::gcread:
      // i8* @llvm.gcread(i8* %ObjPtr, i8** %Ptr)
      return FunctionType::get(
          TypedPointerType::get(Type::getInt8Ty(Ctx),
                                FTy->getReturnType()->getPointerAddressSpace()),
          {TypedPointerType::get(
               Type::getInt8Ty(Ctx),
               FTy->getParamType(0)->getPointerAddressSpace()),
           TypedPointerType::get(
               TypedPointerType::get(
                   Type::getInt8Ty(Ctx),
                   FTy->getParamType(1)->getPointerAddressSpace()),
               FTy->getParamType(1)->getPointerAddressSpace())},
          false);
    case Intrinsic::gcwrite:
      // void @llvm.gcwrite(i8* %P1, i8* %Obj, i8** %P2)
      return FunctionType::get(
          Type::getVoidTy(Ctx),
          {TypedPointerType::get(
               Type::getInt8Ty(Ctx),
               FTy->getParamType(0)->getPointerAddressSpace()),
           TypedPointerType::get(
               Type::getInt8Ty(Ctx),
               FTy->getParamType(1)->getPointerAddressSpace()),
           TypedPointerType::get(
               TypedPointerType::get(
                   Type::getInt8Ty(Ctx),
                   FTy->getParamType(2)->getPointerAddressSpace()),
               FTy->getParamType(2)->getPointerAddressSpace())},
          false);
    case Intrinsic::returnaddress:
      // i8* @llvm.returnaddress(i32 <level>)
      return FunctionType::get(
          TypedPointerType::get(Type::getInt8Ty(Ctx),
                                FTy->getReturnType()->getPointerAddressSpace()),
          {Type::getInt32Ty(Ctx)}, false);
    case Intrinsic::addressofreturnaddress:
      // i8* @llvm.addressofreturnaddress()
      return FunctionType::get(
          TypedPointerType::get(Type::getInt8Ty(Ctx),
                                FTy->getReturnType()->getPointerAddressSpace()),
          {}, false);
    case Intrinsic::sponentry:
      // i8* @llvm.sponentry()
      return FunctionType::get(
          TypedPointerType::get(Type::getInt8Ty(Ctx),
                                FTy->getReturnType()->getPointerAddressSpace()),
          {}, false);
    case Intrinsic::frameaddress:
      // i8* @llvm.frameaddress(i32 <level>)
      return FunctionType::get(
          TypedPointerType::get(Type::getInt8Ty(Ctx),
                                FTy->getReturnType()->getPointerAddressSpace()),
          {Type::getInt32Ty(Ctx)}, false);
    case Intrinsic::stacksave:
      // i8* @llvm.stacksave()
      return FunctionType::get(
          TypedPointerType::get(Type::getInt8Ty(Ctx),
                                FTy->getReturnType()->getPointerAddressSpace()),
          {}, false);
    case Intrinsic::stackrestore:
      // void @llvm.stackrestore(i8* %ptr)
      return FunctionType::get(
          Type::getVoidTy(Ctx),
          {TypedPointerType::get(
              Type::getInt8Ty(Ctx),
              FTy->getParamType(0)->getPointerAddressSpace())},
          false);
    case Intrinsic::prefetch:
      // void @llvm.prefetch(i8* <address>, i32 <rw>, i32 <locality>, i32 <cache
      // type>)
      return FunctionType::get(
          Type::getVoidTy(Ctx),
          {TypedPointerType::get(
               Type::getInt8Ty(Ctx),
               FTy->getParamType(0)->getPointerAddressSpace()),
           Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx), Type::getInt32Ty(Ctx)},
          false);
    case Intrinsic::clear_cache:
      // void @llvm.clear_cache(i8*, i8*)
      return FunctionType::get(
          Type::getVoidTy(Ctx),
          {TypedPointerType::get(
               Type::getInt8Ty(Ctx),
               FTy->getParamType(0)->getPointerAddressSpace()),
           TypedPointerType::get(
               Type::getInt8Ty(Ctx),
               FTy->getParamType(1)->getPointerAddressSpace())},
          false);
    case Intrinsic::thread_pointer:
      // i8* @llvm.thread.pointer()
      return FunctionType::get(
          TypedPointerType::get(Type::getInt8Ty(Ctx),
                                FTy->getReturnType()->getPointerAddressSpace()),
          {}, false);
    case Intrinsic::amdgcn_implicitarg_ptr:
    case Intrinsic::amdgcn_dispatch_ptr:
    case Intrinsic::amdgcn_queue_ptr:
    case Intrinsic::amdgcn_kernarg_segment_ptr:
      // i8 addrspace(4)* @llvm.amdgcn.implicitarg.ptr() etc.; typed LLVM
      // declares these AMDGPU pointer-returning intrinsics with an i8 pointee.
      return FunctionType::get(
          TypedPointerType::get(Type::getInt8Ty(Ctx),
                                FTy->getReturnType()->getPointerAddressSpace()),
          {}, false);
    case Intrinsic::amdgcn_is_shared:
    case Intrinsic::amdgcn_is_private:
      // i1 @llvm.amdgcn.is.shared(i8* nocapture) etc.
      return FunctionType::get(
          Type::getInt1Ty(Ctx),
          {TypedPointerType::get(
              Type::getInt8Ty(Ctx),
              FTy->getParamType(0)->getPointerAddressSpace())},
          false);
    case Intrinsic::memcpy:
    case Intrinsic::memmove:
    case Intrinsic::memset: {
      // void @llvm.memcpy(i8* dst, i8* src, iN len[, i32 align], i1 volatile)
      // void @llvm.memset(i8* dst, i8 val, iN len[, i32 align], i1 volatile)
      // (the explicit align argument only exists in the 5.0 form; see
      // addMemIntrinsicAlignArg)
      SmallVector<Type *, 5> Params(FTy->params().begin(), FTy->params().end());
      unsigned NumPtrArgs = F->getIntrinsicID() == Intrinsic::memset ? 1 : 2;
      for (unsigned i = 0; i < NumPtrArgs; i++)
        Params[i] = TypedPointerType::get(
            Type::getInt8Ty(Ctx), Params[i]->getPointerAddressSpace());
      return FunctionType::get(Type::getVoidTy(Ctx), Params, false);
    }
    }
  }

  auto Args = FTy->params().vec();
  bool Changed = false;

  // look at the !arg_eltypes metadata
  if (MDNode *MD = F->getMetadata("arg_eltypes")) {
    for (unsigned i = 0; i < MD->getNumOperands(); i += 2) {
      auto IdxConstant = cast<ConstantAsMetadata>(MD->getOperand(i))->getValue();
      int Idx = cast<ConstantInt>(IdxConstant)->getZExtValue();
      Type *ElTy =
          cast<ValueAsMetadata>(MD->getOperand(i + 1))->getValue()->getType();

      auto OpaquePtrTy = cast<PointerType>(Args[Idx]);
      auto TypedPtrTy =
          TypedPointerType::get(ElTy, OpaquePtrTy->getAddressSpace());
      Args[Idx] = TypedPtrTy;
      Changed = true;
    }
  }

  // Parameters carrying a pointee-type attribute must be emitted with that
  // pointee: legacy byval/sret/inalloca take their type from the parameter's
  // pointer element type, and the LLVM 14 verifier requires the typed
  // attribute payload (including byref's) to match it.
  AttributeList AL = F->getAttributes();
  for (unsigned i = 0; i < FTy->getNumParams(); i++) {
    if (!isa<PointerType>(Args[i]))
      continue;
    AttributeSet PA = AL.getParamAttrs(i);
    Type *ElTy = PA.getByValType();
    if (!ElTy)
      ElTy = PA.getStructRetType();
    if (!ElTy)
      ElTy = PA.getInAllocaType();
    if (!ElTy)
      ElTy = PA.getByRefType();
    if (!ElTy)
      continue;
    Args[i] = TypedPointerType::get(
        ElTy, cast<PointerType>(Args[i])->getAddressSpace());
    Changed = true;
  }

  if (!Changed)
    return FTy;
  return FunctionType::get(FTy->getReturnType(), Args, FTy->isVarArg());
}

static bool isNoopCast(const Value *V) {
  auto I = cast<Instruction>(V);
  if (I->getOpcode() != Instruction::BitCast)
    return false;
  return I->getOperand(0)->getType() == I->getType();
}

// prepend an instruction's pointer operand with a no-op bitcast
static void prependBitcast(Module &M, Instruction *I, int Idx) {
  Value *V = I->getOperand(Idx);
  assert(V->getType()->isPtrOrPtrVectorTy() && "Expected a pointer operand");

  // Create no-op bitcast
  auto *Cast = CastInst::Create(Instruction::BitCast, V, V->getType());

  if (auto *PHI = dyn_cast<PHINode>(I)) {
    // we can't insert before phis, so rewrite in the incoming block instead
    auto *BB = PHI->getIncomingBlock(Idx);
    Cast->insertBefore(BB->getTerminator()->getIterator());
  } else {
    Cast->insertBefore(I->getIterator());
  }

  I->setOperand(Idx, Cast);
}

// replace all instruction uses of a value with no-op bitcasts, except for
// uses as the callee of a direct call (which must stay a direct call)
static void replaceWithBitcast(Module &M, Value *V) {
  assert(V->getType()->isPtrOrPtrVectorTy() && "Expected a pointer value");

  // Find all uses
  SmallVector<std::pair<Instruction *, unsigned>, 8> Worklist;
  for (Use &Use : V->uses()) {
    auto User = Use.getUser();
    if (auto *CB = dyn_cast<CallBase>(User))
      if (CB->isCallee(&Use))
        continue;
    if (auto *I = dyn_cast<Instruction>(User))
      Worklist.push_back({I, Use.getOperandNo()});
  }

  // Insert no-op bitcasts
  for (auto Item : Worklist) {
    Instruction *I = Item.first;
    int Idx = Item.second;
    prependBitcast(M, I, Idx);
  }
}

// append a single instruction's pointer return value with a no-op bitcast
static void appendBitcast(Module &M, Instruction *I) {
  assert(I->getType()->isPtrOrPtrVectorTy() &&
         "Expected a pointer-returning instruction");

  // Insert no-op bitcast
  Instruction *Cast = CastInst::Create(Instruction::BitCast, I, I->getType());
  Cast->insertBefore(I->getNextNode()->getIterator());

  I->replaceAllUsesWith(Cast);
  // HACK: undo the part of the RAUW which messed with our input argument
  Cast->setOperand(0, I);
}

// bitcast uses of global values (globals, functions, aliases, ifuncs), which
// are emitted with their own typed pointer type rather than the opaque one.
// Direct-callee uses are left alone: the call machinery retypes those itself.
bool bitcastGlobalValues(Module &M) {
  SmallVector<GlobalValue *, 8> Worklist;
  for (GlobalVariable &GV : M.globals())
    Worklist.push_back(&GV);
  for (Function &F : M)
    Worklist.push_back(&F);
  for (GlobalAlias &GA : M.aliases())
    Worklist.push_back(&GA);
  for (GlobalIFunc &GIF : M.ifuncs())
    Worklist.push_back(&GIF);
  if (Worklist.empty())
    return false;

  // Insert bitcasts
  for (GlobalValue *GV : Worklist) {
    replaceWithBitcast(M, GV);
  }

  return true;
}

// Wrap uses of arguments whose parameter is emitted with a typed pointer
// (byval/sret/inalloca or !arg_eltypes), bridging the reader-side typed
// argument back to the opaque pointer the rest of the body expects.
static bool bitcastTypedArguments(Module &M) {
  bool Changed = false;
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    auto *FTy = F.getFunctionType();
    auto *NewFTy = getTypedFunctionType(&F);
    if (FTy == NewFTy)
      continue;
    for (unsigned i = 0; i < FTy->getNumParams(); i++) {
      if (NewFTy->getParamType(i) == FTy->getParamType(i))
        continue;
      replaceWithBitcast(M, F.getArg(i));
      Changed = true;
    }
  }
  return Changed;
}

// bitcast operands to instructions, by infering the element type by inspecting
// the instruction
bool bitcastInstructionOperands(Module &M) {
  // Find all instructions with pointer inputs or outputs
  SmallVector<Instruction *, 8> Worklist;
  for (Function &F : M) {
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *LI = dyn_cast<LoadInst>(&I))
          Worklist.push_back(LI);
        else if (auto *SI = dyn_cast<StoreInst>(&I))
          Worklist.push_back(SI);
        else if (auto *AI = dyn_cast<AtomicRMWInst>(&I))
          Worklist.push_back(AI);
        else if (auto *AI = dyn_cast<AtomicCmpXchgInst>(&I))
          Worklist.push_back(AI);
        else if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          // A vector-of-pointer GEP would need a vector of typed pointers,
          // which TypedPointerType cannot express; the emitted cast would be
          // invalid ("Invalid cast") for the legacy reader.
          if (GEP->getType()->isVectorTy() ||
              GEP->getPointerOperand()->getType()->isVectorTy())
            llvmdg::reportError("vector-of-pointer getelementptr is not "
                               "supported by the IR downgrader");
          Worklist.push_back(GEP);
        } else if (auto *AI = dyn_cast<AllocaInst>(&I))
          Worklist.push_back(AI);
        else if (auto *CI = dyn_cast<CallInst>(&I)) {
          // An indirect (or otherwise non-Function) callee is enumerated with
          // the opaque pointer type, which won't match the call's function
          // type. Retype it with a bitcast, like old typed-pointer IR's
          // `call ... bitcast(callee to FTy*)()` form. Inline asm is excluded:
          // its constant record already carries the function type, and a cast
          // of it would be rejected ("Cannot take the address of an inline
          // asm"); buildPointerMap types the asm value directly instead.
          if (!CI->getCalledFunction() &&
              !isa<InlineAsm>(CI->getCalledOperand()))
            Worklist.push_back(CI);
        }
      }
    }
  }
  if (Worklist.empty())
    return false;

  // Add no-op bitcasts
  for (Instruction *I : Worklist) {
    if (auto *LI = dyn_cast<LoadInst>(I)) {
      prependBitcast(M, LI, LI->getPointerOperandIndex());
    } else if (auto *SI = dyn_cast<StoreInst>(I)) {
      prependBitcast(M, SI, SI->getPointerOperandIndex());
    } else if (auto *AI = dyn_cast<AtomicRMWInst>(I)) {
      prependBitcast(M, AI, AI->getPointerOperandIndex());
    } else if (auto *AI = dyn_cast<AtomicCmpXchgInst>(I)) {
      prependBitcast(M, AI, AI->getPointerOperandIndex());
    } else if (auto *GEP = dyn_cast<GetElementPtrInst>(I)) {
      prependBitcast(M, GEP, GEP->getPointerOperandIndex());
      appendBitcast(M, GEP);
    } else if (auto *AI = dyn_cast<AllocaInst>(I)) {
      appendBitcast(M, AI);
    } else if (auto *CI = dyn_cast<CallInst>(I)) {
      prependBitcast(M, CI, CI->getCalledOperandUse().getOperandNo());
    } else
      llvm_unreachable("Unhandled instruction");
  }

  return true;
}

// bitcast operands to calls, whose type can be altered by metadata attached to
// the function
bool bitcastFunctionOperands(Module &M) {
  for (Function &F : M) {
    auto *FTy = F.getFunctionType();
    auto *NewFTy = getTypedFunctionType(&F);
    if (FTy == NewFTy)
      continue;

    // convert calls to this function
    for (User *U : F.users()) {
      if (auto *CI = dyn_cast<CallInst>(U)) {
        for (unsigned Idx = 0; Idx < CI->arg_size(); Idx++) {
          auto OldTy = FTy->getParamType(Idx);
          auto NewTy = NewFTy->getParamType(Idx);
          if (OldTy == NewTy)
            continue;

          prependBitcast(M, CI, Idx);
        }
        // A retyped return reads back with its typed pointer type; wrap the
        // call's uses so operands the reader type-checks against their
        // user's opaque type (phi incoming values) see the opaque type.
        if (NewFTy->getReturnType() != FTy->getReturnType() &&
            !CI->use_empty())
          appendBitcast(M, CI);
      }
    }
  }

  return false;
}

// The value type a global value is emitted with (the pointee of its typed
// pointer type). For pointer-valued globals and aliases the opaque value type
// carries no information, so it is derived from the initializer/aliasee: the
// reader-side type of `@p = global ptr @g` is `i32** @p = global i32* @g`.
static Type *typedGlobalValueType(const GlobalValue *GV,
                                  SmallPtrSetImpl<const GlobalValue *> &Seen) {
  if (const auto *F = dyn_cast<Function>(GV))
    return getTypedFunctionType(F);

  Type *ValueTy = GV->getValueType();
  if (!ValueTy->isPointerTy())
    return ValueTy;
  if (!Seen.insert(GV).second)
    llvmdg::reportError("cyclic pointer-typed global initializers are not "
                       "supported by the IR downgrader");

  const Constant *Pointee = nullptr;
  if (const auto *GVar = dyn_cast<GlobalVariable>(GV)) {
    if (GVar->hasInitializer())
      Pointee = GVar->getInitializer();
  } else if (const auto *GA = dyn_cast<GlobalAlias>(GV)) {
    Pointee = GA->getAliasee();
  }

  // Without a pointee naming an element type, keep the opaque form; a null or
  // undef initializer is emitted with that same opaque type, so they match.
  if (!Pointee || isa<ConstantPointerNull>(Pointee) || isa<UndefValue>(Pointee))
    return ValueTy;
  if (const auto *PGV = dyn_cast<GlobalValue>(Pointee))
    return TypedPointerType::get(typedGlobalValueType(PGV, Seen),
                                 PGV->getAddressSpace());
  if (const auto *BA = dyn_cast<BlockAddress>(Pointee))
    return TypedPointerType::get(Type::getInt8Ty(GV->getContext()),
                                 BA->getType()->getPointerAddressSpace());
  // Constant-expression initializers containing pointers are rejected by the
  // writers' constant emission; keep the opaque form here.
  return ValueTy;
}

// blockaddress constants are emitted with the legacy i8* type
static TypedPointerType *blockAddressType(const BlockAddress *BA) {
  return TypedPointerType::get(Type::getInt8Ty(BA->getContext()),
                               BA->getType()->getPointerAddressSpace());
}

// visit every blockaddress reachable from C through constant operands
// (stopping at global values, whose initializers are visited separately)
static void
visitBlockAddresses(const Constant *C, SmallPtrSetImpl<const Constant *> &Seen,
                    function_ref<void(const BlockAddress *)> Fn) {
  if (!Seen.insert(C).second || isa<GlobalValue>(C))
    return;
  if (const auto *BA = dyn_cast<BlockAddress>(C)) {
    Fn(BA);
    return;
  }
  for (const Value *Op : C->operands())
    if (const auto *OpC = dyn_cast<Constant>(Op))
      visitBlockAddresses(OpC, Seen, Fn);
}

// build a map of values to typed pointer types
PointerTypeMap PointerRewriter::buildPointerMap(const Module &M) {
  PointerTypeMap PointerMap;

  // globals
  SmallPtrSet<const Constant *, 8> BAVisited;
  for (const GlobalVariable &GV : M.globals()) {
    SmallPtrSet<const GlobalValue *, 4> Seen;
    unsigned AS = GV.getAddressSpace();
    PointerMap[&GV] =
        TypedPointerType::get(typedGlobalValueType(&GV, Seen), AS);
    if (GV.hasInitializer())
      visitBlockAddresses(GV.getInitializer(), BAVisited,
                          [&](const BlockAddress *BA) {
                            PointerMap[BA] = blockAddressType(BA);
                          });
  }

  // aliases and ifuncs
  for (const GlobalAlias &GA : M.aliases()) {
    SmallPtrSet<const GlobalValue *, 4> Seen;
    PointerMap[&GA] = TypedPointerType::get(typedGlobalValueType(&GA, Seen),
                                            GA.getAddressSpace());
  }
  for (const GlobalIFunc &GIF : M.ifuncs())
    PointerMap[&GIF] =
        TypedPointerType::get(GIF.getValueType(), GIF.getAddressSpace());

  // instructions
  for (const Function &F : M) {
    for (const BasicBlock &BB : F) {
      for (const Instruction &I : BB) {
        for (const Use &Op : I.operands())
          if (const auto *BA = dyn_cast<BlockAddress>(Op))
            PointerMap[BA] = blockAddressType(BA);
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          assert(isNoopCast(LI->getPointerOperand()));
          PointerMap[LI->getPointerOperand()] = TypedPointerType::get(
              LI->getType(), LI->getPointerAddressSpace());
        } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
          assert(isNoopCast(SI->getPointerOperand()));
          PointerMap[SI->getPointerOperand()] = TypedPointerType::get(
              SI->getValueOperand()->getType(), SI->getPointerAddressSpace());
        } else if (auto *AI = dyn_cast<AtomicRMWInst>(&I)) {
          assert(isNoopCast(AI->getPointerOperand()));
          PointerMap[AI->getPointerOperand()] = TypedPointerType::get(
              AI->getValOperand()->getType(), AI->getPointerAddressSpace());
        } else if (auto *AI = dyn_cast<AtomicCmpXchgInst>(&I)) {
          assert(isNoopCast(AI->getPointerOperand()));
          PointerMap[AI->getPointerOperand()] = TypedPointerType::get(
              AI->getNewValOperand()->getType(), AI->getPointerAddressSpace());
        } else if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
          assert(isNoopCast(GEP->getPointerOperand()));
          PointerMap[GEP->getPointerOperand()] = TypedPointerType::get(
              GEP->getSourceElementType(), GEP->getAddressSpace());
          assert(GEP->hasOneUse() && isNoopCast(GEP->user_back()));
          PointerMap[GEP] = TypedPointerType::get(GEP->getResultElementType(),
                                                  GEP->getAddressSpace());
        } else if (auto *AI = dyn_cast<AllocaInst>(&I)) {
          assert(AI->hasOneUse() && isNoopCast(AI->user_back()));
          PointerMap[AI] = TypedPointerType::get(AI->getAllocatedType(),
                                                 AI->getAddressSpace());
        } else if (auto *CI = dyn_cast<CallInst>(&I)) {
          if (!CI->getCalledFunction()) {
            // Inline asm callees have no cast (see bitcastInstructionOperands);
            // typing the asm value itself makes its constant record carry the
            // right pointer-to-function type.
            const Value *Callee = CI->getCalledOperand();
            assert(isa<InlineAsm>(Callee) || isNoopCast(Callee));
            PointerMap[Callee] = TypedPointerType::get(
                CI->getFunctionType(),
                Callee->getType()->getPointerAddressSpace());
          }
        }
      }
    }
  }

  // functions
  for (const Function &F : M) {
    auto *FTy = F.getFunctionType();
    auto *NewFTy = getTypedFunctionType(&F);
    PointerMap[&F] = TypedPointerType::get(NewFTy, F.getAddressSpace());
    if (FTy == NewFTy)
      continue;

    for (unsigned int i = 0; i < FTy->getNumParams(); i++) {
      auto OldTy = FTy->getParamType(i);
      auto NewTy = NewFTy->getParamType(i);
      if (OldTy == NewTy)
        continue;

      // arguments of definitions are bridged with casts too
      if (!F.isDeclaration())
        PointerMap[F.getArg(i)] = cast<TypedPointerType>(NewTy);

      for (const User *U : F.users()) {
        if (auto *CI = dyn_cast<CallInst>(U)) {
          // only calls *to* F retype their arguments; other uses of F (e.g.
          // passing it as an argument) are wrapped by bitcastGlobalValues
          if (CI->getCalledOperand() != &F)
            continue;
          assert(isNoopCast(CI->getArgOperand(i)));
          PointerMap[CI->getArgOperand(i)] = cast<TypedPointerType>(NewTy);
        }
      }
    }
  }

  return PointerMap;
}

// Enumerate the typed pointer types in a deterministic module order (globals,
// functions, then instructions and their operands), rather than in `PointerMap`'s
// address-dependent DenseMap order. Every mapped value is a global, a function, an
// instruction, or an instruction operand, so this reaches them all; repeats are
// harmless (the ValueEnumerator dedups types).
SmallVector<TypedPointerType *, 16>
PointerRewriter::orderedPointerTypes(const Module &M,
                                     const PointerTypeMap &PointerMap) {
  SmallVector<TypedPointerType *, 16> Order;
  auto add = [&](const Value *V) {
    if (TypedPointerType *Ty = PointerMap.lookup(V))
      Order.push_back(Ty);
  };
  SmallPtrSet<const Constant *, 8> BAVisited;
  for (const GlobalVariable &GV : M.globals()) {
    add(&GV);
    if (GV.hasInitializer()) {
      add(GV.getInitializer());
      visitBlockAddresses(GV.getInitializer(), BAVisited,
                          [&](const BlockAddress *BA) { add(BA); });
    }
  }
  for (const GlobalAlias &GA : M.aliases())
    add(&GA);
  for (const GlobalIFunc &GIF : M.ifuncs())
    add(&GIF);
  for (const Function &F : M)
    add(&F);
  for (const Function &F : M)
    for (const BasicBlock &BB : F)
      for (const Instruction &I : BB) {
        add(&I);
        for (const Use &Op : I.operands())
          add(Op.get());
      }
  return Order;
}

// The legacy spelling of a pointer-typed intrinsic's name, or empty if the
// intrinsic needs no renaming. LLVM 15+ progressively dropped the pointee
// type from the mangling (p0i8 -> p0) and, for some intrinsics, the mangling
// or a parameter altogether; legacy readers match intrinsics by name, so the
// old spelling has to be restored.
static std::string legacyIntrinsicName(const Function &F,
                                       unsigned TargetMajor) {
  auto *FTy = F.getFunctionType();
  auto ptrSuffix = [&](unsigned ParamIdx) {
    return ".p" +
           std::to_string(
               FTy->getParamType(ParamIdx)->getPointerAddressSpace()) +
           "i8";
  };
  auto intSuffix = [&](unsigned ParamIdx) {
    return ".i" + std::to_string(cast<IntegerType>(FTy->getParamType(ParamIdx))
                                     ->getBitWidth());
  };

  switch (F.getIntrinsicID()) {
  case Intrinsic::vastart:
    return "llvm.va_start";
  case Intrinsic::vaend:
    return "llvm.va_end";
  case Intrinsic::vacopy:
    return "llvm.va_copy";
  case Intrinsic::stacksave:
    return "llvm.stacksave";
  case Intrinsic::stackrestore:
    return "llvm.stackrestore";
  case Intrinsic::thread_pointer:
    return "llvm.thread.pointer";
  case Intrinsic::prefetch:
    // prefetch gained its pointer mangling in LLVM 10
    return TargetMajor >= 10 ? "llvm.prefetch" + ptrSuffix(0)
                             : "llvm.prefetch";
  case Intrinsic::memcpy:
    return "llvm.memcpy" + ptrSuffix(0) + ptrSuffix(1) + intSuffix(2);
  case Intrinsic::memmove:
    return "llvm.memmove" + ptrSuffix(0) + ptrSuffix(1) + intSuffix(2);
  case Intrinsic::memset:
    return "llvm.memset" + ptrSuffix(0) + intSuffix(2);
  default:
    return std::string();
  }
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

// LLVM 7 moved the memory intrinsics' alignment into parameter attributes;
// LLVM 5 requires the explicit i32 align argument, so rebuild declarations
// and calls with it (conservatively the minimum of the pointer alignments).
static bool addMemIntrinsicAlignArg(Module &M) {
  bool Changed = false;
  for (Function &F : llvm::make_early_inc_range(M)) {
    Intrinsic::ID ID = F.getIntrinsicID();
    if (ID != Intrinsic::memcpy && ID != Intrinsic::memmove &&
        ID != Intrinsic::memset)
      continue;
    auto *FTy = F.getFunctionType();
    if (FTy->getNumParams() != 4)
      continue;

    Type *I32Ty = Type::getInt32Ty(M.getContext());
    SmallVector<Type *, 5> Params(FTy->params().begin(), FTy->params().end());
    Params.insert(Params.begin() + 3, I32Ty);
    auto *NewFTy =
        FunctionType::get(FTy->getReturnType(), Params, /*isVarArg=*/false);
    std::string Name = F.getName().str();
    Function *NewF = Function::Create(NewFTy, F.getLinkage(), "", &M);

    for (User *U : llvm::make_early_inc_range(F.users())) {
      auto *CI = cast<CallInst>(U);
      uint64_t Align = CI->getParamAlign(0).valueOrOne().value();
      if (ID != Intrinsic::memset)
        Align = std::min(Align,
                         CI->getParamAlign(1).valueOrOne().value());
      SmallVector<Value *, 5> Args(CI->arg_begin(), CI->arg_end());
      Args.insert(Args.begin() + 3, ConstantInt::get(I32Ty, Align));
      auto *NewCI = CallInst::Create(NewFTy, NewF, Args, "", CI->getIterator());
      NewCI->setDebugLoc(CI->getDebugLoc());
      CI->eraseFromParent();
    }

    F.eraseFromParent();
    NewF->setName(Name);
    Changed = true;
  }
  return Changed;
}

bool PointerRewriter::prepareIntrinsics(Module &M, unsigned TargetMajor) {
  bool Changed = dropLifetimeIntrinsics(M);
  if (TargetMajor < 7)
    Changed |= addMemIntrinsicAlignArg(M);
  for (Function &F : M) {
    if (!F.isIntrinsic())
      continue;
    std::string Name = legacyIntrinsicName(F, TargetMajor);
    if (!Name.empty() && F.getName() != Name) {
      F.setName(Name);
      Changed = true;
    }
  }
  return Changed;
}

bool PointerRewriter::downgradeModuleFlags(Module &M) {
  NamedMDNode *ModFlags = M.getModuleFlagsMetadata();
  if (!ModFlags)
    return false;
  bool Changed = false;
  for (unsigned i = 0, e = ModFlags->getNumOperands(); i != e; ++i) {
    MDNode *Flag = ModFlags->getOperand(i);
    if (Flag->getNumOperands() < 3)
      continue;
    auto *Behavior =
        mdconst::dyn_extract_or_null<ConstantInt>(Flag->getOperand(0));
    if (!Behavior || Behavior->getZExtValue() <= Module::Max)
      continue;
    auto *ID = dyn_cast<MDString>(Flag->getOperand(1));
    if (Behavior->getZExtValue() == Module::Min && ID &&
        ID->getString() == "PIC Level") {
      // Clang emitted "PIC Level" with the Max behavior before LLVM 15
      // introduced Min; rewriting restores the flag's legacy spelling.
      Metadata *Ops[3] = {
          ConstantAsMetadata::get(ConstantInt::get(
              Type::getInt32Ty(M.getContext()), Module::Max)),
          Flag->getOperand(1), Flag->getOperand(2)};
      ModFlags->setOperand(i, MDNode::get(M.getContext(), Ops));
      Changed = true;
      continue;
    }
    llvmdg::reportError(Twine("module flag with a behavior the target LLVM "
                             "cannot represent: ") +
                           (ID ? ID->getString() : "<unnamed>"));
  }
  return Changed;
}

void PointerRewriter::checkIntrinsics(Module &M, unsigned TargetMajor) {
  for (const Function &F : M) {
    if (!F.isIntrinsic() || F.use_empty())
      continue;
    // Before LLVM 7 switched AMDGPU to its current address-space mapping,
    // these intrinsics returned i8 addrspace(2)* rather than i8
    // addrspace(4)*. A module using the current mapping cannot be renumbered
    // locally, so downgrading it below 7.0 is rejected.
    if (TargetMajor < 7) {
      switch (F.getIntrinsicID()) {
      default:
        break;
      case Intrinsic::amdgcn_implicitarg_ptr:
      case Intrinsic::amdgcn_dispatch_ptr:
      case Intrinsic::amdgcn_queue_ptr:
      case Intrinsic::amdgcn_kernarg_segment_ptr:
        llvmdg::reportError(Twine("AMDGPU intrinsic predates the LLVM 7 "
                                 "address-space remapping and cannot be "
                                 "downgraded: ") +
                               F.getName());
      }
    }
    auto *FTy = F.getFunctionType();
    bool HasPtr = FTy->getReturnType()->isPtrOrPtrVectorTy();
    for (Type *P : FTy->params())
      HasPtr |= P->isPtrOrPtrVectorTy();
    if (!HasPtr)
      continue;
    // Intrinsics with a known typed signature are fine; anything else would
    // be emitted with {}*-typed pointers against the old typed signature.
    if (getTypedFunctionType(&F) != FTy)
      continue;
    llvmdg::reportError(Twine("intrinsic with pointer arguments cannot be "
                             "downgraded: ") +
                           F.getName());
  }
}

// bitcast instruction uses of blockaddress constants, which are emitted with
// the legacy i8* type
static bool bitcastBlockAddresses(Module &M) {
  SmallSetVector<BlockAddress *, 8> BAs;
  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        for (const Use &Op : I.operands())
          if (auto *BA = dyn_cast<BlockAddress>(Op))
            BAs.insert(BA);
  for (BlockAddress *BA : BAs)
    replaceWithBitcast(M, BA);
  return !BAs.empty();
}

bool PointerRewriter::run() {
  bool Changed = demotePointerConstants(M);

  // insert no-op bitcasts surrounding pointer values
  Changed |= bitcastGlobalValues(M);
  Changed |= bitcastBlockAddresses(M);
  Changed |= bitcastInstructionOperands(M);
  Changed |= bitcastFunctionOperands(M);
  Changed |= bitcastTypedArguments(M);

  // TODO: remove double bitcasts?

  return Changed;
}
