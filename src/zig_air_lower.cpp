/*
 * Copyright (c) Zig contributors
 *
 * This file is part of zig, which is MIT licensed.
 * See http://opensource.org/licenses/MIT
 */

// The AIR lowerings that run after the optimization pipeline (src/zig_air.cpp), because the
// pipeline would form again what they remove.
//
// 1. Program-scope constants. The Metal Shading Language keeps program-scope data in the
//    `constant` address space, 2, and Apple's compiler takes a global in address space 0 for an
//    external symbol: a kernel that reads a table there fails to link with "Undefined symbols",
//    Zig's anonymous constants appearing as `___anon_N` (issue #18). Every constant global
//    moves to address space 2.
//
//    Zig's pointer types say nothing about constant data, and an Apple GPU has no generic
//    address space to cast a pointer into constant data to, so every pointer that comes from
//    a constant has to be known for one wherever it is used. SROA takes apart the copies a
//    Debug build makes on the stack, and LLVM's address space inference, with address space 0
//    as the flat one, carries address space 2 from the globals through the address
//    arithmetic, the phis, the loads and the copies. Where such a pointer crosses a call, or
//    meets thread memory at the edge of a function (a phi with a local, a store, a return),
//    the call is inlined and inference runs again in the caller, until nothing crosses; the
//    other functions the optimizer left stay as they are. What is left (a pointer into
//    constant data that meets thread memory in a kernel) and a mutable program-scope
//    variable, which has no place on an Apple GPU, are reported by name.
//
//    Apple's toolchain does not relocate the addresses inside constant data: a table of
//    strings or slices fails to link with "Undefined symbols: _unnamed_1", makes the compiler
//    service die, or reads zeros for its strings, depending on the kernel. A constant that
//    holds the address of another global is reported by name until the backend stores such
//    tables without addresses (issue #22).
//
// 2. Wide multiplies. Apple's AIR-to-GPU compiler cannot produce the high 64 bits of a 64x64-bit
//    product: its compiler service dies with XPC_ERROR_CONNECTION_INTERRUPTED instead
//    (issue #18). Two IR shapes need that high half:
//
//    * `llvm.umul.with.overflow.i64` and `llvm.smul.with.overflow.i64` whose overflow flag is
//      used. Zig emits one for every safety-checked multiply of a 64-bit integer, `usize`
//      included, so a Debug kernel that computes `row * cols` has one.
//    * a `mul` of 128-bit integers, which is what Zig's `u128` and `i128` arithmetic lower to,
//      and what the pipeline's instruction combiner forms out of a 64x64 multiply split into
//      32-bit pieces. The same goes for the 128-bit overflow intrinsics.
//
//    Every one of them is rebuilt out of 32x32->64 multiplies, which Apple's compiler lowers.
//    The unsigned high half of a 64x64 product is the four-multiply schoolbook expansion; the
//    signed one is the unsigned one corrected by the operands' signs; unsigned overflow is a
//    non-zero high half, signed overflow a high half that is not the sign extension of the low
//    half. A 128-bit product is its low 128 bits, which do not depend on signedness. What this
//    pass cannot rebuild is reported by name, rather than left for Apple's compiler to crash on.

#include "zig_air.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/Scalar/InferAddressSpaces.h"
#include "llvm/Transforms/Scalar/InstSimplifyPass.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

namespace {

// AIR: the constant address space (doc/proposals/metal.md section 2.2).
constexpr unsigned air_constant_address_space = 2;

std::string describe(const llvm::Value &V) {
    std::string Text;
    llvm::raw_string_ostream OS(Text);
    V.print(OS);
    return OS.str();
}

std::string describeOperand(const llvm::Value &V) {
    std::string Text;
    llvm::raw_string_ostream OS(Text);
    V.printAsOperand(OS, /*PrintType=*/true);
    return OS.str();
}

std::string where(const llvm::Instruction &I) {
    const llvm::Function *F = I.getFunction();
    return F ? F->getName().str() : std::string("?");
}

// Runs module passes the way the pipeline does (src/zig_air.cpp): without a TargetMachine,
// and with analyses of their own, since the IR changes outside of any pass between runs.
void runPasses(llvm::Module &M, llvm::ModulePassManager &MPM) {
    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;
    llvm::PassBuilder PB;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    MPM.run(M, MAM);
}

void runOnFunctions(llvm::Module &M, llvm::FunctionPassManager FPM) {
    llvm::ModulePassManager MPM;
    MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
    runPasses(M, MPM);
}

// ---------------------------------------------------------------------------------------
// 1. Program-scope constants
// ---------------------------------------------------------------------------------------

bool isProgramScopeData(const llvm::GlobalVariable &GV) {
    return GV.getAddressSpace() == 0 && !GV.use_empty() && !GV.getName().starts_with("llvm.");
}

// The global that `C` holds the address of, if any: a pointer to another constant (a table of
// slices), a function, or an address used as an integer. Apple's toolchain does not relocate
// the addresses inside constant data: a kernel that reads such a table fails to link with
// "Undefined symbols: _unnamed_1", makes the compiler service die, or reads zeros where the
// pointers should be, depending on the kernel (issue #22).
const llvm::GlobalValue *addressIn(const llvm::Constant *C) {
    if (auto *GV = llvm::dyn_cast<llvm::GlobalValue>(C)) return GV;
    if (!llvm::isa<llvm::ConstantAggregate>(C) && !llvm::isa<llvm::ConstantExpr>(C)) return nullptr;
    for (const llvm::Use &Op : C->operands())
        if (const llvm::GlobalValue *GV = addressIn(llvm::cast<llvm::Constant>(Op.get()))) return GV;
    return nullptr;
}

// Takes apart the stack copies a Debug build makes, whose stores and loads would hide where a
// pointer came from, and drops the functions that inlining left unused. SROA runs twice: a
// copy out of a phi of two pointers, a local and a constant (a function that returns an error
// constant or a value through the same result pointer), becomes a phi of pointers into the
// new slices, whose loads only the second run moves into the predecessors.
void split(llvm::Module &M) {
    llvm::FunctionPassManager FPM;
    FPM.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
    FPM.addPass(llvm::InstSimplifyPass());
    FPM.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
    FPM.addPass(llvm::InstSimplifyPass());
    llvm::ModulePassManager MPM;
    MPM.addPass(llvm::createModuleToFunctionPassAdaptor(std::move(FPM)));
    MPM.addPass(llvm::GlobalDCEPass());
    runPasses(M, MPM);
}

// Moves the program-scope constants to the constant address space. Each use of one becomes a
// cast of its new address to address space 0, where inference picks it up.
bool moveConstants(llvm::Module &M, std::string &Err) {
    llvm::SmallVector<llvm::GlobalVariable *, 16> Constants;
    for (llvm::GlobalVariable &GV : M.globals()) {
        if (!isProgramScopeData(GV)) continue;
        if (GV.isDeclaration()) {
            Err = "air64: the kernels refer to " + GV.getName().str() +
                  ", data defined outside the library, which an Apple GPU cannot link";
            return false;
        }
        if (!GV.isConstant()) {
            Err = "air64: " + GV.getName().str() +
                  " is a mutable program-scope variable; an Apple GPU keeps program-scope data "
                  "only in the constant address space, and threadgroup memory is a variable "
                  "in the `shared` address space";
            return false;
        }
        if (const llvm::GlobalValue *Target = addressIn(GV.getInitializer())) {
            Err = "air64: the program-scope constant " + GV.getName().str() +
                  " holds the address of " + describeOperand(*Target) +
                  ", and Apple's GPU toolchain does not relocate the addresses inside constant "
                  "data (a table of strings or slices is one; issue #22)";
            return false;
        }
        Constants.push_back(&GV);
    }

    for (llvm::GlobalVariable *GV : Constants) {
        auto *New = new llvm::GlobalVariable(M, GV->getValueType(), /*isConstant=*/true,
                                             GV->getLinkage(), GV->getInitializer(), "", GV,
                                             GV->getThreadLocalMode(), air_constant_address_space);
        New->copyAttributesFrom(GV);
        New->takeName(GV);
        New->copyMetadata(GV, 0);
        GV->replaceAllUsesWith(llvm::ConstantExpr::getAddrSpaceCast(New, GV->getType()));
        GV->eraseFromParent();
    }
    return true;
}

// Carries address space 2 from the moved constants to everything derived from them.
void inferConstantSpace(llvm::Module &M) {
    llvm::FunctionPassManager FPM;
    FPM.addPass(llvm::InstSimplifyPass());
    FPM.addPass(llvm::InferAddressSpacesPass(0));
    runOnFunctions(M, std::move(FPM));
}

// Whether `V` is, or is a constant built out of, a cast of a pointer out of the constant
// address space.
bool castsOutOfConstantSpace(const llvm::Value *V) {
    if (auto *Cast = llvm::dyn_cast<llvm::AddrSpaceCastOperator>(V))
        return Cast->getSrcAddressSpace() == air_constant_address_space;
    if (!llvm::isa<llvm::ConstantAggregate>(V) && !llvm::isa<llvm::ConstantExpr>(V)) return false;
    return llvm::any_of(llvm::cast<llvm::User>(V)->operands(),
                        [](const llvm::Use &Op) { return castsOutOfConstantSpace(Op.get()); });
}

// An address taken as an integer is taken from the pointer into constant data, whose bits it
// is.
void foldAddressCasts(llvm::Module &M) {
    for (llvm::Function &F : M) {
        for (llvm::Instruction &I : llvm::instructions(F)) {
            auto *P2I = llvm::dyn_cast<llvm::PtrToIntInst>(&I);
            if (!P2I) continue;
            auto *Cast = llvm::dyn_cast<llvm::AddrSpaceCastOperator>(P2I->getPointerOperand());
            if (Cast && Cast->getSrcAddressSpace() == air_constant_address_space)
                P2I->setOperand(0, Cast->getPointerOperand());
        }
    }
}

// Whether `I` uses a pointer into constant data as a pointer of address space 0, which an
// Apple GPU does not have for it: every use of a cast out of the constant address space but
// the cast itself.
bool escapesConstantSpace(const llvm::Instruction &I) {
    if (auto *Cast = llvm::dyn_cast<llvm::AddrSpaceCastInst>(&I))
        if (Cast->getSrcAddressSpace() == air_constant_address_space) return false;
    return llvm::any_of(I.operands(),
                        [](const llvm::Use &Op) { return castsOutOfConstantSpace(Op.get()); });
}

// The calls to inline so that inference sees a pointer into constant data where it goes: a
// call that is passed one, and each call of a function where one meets thread memory in
// another way (a phi with a local, a store, a return), since the caller is where the two can
// be told apart. A function is not inlined into itself.
llvm::SmallSetVector<llvm::CallBase *, 16> callsToInline(llvm::Module &M) {
    llvm::SmallSetVector<llvm::CallBase *, 16> Calls;
    for (llvm::Function &F : M) {
        if (F.isDeclaration()) continue;
        bool EscapesHere = false;
        for (llvm::Instruction &I : llvm::instructions(F)) {
            if (!escapesConstantSpace(I)) continue;
            auto *Call = llvm::dyn_cast<llvm::CallBase>(&I);
            llvm::Function *Callee = Call ? Call->getCalledFunction() : nullptr;
            if (Callee && !Callee->isDeclaration() && Callee != &F)
                Calls.insert(Call);
            else
                EscapesHere = true;
        }
        if (!EscapesHere) continue;
        for (llvm::User *User : F.users()) {
            auto *Call = llvm::dyn_cast<llvm::CallBase>(User);
            if (Call && Call->getCalledFunction() == &F && Call->getFunction() != &F)
                Calls.insert(Call);
        }
    }
    return Calls;
}

// What inference and inlining left: the casts nothing uses go, and any other use of a cast out
// of the constant address space needs the generic address space that an Apple GPU does not
// have.
bool checkConstantUses(llvm::Module &M, std::string &Err) {
    for (llvm::Function &F : M) {
        if (F.isDeclaration()) continue;
        llvm::SmallVector<llvm::WeakTrackingVH, 8> Casts;
        for (llvm::Instruction &I : llvm::instructions(F)) {
            if (auto *Cast = llvm::dyn_cast<llvm::AddrSpaceCastInst>(&I)) {
                if (Cast->getSrcAddressSpace() == air_constant_address_space) {
                    Casts.push_back(Cast);
                    continue;
                }
            }
            if (!escapesConstantSpace(I)) continue;
            Err = "air64: " + F.getName().str() +
                  ": a pointer into program-scope constant data meets memory of another address "
                  "space here, and an Apple GPU has no generic address space to hold both: " +
                  describe(I);
            return false;
        }
        llvm::RecursivelyDeleteTriviallyDeadInstructionsPermissive(Casts);
    }
    return true;
}

bool moveConstantsToConstantSpace(llvm::Module &M, std::string &Err) {
    if (llvm::none_of(M.globals(), isProgramScopeData)) return true;
    split(M);
    if (!moveConstants(M, Err)) return false;
    // Each round inlines where a pointer into constant data crossed a call, so that the next
    // one follows it in the caller; the rounds are bounded, since inlining cannot take
    // recursion apart.
    for (unsigned Round = 0;; ++Round) {
        inferConstantSpace(M);
        foldAddressCasts(M);
        if (Round == 32) break;
        bool Inlined = false;
        for (llvm::CallBase *Call : callsToInline(M)) {
            llvm::InlineFunctionInfo Info;
            Inlined |= llvm::InlineFunction(*Call, Info).isSuccess();
        }
        if (!Inlined) break;
        split(M);
    }
    return checkConstantUses(M, Err);
}

// ---------------------------------------------------------------------------------------
// 2. Wide multiplies
// ---------------------------------------------------------------------------------------

using DeadList = llvm::SmallVectorImpl<llvm::WeakTrackingVH>;

// The two 64-bit halves of a 128-bit integer. `Hi == nullptr` means a high half known to be
// zero, so that the terms it would contribute are not emitted at all.
struct Halves {
    llvm::Value *Lo;
    llvm::Value *Hi;
};

bool isInt(const llvm::Type *T, unsigned Bits) { return T->isIntegerTy(Bits); }

// The high 64 bits of the unsigned 128-bit product of two i64 values, out of four 32x32->64
// multiplies, none of which needs a high half (Hacker's Delight, 8-2). This is the sequence
// that ran bit-exactly on the M4 in the bisection of issue #18.
llvm::Value *emitUMulHigh(llvm::IRBuilder<> &B, llvm::Value *X, llvm::Value *Y) {
    llvm::Value *Mask = B.getInt64(0xffffffff);
    llvm::Value *XL = B.CreateAnd(X, Mask);
    llvm::Value *XH = B.CreateLShr(X, 32);
    llvm::Value *YL = B.CreateAnd(Y, Mask);
    llvm::Value *YH = B.CreateLShr(Y, 32);
    llvm::Value *LL = B.CreateMul(XL, YL);
    llvm::Value *LH = B.CreateMul(XL, YH);
    llvm::Value *HL = B.CreateMul(XH, YL);
    llvm::Value *HH = B.CreateMul(XH, YH);
    // The middle column: at most 3 * (2^32 - 1), so it cannot carry out of 64 bits.
    llvm::Value *Mid =
        B.CreateAdd(B.CreateAdd(B.CreateLShr(LL, 32), B.CreateAnd(LH, Mask)), B.CreateAnd(HL, Mask));
    return B.CreateAdd(B.CreateAdd(B.CreateAdd(HH, B.CreateLShr(LH, 32)), B.CreateLShr(HL, 32)),
                       B.CreateLShr(Mid, 32));
}

// The high 64 bits of the signed 128-bit product: the unsigned high half, minus `y` when `x`
// is negative and minus `x` when `y` is negative, in wrapping arithmetic.
llvm::Value *emitSMulHigh(llvm::IRBuilder<> &B, llvm::Value *X, llvm::Value *Y) {
    llvm::Value *High = emitUMulHigh(B, X, Y);
    llvm::Value *YIfXNegative = B.CreateAnd(B.CreateAShr(X, 63), Y);
    llvm::Value *XIfYNegative = B.CreateAnd(B.CreateAShr(Y, 63), X);
    return B.CreateSub(B.CreateSub(High, YIfXNegative), XIfYNegative);
}

// The halves of a 128-bit operand. An operand that is an extension of an integer of 64 bits
// or fewer, or a constant, gives its halves without a 128-bit operation.
Halves splitOperand(llvm::IRBuilder<> &B, llvm::Value *V) {
    llvm::Type *I64 = B.getInt64Ty();
    if (auto *C = llvm::dyn_cast<llvm::ConstantInt>(V)) {
        const llvm::APInt &Value = C->getValue();
        llvm::Value *Lo = llvm::ConstantInt::get(I64, Value.trunc(64));
        llvm::APInt High = Value.lshr(64).trunc(64);
        return {Lo, High.isZero() ? nullptr : llvm::ConstantInt::get(I64, High)};
    }
    if (auto *Z = llvm::dyn_cast<llvm::ZExtInst>(V)) {
        if (Z->getSrcTy()->getScalarSizeInBits() <= 64)
            return {B.CreateZExt(Z->getOperand(0), I64), nullptr};
    }
    if (auto *S = llvm::dyn_cast<llvm::SExtInst>(V)) {
        if (S->getSrcTy()->getScalarSizeInBits() <= 64) {
            llvm::Value *Lo = B.CreateSExt(S->getOperand(0), I64);
            return {Lo, B.CreateAShr(Lo, 63)};
        }
    }
    return {B.CreateTrunc(V, I64), B.CreateTrunc(B.CreateLShr(V, 64), I64)};
}

// Whether a 128-bit operand is known to be a signed 64-bit value: a sign extension of 64 bits
// or fewer, a zero extension of 63 bits or fewer, or such a constant. The product of two of
// them always fits in 128 bits.
bool isSigned64(llvm::Value *V) {
    if (auto *C = llvm::dyn_cast<llvm::ConstantInt>(V)) return C->getValue().isSignedIntN(64);
    if (auto *S = llvm::dyn_cast<llvm::SExtInst>(V)) return S->getSrcTy()->getScalarSizeInBits() <= 64;
    if (auto *Z = llvm::dyn_cast<llvm::ZExtInst>(V)) return Z->getSrcTy()->getScalarSizeInBits() <= 63;
    return false;
}

// The low 128 bits of the product of two 128-bit integers, as halves. They are the same for
// signed and unsigned operands; the product of the two high halves only reaches bit 128.
Halves emitMulLow128(llvm::IRBuilder<> &B, const Halves &A, const Halves &C) {
    llvm::Value *Lo = B.CreateMul(A.Lo, C.Lo);
    llvm::Value *Hi = emitUMulHigh(B, A.Lo, C.Lo);
    if (C.Hi) Hi = B.CreateAdd(Hi, B.CreateMul(A.Lo, C.Hi));
    if (A.Hi) Hi = B.CreateAdd(Hi, B.CreateMul(A.Hi, C.Lo));
    return {Lo, Hi};
}

// 64 bits of a 128-bit value given as halves, starting at bit `Shift` (less than 128), with
// the sign of the value filling in from above for an arithmetic shift.
llvm::Value *bitsAt(llvm::IRBuilder<> &B, const Halves &V, uint64_t Shift, bool Arithmetic) {
    llvm::Value *Zero = B.getInt64(0);
    llvm::Value *Hi = V.Hi ? V.Hi : Zero;
    if (Shift == 0) return V.Lo;
    if (Shift < 64) return B.CreateOr(B.CreateLShr(V.Lo, Shift), B.CreateShl(Hi, 64 - Shift));
    if (Shift == 64) return Hi;
    return Arithmetic ? B.CreateAShr(Hi, Shift - 64) : B.CreateLShr(Hi, Shift - 64);
}

// Replaces every use of the 128-bit value `Old` by the value that `New` holds as halves.
// The uses the kernels of std.fmt and of 128-bit arithmetic make, a truncation to the low
// half and a truncation of a shift right (logical or arithmetic), become 64-bit operations;
// anything else gets the 128-bit value put back together once. The instructions whose uses
// were replaced go to `Dead`.
void replaceWide(llvm::Instruction *Old, const Halves &New, DeadList &Dead) {
    llvm::IRBuilder<> B(Old->getNextNode());
    llvm::Value *Whole = nullptr;
    auto whole = [&]() -> llvm::Value * {
        if (Whole) return Whole;
        llvm::Type *I128 = Old->getType();
        llvm::Value *Low = B.CreateZExt(New.Lo, I128);
        if (!New.Hi) return Whole = Low;
        return Whole = B.CreateOr(B.CreateShl(B.CreateZExt(New.Hi, I128), 64), Low);
    };

    llvm::SmallVector<llvm::Use *, 8> Uses;
    for (llvm::Use &U : Old->uses()) Uses.push_back(&U);
    for (llvm::Use *U : Uses) {
        auto *User = llvm::cast<llvm::Instruction>(U->getUser());
        llvm::IRBuilder<> At(User);
        if (auto *T = llvm::dyn_cast<llvm::TruncInst>(User)) {
            if (T->getType()->getIntegerBitWidth() <= 64) {
                T->replaceAllUsesWith(At.CreateZExtOrTrunc(New.Lo, T->getType()));
                Dead.push_back(T);
                continue;
            }
        }
        if (auto *Shift = llvm::dyn_cast<llvm::BinaryOperator>(User)) {
            auto *Amount = llvm::dyn_cast<llvm::ConstantInt>(Shift->getOperand(1));
            bool Arithmetic = Shift->getOpcode() == llvm::Instruction::AShr;
            bool OnlyTruncated = (Arithmetic || Shift->getOpcode() == llvm::Instruction::LShr) &&
                                 Amount && Amount->getValue().ult(128) && U->getOperandNo() == 0;
            for (llvm::User *ShiftUser : Shift->users()) {
                auto *T = llvm::dyn_cast<llvm::TruncInst>(ShiftUser);
                if (!T || T->getType()->getIntegerBitWidth() > 64) OnlyTruncated = false;
            }
            if (OnlyTruncated) {
                llvm::SmallVector<llvm::TruncInst *, 4> Truncs;
                for (llvm::User *ShiftUser : Shift->users())
                    Truncs.push_back(llvm::cast<llvm::TruncInst>(ShiftUser));
                for (llvm::TruncInst *T : Truncs) {
                    llvm::IRBuilder<> AtTrunc(T);
                    llvm::Value *Bits = bitsAt(AtTrunc, New, Amount->getZExtValue(), Arithmetic);
                    T->replaceAllUsesWith(AtTrunc.CreateZExtOrTrunc(Bits, T->getType()));
                    Dead.push_back(T);
                }
                Dead.push_back(Shift);
                continue;
            }
        }
        U->set(whole());
    }
}

// `llvm.{u,s}mul.with.overflow.iN` with 32 < N <= 64: the product, and the overflow flag from
// the product of the operands extended to 64 bits. Unsigned, the product overflows when its
// high 64 bits are not zero or, below 64 bits, when its low half has bits above bit N; signed,
// when the high half is not the sign extension of the low half or, below 64 bits, when the low
// half is not the sign extension of its own low N bits. Only the uses of the flag pay for the
// high half.
void expandMulOverflowScalar(llvm::IntrinsicInst *II, DeadList &Dead) {
    bool Signed = II->getIntrinsicID() == llvm::Intrinsic::smul_with_overflow;
    llvm::Type *T = II->getArgOperand(0)->getType();
    unsigned Bits = T->getIntegerBitWidth();
    llvm::IRBuilder<> B(II);
    llvm::Type *I64 = B.getInt64Ty();
    auto widen = [&](llvm::Value *V) {
        return Bits == 64 ? V : Signed ? B.CreateSExt(V, I64) : B.CreateZExt(V, I64);
    };
    llvm::Value *X = widen(II->getArgOperand(0));
    llvm::Value *Y = widen(II->getArgOperand(1));
    llvm::Value *Wide = B.CreateMul(X, Y);
    llvm::Value *Product = Bits == 64 ? Wide : B.CreateTrunc(Wide, T);
    llvm::Value *Overflow = nullptr;
    auto overflow = [&]() -> llvm::Value * {
        if (Overflow) return Overflow;
        if (Signed) {
            Overflow = B.CreateICmpNE(emitSMulHigh(B, X, Y), B.CreateAShr(Wide, 63));
            if (Bits < 64)
                Overflow = B.CreateOr(Overflow,
                                      B.CreateICmpNE(B.CreateSExt(Product, I64), Wide));
        } else {
            Overflow = B.CreateICmpNE(emitUMulHigh(B, X, Y), B.getInt64(0));
            if (Bits < 64)
                Overflow = B.CreateOr(Overflow,
                                      B.CreateICmpNE(B.CreateLShr(Wide, Bits), B.getInt64(0)));
        }
        return Overflow;
    };

    llvm::SmallVector<llvm::User *, 4> Users(II->users());
    for (llvm::User *User : Users) {
        auto *EV = llvm::dyn_cast<llvm::ExtractValueInst>(User);
        if (!EV || EV->getNumIndices() != 1) {
            // A use of the whole pair: rebuild it.
            llvm::Value *Pair = llvm::PoisonValue::get(II->getType());
            Pair = B.CreateInsertValue(Pair, Product, {0});
            Pair = B.CreateInsertValue(Pair, overflow(), {1});
            User->replaceUsesOfWith(II, Pair);
            continue;
        }
        EV->replaceAllUsesWith(EV->getIndices()[0] == 0 ? Product : overflow());
        Dead.push_back(EV);
    }
    Dead.push_back(II);
    // The product when only the flag was used.
    if (auto *P = llvm::dyn_cast<llvm::Instruction>(Product)) Dead.push_back(P);
}

// `llvm.umul.with.overflow.i128`: the low 128 bits of the product, and whether the full product
// needs more than 128 bits. With A = Ah:Al and C = Ch:Cl, it overflows when both high halves are
// non-zero, when a cross product Ah*Cl or Al*Ch needs more than 64 bits, or when adding the
// cross products' low halves to the high half of Al*Cl carries out of 64 bits.
//
// `llvm.smul.with.overflow.i128` is rebuilt only where both operands are known to be signed
// 64-bit values (isSigned64), whose product always fits in 128 bits.
bool expandMulOverflow128(llvm::IntrinsicInst *II, DeadList &Dead, std::string &Err) {
    bool Signed = II->getIntrinsicID() == llvm::Intrinsic::smul_with_overflow;
    if (Signed && (!isSigned64(II->getArgOperand(0)) || !isSigned64(II->getArgOperand(1)))) {
        Err = "air64: " + where(*II) +
              ": Apple's GPU compiler cannot lower a checked multiply of 128-bit signed "
              "integers, and it is only rebuilt for operands that are 64-bit values: " +
              describe(*II);
        return false;
    }
    llvm::IRBuilder<> B(II);
    Halves A = splitOperand(B, II->getArgOperand(0));
    Halves C = splitOperand(B, II->getArgOperand(1));
    Halves Product = emitMulLow128(B, A, C);

    llvm::Value *Overflow = B.getFalse();
    if (!Signed && (A.Hi || C.Hi)) {
        llvm::Value *Zero = B.getInt64(0);
        llvm::Value *Carry = B.getFalse();
        llvm::Value *Sum = emitUMulHigh(B, A.Lo, C.Lo);
        llvm::Value *Wide = B.getFalse();
        auto addCross = [&](llvm::Value *P, llvm::Value *Q) {
            llvm::Value *CrossLo = B.CreateMul(P, Q);
            Wide = B.CreateOr(Wide, B.CreateICmpNE(emitUMulHigh(B, P, Q), Zero));
            llvm::Value *Next = B.CreateAdd(Sum, CrossLo);
            Carry = B.CreateOr(Carry, B.CreateICmpULT(Next, Sum));
            Sum = Next;
        };
        if (A.Hi) addCross(A.Hi, C.Lo);
        if (C.Hi) addCross(A.Lo, C.Hi);
        llvm::Value *BothHigh = B.getFalse();
        if (A.Hi && C.Hi)
            BothHigh = B.CreateAnd(B.CreateICmpNE(A.Hi, Zero), B.CreateICmpNE(C.Hi, Zero));
        Overflow = B.CreateOr(B.CreateOr(BothHigh, Wide), Carry);
    }

    llvm::SmallVector<llvm::User *, 4> Users(II->users());
    for (llvm::User *User : Users) {
        auto *EV = llvm::dyn_cast<llvm::ExtractValueInst>(User);
        if (EV && EV->getNumIndices() == 1 && EV->getIndices()[0] == 1) {
            EV->replaceAllUsesWith(Overflow);
            Dead.push_back(EV);
        } else if (EV && EV->getNumIndices() == 1) {
            replaceWide(EV, Product, Dead);
            Dead.push_back(EV);
        } else {
            llvm::Type *I128 = II->getArgOperand(0)->getType();
            llvm::Value *Value = B.CreateZExt(Product.Lo, I128);
            if (Product.Hi)
                Value = B.CreateOr(B.CreateShl(B.CreateZExt(Product.Hi, I128), 64), Value);
            llvm::Value *Pair = llvm::PoisonValue::get(II->getType());
            Pair = B.CreateInsertValue(Pair, Value, {0});
            Pair = B.CreateInsertValue(Pair, Overflow, {1});
            User->replaceUsesOfWith(II, Pair);
        }
    }
    Dead.push_back(II);
    return true;
}

// The overflow intrinsics of vectors of i64 (Zig's checked multiply of `@Vector(n, u64)`),
// one lane at a time.
void expandMulOverflowVector(llvm::IntrinsicInst *II, DeadList &Dead) {
    bool Signed = II->getIntrinsicID() == llvm::Intrinsic::smul_with_overflow;
    auto *VT = llvm::cast<llvm::FixedVectorType>(II->getArgOperand(0)->getType());
    auto *FlagsT = llvm::FixedVectorType::get(llvm::Type::getInt1Ty(II->getContext()),
                                              VT->getNumElements());
    llvm::IRBuilder<> B(II);
    llvm::Value *Products = llvm::PoisonValue::get(VT);
    llvm::Value *Flags = llvm::PoisonValue::get(FlagsT);
    for (unsigned Lane = 0; Lane < VT->getNumElements(); ++Lane) {
        llvm::Value *X = B.CreateExtractElement(II->getArgOperand(0), Lane);
        llvm::Value *Y = B.CreateExtractElement(II->getArgOperand(1), Lane);
        llvm::Value *Product = B.CreateMul(X, Y);
        llvm::Value *Overflow =
            Signed ? B.CreateICmpNE(emitSMulHigh(B, X, Y), B.CreateAShr(Product, 63))
                   : B.CreateICmpNE(emitUMulHigh(B, X, Y), B.getInt64(0));
        Products = B.CreateInsertElement(Products, Product, Lane);
        Flags = B.CreateInsertElement(Flags, Overflow, Lane);
    }
    llvm::Value *Pair = llvm::PoisonValue::get(II->getType());
    Pair = B.CreateInsertValue(Pair, Products, {0});
    Pair = B.CreateInsertValue(Pair, Flags, {1});
    II->replaceAllUsesWith(Pair);
    Dead.push_back(II);
}

bool expandWideMultiplies(llvm::Module &M, std::string &Err) {
    // A Debug build leaves constant operations unfolded, like the zero extension of the shift
    // amount of a u128 `>> 64`, which would hide the shapes matched below.
    {
        llvm::FunctionPassManager FPM;
        FPM.addPass(llvm::InstSimplifyPass());
        runOnFunctions(M, std::move(FPM));
    }

    llvm::SmallVector<llvm::Instruction *, 16> Work;
    for (llvm::Function &F : M) {
        if (F.isDeclaration()) continue;
        for (llvm::Instruction &I : llvm::instructions(F)) {
            if (auto *II = llvm::dyn_cast<llvm::IntrinsicInst>(&I)) {
                llvm::Intrinsic::ID ID = II->getIntrinsicID();
                if (ID == llvm::Intrinsic::umul_with_overflow ||
                    ID == llvm::Intrinsic::smul_with_overflow) {
                    if (II->getArgOperand(0)->getType()->getScalarSizeInBits() > 32)
                        Work.push_back(II);
                }
                continue;
            }
            if (I.getOpcode() == llvm::Instruction::Mul && isInt(I.getType(), 128))
                Work.push_back(&I);
        }
    }

    llvm::SmallVector<llvm::WeakTrackingVH, 32> Dead;
    for (llvm::Instruction *I : Work) {
        if (auto *II = llvm::dyn_cast<llvm::IntrinsicInst>(I)) {
            llvm::Type *T = II->getArgOperand(0)->getType();
            if (T->isIntegerTy() && T->getIntegerBitWidth() <= 64) {
                expandMulOverflowScalar(II, Dead);
            } else if (isInt(T, 128)) {
                if (!expandMulOverflow128(II, Dead, Err)) return false;
            } else if (auto *VT = llvm::dyn_cast<llvm::FixedVectorType>(T);
                       VT && isInt(VT->getElementType(), 64)) {
                expandMulOverflowVector(II, Dead);
            }
            continue;
        }
        llvm::IRBuilder<> B(I);
        Halves A = splitOperand(B, I->getOperand(0));
        Halves C = splitOperand(B, I->getOperand(1));
        replaceWide(I, emitMulLow128(B, A, C), Dead);
        Dead.push_back(I);
    }
    llvm::RecursivelyDeleteTriviallyDeadInstructionsPermissive(Dead);
    for (llvm::Function &F : llvm::make_early_inc_range(M)) {
        llvm::Intrinsic::ID ID = F.getIntrinsicID();
        if (F.use_empty() && (ID == llvm::Intrinsic::umul_with_overflow ||
                              ID == llvm::Intrinsic::smul_with_overflow))
            F.eraseFromParent();
    }

    // What is left needs a high half this pass does not rebuild: say which, rather than let
    // Apple's compiler crash on it.
    for (llvm::Function &F : M) {
        if (F.isDeclaration()) continue;
        for (llvm::Instruction &I : llvm::instructions(F)) {
            bool Wide = false;
            if (I.getOpcode() == llvm::Instruction::Mul)
                Wide = I.getType()->getScalarSizeInBits() > 64;
            else if (auto *II = llvm::dyn_cast<llvm::IntrinsicInst>(&I))
                Wide = (II->getIntrinsicID() == llvm::Intrinsic::umul_with_overflow ||
                        II->getIntrinsicID() == llvm::Intrinsic::smul_with_overflow) &&
                       II->getArgOperand(0)->getType()->getScalarSizeInBits() > 32;
            if (Wide) {
                Err = "air64: " + F.getName().str() +
                      ": Apple's GPU compiler cannot lower a multiply whose high 64 bits are "
                      "needed, and this one has no 32-bit expansion: " +
                      describe(I);
                return false;
            }
        }
    }
    return true;
}

}  // namespace

bool zigAirLowerLate(llvm::Module &M, std::string &Err) {
    if (!moveConstantsToConstantSpace(M, Err)) return false;
    return expandWideMultiplies(M, Err);
}
