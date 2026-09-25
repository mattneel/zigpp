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
//    Apple's toolchain does not relocate the addresses inside constant data either (issue #22),
//    so constant data is made position-independent: the constants that hold pointers to other
//    constants (a table of slices, like std.fmt.parseFloat's powers of five written out as
//    decimal strings or the table behind `@errorName`) and the constants they point to are laid
//    out in one global, and each such pointer is stored as its offset into that global. A
//    pointer loaded out of it is loaded as the offset and becomes the global's address plus the
//    offset; the only pointers in constant data are the ones the compiler put there, so this is
//    exact. A pointer that is a number and not an address, like the pointer of an empty slice,
//    needs no relocation and is kept as the integer of the same bits.
//
//    Zig's pointer types say nothing about constant data, and an Apple GPU has no generic
//    address space to cast a pointer into constant data to, so every pointer that comes from
//    a constant has to be known for one wherever it is used. SROA takes apart the copies a
//    Debug build makes on the stack, and LLVM's address space inference, with address space 0
//    as the flat one, carries address space 2 from the globals through the address
//    arithmetic, the phis, the loads and the copies. Where such a pointer crosses a call, or
//    meets thread memory at the edge of a function (a phi with a local, a store, a return),
//    the call is inlined and inference runs again in the caller, until nothing crosses; the
//    other functions the optimizer left stay as they are. A table of function pointers, like
//    an allocator's vtable, gets the same treatment: once the calls that carry it are inlined,
//    the loads of its entries are constant and fold into direct calls.
//
//    What is left is reported by name: a pointer into constant data that meets thread memory
//    in a kernel, the bytes of a pointer in constant data read as something else, a table of
//    functions that a kernel still calls through, and a mutable program-scope variable, which
//    has no place on an Apple GPU.
//
// 2. Arithmetic. Apple's AIR-to-GPU compiler cannot produce the high 64 bits of a 64x64-bit
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
//
//    The add and subtract overflow intrinsics, `llvm.{s,u}{add,sub}.with.overflow`, Apple's
//    compiler gets wrong in another way: the flag of the signed ones is wrong (0 - 10 overflows
//    an i64, it says), and the 8-bit ones make its compiler service die (issue #26). Every
//    checked add and subtract of a Debug kernel is one, so each becomes the operation and a
//    comparison of its operands and result.

#include "zig_air.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
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
#include <optional>
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

// The type of a constant's value with each pointer it holds replaced by an offset into the
// blob: a pointer in address space 0 becomes an i64, of the same size and alignment in AIR, so
// the layout does not move.
llvm::Type *offsetType(llvm::Type *T) {
    llvm::LLVMContext &Ctx = T->getContext();
    if (auto *PT = llvm::dyn_cast<llvm::PointerType>(T))
        return PT->getAddressSpace() == 0 ? llvm::Type::getInt64Ty(Ctx) : T;
    if (auto *VT = llvm::dyn_cast<llvm::VectorType>(T)) {
        llvm::Type *Element = offsetType(VT->getElementType());
        return Element == VT->getElementType() ? T
                                               : llvm::VectorType::get(Element, VT->getElementCount());
    }
    if (auto *AT = llvm::dyn_cast<llvm::ArrayType>(T)) {
        llvm::Type *Element = offsetType(AT->getElementType());
        return Element == AT->getElementType() ? T
                                               : llvm::ArrayType::get(Element, AT->getNumElements());
    }
    if (auto *ST = llvm::dyn_cast<llvm::StructType>(T)) {
        llvm::SmallVector<llvm::Type *, 8> Elements;
        bool Changed = false;
        for (llvm::Type *Element : ST->elements()) {
            Elements.push_back(offsetType(Element));
            Changed |= Elements.back() != Element;
        }
        return Changed ? llvm::StructType::get(Ctx, Elements, ST->isPacked()) : T;
    }
    return T;
}

// Whether a `T` holds a pointer in address space 0.
bool holdsFlatPointer(llvm::Type *T) { return offsetType(T) != T; }

// The globals whose addresses `C` holds.
void addressesIn(const llvm::Constant *C, llvm::SmallPtrSetImpl<const llvm::GlobalValue *> &Out) {
    if (auto *GV = llvm::dyn_cast<llvm::GlobalValue>(C)) {
        Out.insert(GV);
        return;
    }
    if (!llvm::isa<llvm::ConstantAggregate>(C) && !llvm::isa<llvm::ConstantExpr>(C)) return;
    for (const llvm::Use &Op : C->operands()) addressesIn(llvm::cast<llvm::Constant>(Op.get()), Out);
}

// Apple's toolchain does not relocate the addresses inside constant data: a table of slices,
// like std.fmt.parseFloat's powers of five written out as decimal strings or the table behind
// `@errorName`, fails to link with "Undefined symbols: _unnamed_1", makes the compiler service
// die, or reads zeros where its pointers should be, depending on the kernel (issue #22). So the
// constants that hold pointers to other constants, and the constants they point to, are laid
// out in one global, the blob, and each such pointer is stored as an offset into it; 0 stays
// null, because the blob starts with a byte that no constant is placed at. The only pointers
// in constant data are the ones the compiler put there, so a pointer loaded out of the blob is
// always one of these offsets.
struct Blob {
    llvm::GlobalVariable *Global = nullptr;
    // Where each constant that moved into the blob is.
    llvm::DenseMap<const llvm::GlobalVariable *, uint64_t> Offsets;
    // The members in the order of their offsets, with their value types as they were, pointers
    // and all, for telling which bytes are the offsets of pointers.
    llvm::SmallVector<std::pair<uint64_t, llvm::Type *>, 16> Layout;
    // Whether a constant that a pointer in the blob points at holds pointers itself: whether the
    // bytes behind a pointer loaded out of the blob may be offsets.
    bool PointedHoldPointers = false;
};

constexpr const char *blob_name = "__zig_air_constants";

// The metadata that marks what loadConstantPointers makes, until the checks after it have run:
// the loads of offsets, which read the blob as the offsets it holds, and the pointers decoded
// from them.
constexpr const char *offset_load_kind = "zig.air.offset";
constexpr const char *decoded_kind = "zig.air.decoded";

// `C` with the pointers it holds replaced by their offsets into the blob, of the type offsetType
// gives. Returns nullptr, with the offending part in `Bad`, for an address that is not one of a
// member of the blob, or one used as an integer.
llvm::Constant *offsetConstant(llvm::Constant *C, const Blob &B, const llvm::DataLayout &DL,
                               const llvm::Constant *&Bad) {
    llvm::Type *T = C->getType();
    llvm::Type *NewT = offsetType(T);
    llvm::Type *I64 = llvm::Type::getInt64Ty(C->getContext());
    if (auto *GEP = llvm::dyn_cast<llvm::GEPOperator>(C); GEP || llvm::isa<llvm::GlobalValue>(C)) {
        llvm::APInt Offset(64, 0);
        const llvm::Value *Base = C->stripAndAccumulateConstantOffsets(DL, Offset, true);
        auto *Member = llvm::dyn_cast<llvm::GlobalVariable>(Base);
        auto It = Member ? B.Offsets.find(Member) : B.Offsets.end();
        if (It == B.Offsets.end() || NewT != I64) {
            Bad = C;
            return nullptr;
        }
        return llvm::ConstantInt::get(I64, It->second + Offset.getSExtValue());
    }
    if (llvm::isa<llvm::ConstantAggregate>(C)) {
        llvm::SmallVector<llvm::Constant *, 16> Ops;
        for (llvm::Use &Op : C->operands()) {
            llvm::Constant *New = offsetConstant(llvm::cast<llvm::Constant>(Op.get()), B, DL, Bad);
            if (!New) return nullptr;
            Ops.push_back(New);
        }
        if (llvm::isa<llvm::ConstantStruct>(C))
            return llvm::ConstantStruct::get(llvm::cast<llvm::StructType>(NewT), Ops);
        if (llvm::isa<llvm::ConstantArray>(C))
            return llvm::ConstantArray::get(llvm::cast<llvm::ArrayType>(NewT), Ops);
        return llvm::ConstantVector::get(Ops);
    }
    llvm::SmallPtrSet<const llvm::GlobalValue *, 4> Held;
    addressesIn(C, Held);
    if (!Held.empty()) {
        Bad = C;
        return nullptr;
    }
    if (NewT == T) return C;
    if (llvm::isa<llvm::PoisonValue>(C)) return llvm::PoisonValue::get(NewT);
    if (llvm::isa<llvm::UndefValue>(C)) return llvm::UndefValue::get(NewT);
    if (C->isNullValue()) return llvm::Constant::getNullValue(NewT);
    Bad = C;
    return nullptr;
}

// Whether `C` holds a pointer that is a constant expression, which the downgrader to the typed
// pointers of the bitcode Apple's reader takes cannot write into an initializer.
bool holdsPointerExpr(const llvm::Constant *C) {
    if (llvm::isa<llvm::ConstantExpr>(C) && C->getType()->isPtrOrPtrVectorTy()) return true;
    if (!llvm::isa<llvm::ConstantAggregate>(C) && !llvm::isa<llvm::ConstantExpr>(C)) return false;
    return llvm::any_of(C->operands(), [](const llvm::Use &Op) {
        return holdsPointerExpr(llvm::cast<llvm::Constant>(Op.get()));
    });
}

// `C`, which holds no address, with its pointers (null, undefined, or made from an integer, like
// the pointer of an empty slice) replaced by the integers of the same bits, of the type
// offsetType gives. Returns nullptr for a pointer that is none of those.
llvm::Constant *integerPointers(llvm::Constant *C) {
    llvm::Type *T = C->getType();
    llvm::Type *NewT = offsetType(T);
    if (NewT == T) return C;
    if (llvm::isa<llvm::PoisonValue>(C)) return llvm::PoisonValue::get(NewT);
    if (llvm::isa<llvm::UndefValue>(C)) return llvm::UndefValue::get(NewT);
    if (C->isNullValue()) return llvm::Constant::getNullValue(NewT);
    if (auto *CE = llvm::dyn_cast<llvm::ConstantExpr>(C);
        CE && CE->getOpcode() == llvm::Instruction::IntToPtr && NewT->isIntegerTy()) {
        if (auto *Int = llvm::dyn_cast<llvm::ConstantInt>(CE->getOperand(0)))
            return llvm::ConstantInt::get(
                NewT, Int->getValue().zextOrTrunc(NewT->getIntegerBitWidth()));
        return nullptr;
    }
    if (!llvm::isa<llvm::ConstantAggregate>(C)) return nullptr;
    llvm::SmallVector<llvm::Constant *, 16> Ops;
    for (llvm::Use &Op : C->operands()) {
        llvm::Constant *New = integerPointers(llvm::cast<llvm::Constant>(Op.get()));
        if (!New) return nullptr;
        Ops.push_back(New);
    }
    if (llvm::isa<llvm::ConstantStruct>(C))
        return llvm::ConstantStruct::get(llvm::cast<llvm::StructType>(NewT), Ops);
    if (llvm::isa<llvm::ConstantArray>(C))
        return llvm::ConstantArray::get(llvm::cast<llvm::ArrayType>(NewT), Ops);
    return llvm::ConstantVector::get(Ops);
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

// Moves the program-scope constants to the constant address space: the ones that hold pointers
// to other constants, and the ones those point to, into the blob, and the others each on its
// own. Each use of one becomes a cast of its new address to address space 0, where inference
// picks it up.
//
// A constant that holds the addresses of functions, like an allocator's vtable, moves on its
// own: inlining is what makes the loads of its entries constant, so that they fold into direct
// calls, and checkFunctionTables reports the ones still used after that. The blob is not marked
// constant until the loads out of it have been rewritten, so that nothing folds a pointer out of
// it into the bare offset it holds.
bool moveConstants(llvm::Module &M, Blob &B, std::string &Err) {
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
        Constants.push_back(&GV);
    }

    llvm::SmallPtrSet<const llvm::GlobalVariable *, 16> IsConstant(Constants.begin(), Constants.end());
    llvm::SmallPtrSet<const llvm::GlobalVariable *, 16> InBlob, Pointed, HoldsFunctions;
    for (llvm::GlobalVariable *GV : Constants) {
        llvm::SmallPtrSet<const llvm::GlobalValue *, 8> Held;
        addressesIn(GV->getInitializer(), Held);
        bool HoldsData = false;
        for (const llvm::GlobalValue *Address : Held) {
            if (llvm::isa<llvm::Function>(Address)) {
                HoldsFunctions.insert(GV);
                continue;
            }
            auto *Target = llvm::dyn_cast<llvm::GlobalVariable>(Address);
            if (!Target || !IsConstant.contains(Target)) {
                Err = "air64: the program-scope constant " + GV->getName().str() +
                      " holds the address of " + describeOperand(*Address) +
                      ", which an Apple GPU cannot keep in constant data: only the addresses of "
                      "other constants are";
                return false;
            }
            InBlob.insert(Target);
            Pointed.insert(Target);
            HoldsData = true;
        }
        if (HoldsData) InBlob.insert(GV);
    }
    for (const llvm::GlobalVariable *GV : InBlob) {
        if (!HoldsFunctions.contains(GV)) continue;
        Err = "air64: the program-scope constant " + GV->getName().str() +
              " holds the address of a function, and it " +
              (Pointed.contains(GV) ? "is pointed at by other constant data"
                                    : "holds pointers to other constants too") +
              "; on an Apple GPU such pointers are offsets into constant data, which the address "
              "of a function cannot be";
        return false;
    }
    for (const llvm::GlobalVariable *GV : Pointed) {
        llvm::SmallPtrSet<const llvm::GlobalValue *, 8> Held;
        addressesIn(GV->getInitializer(), Held);
        if (!Held.empty()) B.PointedHoldPointers = true;
    }

    const llvm::DataLayout &DL = M.getDataLayout();
    llvm::LLVMContext &Ctx = M.getContext();
    if (!InBlob.empty()) {
        llvm::Type *I8 = llvm::Type::getInt8Ty(Ctx);
        llvm::SmallVector<llvm::Type *, 32> Fields{I8};
        llvm::SmallVector<llvm::GlobalVariable *, 16> Members;
        llvm::SmallVector<unsigned, 16> MemberField;
        uint64_t Offset = 1;
        llvm::Align BlobAlign(1);
        for (llvm::GlobalVariable *GV : Constants) {
            if (!InBlob.contains(GV)) continue;
            llvm::Align A = DL.getPreferredAlign(GV);
            uint64_t At = llvm::alignTo(Offset, A);
            if (At > Offset) Fields.push_back(llvm::ArrayType::get(I8, At - Offset));
            B.Offsets[GV] = At;
            B.Layout.push_back({At, GV->getValueType()});
            MemberField.push_back((unsigned)Fields.size());
            Fields.push_back(offsetType(GV->getValueType()));
            Members.push_back(GV);
            Offset = At + DL.getTypeAllocSize(GV->getValueType()).getFixedValue();
            BlobAlign = std::max(BlobAlign, A);
        }
        auto *BlobType = llvm::StructType::get(Ctx, Fields, /*isPacked=*/true);
        llvm::SmallVector<llvm::Constant *, 32> Inits;
        for (llvm::Type *Field : Fields) Inits.push_back(llvm::Constant::getNullValue(Field));
        for (size_t I = 0; I < Members.size(); ++I) {
            const llvm::Constant *Bad = nullptr;
            llvm::Constant *Init = offsetConstant(Members[I]->getInitializer(), B, DL, Bad);
            if (!Init) {
                Err = "air64: the program-scope constant " + Members[I]->getName().str() +
                      " holds " + describeOperand(*Bad) +
                      ", an address that has no offset in the constant data of an Apple GPU";
                return false;
            }
            Inits[MemberField[I]] = Init;
        }
        B.Global = new llvm::GlobalVariable(M, BlobType, /*isConstant=*/false,
                                            llvm::GlobalValue::PrivateLinkage,
                                            llvm::ConstantStruct::get(BlobType, Inits), blob_name,
                                            nullptr, llvm::GlobalValue::NotThreadLocal,
                                            air_constant_address_space);
        B.Global->setAlignment(BlobAlign);
        B.Global->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
        for (llvm::GlobalVariable *GV : Members) {
            llvm::Constant *Address = llvm::ConstantExpr::getInBoundsGetElementPtr(
                I8, B.Global, llvm::ConstantInt::get(llvm::Type::getInt64Ty(Ctx), B.Offsets[GV]));
            GV->replaceAllUsesWith(llvm::ConstantExpr::getAddrSpaceCast(Address, GV->getType()));
            GV->eraseFromParent();
        }
    }

    for (llvm::GlobalVariable *GV : Constants) {
        if (InBlob.contains(GV)) continue;
        // A pointer that is a number and not an address, like the pointer of an empty slice,
        // needs no relocation; it is kept as the integer of the same bits, because the
        // downgrader writes no pointer expression into an initializer. A load of it as a pointer
        // reads the same bits. The addresses in a table of functions fold away or are reported.
        llvm::Constant *Init = GV->getInitializer();
        if (!HoldsFunctions.contains(GV) && holdsPointerExpr(Init)) {
            Init = integerPointers(Init);
            if (!Init) {
                Err = "air64: the program-scope constant " + GV->getName().str() +
                      " holds a pointer that is neither the address of a constant nor a number, "
                      "which an Apple GPU cannot keep in constant data: " +
                      describe(*GV->getInitializer());
                return false;
            }
        }
        auto *New = new llvm::GlobalVariable(M, Init->getType(), /*isConstant=*/true,
                                             GV->getLinkage(), Init, "", GV,
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

// The value of type `T` that the offsets in `V`, of type offsetType(T), stand for: a pointer
// into the blob for each non-zero offset, null for zero, cast to the address space 0 of `T`
// until inference carries address space 2 into the uses.
llvm::Value *fromOffsets(llvm::IRBuilder<> &B, llvm::Value *V, llvm::Type *T,
                         llvm::GlobalVariable *Blob, unsigned DecodedKind) {
    if (V->getType() == T) return V;
    if (T->isPtrOrPtrVectorTy()) {
        llvm::LLVMContext &Ctx = T->getContext();
        llvm::Type *ConstantPtr = llvm::PointerType::get(Ctx, air_constant_address_space);
        if (auto *VT = llvm::dyn_cast<llvm::VectorType>(T))
            ConstantPtr = llvm::VectorType::get(ConstantPtr, VT->getElementCount());
        llvm::MDNode *Mark = llvm::MDNode::get(Ctx, {});
        llvm::Value *Address = B.CreateInBoundsGEP(B.getInt8Ty(), Blob, V);
        llvm::Value *IsNull = B.CreateICmpEQ(V, llvm::Constant::getNullValue(V->getType()));
        llvm::Value *Pointer =
            B.CreateSelect(IsNull, llvm::Constant::getNullValue(ConstantPtr), Address);
        for (llvm::Value *Made : {Address, Pointer})
            if (auto *I = llvm::dyn_cast<llvm::Instruction>(Made)) I->setMetadata(DecodedKind, Mark);
        return B.CreateAddrSpaceCast(Pointer, T);
    }
    llvm::Value *Result = llvm::PoisonValue::get(T);
    if (auto *ST = llvm::dyn_cast<llvm::StructType>(T)) {
        for (unsigned I = 0, E = ST->getNumElements(); I != E; ++I)
            Result = B.CreateInsertValue(
                Result,
                fromOffsets(B, B.CreateExtractValue(V, I), ST->getElementType(I), Blob, DecodedKind),
                I);
        return Result;
    }
    auto *AT = llvm::cast<llvm::ArrayType>(T);
    for (unsigned I = 0, E = (unsigned)AT->getNumElements(); I != E; ++I)
        Result = B.CreateInsertValue(
            Result,
            fromOffsets(B, B.CreateExtractValue(V, I), AT->getElementType(), Blob, DecodedKind), I);
    return Result;
}

// Whether a pointer loaded through `Ptr` is one of the offsets in the blob: whether `Ptr` points
// into the blob (or is null) and nowhere else. A kernel's scalar parameters are in the constant
// address space too, and a pointer loaded out of one of those is a real address.
bool readsBlobOnly(const llvm::Value *Ptr, const llvm::GlobalVariable *Blob) {
    llvm::SmallVector<const llvm::Value *, 4> Objects;
    llvm::getUnderlyingObjects(Ptr, Objects, nullptr, 0);
    bool FromBlob = false;
    for (const llvm::Value *Object : Objects) {
        if (Object == Blob)
            FromBlob = true;
        else if (!llvm::isa<llvm::ConstantPointerNull>(Object) && !llvm::isa<llvm::UndefValue>(Object))
            return false;
    }
    return FromBlob;
}

// A pointer that a kernel loads out of the blob is an offset into it: the load loads the offset,
// and the pointer is the blob's address plus it. Returns whether any load changed.
bool loadConstantPointers(llvm::Module &M, const Blob &TheBlob) {
    if (!TheBlob.Global) return false;
    llvm::LLVMContext &Ctx = M.getContext();
    unsigned OffsetKind = Ctx.getMDKindID(offset_load_kind);
    unsigned DecodedKind = Ctx.getMDKindID(decoded_kind);
    llvm::SmallVector<llvm::LoadInst *, 8> Loads;
    for (llvm::Function &F : M) {
        for (llvm::Instruction &I : llvm::instructions(F)) {
            auto *L = llvm::dyn_cast<llvm::LoadInst>(&I);
            if (L && holdsFlatPointer(L->getType()) &&
                readsBlobOnly(L->getPointerOperand(), TheBlob.Global))
                Loads.push_back(L);
        }
    }
    for (llvm::LoadInst *L : Loads) {
        llvm::IRBuilder<> B(L);
        llvm::LoadInst *New = B.CreateAlignedLoad(offsetType(L->getType()), L->getPointerOperand(),
                                                  L->getAlign(), L->isVolatile());
        New->setAtomic(L->getOrdering(), L->getSyncScopeID());
        New->copyMetadata(*L, {llvm::LLVMContext::MD_tbaa, llvm::LLVMContext::MD_alias_scope,
                               llvm::LLVMContext::MD_noalias});
        New->setMetadata(OffsetKind, llvm::MDNode::get(Ctx, {}));
        New->takeName(L);
        L->replaceAllUsesWith(fromOffsets(B, New, L->getType(), TheBlob.Global, DecodedKind));
        L->eraseFromParent();
    }
    return !Loads.empty();
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
// call that is passed one; a call that is passed memory on the stack that one was stored in,
// like a struct that holds an allocator's vtable; and each call of a function where one meets
// thread memory in another way (a phi with a local, a store to memory of the caller, a return),
// since the caller is where the two can be told apart. A function is not inlined into itself.
llvm::SmallSetVector<llvm::CallBase *, 16> callsToInline(llvm::Module &M) {
    llvm::SmallSetVector<llvm::CallBase *, 16> Calls;
    auto inlinable = [](const llvm::CallBase &Call, const llvm::Function &Caller) {
        const llvm::Function *Callee = Call.getCalledFunction();
        return Callee && !Callee->isDeclaration() && Callee != &Caller;
    };
    for (llvm::Function &F : M) {
        if (F.isDeclaration()) continue;
        bool EscapesHere = false;
        llvm::SmallPtrSet<const llvm::Value *, 4> Holders;
        for (llvm::Instruction &I : llvm::instructions(F)) {
            if (!escapesConstantSpace(I)) continue;
            if (auto *Call = llvm::dyn_cast<llvm::CallBase>(&I); Call && inlinable(*Call, F)) {
                Calls.insert(Call);
                continue;
            }
            if (auto *Store = llvm::dyn_cast<llvm::StoreInst>(&I)) {
                const llvm::Value *Object = llvm::getUnderlyingObject(Store->getPointerOperand(), 0);
                if (llvm::isa<llvm::AllocaInst>(Object)) {
                    Holders.insert(Object);
                    continue;
                }
            }
            EscapesHere = true;
        }
        if (!Holders.empty()) {
            for (llvm::Instruction &I : llvm::instructions(F)) {
                auto *Call = llvm::dyn_cast<llvm::CallBase>(&I);
                if (!Call || !inlinable(*Call, F)) continue;
                for (const llvm::Use &Arg : Call->args()) {
                    if (Arg->getType()->isPointerTy() &&
                        Holders.contains(llvm::getUnderlyingObject(Arg.get(), 0))) {
                        Calls.insert(Call);
                        break;
                    }
                }
            }
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

// Whether bytes [Begin, End) of a `T` hold part of a pointer in address space 0: in the blob,
// part of the offset that stands for one.
bool bytesHoldFlatPointer(const llvm::DataLayout &DL, llvm::Type *T, uint64_t Begin,
                          uint64_t End) {
    if (Begin >= End || !holdsFlatPointer(T)) return false;
    if (T->isPtrOrPtrVectorTy()) return true;
    if (auto *ST = llvm::dyn_cast<llvm::StructType>(T)) {
        const llvm::StructLayout *Layout = DL.getStructLayout(ST);
        for (unsigned I = 0, E = ST->getNumElements(); I != E; ++I) {
            uint64_t Offset = Layout->getElementOffset(I).getFixedValue();
            uint64_t Size = DL.getTypeStoreSize(ST->getElementType(I)).getFixedValue();
            uint64_t From = std::max(Begin, Offset);
            uint64_t To = std::min(End, Offset + Size);
            if (From < To &&
                bytesHoldFlatPointer(DL, ST->getElementType(I), From - Offset, To - Offset))
                return true;
        }
        return false;
    }
    auto *AT = llvm::cast<llvm::ArrayType>(T);
    uint64_t Stride = DL.getTypeAllocSize(AT->getElementType()).getFixedValue();
    if (Stride == 0) return false;
    uint64_t First = Begin / Stride;
    uint64_t Last = (End - 1) / Stride;
    // A whole element in between holds its pointer.
    if (Last - First >= 2) return true;
    for (uint64_t Index : {First, Last}) {
        uint64_t Offset = Index * Stride;
        uint64_t From = std::max(Begin, Offset);
        uint64_t To = std::min(End, Offset + Stride);
        if (From < To &&
            bytesHoldFlatPointer(DL, AT->getElementType(), From - Offset, To - Offset))
            return true;
    }
    return false;
}

// Whether a read of `Size` bytes (unknown when empty) at `Constant` plus `Variable` bytes into
// the blob may cover part of an offset. With `AnyOffset`, the read may be anywhere in the member
// that `Constant` is in, because a loop steps the pointer.
bool readCoversOffset(const llvm::DataLayout &DL, const Blob &B, const llvm::APInt &Constant,
                      const llvm::SmallMapVector<llvm::Value *, llvm::APInt, 4> &Variable,
                      std::optional<uint64_t> Size, bool AnyOffset) {
    if (Constant.isNegative()) return true;
    uint64_t Offset = Constant.getZExtValue();
    const std::pair<uint64_t, llvm::Type *> *Member = nullptr;
    for (const auto &Entry : B.Layout)
        if (Offset >= Entry.first &&
            Offset < Entry.first + DL.getTypeAllocSize(Entry.second).getFixedValue())
            Member = &Entry;
    if (AnyOffset) return !Member || holdsFlatPointer(Member->second);
    if (!Size) return true;
    if (Variable.empty()) {
        for (const auto &[Start, T] : B.Layout) {
            uint64_t End = Start + DL.getTypeAllocSize(T).getFixedValue();
            uint64_t From = std::max(Offset, Start);
            uint64_t To = std::min(Offset + *Size, End);
            if (From < To && bytesHoldFlatPointer(DL, T, From - Start, To - Start)) return true;
        }
        return false;
    }
    // A runtime index steps over whole elements of an array in the member, and the read has to
    // stay inside one element.
    if (!Member) return true;
    llvm::Type *T = Member->second;
    uint64_t Within = Offset - Member->first;
    while (auto *AT = llvm::dyn_cast<llvm::ArrayType>(T)) {
        uint64_t Stride = DL.getTypeAllocSize(AT->getElementType()).getFixedValue();
        if (Stride == 0) return true;
        bool Whole = llvm::all_of(Variable, [&](const auto &Entry) {
            return Entry.second.srem((int64_t)Stride) == 0;
        });
        if (Whole) {
            uint64_t In = Within % Stride;
            if (In + *Size > Stride) return true;
            return bytesHoldFlatPointer(DL, AT->getElementType(), In, In + *Size);
        }
        Within %= Stride;
        T = AT->getElementType();
    }
    return true;
}

// Whether `Phi` is defined, through GEPs, phis and selects, in terms of itself: a pointer that a
// loop steps.
bool steppedInLoop(const llvm::Value *Phi) {
    llvm::SmallVector<const llvm::Value *, 8> Work{Phi};
    llvm::SmallPtrSet<const llvm::Value *, 16> Seen;
    while (!Work.empty()) {
        const llvm::Value *V = Work.pop_back_val();
        if (!Seen.insert(V).second) {
            if (V == Phi) return true;
            continue;
        }
        if (auto *GEP = llvm::dyn_cast<llvm::GEPOperator>(V)) {
            Work.push_back(GEP->getPointerOperand());
        } else if (auto *P = llvm::dyn_cast<llvm::PHINode>(V)) {
            for (const llvm::Use &In : P->incoming_values()) Work.push_back(In.get());
        } else if (auto *S = llvm::dyn_cast<llvm::SelectInst>(V)) {
            Work.push_back(S->getTrueValue());
            Work.push_back(S->getFalseValue());
        }
    }
    return false;
}

// Whether a read of `Size` bytes at `Ptr`, `Constant` and `Variable` bytes further on, may read
// part of an offset in the blob as something other than the pointer it stands for.
bool mayReadOffset(const llvm::DataLayout &DL, const Blob &B, unsigned DecodedKind,
                   const llvm::Value *Ptr, llvm::APInt Constant,
                   llvm::SmallMapVector<llvm::Value *, llvm::APInt, 4> Variable,
                   std::optional<uint64_t> Size, bool AnyOffset,
                   llvm::SmallPtrSetImpl<const llvm::Value *> &Visiting) {
    const llvm::Value *Base = Ptr;
    for (;;) {
        // A pointer decoded from an offset: into a constant that a pointer in the blob points at.
        if (auto *I = llvm::dyn_cast<llvm::Instruction>(Base); I && I->getMetadata(DecodedKind))
            return B.PointedHoldPointers;
        auto *GEP = llvm::dyn_cast<llvm::GEPOperator>(Base);
        if (!GEP) break;
        if (!GEP->collectOffset(DL, Constant.getBitWidth(), Variable, Constant)) return true;
        Base = GEP->getPointerOperand();
    }
    if (Base == B.Global) return readCoversOffset(DL, B, Constant, Variable, Size, AnyOffset);
    if (llvm::isa<llvm::ConstantPointerNull>(Base) || llvm::isa<llvm::UndefValue>(Base)) return false;
    auto *Merge = llvm::dyn_cast<llvm::Instruction>(Base);
    if (Merge && (llvm::isa<llvm::PHINode>(Merge) || llvm::isa<llvm::SelectInst>(Merge))) {
        // Back at the phi of a loop the walk is in: its other incoming values tell where.
        if (Visiting.contains(Merge)) return false;
        bool Stepped = AnyOffset || (llvm::isa<llvm::PHINode>(Merge) && steppedInLoop(Merge));
        Visiting.insert(Merge);
        llvm::SmallVector<const llvm::Value *, 4> Incoming;
        if (auto *P = llvm::dyn_cast<llvm::PHINode>(Merge)) {
            for (const llvm::Use &In : P->incoming_values()) Incoming.push_back(In.get());
        } else {
            Incoming.push_back(Merge->getOperand(1));
            Incoming.push_back(Merge->getOperand(2));
        }
        bool May = llvm::any_of(Incoming, [&](const llvm::Value *In) {
            return mayReadOffset(DL, B, DecodedKind, In, Constant, Variable, Size, Stepped, Visiting);
        });
        Visiting.erase(Merge);
        return May;
    }
    llvm::SmallVector<const llvm::Value *, 4> Objects;
    llvm::getUnderlyingObjects(Base, Objects, nullptr, 0);
    return llvm::is_contained(Objects, B.Global);
}

// A copy or a plain load of the bytes of an offset carries the offset, not the pointer, out of
// the blob; only the loads of offsets that loadConstantPointers made may read them. The marks
// come off once the reads have been checked, and the blob becomes constant.
bool checkOffsetReads(llvm::Module &M, Blob &B, std::string &Err) {
    if (!B.Global) return true;
    const llvm::DataLayout &DL = M.getDataLayout();
    llvm::LLVMContext &Ctx = M.getContext();
    unsigned OffsetKind = Ctx.getMDKindID(offset_load_kind);
    unsigned DecodedKind = Ctx.getMDKindID(decoded_kind);
    for (llvm::Function &F : M) {
        for (llvm::Instruction &I : llvm::instructions(F)) {
            llvm::Value *Ptr = nullptr;
            std::optional<uint64_t> Size;
            if (auto *L = llvm::dyn_cast<llvm::LoadInst>(&I)) {
                if (L->getMetadata(OffsetKind)) continue;
                Ptr = L->getPointerOperand();
                Size = DL.getTypeStoreSize(L->getType()).getFixedValue();
            } else if (auto *MT = llvm::dyn_cast<llvm::MemTransferInst>(&I)) {
                Ptr = MT->getRawSource();
                if (auto *Length = llvm::dyn_cast<llvm::ConstantInt>(MT->getLength()))
                    Size = Length->getZExtValue();
            } else {
                continue;
            }
            llvm::SmallPtrSet<const llvm::Value *, 8> Visiting;
            llvm::APInt Constant(DL.getIndexTypeSizeInBits(Ptr->getType()), 0);
            if (!mayReadOffset(DL, B, DecodedKind, Ptr, Constant, {}, Size, false, Visiting)) continue;
            Err = "air64: " + F.getName().str() +
                  ": this reads a pointer out of program-scope constant data as plain bytes; on "
                  "an Apple GPU such a pointer is an offset into the constant data, so the bytes "
                  "are not the pointer: " +
                  describe(I);
            return false;
        }
    }
    for (llvm::Function &F : M) {
        for (llvm::Instruction &I : llvm::instructions(F)) {
            I.setMetadata(OffsetKind, nullptr);
            I.setMetadata(DecodedKind, nullptr);
        }
    }
    B.Global->setConstant(true);
    return true;
}

// A table of functions, like an allocator's vtable, that is still used once inlining and folding
// have turned the calls through it into direct ones: Apple's toolchain relocates no address in
// constant data, and an Apple GPU makes no call through a function pointer.
bool checkFunctionTables(llvm::Module &M, std::string &Err) {
    llvm::ModulePassManager MPM;
    MPM.addPass(llvm::GlobalDCEPass());
    runPasses(M, MPM);
    for (llvm::GlobalVariable &GV : M.globals()) {
        if (GV.getAddressSpace() != air_constant_address_space || !GV.hasInitializer()) continue;
        llvm::SmallPtrSet<const llvm::GlobalValue *, 4> Held;
        addressesIn(GV.getInitializer(), Held);
        for (const llvm::GlobalValue *Address : Held) {
            if (!llvm::isa<llvm::Function>(Address)) continue;
            Err = "air64: the program-scope constant " + GV.getName().str() +
                  " holds the address of " + describeOperand(*Address) +
                  ", and a kernel calls through it where inlining did not make the call direct; "
                  "Apple's GPU toolchain relocates no address in constant data, and an Apple GPU "
                  "makes no indirect calls";
            return false;
        }
    }
    return true;
}

bool moveConstantsToConstantSpace(llvm::Module &M, std::string &Err) {
    if (llvm::none_of(M.globals(), isProgramScopeData)) return true;
    split(M);
    Blob TheBlob;
    if (!moveConstants(M, TheBlob, Err)) return false;
    // Each round inlines where a pointer into constant data crossed a call, so that the next
    // one follows it in the caller; the rounds are bounded, since inlining cannot take
    // recursion apart. The pointers loaded out of the blob are rewritten before anything else
    // runs, and again for each level of them that inference reaches.
    for (unsigned Round = 0;; ++Round) {
        while (loadConstantPointers(M, TheBlob)) inferConstantSpace(M);
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
        // The GlobalDCE of split drops the blob once nothing uses it.
        TheBlob.Global = M.getGlobalVariable(blob_name, /*AllowInternal=*/true);
    }
    if (!checkConstantUses(M, Err)) return false;
    if (!checkOffsetReads(M, TheBlob, Err)) return false;
    return checkFunctionTables(M, Err);
}

// ---------------------------------------------------------------------------------------
// 2. Arithmetic
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

bool isAddSubOverflow(llvm::Intrinsic::ID ID) {
    return ID == llvm::Intrinsic::uadd_with_overflow || ID == llvm::Intrinsic::sadd_with_overflow ||
           ID == llvm::Intrinsic::usub_with_overflow || ID == llvm::Intrinsic::ssub_with_overflow;
}

// `llvm.{s,u}{add,sub}.with.overflow`, of every width and of vectors: Apple's compiler computes
// the flag of the signed ones wrong (0 - 10 overflows an i64, it says) and dies on the 8-bit ones
// (issue #26), so each becomes the operation and a comparison. Unsigned, a sum overflows when it
// is below an operand, and a difference when the subtrahend is above the minuend; signed, a sum
// overflows when both operands have the sign that the sum does not, and a difference when the
// operands' signs differ and the result's sign is not the minuend's.
void expandAddSubOverflow(llvm::IntrinsicInst *II, DeadList &Dead) {
    llvm::Intrinsic::ID ID = II->getIntrinsicID();
    bool Signed =
        ID == llvm::Intrinsic::sadd_with_overflow || ID == llvm::Intrinsic::ssub_with_overflow;
    bool Sub = ID == llvm::Intrinsic::usub_with_overflow || ID == llvm::Intrinsic::ssub_with_overflow;
    llvm::IRBuilder<> B(II);
    llvm::Value *X = II->getArgOperand(0);
    llvm::Value *Y = II->getArgOperand(1);
    llvm::Value *Result = Sub ? B.CreateSub(X, Y) : B.CreateAdd(X, Y);
    llvm::Value *Overflow;
    if (!Signed) {
        Overflow = Sub ? B.CreateICmpULT(X, Y) : B.CreateICmpULT(Result, X);
    } else {
        llvm::Value *Signs = Sub ? B.CreateAnd(B.CreateXor(X, Y), B.CreateXor(X, Result))
                                 : B.CreateAnd(B.CreateXor(X, Result), B.CreateXor(Y, Result));
        Overflow = B.CreateICmpSLT(Signs, llvm::Constant::getNullValue(X->getType()));
    }
    llvm::SmallVector<llvm::User *, 4> Users(II->users());
    for (llvm::User *User : Users) {
        auto *EV = llvm::dyn_cast<llvm::ExtractValueInst>(User);
        if (EV && EV->getNumIndices() == 1) {
            EV->replaceAllUsesWith(EV->getIndices()[0] == 0 ? Result : Overflow);
            Dead.push_back(EV);
            continue;
        }
        // A use of the whole pair: rebuild it.
        llvm::Value *Pair = llvm::PoisonValue::get(II->getType());
        Pair = B.CreateInsertValue(Pair, Result, {0});
        Pair = B.CreateInsertValue(Pair, Overflow, {1});
        User->replaceUsesOfWith(II, Pair);
    }
    Dead.push_back(II);
    // The operation or the flag, when only the other was used.
    for (llvm::Value *Made : {Result, Overflow})
        if (auto *I = llvm::dyn_cast<llvm::Instruction>(Made)) Dead.push_back(I);
}

bool expandArithmetic(llvm::Module &M, std::string &Err) {
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
                } else if (isAddSubOverflow(ID)) {
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
            if (isAddSubOverflow(II->getIntrinsicID())) {
                expandAddSubOverflow(II, Dead);
                continue;
            }
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
                              ID == llvm::Intrinsic::smul_with_overflow || isAddSubOverflow(ID)))
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
    return expandArithmetic(M, Err);
}
