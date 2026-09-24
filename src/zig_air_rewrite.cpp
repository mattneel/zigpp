/*
 * Copyright (c) Zig contributors
 *
 * This file is part of zig, which is MIT licensed.
 * See http://opensource.org/licenses/MIT
 */

// Rewrite a module the `air64` backend produced into the shape Apple's AIR reader accepts.
//
// This is the port of the spike's worktree tool
// (test/standalone/metal_spike/air-rewrite.cpp; every step there carries a "// AIR:"
// comment naming the measured fact behind it), with one difference of *input*: the spike
// read the kernel argument list from a hand-written spec file, while here every fact comes
// from the module itself — `!air.kernel` metadata the Zig emitter produces, whose argument
// nodes are already in Apple's buffer/scalar shape, plus the `zig.air.builtin.<name>.<dim>`
// placeholder declarations the emitter lowers builtin reads to.
//
// Steps (doc/proposals/metal.md sections 2, 4 and 6.2):
//
//   2. sanitize(): drop the target-specific attribute/metadata/global baggage, make every
//      kernel `ccc` with `nounwind` (+ `convergent` when its closure contains barriers,
//      SIMD-group operations or atomics), and give the `air.*` declarations Apple's
//      parameter shapes.
//   3. The kernel ABI: append the builtins each kernel uses as trailing value arguments,
//      threading them through every out-of-line caller (a fixed point over the call graph),
//      replace the placeholder calls by reads of those arguments, and turn a kernel's
//      scalar parameter into a constant-buffer pointer plus an entry-block load.
//   4. Metadata: `!air.version`, `!air.language_version`, `!air.compile_options`,
//      Apple's `!llvm.module.flags`, `!llvm.ident`, `!air.source_file_name`, and
//      `!air.kernel` rebuilt with the appended builtin arguments (one kernel node per kernel
//      *directly*; Apple's air-opt reports "metadata AIKernelFunction is corrupted"
//      otherwise).
//   5. Typed-pointer hints: `!arg_eltypes` for pointer parameters whose pointee the
//      downgrader cannot infer, and element-indexed GEPs on buffer parameters retyped to
//      the buffer's element type (the downgrader recovers pointer pointee types from GEPs).
//
// The optimization pipeline and the bitcode downgrade live in src/zig_air.cpp.

#include "zig_air.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

// AIR: the module data layout of every AIR module the spike inspected, byte for byte
// (vadd.ll:2, reduce.ll:3, parse.ll:2, ref2.ll:2, ref3.ll:2). Pointers are 64-bit in every
// address space, unlike the nvptx/amdgcn stand-ins'.
const char *AirDataLayout =
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-"
    "v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-"
    "v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

// The builtins a kernel can take, in the fixed order the appended arguments use: first by
// this order, then by dimension x < y < z. The names are the `air.<name>` metadata spelling
// (doc section 2.5); the short names are the `air.arg_name` values Apple's frontend uses.
struct BuiltinDef {
    const char *AirName;
    const char *ShortName;
};
const BuiltinDef BuiltinTable[] = {
    {"thread_position_in_grid", "gid"},
    {"threadgroup_position_in_grid", "tgid"},
    {"thread_position_in_threadgroup", "tid"},
    {"thread_index_in_simdgroup", "lane"},
    {"simdgroup_index_in_threadgroup", "sgid"},
    {"threads_per_threadgroup", "ntid"},
    {"threadgroups_per_grid", "ngid"},
    {"threads_per_simdgroup", "simd_size"},
    {"simdgroup_per_threadgroup", "nsg"},
};
enum { NumBuiltins = sizeof(BuiltinTable) / sizeof(BuiltinTable[0]) };

// The placeholder declarations the emitter lowers builtin reads to:
// `declare i32 @"zig.air.builtin.<air_name>.<x|y|z>"()`.
const char *PlaceholderPrefix = "zig.air.builtin.";

int builtinIndex(llvm::StringRef AirName) {
    for (int I = 0; I < NumBuiltins; ++I)
        if (AirName == BuiltinTable[I].AirName) return I;
    return -1;
}

// One parameter per builtin name: `i32` while only the x component is used, `<3 x i32>` as
// soon as y or z is (Apple's frontend emits `uint`/`uint3` for the 1D/3D builtins; the same
// principle, doc section 2.4).
enum class Shape : unsigned { None = 0, X = 1, Vec = 2 };

Shape mergeShapes(Shape A, Shape B) { return (unsigned)A >= (unsigned)B ? A : B; }
Shape shapeOfDim(unsigned Dim) { return Dim == 0 ? Shape::X : Shape::Vec; }

std::string toString(const llvm::Type &Ty) {
    std::string Text;
    llvm::raw_string_ostream OS(Text);
    Ty.print(OS);
    return OS.str();
}

std::string toString(const llvm::Value &V) { return toString(*V.getType()); }

// ---------------------------------------------------------------------------------------
// `!air.kernel`: one node per kernel, `!{ptr @fn, <stages>, !{<arg nodes>}}`, where a
// buffer/scalar argument node is a flat key/value list (doc section 2.5).
// ---------------------------------------------------------------------------------------

struct KernelArg {
    unsigned ParamIndex = 0;
    llvm::MDNode *Node = nullptr;
    bool IsBuffer = false;
    llvm::StringRef Kind;  // "air.buffer" or "air.<builtin>"
    // Buffer fields (the emitter's / Apple's final shape).
    unsigned AddressSpace = 0;
    unsigned TypeSize = 0;
    unsigned TypeAlign = 0;
    llvm::StringRef TypeName;
    llvm::StringRef Access;  // "air.read" | "air.write" | "air.read_write"
    llvm::StringRef ArgName;
};

struct KernelInfo {
    llvm::Function *Fn = nullptr;
    llvm::MDNode *Node = nullptr;
    llvm::MDNode *ArgList = nullptr;
    llvm::SmallVector<KernelArg, 8> Args;
};

bool getI32Operand(llvm::Metadata *MD, uint64_t &Out) {
    auto *C = llvm::dyn_cast<llvm::ConstantAsMetadata>(MD);
    if (!C) return false;
    auto *CI = llvm::dyn_cast<llvm::ConstantInt>(C->getValue());
    if (!CI) return false;
    Out = CI->getZExtValue();
    return true;
}

llvm::StringRef getStringOperand(llvm::Metadata *MD) {
    if (auto *S = llvm::dyn_cast<llvm::MDString>(MD)) return S->getString();
    return llvm::StringRef();
}

bool isMetadataValueKey(llvm::StringRef Key) {
    return Key == "air.location_index" || Key == "air.address_space" ||
           Key == "air.arg_type_size" || Key == "air.arg_type_align_size" ||
           Key == "air.arg_type_name" || Key == "air.arg_name" || Key == "air.buffer_size";
}

// Parse one argument node. Every node starts with the parameter index and the kind; the
// rest is a flat list of `"key", value` pairs, where the access mode is a key of its own
// ("air.read"/"air.write"/"air.read_write") and the spare `i32` the AIR writer leaves after
// `air.location_index` is ignored.
bool parseArgNode(llvm::MDNode *Node, KernelArg &Out, std::string &Err) {
    if (Node->getNumOperands() < 2) {
        Err = "an argument node has fewer than two operands";
        return false;
    }
    uint64_t Index = 0;
    if (!getI32Operand(Node->getOperand(0), Index)) {
        Err = "an argument node does not start with an i32 parameter index";
        return false;
    }
    Out.Node = Node;
    Out.ParamIndex = (unsigned)Index;
    Out.Kind = getStringOperand(Node->getOperand(1));
    if (Out.Kind.empty()) {
        Err = "an argument node's second operand is not a string";
        return false;
    }
    Out.IsBuffer = Out.Kind == "air.buffer";
    if (!Out.IsBuffer) {
        if (Out.Kind.starts_with("air.") && builtinIndex(Out.Kind.drop_front(4)) >= 0)
            return true;  // A builtin node, already in Apple's shape.
        Err = "an argument node's kind is neither \"air.buffer\" nor a known builtin name, "
              "but '" +
              Out.Kind.str() + "'";
        return false;
    }

    for (unsigned I = 2, E = Node->getNumOperands(); I != E; ++I) {
        llvm::Metadata *MD = Node->getOperand(I);
        if (llvm::isa<llvm::ConstantAsMetadata>(MD))
            continue;  // The bare `i32 1` after air.location_index (writer spare).
        llvm::StringRef Key = getStringOperand(MD);
        if (Key.empty()) continue;
        llvm::Metadata *Value = I + 1 < E ? Node->getOperand(I + 1).get() : nullptr;

        if (Key == "air.read" || Key == "air.write" || Key == "air.read_write") {
            Out.Access = Key;
            continue;
        }
        if (!isMetadataValueKey(Key) || !Value) continue;

        uint64_t Number = 0;
        if (getI32Operand(Value, Number)) {
            if (Key == "air.address_space")
                Out.AddressSpace = (unsigned)Number;
            else if (Key == "air.arg_type_size")
                Out.TypeSize = (unsigned)Number;
            else if (Key == "air.arg_type_align_size")
                Out.TypeAlign = (unsigned)Number;
            // air.location_index is kept as it is: the index equals the parameter index,
            // because the host binds bytes at that index.
            ++I;
            continue;
        }
        llvm::StringRef Text = getStringOperand(Value);
        if (!Text.empty()) {
            if (Key == "air.arg_type_name")
                Out.TypeName = Text;
            else if (Key == "air.arg_name")
                Out.ArgName = Text;
            ++I;
        }
    }
    return true;
}

// Read `!air.kernel`. Apple's reader rejects a kernel node nested under another node, so the
// list must hold the kernel nodes directly.
bool readKernelMetadata(llvm::Module &M, std::vector<KernelInfo> &Kernels, std::string &Err) {
    llvm::NamedMDNode *AirKernel = M.getNamedMetadata("air.kernel");
    if (!AirKernel) {
        Err = "the module has no !air.kernel metadata; the Metal backend must mark its "
              "kernels";
        return false;
    }
    if (AirKernel->getNumOperands() == 0) {
        Err = "!air.kernel is empty: a Metal module must name at least one kernel";
        return false;
    }
    for (unsigned I = 0, E = AirKernel->getNumOperands(); I != E; ++I) {
        llvm::MDNode *Node = llvm::dyn_cast<llvm::MDNode>(AirKernel->getOperand(I));
        if (!Node || Node->getNumOperands() != 3) {
            Err = "!air.kernel: operand " + std::to_string(I) +
                  " is not a { function, stages, arguments } node";
            return false;
        }
        auto *VM = llvm::dyn_cast<llvm::ValueAsMetadata>(Node->getOperand(0));
        auto *Fn = VM ? llvm::dyn_cast<llvm::Function>(VM->getValue()) : nullptr;
        if (!Fn) {
            Err = "!air.kernel: operand " + std::to_string(I) +
                  " does not start with a function";
            return false;
        }
        llvm::MDNode *ArgList = llvm::dyn_cast<llvm::MDNode>(Node->getOperand(2));
        if (!ArgList) {
            Err = "!air.kernel: kernel '" + Fn->getName().str() +
                  "' has no argument list node";
            return false;
        }
        KernelInfo Info;
        Info.Fn = Fn;
        Info.Node = Node;
        Info.ArgList = ArgList;
        for (unsigned A = 0, AE = ArgList->getNumOperands(); A != AE; ++A) {
            llvm::MDNode *ArgNode = llvm::dyn_cast<llvm::MDNode>(ArgList->getOperand(A));
            if (!ArgNode) {
                Err = "!air.kernel: kernel '" + Fn->getName().str() + "' argument " +
                      std::to_string(A) + " is not a node";
                return false;
            }
            KernelArg Arg;
            if (!parseArgNode(ArgNode, Arg, Err)) {
                Err = "!air.kernel: kernel '" + Fn->getName().str() + "': " + Err;
                return false;
            }
            Info.Args.push_back(Arg);
        }
        for (const KernelInfo &Other : Kernels)
            if (Other.Fn == Fn) {
                Err = "!air.kernel lists '" + Fn->getName().str() + "' more than once";
                return false;
            }
        // One argument node per parameter: the node's first operand is the parameter index
        // the host binds by, so a repeated index would bind two arguments to one slot.
        for (unsigned A = 0; A < Info.Args.size(); ++A)
            for (unsigned B = A + 1; B < Info.Args.size(); ++B)
                if (Info.Args[A].ParamIndex == Info.Args[B].ParamIndex) {
                    Err = "!air.kernel: kernel '" + Fn->getName().str() +
                          "' has two argument nodes for parameter " +
                          std::to_string(Info.Args[A].ParamIndex);
                    return false;
                }
        Kernels.push_back(std::move(Info));
    }
    return true;
}

// The MSL scalar type name of an `air.arg_type_name` (the spike's airElementType).
llvm::Type *airElementType(llvm::LLVMContext &Ctx, llvm::StringRef Name) {
    return llvm::StringSwitch<llvm::Type *>(Name)
        .Cases({"bool", "char", "uchar"}, llvm::Type::getInt8Ty(Ctx))
        .Cases({"short", "ushort"}, llvm::Type::getInt16Ty(Ctx))
        .Cases({"int", "uint"}, llvm::Type::getInt32Ty(Ctx))
        .Cases({"long", "ulong"}, llvm::Type::getInt64Ty(Ctx))
        .Case("half", llvm::Type::getHalfTy(Ctx))
        .Case("float", llvm::Type::getFloatTy(Ctx))
        .Case("double", llvm::Type::getDoubleTy(Ctx))
        .Default(nullptr);
}

// ---------------------------------------------------------------------------------------
// Step 2: sanitize
// ---------------------------------------------------------------------------------------

// The named metadata AIR modules carry (notes/air_conventions.md section 7 plus the ones
// this pass writes); anything else is host-only baggage (`!nvvm.annotations`, `!llvm.used`,
// `!llvm.dbg.cu`, `!llvm.embedded.module`, ...).
bool keepNamedMetadata(llvm::StringRef Name) {
    return Name == "llvm.module.flags" || Name == "llvm.ident" || Name.starts_with("air.");
}

// Does this function itself contain a barrier, a SIMD-group operation or an atomic?
bool hasConvergentOp(const llvm::Function &F) {
    for (const llvm::BasicBlock &BB : F)
        for (const llvm::Instruction &I : BB) {
            if (llvm::isa<llvm::AtomicRMWInst>(I) || llvm::isa<llvm::AtomicCmpXchgInst>(I) ||
                llvm::isa<llvm::FenceInst>(I))
                return true;
            const auto *CB = llvm::dyn_cast<llvm::CallBase>(&I);
            if (!CB) continue;
            auto *Callee =
                llvm::dyn_cast<llvm::Function>(CB->getCalledOperand()->stripPointerCasts());
            if (!Callee) continue;
            llvm::StringRef Name = Callee->getName();
            if (Name.contains("barrier") || Name.contains("simd") || Name.contains("quad") ||
                Name.starts_with("air.atomic.") || Name.starts_with("air.fence"))
                return true;
        }
    return false;
}

// The functions whose transitive closure contains a convergent operation: a kernel that
// calls a helper doing a barrier is convergent as well.
void collectConvergentFunctions(llvm::Module &M, llvm::SmallPtrSetImpl<llvm::Function *> &Out) {
    llvm::DenseMap<llvm::Function *, llvm::SmallVector<llvm::Function *, 4>> Callers;
    llvm::SmallVector<llvm::Function *, 16> Worklist;
    for (llvm::Function &F : M) {
        if (F.isDeclaration() || !hasConvergentOp(F)) continue;
        if (Out.insert(&F).second) Worklist.push_back(&F);
    }
    for (llvm::Function &F : M) {
        if (F.isDeclaration()) continue;
        for (llvm::BasicBlock &BB : F)
            for (llvm::Instruction &I : BB) {
                const auto *CB = llvm::dyn_cast<llvm::CallBase>(&I);
                if (!CB) continue;
                auto *Callee = llvm::dyn_cast<llvm::Function>(
                    CB->getCalledOperand()->stripPointerCasts());
                if (Callee && !Callee->isDeclaration()) Callers[Callee].push_back(&F);
            }
    }
    while (!Worklist.empty()) {
        llvm::Function *F = Worklist.pop_back_val();
        for (llvm::Function *Caller : Callers[F])
            if (Out.insert(Caller).second) Worklist.push_back(Caller);
    }
}

// AIR: kernel parameters carry only their access mode (`readonly` for `air.read`,
// `writeonly` for `air.write`, nothing for `air.read_write`) and `noundef`: Apple's own
// output has `ptr addrspace(1) noundef readonly "air-buffer-no-alias" %0` (vadd.ll:10, doc
// section 2.4). Everything else (the optimizer's captures/nofree/align hints, and any
// attribute of the stand-in backend) is dropped.
void setKernelParamAttrs(llvm::Function &F, const KernelInfo &K) {
    llvm::LLVMContext &Ctx = F.getContext();
    for (unsigned I = 0, E = (unsigned)F.arg_size(); I != E; ++I)
        F.setAttributes(F.getAttributes().removeParamAttributes(Ctx, I));
    for (const KernelArg &A : K.Args) {
        if (A.ParamIndex >= F.arg_size()) continue;
        F.addParamAttr(A.ParamIndex, llvm::Attribute::NoUndef);
        if (A.Access == "air.read")
            F.addParamAttr(A.ParamIndex, llvm::Attribute::ReadOnly);
        else if (A.Access == "air.write")
            F.addParamAttr(A.ParamIndex, llvm::Attribute::WriteOnly);
    }
}

void sanitize(llvm::Module &M, const std::vector<KernelInfo> &Kernels) {
    llvm::SmallPtrSet<llvm::Function *, 16> KernelFns;
    llvm::SmallPtrSet<llvm::Function *, 16> Convergent;
    for (const KernelInfo &K : Kernels) KernelFns.insert(K.Fn);
    collectConvergentFunctions(M, Convergent);

    // AIR: nvvm/amdgcn annotations have no meaning in AIR, and neither has anything else the
    // host compiler keeps for itself.
    llvm::SmallVector<llvm::NamedMDNode *, 8> DropMetadata;
    for (llvm::NamedMDNode &N : M.named_metadata())
        if (!keepNamedMetadata(N.getName())) DropMetadata.push_back(&N);
    for (llvm::NamedMDNode *N : DropMetadata) M.eraseNamedMetadata(N);

    // Host-only data: the appending lists first, so that the `%Target.*` globals they keep
    // alive are dead when they are dropped (doc section 6.2, "the @Target.* globals the
    // backend emits for runtime target queries must be dropped for kernel modules").
    llvm::SmallVector<llvm::GlobalVariable *, 8> DropGlobals;
    for (llvm::GlobalVariable &GV : M.globals()) {
        llvm::StringRef Name = GV.getName();
        if (Name == "llvm.used" || Name == "llvm.compiler.used" ||
            Name == "llvm.global_ctors" || Name == "llvm.global_dtors" ||
            Name == "llvm.embedded.module" || Name == "llvm.init_array" ||
            Name == "llvm.fini_array")
            DropGlobals.push_back(&GV);
    }
    for (llvm::GlobalVariable *GV : DropGlobals) GV->eraseFromParent();
    // Then the `Target.*` globals that nothing uses any more. One can hold another in its
    // initializer, so this repeats until a round erases none. A global that code still reads
    // stays: erasing it would leave its users pointing at freed memory. The optimization
    // pipeline's GlobalDCE removes it later if that code turns out to be dead.
    for (bool Erased = true; Erased;) {
        Erased = false;
        DropGlobals.clear();
        for (llvm::GlobalVariable &GV : M.globals())
            if (GV.getName().starts_with("Target.")) DropGlobals.push_back(&GV);
        for (llvm::GlobalVariable *GV : DropGlobals) {
            GV->removeDeadConstantUsers();
            if (!GV->use_empty()) continue;
            GV->eraseFromParent();
            Erased = true;
        }
    }

    for (llvm::Function &F : M) {
        // AIR: Apple's functions carry no target-cpu/target-features: both are attributes of
        // the stand-in backend's functions (vadd.ll attributes #0 have neither).
        F.removeFnAttr("target-cpu");
        F.removeFnAttr("target-features");
        F.removeFnAttr("tune-cpu");

        if (F.isDeclaration()) {
            if (!F.getName().starts_with("air.")) continue;
            // AIR: the air.* declarations in Apple's modules have no parameter attributes
            // (reduce.ll:57-62, ref2.ll:105-111), are local_unnamed_addr and nounwind;
            // barriers, SIMD-group operations and quadgroup permutes are convergent, the
            // atomics are not (reduce.ll attributes #1 vs #2, notes section 7).
            F.setAttributes(llvm::AttributeList());
            F.addFnAttr(llvm::Attribute::NoUnwind);
            llvm::StringRef Name = F.getName();
            if (Name.contains("barrier") || Name.contains("simd") || Name.contains("quad"))
                F.addFnAttr(llvm::Attribute::Convergent);
            F.setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);
            continue;
        }

        if (KernelFns.count(&F)) {
            // AIR: kernels use the C calling convention; a GPU calling convention only means
            // something to the stand-in backend (doc section 2.4).
            F.setCallingConv(llvm::CallingConv::C);
            // AIR: kernels are nounwind, and convergent when they contain barriers or
            // SIMD-group operations (doc sections 2.4 and 2.6); the atomics count as well
            // because a kernel that uses them with barriers must not be reordered around
            // them.
            F.addFnAttr(llvm::Attribute::NoUnwind);
            if (Convergent.count(&F)) F.addFnAttr(llvm::Attribute::Convergent);
            // AIR: kernels are local_unnamed_addr (vadd.ll:10, reduce.ll:14).
            F.setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);
            continue;
        }

        switch (F.getLinkage()) {
            case llvm::GlobalValue::PrivateLinkage:
            case llvm::GlobalValue::LinkOnceAnyLinkage:
            case llvm::GlobalValue::LinkOnceODRLinkage:
            case llvm::GlobalValue::WeakAnyLinkage:
            case llvm::GlobalValue::WeakODRLinkage:
                // AIR: functions defined in the module are internal. Zig emits its outlined
                // helpers as private; the bodies, the fastcc convention and the parameter
                // attributes are left alone (Apple's reference modules contain no
                // non-kernel definitions to copy).
                F.setLinkage(llvm::GlobalValue::InternalLinkage);
                break;
            default:
                break;
        }
    }

    for (llvm::GlobalVariable &GV : M.globals()) {
        if (GV.getAddressSpace() == 0 || GV.getLinkage() != llvm::GlobalValue::PrivateLinkage)
            continue;
        // AIR: threadgroup (addrspace(3)) memory is an internal global in Apple's AIR
        // (reduce.ll:8).
        GV.setLinkage(llvm::GlobalValue::InternalLinkage);
    }
}

// ---------------------------------------------------------------------------------------
// Step 3: the kernel ABI (appended builtins, scalar parameters as constant buffers)
// ---------------------------------------------------------------------------------------

struct FuncState {
    llvm::Function *Old = nullptr;  // valid until the function is rebuilt
    llvm::Function *New = nullptr;
    Shape B[NumBuiltins] = {};
    bool Kernel = false;
    unsigned KernelIndex = 0;
    unsigned OldParamCount = 0;  // captured before the rebuild (Old dies with it)
    // The new parameter list, filled in while the function is planned.
    llvm::SmallVector<llvm::Type *, 8> NewParamTys;
    bool Appended[NumBuiltins] = {};
    unsigned AppendIndex[NumBuiltins] = {};
    // Kernel parameter index -> element type of the scalar that becomes a constant buffer.
    llvm::DenseMap<unsigned, llvm::Type *> ConstLoads;
};

bool parsePlaceholderName(llvm::StringRef Name, unsigned &BuiltinOut, unsigned &DimOut) {
    if (!Name.starts_with(PlaceholderPrefix)) return false;
    llvm::StringRef Rest = Name.drop_front(llvm::StringRef(PlaceholderPrefix).size());
    size_t Dot = Rest.rfind('.');
    if (Dot == llvm::StringRef::npos) return false;
    llvm::StringRef AirName = Rest.take_front(Dot);
    llvm::StringRef DimName = Rest.drop_front(Dot + 1);
    int B = builtinIndex(AirName);
    if (B < 0) return false;
    unsigned Dim;
    if (DimName == "x")
        Dim = 0;
    else if (DimName == "y")
        Dim = 1;
    else if (DimName == "z")
        Dim = 2;
    else
        return false;
    BuiltinOut = (unsigned)B;
    DimOut = Dim;
    return true;
}

class KernelAbi {
  public:
    KernelAbi(llvm::Module &M, std::vector<KernelInfo> &Kernels, std::string &Err)
        : M(M), Kernels(Kernels), Err(Err) {}

    bool run();

  private:
    // Plan for `F`, created on first use. The old function object is gone once the function
    // has been rebuilt, so nothing may dereference `State->Old` afterwards.
    FuncState *planFor(llvm::Function *F) {
        auto It = PlanIndex.find(F);
        if (It != PlanIndex.end()) return It->second;
        Storage.push_back(std::make_unique<FuncState>());
        FuncState *S = Storage.back().get();
        S->Old = F;
        S->OldParamCount = F->isDeclaration() ? 0 : (unsigned)F->arg_size();
        for (unsigned I = 0; I < Kernels.size(); ++I)
            if (Kernels[I].Fn == F) {
                S->Kernel = true;
                S->KernelIndex = I;
            }
        PlanIndex[F] = S;
        return S;
    }

    llvm::Function *directCallee(const llvm::CallBase *CB) {
        return llvm::dyn_cast<llvm::Function>(CB->getCalledOperand()->stripPointerCasts());
    }

    bool collectPlaceholders();
    bool computeSets();
    bool planFunctions();
    void rebuild(FuncState &S);
    bool rewriteCallSites();
    bool rebuildKernelMetadata();

    llvm::Module &M;
    std::vector<KernelInfo> &Kernels;
    std::string &Err;
    llvm::DenseMap<llvm::Function *, std::pair<unsigned, unsigned>> Placeholders;
    std::vector<std::unique_ptr<FuncState>> Storage;
    llvm::DenseMap<llvm::Function *, FuncState *> PlanIndex;
};

// Find the emitter's placeholder declarations and check their shape.
bool KernelAbi::collectPlaceholders() {
    for (llvm::Function &F : M) {
        llvm::StringRef Name = F.getName();
        if (!Name.starts_with(PlaceholderPrefix)) continue;
        if (!F.isDeclaration()) {
            Err = "builtin placeholder '" + Name.str() + "' is a definition";
            return false;
        }
        if (F.arg_size() != 0 || !F.getReturnType()->isIntegerTy(32)) {
            Err = "builtin placeholder '" + Name.str() +
                  "' must be `declare i32 @\"...\"()`";
            return false;
        }
        unsigned Builtin = 0, Dim = 0;
        if (!parsePlaceholderName(Name, Builtin, Dim)) {
            Err = "builtin placeholder '" + Name.str() +
                  "' does not name one of the AIR builtins in x/y/z";
            return false;
        }
        Placeholders[&F] = {Builtin, Dim};
    }
    return true;
}

// The fixed point over the call graph: a function needs a builtin if it reads it, if one of
// its callees needs it (it has to pass the value on), and a callee's shape has to match its
// callers' (the value is passed along unchanged, so the parameter type is the caller's
// type). Cycles are fine: every step only moves a cell up the lattice
// none < i32 < <3 x i32>, so this terminates.
bool KernelAbi::computeSets() {
    // Direct placeholder reads.
    for (llvm::Function &F : M) {
        if (F.isDeclaration()) continue;
        for (llvm::Instruction &I : llvm::instructions(F)) {
            const auto *CB = llvm::dyn_cast<llvm::CallBase>(&I);
            if (!CB) continue;
            llvm::Function *Callee = directCallee(CB);
            if (!Callee) continue;
            auto It = Placeholders.find(Callee);
            if (It == Placeholders.end()) continue;
            FuncState *S = planFor(&F);
            S->B[It->second.first] =
                mergeShapes(S->B[It->second.first], shapeOfDim(It->second.second));
        }
    }

    // Call graph edges between definitions.
    std::vector<std::pair<llvm::Function *, llvm::Function *>> Edges;
    for (llvm::Function &F : M) {
        if (F.isDeclaration()) continue;
        for (llvm::Instruction &I : llvm::instructions(F)) {
            const auto *CB = llvm::dyn_cast<llvm::CallBase>(&I);
            if (!CB) continue;
            llvm::Function *Callee = directCallee(CB);
            if (Callee && !Callee->isDeclaration()) Edges.push_back({&F, Callee});
        }
    }

    bool Changed = true;
    unsigned Iterations = 0;
    while (Changed) {
        Changed = false;
        if (++Iterations > 4 * (unsigned)Storage.size() * NumBuiltins + 8) {
            Err = "the builtin argument fixed point does not converge";
            return false;
        }
        for (const auto &Edge : Edges) {
            FuncState *Caller = planFor(Edge.first);
            FuncState *Callee = planFor(Edge.second);
            for (unsigned B = 0; B < NumBuiltins; ++B) {
                Shape Joined = mergeShapes(Caller->B[B], Callee->B[B]);
                if (Joined != Caller->B[B]) {
                    Caller->B[B] = Joined;
                    Changed = true;
                }
                if (Callee->B[B] != Shape::None) {
                    Shape Matched = mergeShapes(Callee->B[B], Caller->B[B]);
                    if (Matched != Callee->B[B]) {
                        Callee->B[B] = Matched;
                        Changed = true;
                    }
                }
            }
        }
    }
    return true;
}

// Decide what every function looks like after the ABI pass, and build its new parameter
// list.
bool KernelAbi::planFunctions() {
    for (KernelInfo &K : Kernels) {
        if (K.Fn->isDeclaration()) {
            Err = "kernel '" + K.Fn->getName().str() + "' is only declared";
            return false;
        }
    }

    for (const std::unique_ptr<FuncState> &Ptr : Storage) {
        FuncState &S = *Ptr;
        llvm::Function *F = S.Old;
        if (F->isDeclaration()) continue;

        // Every builtin the function needs becomes one trailing argument, in the fixed
        // order (builtin table first, then x < y < z).
        bool Rebuild = false;
        for (unsigned B = 0; B < NumBuiltins; ++B)
            if (S.B[B] != Shape::None) {
                S.Appended[B] = true;
                Rebuild = true;
            }
        if (!Rebuild && !S.Kernel) continue;

        S.NewParamTys.clear();
        for (unsigned I = 0, E = (unsigned)F->arg_size(); I != E; ++I)
            S.NewParamTys.push_back(F->getArg(I)->getType());

        if (S.Kernel) {
            // A kernel's scalar parameter (a value in Zig, bound by the host at that
            // location_index) is really a constant-buffer pointer: change it to
            // `ptr addrspace(N)` and read the value out of it in the entry block (doc
            // section 2.4: AIR has no kernel scalar argument).
            for (const KernelArg &Arg : Kernels[S.KernelIndex].Args) {
                if (!Arg.IsBuffer) continue;
                if (Arg.ParamIndex >= F->arg_size()) {
                    Err = "!air.kernel: kernel '" + F->getName().str() + "' argument " +
                          std::to_string(Arg.ParamIndex) + " is out of range";
                    return false;
                }
                llvm::Type *Ty = F->getArg(Arg.ParamIndex)->getType();
                if (Ty->isPointerTy()) continue;
                if (!Ty->isIntegerTy() && !Ty->isFloatingPointTy()) {
                    Err = "kernel '" + F->getName().str() + "': parameter " +
                          std::to_string(Arg.ParamIndex) +
                          " is not a pointer but is typed '" + toString(*Ty) +
                          "', which the host cannot bind to a buffer index";
                    return false;
                }
                S.ConstLoads[Arg.ParamIndex] = Ty;
                S.NewParamTys[Arg.ParamIndex] =
                    llvm::PointerType::get(F->getContext(), Arg.AddressSpace);
                Rebuild = true;
            }
        }
        if (!Rebuild) continue;

        for (unsigned B = 0; B < NumBuiltins; ++B) {
            if (!S.Appended[B]) continue;
            llvm::Type *Ty;
            if (S.B[B] == Shape::Vec)
                Ty = llvm::FixedVectorType::get(llvm::Type::getInt32Ty(F->getContext()), 3);
            else
                Ty = llvm::Type::getInt32Ty(F->getContext());
            S.AppendIndex[B] = (unsigned)S.NewParamTys.size();
            S.NewParamTys.push_back(Ty);
        }
    }
    return true;
}

// Replace the function by one with the planned signature, keeping its body. LLVM 23 has no
// way to change a function's type in place (the argument list is an array built from the
// FunctionType), so this is the usual create/splice/RAUW dance.
void KernelAbi::rebuild(FuncState &S) {
    llvm::Function *Old = S.Old;
    llvm::FunctionType *OldFT = Old->getFunctionType();
    llvm::FunctionType *NewFT =
        llvm::FunctionType::get(OldFT->getReturnType(), S.NewParamTys, OldFT->isVarArg());
    llvm::Function *New =
        llvm::Function::Create(NewFT, Old->getLinkage(), Old->getAddressSpace(), "", &M);
    New->copyAttributesFrom(Old);
    New->copyMetadata(Old, 0);
    New->setCallingConv(Old->getCallingConv());
    New->setUnnamedAddr(Old->getUnnamedAddr());
    New->setVisibility(Old->getVisibility());
    New->setDLLStorageClass(Old->getDLLStorageClass());
    New->setDSOLocal(Old->isDSOLocal());
    New->setAlignment(Old->getAlign());
    if (Old->hasComdat()) New->setComdat(Old->getComdat());
    if (!Old->getSection().empty()) New->setSection(Old->getSection());

    New->splice(New->begin(), Old);
    for (unsigned I = 0, E = S.OldParamCount; I != E; ++I) {
        llvm::Argument *OldArg = Old->getArg(I);
        llvm::Argument *NewArg = New->getArg(I);
        NewArg->setName(OldArg->getName());
        OldArg->replaceAllUsesWith(NewArg);
    }

    Old->replaceAllUsesWith(New);
    New->takeName(Old);
    Old->eraseFromParent();
    S.New = New;
    // Call sites, the kernel metadata and the checks below find this plan by the function
    // object that is alive; the old (freed) pointer must not stay a key, or a later
    // allocation could land on it and pick up the wrong plan.
    PlanIndex.erase(Old);
    PlanIndex[New] = &S;
    if (S.Kernel) Kernels[S.KernelIndex].Fn = New;

    // The loads that read a scalar parameter out of constant memory (doc section 2.4:
    // constant buffers; the host binds the bytes at air.location_index).
    for (const auto &Entry : S.ConstLoads) {
        unsigned Index = Entry.first;
        llvm::Type *Ty = Entry.second;
        llvm::Align Align;
        bool HasAlign = false;
        for (const KernelArg &Candidate : Kernels[S.KernelIndex].Args)
            if (Candidate.IsBuffer && Candidate.ParamIndex == Index &&
                Candidate.TypeAlign != 0) {
                Align = llvm::Align(Candidate.TypeAlign);
                HasAlign = true;
            }
        if (!HasAlign) Align = M.getDataLayout().getABITypeAlign(Ty);

        llvm::BasicBlock &EntryBlock = New->getEntryBlock();
        llvm::BasicBlock::iterator IP = EntryBlock.getFirstInsertionPt();
        while (IP != EntryBlock.end() && llvm::isa<llvm::AllocaInst>(&*IP)) ++IP;
        llvm::Argument *Ptr = New->getArg(Index);
        llvm::LoadInst *Load =
            new llvm::LoadInst(Ty, Ptr, Ptr->getName() + ".val", false, Align, &*IP);
        Ptr->replaceAllUsesWith(Load);
        Load->setOperand(0, Ptr);
    }
}

// Call sites: replace placeholder calls by reads of the caller's appended argument, and give
// every call to a rebuilt function the callee's new signature plus the values it passes on.
bool KernelAbi::rewriteCallSites() {
    for (llvm::Function &F : M) {
        if (F.isDeclaration()) continue;
        for (llvm::BasicBlock &BB : F) {
            for (llvm::BasicBlock::iterator It = BB.begin(), End = BB.end(); It != End;) {
                llvm::Instruction *I = &*It++;
                auto *CB = llvm::dyn_cast<llvm::CallBase>(I);
                if (!CB) continue;
                llvm::Function *Callee = directCallee(CB);
                if (!Callee) continue;
                llvm::Function *Parent = BB.getParent();

                auto Placeholder = Placeholders.find(Callee);
                if (Placeholder != Placeholders.end()) {
                    auto *CI = llvm::dyn_cast<llvm::CallInst>(CB);
                    if (!CI) {
                        Err = "builtin placeholder '" + Callee->getName().str() +
                              "' is called by something other than a call instruction";
                        return false;
                    }
                    auto CallerState = PlanIndex.find(Parent);
                    if (CallerState == PlanIndex.end() || !CallerState->second->New ||
                        !CallerState->second->Appended[Placeholder->second.first]) {
                        Err = "function '" + Parent->getName().str() + "' calls '" +
                              Callee->getName().str() +
                              "' but has no builtin argument for it";
                        return false;
                    }
                    FuncState *CallerPlan = CallerState->second;
                    unsigned B = Placeholder->second.first;
                    unsigned Dim = Placeholder->second.second;
                    llvm::Value *Value =
                        CallerPlan->New->getArg(CallerPlan->AppendIndex[B]);
                    if (CallerPlan->B[B] == Shape::Vec) {
                        llvm::Value *Index = llvm::ConstantInt::get(
                            llvm::Type::getInt32Ty(M.getContext()), Dim);
                        Value = llvm::ExtractElementInst::Create(Value, Index, "", CI);
                    }
                    CI->replaceAllUsesWith(Value);
                    CI->eraseFromParent();
                    continue;
                }

                auto CalleeState = PlanIndex.find(Callee);
                if (CalleeState == PlanIndex.end() || !CalleeState->second->New) continue;
                FuncState &CalleePlan = *CalleeState->second;

                auto CallerState = PlanIndex.find(Parent);
                if (CallerState == PlanIndex.end() || !CallerState->second->New) {
                    Err = "function '" + Parent->getName().str() + "' calls '" +
                          Callee->getName().str() +
                          "', whose signature gains builtin arguments, but was not planned";
                    return false;
                }
                FuncState &CallerPlan = *CallerState->second;

                llvm::SmallVector<llvm::Value *, 8> Args(CB->arg_begin(), CB->arg_end());
                unsigned Prefix = CalleePlan.OldParamCount;
                if (Args.size() != Prefix) {
                    Err = "the call to '" + Callee->getName().str() + "' in '" +
                          Parent->getName().str() + "' has " + std::to_string(Args.size()) +
                          " arguments, the callee has " + std::to_string(Prefix);
                    return false;
                }
                // A kernel's scalar parameters change type as well; nothing calls a kernel
                // from inside the module, so a mismatch here is a real error.
                for (unsigned I = 0; I < Prefix; ++I)
                    if (Args[I]->getType() != CalleePlan.New->getArg(I)->getType()) {
                        Err = "the call to '" + Callee->getName().str() + "' in '" +
                              Parent->getName().str() + "' passes '" + toString(*Args[I]) +
                              "' for parameter " + std::to_string(I) + ", which is now '" +
                              toString(*CalleePlan.New->getArg(I)) + "'";
                        return false;
                    }
                for (unsigned B = 0; B < NumBuiltins; ++B) {
                    if (!CalleePlan.Appended[B]) continue;
                    if (!CallerPlan.Appended[B]) {
                        Err = "function '" + Parent->getName().str() + "' calls '" +
                              Callee->getName().str() + "' without an argument for builtin air." +
                              BuiltinTable[B].AirName;
                        return false;
                    }
                    Args.push_back(CallerPlan.New->getArg(CallerPlan.AppendIndex[B]));
                }

                auto *CI = llvm::dyn_cast<llvm::CallInst>(CB);
                if (!CI) {
                    Err = "the call to '" + Callee->getName().str() + "' in '" +
                          Parent->getName().str() +
                          "' is not a call instruction, which the builtin arguments cannot be "
                          "appended to";
                    return false;
                }
                llvm::CallInst *New = llvm::CallInst::Create(
                    CalleePlan.New->getFunctionType(), CalleePlan.New, Args, "", CI);
                New->setCallingConv(CI->getCallingConv());
                New->setTailCallKind(CI->getTailCallKind());
                New->setAttributes(CI->getAttributes());
                New->copyMetadata(*CI, 0);
                CI->replaceAllUsesWith(New);
                CI->eraseFromParent();
            }
        }
    }

    for (auto &Entry : Placeholders) {
        llvm::Function *Placeholder = Entry.first;
        if (!Placeholder->use_empty()) {
            std::string Users;
            for (llvm::User *U : Placeholder->users()) {
                if (!Users.empty()) Users += ", ";
                if (auto *F = llvm::dyn_cast<llvm::Function>(U))
                    Users += F->getName().str();
                else
                    Users += "an instruction";
            }
            Err = "builtin placeholder '" + Placeholder->getName().str() +
                  "' still has users after the ABI rewrite (" + Users + ")";
            return false;
        }
        Placeholder->eraseFromParent();
    }
    return true;
}

// `!air.kernel` with the appended builtin arguments: one node per kernel directly, each new
// argument node `{i32 <parameter index>, !"air.<builtin>", "air.arg_type_name",
// !"uint"|!"uint3", "air.arg_name", !"<short name>"}` (doc section 2.5).
bool KernelAbi::rebuildKernelMetadata() {
    llvm::LLVMContext &Ctx = M.getContext();
    llvm::NamedMDNode *AirKernel = M.getNamedMetadata("air.kernel");
    std::vector<llvm::MDNode *> KernelNodes;
    for (KernelInfo &K : Kernels) {
        auto It = PlanIndex.find(K.Fn);
        FuncState *Plan = It == PlanIndex.end() ? nullptr : It->second;
        if (!Plan || !Plan->New) {
            KernelNodes.push_back(K.Node);
            continue;
        }
        llvm::SmallVector<llvm::Metadata *, 16> ArgNodes(K.ArgList->op_begin(),
                                                         K.ArgList->op_end());
        for (unsigned B = 0; B < NumBuiltins; ++B) {
            if (!Plan->Appended[B]) continue;
            llvm::Metadata *TypeName =
                llvm::MDString::get(Ctx, Plan->B[B] == Shape::Vec ? "uint3" : "uint");
            ArgNodes.push_back(llvm::MDNode::get(
                Ctx,
                {llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                     llvm::Type::getInt32Ty(Ctx), Plan->AppendIndex[B])),
                 llvm::MDString::get(Ctx, std::string("air.") + BuiltinTable[B].AirName),
                 llvm::MDString::get(Ctx, "air.arg_type_name"), TypeName,
                 llvm::MDString::get(Ctx, "air.arg_name"),
                 llvm::MDString::get(Ctx, BuiltinTable[B].ShortName)}));
        }
        llvm::MDNode *ArgList = llvm::MDNode::get(Ctx, ArgNodes);
        llvm::MDNode *KernelNode =
            llvm::MDNode::get(Ctx, {K.Node->getOperand(0), K.Node->getOperand(1), ArgList});
        KernelNodes.push_back(KernelNode);
    }
    AirKernel->clearOperands();
    for (llvm::MDNode *Node : KernelNodes) AirKernel->addOperand(Node);
    return true;
}

bool KernelAbi::run() {
    if (!collectPlaceholders()) return false;
    if (!computeSets()) return false;
    if (!planFunctions()) return false;

    // Rebuilding moves the body to a new function object; the plans are keyed by both the
    // old and the new function, so the order does not matter.
    std::vector<FuncState *> RebuildOrder;
    for (const std::unique_ptr<FuncState> &Ptr : Storage)
        if (Ptr->New == nullptr && !Ptr->NewParamTys.empty()) RebuildOrder.push_back(Ptr.get());
    for (FuncState *S : RebuildOrder) rebuild(*S);

    if (!rewriteCallSites()) return false;
    return rebuildKernelMetadata();
}

// A buffer/scalar argument node must describe the parameter it indexes: a pointer in the
// address space the metadata names (doc sections 2.4 and 2.5).
bool checkKernelArgs(const std::vector<KernelInfo> &Kernels, std::string &Err) {
    for (const KernelInfo &K : Kernels) {
        for (const KernelArg &Arg : K.Args) {
            if (!Arg.IsBuffer) continue;
            if (Arg.ParamIndex >= K.Fn->arg_size()) {
                Err = "!air.kernel: kernel '" + K.Fn->getName().str() + "' argument " +
                      std::to_string(Arg.ParamIndex) + " is out of range";
                return false;
            }
            llvm::Type *Ty = K.Fn->getArg(Arg.ParamIndex)->getType();
            if (!Ty->isPointerTy()) {
                Err = "!air.kernel: kernel '" + K.Fn->getName().str() + "' argument " +
                      std::to_string(Arg.ParamIndex) + " is a buffer but is typed '" +
                      toString(*Ty) + "'";
                return false;
            }
            if (Ty->getPointerAddressSpace() != Arg.AddressSpace) {
                Err = "!air.kernel: kernel '" + K.Fn->getName().str() + "' argument " +
                      std::to_string(Arg.ParamIndex) + " is in address space " +
                      std::to_string(Ty->getPointerAddressSpace()) +
                      ", the metadata says " + std::to_string(Arg.AddressSpace);
                return false;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------
// Step 4: metadata
// ---------------------------------------------------------------------------------------

llvm::Metadata *i32MD(llvm::LLVMContext &Ctx, uint64_t Value) {
    return llvm::ConstantAsMetadata::get(
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(Ctx), Value));
}

llvm::Metadata *strMD(llvm::LLVMContext &Ctx, llvm::StringRef Text) {
    return llvm::MDString::get(Ctx, Text);
}

// Does any FP instruction in the module carry a fast-math flag?
bool hasFastMath(const llvm::Module &M) {
    for (const llvm::Function &F : M)
        for (const llvm::BasicBlock &BB : F)
            for (const llvm::Instruction &I : BB)
                if (const auto *FP = llvm::dyn_cast<llvm::FPMathOperator>(&I))
                    if (FP->getFastMathFlags().any()) return true;
    return false;
}

void setSingleNodeNamedMetadata(llvm::Module &M, llvm::StringRef Name,
                                llvm::ArrayRef<llvm::Metadata *> Operands) {
    if (llvm::NamedMDNode *Existing = M.getNamedMetadata(Name)) M.eraseNamedMetadata(Existing);
    if (Operands.empty()) return;
    M.getOrInsertNamedMetadata(Name)->addOperand(llvm::MDNode::get(M.getContext(), Operands));
}

void attachMetadata(llvm::Module &M, const ZigLLVMAirOptions &Options) {
    llvm::LLVMContext &Ctx = M.getContext();

    // AIR: !air.version = !{!{i32 2, i32 8, i32 0}} (vadd.ll:20).
    setSingleNodeNamedMetadata(M, "air.version",
                               {i32MD(Ctx, Options.air_major), i32MD(Ctx, Options.air_minor),
                                i32MD(Ctx, Options.air_patch)});

    // AIR: !air.language_version = !{!{!"Metal", i32 4, i32 0, i32 0}} (vadd.ll:21).
    setSingleNodeNamedMetadata(M, "air.language_version",
                               {strMD(Ctx, "Metal"), i32MD(Ctx, Options.metal_major),
                                i32MD(Ctx, Options.metal_minor),
                                i32MD(Ctx, Options.metal_patch)});

    // AIR: !air.compile_options (vadd.ll:16-18). denorms_disable and
    // framebuffer_fetch_enable are driver-level defaults Apple emits for every module, but
    // fast_math_enable is a permission for the later AIR pipelines to assume fast math, and
    // Apple only ever emits it with fast-math flags on the FP instructions themselves
    // (vadd.ll:10 `fadd fast` + attributes #0's no-nans/unsafe-fp-math, parse.ll the same).
    // Claiming it for precise FP would silently change the meaning of those operations, so
    // it is emitted only when the module's own FP instructions carry fast-math flags.
    if (llvm::NamedMDNode *Existing = M.getNamedMetadata("air.compile_options"))
        M.eraseNamedMetadata(Existing);
    llvm::NamedMDNode *CompileOptions = M.getOrInsertNamedMetadata("air.compile_options");
    CompileOptions->addOperand(
        llvm::MDNode::get(Ctx, {strMD(Ctx, "air.compile.denorms_disable")}));
    if (hasFastMath(M))
        CompileOptions->addOperand(
            llvm::MDNode::get(Ctx, {strMD(Ctx, "air.compile.fast_math_enable")}));
    CompileOptions->addOperand(
        llvm::MDNode::get(Ctx, {strMD(Ctx, "air.compile.framebuffer_fetch_enable")}));

    // AIR: !llvm.module.flags is Apple's own set (vadd.ll:0-8): SDK Version (Warning,
    // [3 x i32]), wchar_size (Error, i32 4), frame-pointer (Max, i32 2) and the six Max
    // air.max_* limits (31 device/constant/threadgroup buffers, 128 textures, 8 read-write
    // textures, 16 samplers).
    if (llvm::NamedMDNode *Flags = M.getNamedMetadata("llvm.module.flags"))
        M.eraseNamedMetadata(Flags);
    uint32_t SDK[3] = {Options.sdk_major, Options.sdk_minor, Options.sdk_patch};
    M.addModuleFlag(llvm::Module::Warning, "SDK Version",
                    llvm::ConstantDataArray::get(Ctx, SDK));
    M.addModuleFlag(llvm::Module::Error, "wchar_size", uint32_t(4));
    M.addModuleFlag(llvm::Module::Max, "frame-pointer", uint32_t(2));
    M.addModuleFlag(llvm::Module::Max, "air.max_device_buffers", uint32_t(31));
    M.addModuleFlag(llvm::Module::Max, "air.max_constant_buffers", uint32_t(31));
    M.addModuleFlag(llvm::Module::Max, "air.max_threadgroup_buffers", uint32_t(31));
    M.addModuleFlag(llvm::Module::Max, "air.max_textures", uint32_t(128));
    M.addModuleFlag(llvm::Module::Max, "air.max_read_write_textures", uint32_t(8));
    M.addModuleFlag(llvm::Module::Max, "air.max_samplers", uint32_t(16));

    // AIR: !air.source_file_name = !{!{!"/path/to/file.metal"}} (vadd.ll:22); Apple also
    // sets the module's source_filename to the file's base name (vadd.ll:1).
    if (Options.source_name) {
        setSingleNodeNamedMetadata(M, "air.source_file_name", {strMD(Ctx, Options.source_name)});
        M.setSourceFileName(llvm::sys::path::filename(Options.source_name));
    } else if (llvm::NamedMDNode *Existing = M.getNamedMetadata("air.source_file_name")) {
        M.eraseNamedMetadata(Existing);
    }

    // AIR: !llvm.ident = !{!{!"Apple metal version ..."}} (vadd.ll:19). The compiler writes
    // its own identity here; the option decides what the final module says (and
    // src/zig_air.cpp re-applies it after the optimization pipeline).
    if (llvm::NamedMDNode *Ident = M.getNamedMetadata("llvm.ident"))
        M.eraseNamedMetadata(Ident);
    if (Options.ident)
        M.getOrInsertNamedMetadata("llvm.ident")
            ->addOperand(llvm::MDNode::get(Ctx, {strMD(Ctx, Options.ident)}));
}

// AIR: triple and data layout of every reference module (vadd.ll:2-3
// "air64_v28-apple-macosx26.0.0"; notes/air_conventions.md sections 1 and 2). The arch
// component is "air64_v<major><minor>" from AIR 2.6 on and "air64" below that; the OS
// component is the deployment target the emitter chose, so the module keeps it.
bool setTripleAndLayout(llvm::Module &M, const ZigLLVMAirOptions &Options, std::string &Err) {
    llvm::Triple Input(M.getTargetTriple());
    std::string Arch = "air64";
    if (Options.air_major > 2 || (Options.air_major == 2 && Options.air_minor >= 6))
        Arch += "_v" + std::to_string(Options.air_major) + std::to_string(Options.air_minor);

    llvm::StringRef Vendor = Input.getVendorName();
    std::string VendorText =
        (Vendor.empty() || Vendor == "unknown") ? std::string("apple") : Vendor.str();
    llvm::StringRef OSName = Input.getOSAndEnvironmentName();
    std::string OS;
    if (OSName.empty() || OSName.starts_with("unknown"))
        OS = "macosx" + std::to_string(Options.sdk_major) + "." +
             std::to_string(Options.sdk_minor) + "." + std::to_string(Options.sdk_patch);
    else
        OS = OSName.str();
    std::string TripleText = Arch + "-" + VendorText + "-" + OS;

    M.setTargetTriple(llvm::Triple(TripleText));
    M.setDataLayout(AirDataLayout);
    if (M.getTargetTriple().str() != TripleText) {
        Err = "LLVM rewrote the AIR triple '" + TripleText + "' to '" +
              M.getTargetTriple().str() + "'";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------------------
// Step 5: typed-pointer hints
// ---------------------------------------------------------------------------------------

// The element type an air.* intrinsic's name encodes: "air.atomic.global.add.u.i32" -> i32,
// "air.simd_sum.f32" -> float, "air.wg.barrier" -> nothing (notes/air_conventions.md section
// 4 lists the families; every pointer-taking name we emit ends in ".iN"/".fN").
llvm::Type *airIntrinsicPointee(llvm::LLVMContext &Ctx, llvm::StringRef Name) {
    size_t Dot = Name.rfind('.');
    if (Dot == llvm::StringRef::npos || Dot + 2 >= Name.size()) return nullptr;
    llvm::StringRef Tail = Name.drop_front(Dot + 1);
    char Kind = Tail[0];
    if (Kind != 'i' && Kind != 'f') return nullptr;
    unsigned Bits = 0;
    for (char C : Tail.drop_front()) {
        if (C < '0' || C > '9') return nullptr;
        Bits = Bits * 10 + unsigned(C - '0');
    }
    if (Kind == 'i') return Bits ? llvm::IntegerType::get(Ctx, Bits) : nullptr;
    if (Bits == 16) return llvm::Type::getHalfTy(Ctx);
    if (Bits == 32) return llvm::Type::getFloatTy(Ctx);
    if (Bits == 64) return llvm::Type::getDoubleTy(Ctx);
    return nullptr;
}

llvm::Metadata *zeroOf(llvm::Type *Ty) {
    return llvm::ConstantAsMetadata::get(llvm::Constant::getNullValue(Ty));
}

// !arg_eltypes (notes/air_conventions.md section 5): a list of `{i32 <parameter index>,
// <zero of the pointee element type>}` pairs that tells the typed-pointer downgrader which
// pointee to give a pointer parameter. It infers most pointees itself from
// loads/stores/GEPs, but a pointer that only ever goes into an intrinsic has no element type
// anywhere in the IR (reduce's counter is handed straight to air.atomic.global.add.u.i32,
// and the air.* declarations' own pointer parameters are never dereferenced at all). Without
// the hints the LLVM-14 bitcode types those as pointers to the empty struct, which
// contradicts the air.arg_type_name/air.arg_type_size metadata describing the same argument
// as a uint.
void attachArgElTypes(llvm::Module &M, const std::vector<KernelInfo> &Kernels) {
    llvm::LLVMContext &Ctx = M.getContext();

    // Kernels: every buffer parameter gets the element type its metadata names.
    for (const KernelInfo &K : Kernels) {
        std::vector<llvm::Metadata *> Pairs;
        for (const KernelArg &Arg : K.Args) {
            if (!Arg.IsBuffer) continue;
            llvm::Type *ElementTy = airElementType(Ctx, Arg.TypeName);
            if (!ElementTy) continue;
            Pairs.push_back(i32MD(Ctx, Arg.ParamIndex));
            Pairs.push_back(zeroOf(ElementTy));
        }
        if (!Pairs.empty()) K.Fn->setMetadata("arg_eltypes", llvm::MDNode::get(Ctx, Pairs));
    }

    // air.* declarations: the pointee is the element type in the intrinsic's name; names that
    // encode no element type are skipped.
    for (llvm::Function &F : M) {
        if (!F.isDeclaration() || !F.getName().starts_with("air.")) continue;
        llvm::Type *ElementTy = airIntrinsicPointee(Ctx, F.getName());
        if (!ElementTy) continue;
        std::vector<llvm::Metadata *> Pairs;
        for (unsigned I = 0, E = (unsigned)F.arg_size(); I != E; ++I)
            if (F.getArg(I)->getType()->isPointerTy()) {
                Pairs.push_back(i32MD(Ctx, I));
                Pairs.push_back(zeroOf(ElementTy));
            }
        if (!Pairs.empty()) F.setMetadata("arg_eltypes", llvm::MDNode::get(Ctx, Pairs));
    }
}

// Walk buffer elements with the buffer's real element type.
//
// Zig emits element-indexed buffer accesses as
// `getelementptr inbounds [4 x i8], ptr addrspace(1) %2, i64 %5`, i.e. an array of
// @sizeOf(element) bytes indexed by the element index; Apple's IR uses the element type
// itself: `getelementptr inbounds float, ptr addrspace(1) %2, i64 %5` (vadd.ll:11, where the
// metadata says air.arg_type_name "float"). The two agree byte for byte as long as the
// element sizes match, which is exactly the condition for the rewrite, so the addresses
// cannot change; it makes the buffer walks agree with the air.arg_type_name/
// air.arg_type_size metadata and gives the typed-pointer downgrade something to work with.
// GEPs that address bytes (a byte-sized element type, as parsef's per-thread
// `text + gid * 32` does) are left alone, and so are GEPs that reach the buffer through the
// staging alloca of Zig's calling convention: this walks the parameter's own uses, which is
// what SROA leaves behind (at opt_level 0 the staging still exists and those GEPs keep the
// [N x i8] form).
void retypeBufferGEPs(llvm::Module &M, const std::vector<KernelInfo> &Kernels) {
    const llvm::DataLayout &DL = M.getDataLayout();
    llvm::LLVMContext &Ctx = M.getContext();
    for (const KernelInfo &K : Kernels) {
        llvm::Function *F = K.Fn;
        if (F->isDeclaration()) continue;
        for (const KernelArg &Arg : K.Args) {
            if (!Arg.IsBuffer) continue;
            if (Arg.ParamIndex >= F->arg_size()) continue;
            llvm::Value *Buffer = F->getArg(Arg.ParamIndex);
            if (!Buffer->getType()->isPointerTy()) continue;
            llvm::Type *ElementTy = airElementType(Ctx, Arg.TypeName);
            if (!ElementTy) continue;
            // Byte-element types stay as they are: an i8/[1 x i8] GEP already is what the
            // element access looks like.
            if (DL.getTypeAllocSize(ElementTy) == 1) continue;
            for (llvm::Use &U : Buffer->uses()) {
                auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(U.getUser());
                if (!GEP || GEP->getPointerOperand() != Buffer || GEP->getNumIndices() != 1)
                    continue;
                if (DL.getTypeAllocSize(GEP->getSourceElementType()) !=
                    DL.getTypeAllocSize(ElementTy))
                    continue;
                // Both fields have to move together: GetElementPtrInst keeps a source and a
                // result element type, and the verifier checks them against each other.
                GEP->setSourceElementType(ElementTy);
                GEP->setResultElementType(ElementTy);
            }
        }
    }
}

}  // namespace

// Re-normalize the kernel signatures and the buffer GEPs after the optimization pipeline: O3
// infers parameter attributes of its own, and SROA turns Zig's staging allocas into GEPs on
// the parameters (the spike's steps 5b and 5c).
void zigAirRetypeBufferGEPs(llvm::Module &M, std::string &Err) {
    std::vector<KernelInfo> Kernels;
    if (!readKernelMetadata(M, Kernels, Err)) return;
    retypeBufferGEPs(M, Kernels);
    for (const KernelInfo &K : Kernels) {
        if (K.Fn->isDeclaration()) continue;
        K.Fn->addFnAttr(llvm::Attribute::NoUnwind);
        K.Fn->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Local);
        setKernelParamAttrs(*K.Fn, K);
    }
    Err.clear();
}

bool zigAirRewrite(llvm::Module &M, const ZigLLVMAirOptions &Options, std::string &Err) {
    std::vector<KernelInfo> Kernels;
    if (!readKernelMetadata(M, Kernels, Err)) return false;

    sanitize(M, Kernels);

    KernelAbi Abi(M, Kernels, Err);
    if (!Abi.run()) return false;

    // The kernel functions may have been replaced by ones with the appended builtin
    // arguments, and the metadata has been rebuilt with them; re-read it for the checks and
    // the hints below.
    Kernels.clear();
    if (!readKernelMetadata(M, Kernels, Err)) return false;
    if (!checkKernelArgs(Kernels, Err)) return false;

    if (!setTripleAndLayout(M, Options, Err)) return false;
    attachMetadata(M, Options);
    attachArgElTypes(M, Kernels);
    retypeBufferGEPs(M, Kernels);

    std::string VerifyMessage;
    llvm::raw_string_ostream OS(VerifyMessage);
    if (llvm::verifyModule(M, &OS)) {
        Err = "the rewritten module does not verify:\n" + OS.str();
        return false;
    }
    return true;
}
