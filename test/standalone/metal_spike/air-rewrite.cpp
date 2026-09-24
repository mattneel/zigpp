// air-rewrite: rewrite a Zig++ bitcode module (built for the nvptx64 stand-in target)
// into Apple's AIR conventions, i.e. the shapes `xcrun metal -c` emits.
//
// Usage:
//   air-rewrite <input.bc> -o <output.bc> --spec <spec> [--air 2.8] [--metal 4.0]
//               [--deploy 26.0.0] [--sdk 26.5.0] [--opt O3|O0] [--ident <string>]
//               [--source-name <path>]
//
// The spec is the file documented at the top of spike/kernels.spec: `kernel <name>`
// followed by one `buffer`/`builtin` line per kernel argument, in IR parameter order.
//
// The reference modules this follows are Apple's own toolchain output (Xcode 26.6 Metal,
// disassembled with LLVM 23 llvm-dis) in /tmp/metal-spike: vadd.ll, reduce.ll, parse.ll,
// ref2.ll, ref3.ll. Every convention below carries a "// AIR:" comment naming the fact
// and its reference; conventions that no reference shows literally are marked
// "(not in the references)", because there is no Apple compiler on this machine to check
// them against.

#include "llvm/ADT/StringSwitch.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace llvm;

namespace {

// AIR: the module data layout of every AIR module we inspected, byte for byte
// (vadd.ll:2, reduce.ll:3, parse.ll:2, ref2.ll:2, ref3.ll:2).
const char *AirDataLayout =
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-"
    "v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-"
    "v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32";

std::string toString(const Type &Ty) {
  std::string Text;
  raw_string_ostream OS(Text);
  Ty.print(OS);
  return OS.str();
}

enum class ArgKind { Buffer, Builtin };

// One line of the spec file.
struct ArgSpec {
  ArgKind Kind = ArgKind::Buffer;
  unsigned Index = 0;     // air.location_index for buffers; the AIR index for builtins
  std::string Access;     // read | write | read_write (buffers)
  std::string Intrinsic;  // AIR builtin name (builtins)
  std::string TypeName;   // air.arg_type_name
  unsigned TypeSize = 0;  // air.arg_type_size (buffers)
  unsigned TypeAlign = 0; // air.arg_type_align_size (buffers)
  std::string Name;       // air.arg_name
};

struct KernelSpec {
  std::string Name;
  std::vector<ArgSpec> Args;
};

struct Options {
  std::string Input, Output, SpecPath;
  unsigned Air[3] = {2, 8, 0};   // --air
  unsigned Metal[3] = {4, 0, 0}; // --metal
  std::string Deploy = "26.0.0"; // --deploy
  unsigned Sdk[3] = {26, 5, 0};  // --sdk
  bool Optimize = true;          // --opt O3 (default) vs --opt O0
  std::string Ident;             // --ident
  std::string SourceName;        // --source-name
};

[[noreturn]] void usage(const std::string &Message) {
  if (!Message.empty())
    errs() << "air-rewrite: " << Message << "\n";
  errs() << "usage: air-rewrite <input.bc> -o <output.bc> --spec <spec> [options]\n"
            "\n"
            "  --air <major.minor[.patch]>    AIR version to declare (default 2.8)\n"
            "  --metal <major.minor[.patch]>  Metal language version (default 4.0)\n"
            "  --deploy <major.minor.patch>   macOS deployment target (default 26.0.0)\n"
            "  --sdk <major.minor[.patch]>    SDK version module flag (default 26.5.0)\n"
            "  --opt O3|O0                    standard O3 pipeline (default) or GlobalDCE\n"
            "  --ident <string>               !llvm.ident value (default: no such metadata)\n"
            "  --source-name <path>           !air.source_file_name value (default: none)\n";
  exit(1);
}

// "2.8" -> {2, 8, 0}; also accepts "M" and "M.m.p".
bool parseVersion(const std::string &Text, unsigned Out[3]) {
  Out[0] = Out[1] = Out[2] = 0;
  unsigned Slot = 0;
  std::istringstream In(Text);
  std::string Component;
  while (std::getline(In, Component, '.')) {
    if (Slot == 3 || Component.empty() ||
        Component.find_first_not_of("0123456789") != std::string::npos)
      return false;
    Out[Slot++] = std::stoul(Component);
  }
  return Slot > 0;
}

bool parseCommandLine(int Argc, char **Argv, Options &O, std::string &Err) {
  for (int I = 1; I < Argc; ++I) {
    std::string Arg = Argv[I];
    auto Next = [&](std::string &Out) {
      if (I + 1 >= Argc)
        usage("missing value for " + Arg);
      Out = Argv[++I];
    };
    if (Arg == "-h" || Arg == "--help") {
      usage("");
    } else if (Arg == "-o") {
      Next(O.Output);
    } else if (Arg == "--spec") {
      Next(O.SpecPath);
    } else if (Arg == "--air") {
      std::string Value;
      Next(Value);
      if (!parseVersion(Value, O.Air))
        return Err = "bad --air version '" + Value + "'", false;
    } else if (Arg == "--metal") {
      std::string Value;
      Next(Value);
      if (!parseVersion(Value, O.Metal))
        return Err = "bad --metal version '" + Value + "'", false;
    } else if (Arg == "--deploy") {
      Next(O.Deploy);
    } else if (Arg == "--sdk") {
      std::string Value;
      Next(Value);
      if (!parseVersion(Value, O.Sdk))
        return Err = "bad --sdk version '" + Value + "'", false;
    } else if (Arg == "--opt") {
      std::string Value;
      Next(Value);
      if (Value != "O3" && Value != "O0")
        return Err = "--opt takes O3 or O0, not '" + Value + "'", false;
      O.Optimize = Value == "O3";
    } else if (Arg == "--ident") {
      Next(O.Ident);
    } else if (Arg == "--source-name") {
      Next(O.SourceName);
    } else if (!Arg.empty() && Arg[0] == '-') {
      return Err = "unknown option '" + Arg + "'", false;
    } else if (O.Input.empty()) {
      O.Input = Arg;
    } else {
      return Err = "more than one input file ('" + O.Input + "', '" + Arg + "')", false;
    }
  }
  if (O.Input.empty())
    return Err = "no input bitcode file", false;
  if (O.Output.empty())
    return Err = "no output bitcode file (-o)", false;
  if (O.SpecPath.empty())
    return Err = "no kernel argument spec (--spec)", false;
  return true;
}

// Step 1: read the argument spec. The format is documented at the top of
// spike/kernels.spec:
//   kernel  <name>
//   buffer  <index> <read|write|read_write> <air.arg_type_name> <size> <align> <name>
//   builtin <index> <air builtin name>      <air.arg_type_name>            <name>
bool parseSpec(const std::string &Path, std::vector<KernelSpec> &Kernels,
               std::string &Err) {
  std::ifstream File(Path);
  if (!File)
    return Err = "cannot open spec file '" + Path + "'", false;

  std::string Line;
  unsigned LineNo = 0;
  while (std::getline(File, Line)) {
    ++LineNo;
    if (size_t Comment = Line.find('#'); Comment != std::string::npos)
      Line.resize(Comment);
    std::istringstream Fields(Line);
    std::string Kind;
    if (!(Fields >> Kind))
      continue;

    if (Kind == "kernel") {
      KernelSpec K;
      if (!(Fields >> K.Name))
        return Err = "spec:" + std::to_string(LineNo) + ": kernel without a name", false;
      Kernels.push_back(std::move(K));
      continue;
    }
    if (Kernels.empty())
      return Err = "spec:" + std::to_string(LineNo) + ": '" + Kind +
                   "' before any 'kernel' line",
             false;

    ArgSpec A;
    if (Kind == "buffer") {
      A.Kind = ArgKind::Buffer;
      if (!(Fields >> A.Index >> A.Access >> A.TypeName >> A.TypeSize >> A.TypeAlign >>
            A.Name))
        return Err = "spec:" + std::to_string(LineNo) +
                     ": buffer needs <index> <read|write|read_write> <type> <size> "
                     "<align> <name>",
               false;
      if (A.Access != "read" && A.Access != "write" && A.Access != "read_write")
        return Err = "spec:" + std::to_string(LineNo) + ": bad buffer access mode '" +
                     A.Access + "'",
               false;
      if (A.TypeSize == 0 || A.TypeAlign == 0)
        return Err = "spec:" + std::to_string(LineNo) +
                     ": buffer size and alignment must be nonzero",
               false;
    } else if (Kind == "builtin") {
      A.Kind = ArgKind::Builtin;
      if (!(Fields >> A.Index >> A.Intrinsic >> A.TypeName >> A.Name))
        return Err = "spec:" + std::to_string(LineNo) +
                     ": builtin needs <index> <air name> <type> <name>",
               false;
      if (!StringRef(A.Intrinsic).starts_with("air."))
        return Err = "spec:" + std::to_string(LineNo) + ": builtin name '" + A.Intrinsic +
                     "' is not an air.* name",
               false;
    } else {
      return Err = "spec:" + std::to_string(LineNo) + ": unknown kind '" + Kind + "'",
             false;
    }

    // The spec header promises IR parameter order; holding it to that keeps one meaning
    // of the index (AIR's air.location_index) without guessing.
    unsigned Position = Kernels.back().Args.size();
    if (A.Index != Position)
      return Err = "spec:" + std::to_string(LineNo) + ": argument " +
                   std::to_string(Position) + " of kernel '" + Kernels.back().Name +
                   "' has index " + std::to_string(A.Index) +
                   "; arguments must be listed in IR parameter order with 0-based "
                   "consecutive indices",
             false;
    Kernels.back().Args.push_back(std::move(A));
  }

  if (Kernels.empty())
    return Err = "spec '" + Path + "' defines no kernels", false;
  return true;
}

// Step 1b: the spec must describe the module's kernels, and every kernel the module
// defines must be in the spec, so that no ptx_kernel function slips through unrewritten.
bool checkKernelSignatures(Module &M, const std::vector<KernelSpec> &Kernels,
                           std::string &Err) {
  for (const KernelSpec &K : Kernels) {
    Function *F = M.getFunction(K.Name);
    if (!F)
      return Err = "kernel '" + K.Name + "': no such function in the module", false;
    if (F->isDeclaration())
      return Err = "kernel '" + K.Name + "': the module only declares it", false;
    if (F->arg_size() != K.Args.size())
      return Err = "kernel '" + K.Name + "': module has " +
                   std::to_string(F->arg_size()) + " arguments, spec has " +
                   std::to_string(K.Args.size()),
             false;

    for (unsigned I = 0, E = F->arg_size(); I != E; ++I) {
      Type *Ty = F->getArg(I)->getType();
      const ArgSpec &A = K.Args[I];
      if (A.Kind == ArgKind::Buffer && !Ty->isPointerTy())
        return Err = "kernel '" + K.Name + "': argument " + std::to_string(I) + " is '" +
                     A.Name + "' in the spec, so it must be a pointer, but the module "
                     "declares '" + toString(*Ty) + "'",
               false;
      if (A.Kind == ArgKind::Builtin && !Ty->isIntegerTy())
        return Err = "kernel '" + K.Name + "': argument " + std::to_string(I) + " is '" +
                     A.Name + "' in the spec, so it must be an integer builtin, but the "
                     "module declares '" + toString(*Ty) + "'",
               false;
    }
  }

  for (Function &F : M) {
    if (F.getCallingConv() != CallingConv::PTX_Kernel)
      continue;
    bool Listed = false;
    for (const KernelSpec &K : Kernels)
      Listed |= K.Name == F.getName();
    if (!Listed)
      return Err = "function '" + F.getName().str() +
                   "' uses the ptx_kernel calling convention but is not in the spec",
             false;
  }
  return true;
}

// Step 2: AIR triple and data layout.
std::string airTriple(const Options &O) {
  // AIR: the arch component is "air64_v<major><minor>" for AIR >= 2.6 and "air64" below
  // that; the OS component is the macOS deployment target (vadd.ll:3
  // "air64_v28-apple-macosx26.0.0"; notes/air_conventions.md section 1, 6).
  std::string Arch = "air64";
  if (O.Air[0] > 2 || (O.Air[0] == 2 && O.Air[1] >= 6))
    Arch += "_v" + std::to_string(O.Air[0]) + std::to_string(O.Air[1]);
  return Arch + "-apple-macosx" + O.Deploy;
}

// AIR: kernel parameters carry only their access mode: `readonly` for spec access `read`,
// `writeonly` for `write`, nothing for `read_write`. Apple's own output additionally has
// noundef, captures(none) and "air-buffer-no-alias" (vadd.ll:10), and O3 infers
// nofree/writeonly/captures on its own; the spike's contract is the bare parameter list,
// so everything except the access mode is dropped here (not in the references: dropping
// noundef/captures/air-buffer-no-alias).
void setKernelParamAttrs(Function &F, const KernelSpec &K) {
  for (unsigned I = 0, E = F.arg_size(); I != E; ++I)
    F.setAttributes(F.getAttributes().removeParamAttributes(F.getContext(), I));
  for (unsigned I = 0, E = F.arg_size(); I != E; ++I) {
    const ArgSpec &A = K.Args[I];
    if (A.Kind != ArgKind::Buffer)
      continue;
    if (A.Access == "read")
      F.addParamAttr(I, Attribute::ReadOnly);
    else if (A.Access == "write")
      F.addParamAttr(I, Attribute::WriteOnly);
  }
}

// Step 3: make the module look like AIR rather than like nvptx.
void sanitize(Module &M, const std::vector<KernelSpec> &Kernels) {
  // AIR: nvptx puts its annotations in !nvvm.annotations; AIR has no such metadata, and
  // it would be meaningless to the AIR compiler (notes/air_conventions.md section 7 lists
  // the metadata AIR modules carry and it is not among them).
  if (NamedMDNode *NVVM = M.getNamedMetadata("nvvm.annotations"))
    M.eraseNamedMetadata(NVVM);

  for (Function &F : M) {
    // AIR: Apple's functions carry no target-cpu/target-features. Both are string
    // attributes of the nvptx backend (vadd.ll attributes #0 have neither).
    F.removeFnAttr("target-cpu");
    F.removeFnAttr("target-features");
  }

  for (const KernelSpec &K : Kernels) {
    Function *F = M.getFunction(K.Name);
    // AIR: kernels use the C calling convention; the Zig input is ptx_kernel, which only
    // means something to the nvptx backend.
    F->setCallingConv(CallingConv::C);
    // AIR: kernels are convergent and nounwind (reduce.ll attributes #0, vadd.ll #0).
    F->addFnAttr(Attribute::Convergent);
    F->addFnAttr(Attribute::NoUnwind);
    // AIR: kernels are local_unnamed_addr (vadd.ll:10, reduce.ll:14).
    F->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);
    setKernelParamAttrs(*F, K);
  }

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    bool IsKernel = false;
    for (const KernelSpec &K : Kernels)
      IsKernel |= K.Name == F.getName();
    if (IsKernel)
      continue;
    switch (F.getLinkage()) {
    case GlobalValue::PrivateLinkage:
    case GlobalValue::LinkOnceAnyLinkage:
    case GlobalValue::LinkOnceODRLinkage:
    case GlobalValue::WeakAnyLinkage:
    case GlobalValue::WeakODRLinkage:
      // AIR: functions defined in the module are internal. Zig emits its outlined
      // helpers as private + fastcc; the bodies, the fastcc convention and the parameter
      // attributes are left alone (not in the references: Apple's reference modules
      // contain no non-kernel definitions to copy).
      F.setLinkage(GlobalValue::InternalLinkage);
      break;
    default:
      break;
    }
  }

  for (GlobalVariable &GV : M.globals()) {
    if (GV.getAddressSpace() == 0 || GV.getLinkage() != GlobalValue::PrivateLinkage)
      continue;
    // AIR: threadgroup (addrspace(3)) memory is an internal global in Apple's AIR
    // (reduce.ll:8); Zig's nvptx output makes it private.
    GV.setLinkage(GlobalValue::InternalLinkage);
  }

  for (Function &F : M) {
    if (!F.isDeclaration() || !F.getName().starts_with("air."))
      continue;
    // AIR: the air.* declarations in Apple's modules have no parameter attributes
    // (reduce.ll:57-62, ref2.ll:105-111; the references keep captures(none) on pointer
    // arguments, which the spike's contract drops as well), and are local_unnamed_addr.
    F.setAttributes(AttributeList());
    F.addFnAttr(Attribute::NoUnwind);
    // AIR: barriers and SIMD-group operations are convergent; the atomics are not
    // (reduce.ll attributes #1 vs #2).
    if (F.getName().contains("barrier") || F.getName().contains("simd"))
      F.addFnAttr(Attribute::Convergent);
    F.setUnnamedAddr(GlobalValue::UnnamedAddr::Local);
  }
}

Metadata *i32MD(LLVMContext &Ctx, uint64_t Value) {
  return ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(Ctx), Value));
}

Metadata *strMD(LLVMContext &Ctx, StringRef Text) { return MDString::get(Ctx, Text); }

// Does any FP instruction in the module carry a fast-math flag?
bool hasFastMath(const Module &M) {
  for (const Function &F : M)
    for (const BasicBlock &BB : F)
      for (const Instruction &I : BB)
        if (const auto *FP = dyn_cast<FPMathOperator>(&I))
          if (FP->getFastMathFlags().any())
            return true;
  return false;
}

// AIR: !air.kernel is a list of one kernel node per kernel, each node
// {ptr @k, <stages>, <argument list>} (ref2.ll:138 holds nine such nodes; Apple does not
// nest them under a single node, and the loader reads each list element as a kernel whose
// first operand is the function).
void attachMetadata(Module &M, const std::vector<KernelSpec> &Kernels,
                    const Options &O) {
  LLVMContext &Ctx = M.getContext();

  // !air.version = !{!{i32 2, i32 8, i32 0}} (vadd.ll:20)
  M.getOrInsertNamedMetadata("air.version")
      ->addOperand(MDNode::get(Ctx, {i32MD(Ctx, O.Air[0]), i32MD(Ctx, O.Air[1]),
                                     i32MD(Ctx, O.Air[2])}));

  // !air.language_version = !{!{!"Metal", i32 4, i32 0, i32 0}} (vadd.ll:21)
  M.getOrInsertNamedMetadata("air.language_version")
      ->addOperand(MDNode::get(Ctx, {strMD(Ctx, "Metal"), i32MD(Ctx, O.Metal[0]),
                                     i32MD(Ctx, O.Metal[1]), i32MD(Ctx, O.Metal[2])}));

  // !air.compile_options = !{!{!"air.compile.denorms_disable"}, ...} (vadd.ll:16-18).
  // denorms_disable and framebuffer_fetch_enable are driver-level defaults (Apple emits
  // them for every module, including ones whose IR says nothing about denormals), but
  // fast_math_enable is a permission for the later AIR pipelines to assume fast math, and
  // Apple only ever emits it with fast-math flags on the FP instructions themselves
  // (vadd.ll:10 `fadd fast` + attributes #0's no-nans/unsafe-fp-math, parse.ll the same).
  // Claiming it for precise FP would silently change the meaning of those operations, so
  // it is emitted only when the module's own FP instructions carry fast-math flags.
  NamedMDNode *CompileOptions = M.getOrInsertNamedMetadata("air.compile_options");
  CompileOptions->addOperand(MDNode::get(Ctx, {strMD(Ctx, "air.compile.denorms_disable")}));
  if (hasFastMath(M))
    CompileOptions->addOperand(MDNode::get(Ctx, {strMD(Ctx, "air.compile.fast_math_enable")}));
  CompileOptions->addOperand(
      MDNode::get(Ctx, {strMD(Ctx, "air.compile.framebuffer_fetch_enable")}));

  // !air.kernel = !{!K1, !K2, ...} with !Ki = !{ptr @k, !{}, !{arg nodes...}}
  // (vadd.ll:9-15, ref2.ll:138/155-171)
  std::vector<MDNode *> KernelNodes;
  for (const KernelSpec &K : Kernels) {
    Function *F = M.getFunction(K.Name);
    std::vector<Metadata *> ArgNodes;
    for (unsigned I = 0, E = K.Args.size(); I != E; ++I) {
      const ArgSpec &A = K.Args[I];
      if (A.Kind == ArgKind::Buffer) {
        // AIR: a buffer node is
        // {i32 <parameter index>, "air.buffer", "air.location_index", i32 <index>, i32 1,
        //  "air.<access>", "air.address_space", i32 <as>, "air.arg_type_size", i32 <n>,
        //  "air.arg_type_align_size", i32 <n>, "air.arg_type_name", !"<type>",
        //  "air.arg_name", !"<name>"} (vadd.ll:12). The constant i32 1 after the location
        // index is what all our references carry; the address space is not in the spec,
        // so it is taken from the parameter's pointer type (1 = device for all our
        // kernels, matching vadd.ll).
        unsigned AddressSpace =
            cast<PointerType>(F->getArg(I)->getType())->getAddressSpace();
        ArgNodes.push_back(MDNode::get(
            Ctx, {i32MD(Ctx, I), strMD(Ctx, "air.buffer"), strMD(Ctx, "air.location_index"),
                  i32MD(Ctx, A.Index), i32MD(Ctx, 1), strMD(Ctx, "air." + A.Access),
                  strMD(Ctx, "air.address_space"), i32MD(Ctx, AddressSpace),
                  strMD(Ctx, "air.arg_type_size"), i32MD(Ctx, A.TypeSize),
                  strMD(Ctx, "air.arg_type_align_size"), i32MD(Ctx, A.TypeAlign),
                  strMD(Ctx, "air.arg_type_name"), strMD(Ctx, A.TypeName),
                  strMD(Ctx, "air.arg_name"), strMD(Ctx, A.Name)}));
      } else {
        // AIR: a builtin node is
        // {i32 <parameter index>, !"<air builtin>", "air.arg_type_name", !"<type>",
        //  "air.arg_name", !"<name>"} (vadd.ll:15, reduce.ll:16-19).
        ArgNodes.push_back(MDNode::get(
            Ctx, {i32MD(Ctx, I), strMD(Ctx, A.Intrinsic), strMD(Ctx, "air.arg_type_name"),
                  strMD(Ctx, A.TypeName), strMD(Ctx, "air.arg_name"), strMD(Ctx, A.Name)}));
      }
    }
    KernelNodes.push_back(
        MDNode::get(Ctx, {ValueAsMetadata::get(F), MDNode::get(Ctx, ArrayRef<Metadata *>()),
                          MDNode::get(Ctx, ArgNodes)}));
  }
  NamedMDNode *AirKernel = M.getOrInsertNamedMetadata("air.kernel");
  AirKernel->clearOperands();
  for (MDNode *Kernel : KernelNodes)
    AirKernel->addOperand(Kernel);

  // !llvm.module.flags = !{!0, ..., !8} (vadd.ll:0-8, reduce.ll:0-8). The flags and
  // their behaviors are copied from Apple: SDK Version (Warning, [3 x i32]),
  // wchar_size (Error, i32 4), frame-pointer (Max, i32 2), and the six Max air.max_*
  // limits (31 device/constant/threadgroup buffers, 128 textures, 8 read-write
  // textures, 16 samplers).
  if (NamedMDNode *Flags = M.getNamedMetadata("llvm.module.flags"))
    M.eraseNamedMetadata(Flags);
  uint32_t Sdk[3] = {O.Sdk[0], O.Sdk[1], O.Sdk[2]};
  M.addModuleFlag(Module::Warning, "SDK Version", ConstantDataArray::get(Ctx, Sdk));
  M.addModuleFlag(Module::Error, "wchar_size", uint32_t(4));
  M.addModuleFlag(Module::Max, "frame-pointer", uint32_t(2));
  M.addModuleFlag(Module::Max, "air.max_device_buffers", uint32_t(31));
  M.addModuleFlag(Module::Max, "air.max_constant_buffers", uint32_t(31));
  M.addModuleFlag(Module::Max, "air.max_threadgroup_buffers", uint32_t(31));
  M.addModuleFlag(Module::Max, "air.max_textures", uint32_t(128));
  M.addModuleFlag(Module::Max, "air.max_read_write_textures", uint32_t(8));
  M.addModuleFlag(Module::Max, "air.max_samplers", uint32_t(16));

  // !air.source_file_name = !{!{!"/path/to/file.metal"}} (vadd.ll:22); Apple also sets
  // the module's source_filename to the file's base name (vadd.ll:1).
  if (!O.SourceName.empty()) {
    M.getOrInsertNamedMetadata("air.source_file_name")
        ->addOperand(MDNode::get(Ctx, {strMD(Ctx, O.SourceName)}));
    M.setSourceFileName(sys::path::filename(O.SourceName));
  }

  // !llvm.ident = !{!{!"Apple metal version ..."}} (vadd.ll:19). Apple always emits its
  // own version string; this tool only emits the flag when asked, so the output does not
  // claim to be Apple's compiler.
  if (!O.Ident.empty())
    M.getOrInsertNamedMetadata("llvm.ident")
        ->addOperand(MDNode::get(Ctx, {strMD(Ctx, O.Ident)}));
}

// The LLVM type of an air.arg_type_name from the spec (MSL's scalar type names).
Type *airElementType(LLVMContext &Ctx, StringRef Name) {
  return StringSwitch<Type *>(Name)
      .Cases({"bool", "char", "uchar"}, Type::getInt8Ty(Ctx))
      .Cases({"short", "ushort"}, Type::getInt16Ty(Ctx))
      .Cases({"int", "uint"}, Type::getInt32Ty(Ctx))
      .Cases({"long", "ulong"}, Type::getInt64Ty(Ctx))
      .Case("half", Type::getHalfTy(Ctx))
      .Case("float", Type::getFloatTy(Ctx))
      .Case("double", Type::getDoubleTy(Ctx))
      .Default(nullptr);
}

// The element type an air.* intrinsic's name encodes: "air.atomic.global.add.u.i32" -> i32,
// "air.simd_sum.f32" -> float, "air.wg.barrier" -> nothing. notes/air_conventions.md
// section 4 lists the families; every pointer-taking name we emit ends in ".iN"/".fN".
Type *airIntrinsicPointee(LLVMContext &Ctx, StringRef Name) {
  size_t Dot = Name.rfind('.');
  if (Dot == StringRef::npos || Dot + 2 >= Name.size())
    return nullptr;
  StringRef Tail = Name.drop_front(Dot + 1);
  char Kind = Tail[0];
  if (Kind != 'i' && Kind != 'f')
    return nullptr;
  unsigned Bits = 0;
  for (char C : Tail.drop_front())
    if (C < '0' || C > '9')
      return nullptr;
    else
      Bits = Bits * 10 + unsigned(C - '0');
  if (Kind == 'i')
    return Bits ? IntegerType::get(Ctx, Bits) : nullptr;
  if (Bits == 16)
    return Type::getHalfTy(Ctx);
  if (Bits == 32)
    return Type::getFloatTy(Ctx);
  if (Bits == 64)
    return Type::getDoubleTy(Ctx);
  return nullptr;
}

Metadata *zeroOf(LLVMContext &Ctx, Type *Ty) {
  return ConstantAsMetadata::get(Constant::getNullValue(Ty));
}

// !arg_eltypes (notes/air_conventions.md section 5): a list of
// {i32 <parameter index>, <zero of the pointee element type>} pairs that tells the
// typed-pointer downgrader which pointee to give a pointer parameter. It infers most
// pointees itself from loads/stores/GEPs, but a pointer that only ever goes into an
// intrinsic has no element type anywhere in the IR: reduce's counter is handed straight to
// air.atomic.global.add.u.i32, and the air.* declarations' own pointer parameters are
// never dereferenced at all. Without the hints the LLVM-14 bitcode types those as
// pointers to the empty struct ({} addrspace(1)*), which contradicts the
// air.arg_type_name/air.arg_type_size metadata describing the same argument as a uint.
// (not in the references: Apple's own modules are already typed-pointer IR and carry no
// such metadata, so there is nothing to copy but the format from section 5.)
void attachArgElTypes(Module &M, const std::vector<KernelSpec> &Kernels) {
  LLVMContext &Ctx = M.getContext();

  // Kernels: every buffer parameter gets the element type the spec names for it, which is
  // also what the downgrader infers for the ones it can see (float/i32/i8).
  for (const KernelSpec &K : Kernels) {
    Function *F = M.getFunction(K.Name);
    std::vector<Metadata *> Pairs;
    for (unsigned I = 0, E = K.Args.size(); I != E; ++I) {
      Type *ElementTy = K.Args[I].Kind == ArgKind::Buffer
                            ? airElementType(Ctx, K.Args[I].TypeName)
                            : nullptr;
      if (!ElementTy)
        continue;
      Pairs.push_back(i32MD(Ctx, I));
      Pairs.push_back(zeroOf(Ctx, ElementTy));
    }
    if (!Pairs.empty())
      F->setMetadata("arg_eltypes", MDNode::get(Ctx, Pairs));
  }

  // air.* declarations: the pointee is the element type in the intrinsic's name.
  for (Function &F : M) {
    if (!F.isDeclaration() || !F.getName().starts_with("air."))
      continue;
    Type *ElementTy = airIntrinsicPointee(Ctx, F.getName());
    if (!ElementTy)
      continue;
    std::vector<Metadata *> Pairs;
    for (unsigned I = 0, E = F.arg_size(); I != E; ++I)
      if (F.getArg(I)->getType()->isPointerTy()) {
        Pairs.push_back(i32MD(Ctx, I));
        Pairs.push_back(zeroOf(Ctx, ElementTy));
      }
    if (!Pairs.empty())
      F.setMetadata("arg_eltypes", MDNode::get(Ctx, Pairs));
  }
}

// Step 5b: walk buffer elements with the buffer's real element type.
//
// Zig emits element-indexed buffer accesses as
// `getelementptr inbounds [4 x i8], ptr addrspace(1) %2, i64 %5` (kernels_rf.ll:5984), i.e.
// an array of @sizeOf(element) bytes indexed by the element index; Apple's IR uses the
// element type itself: `getelementptr inbounds float, ptr addrspace(1) %2, i64 %5`
// (vadd.ll:11, where the spec says air.arg_type_name "float"). The two agree byte for byte
// as long as the element sizes match, which is exactly the condition for the rewrite, so
// the addresses cannot change; it makes the buffer walks agree with the
// air.arg_type_name/air.arg_type_size metadata and gives the typed-pointer downgrade
// something to work with. GEPs that address bytes instead (an i8 element type, as parsef's
// per-thread `text + gid * 32` does) are left alone, since their sizes differ, and so are
// GEPs that reach the buffer through the staging alloca of Zig's calling convention: this
// walks the parameter's own uses, which is what O3's SROA leaves behind (at --opt O0 the
// staging still exists and those GEPs keep the [N x i8] form).
// (not literally in the references: Apple's frontend never emits the [N x i8] form, so
// there is nothing to copy but the result shape.)
void retypeBufferGEPs(Module &M, const std::vector<KernelSpec> &Kernels) {
  const DataLayout &DL = M.getDataLayout();
  for (const KernelSpec &K : Kernels) {
    Function *F = M.getFunction(K.Name);
    for (unsigned I = 0, E = K.Args.size(); I != E; ++I) {
      const ArgSpec &A = K.Args[I];
      if (A.Kind != ArgKind::Buffer)
        continue;
      Type *ElementTy = airElementType(M.getContext(), A.TypeName);
      if (!ElementTy)
        continue;
      Value *Buffer = F->getArg(I);
      for (Use &U : Buffer->uses()) {
        auto *GEP = dyn_cast<GetElementPtrInst>(U.getUser());
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

// Step 5: the LLVM middle end.
//
// The pipeline runs with the *input* module's target machine, i.e. the one for nvptx64,
// the stand-in target the Zig kernels are compiled for (its address spaces are AIR's).
// This is not cosmetic: inline costs and thresholds come from the target's cost model,
// and with no target machine at all (the AIR triple has no LLVM target) the generic model
// leaves the outlined fmt.parse_float.* helpers un-inlined (they cost 155..395 against a
// threshold of 45..250). Apple's equivalent, metalfe, is a GPU compiler with a GPU cost
// model; `opt -passes='default<O3>'` on this same module (which builds the nvptx target
// machine from the module triple) is what inlines them, and this reproduces that.
// Nothing target-specific is added to the pipeline: NVPTX registers no PassBuilder
// callbacks, so the pass list is byte-for-byte the generic default<O3> pipeline.
std::unique_ptr<TargetMachine> makeStandInTargetMachine(Module &M) {
  Triple Input(M.getTargetTriple());
  if (Input.getArchName().empty())
    return nullptr;
  // Only the stand-in target is initialised; pulling in every backend would not change
  // the pipeline and only inflate the tool.
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetMC();
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(Input.getArchName(), Input, Error);
  if (!TheTarget) {
    errs() << "air-rewrite: no target machine for '" << Input.str() << "' (" << Error
           << "); running with the generic cost model\n";
    return nullptr;
  }
  TargetOptions Options;
  return std::unique_ptr<TargetMachine>(TheTarget->createTargetMachine(
      Input, /*CPU=*/"", /*Features=*/"", Options, /*RM=*/std::nullopt));
}

void runMiddleEnd(Module &M, bool Optimize, TargetMachine *TM) {
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PipelineTuningOptions PTO;
  PassBuilder PB(TM, PTO);
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager MPM;
  if (Optimize) {
    // AIR: Apple's Metal path runs the standard module inliner/instcombine/SimplifyCFG
    // (notes/air_conventions.md section 7), so this is the generic default<O3> pipeline.
    // It respects optnone, which leaves any function marked optnone unsimplified and
    // uninlinable.
    MPM = PB.buildPerModuleDefaultPipeline(OptimizationLevel::O3);
  } else {
    // (not in the references) With --opt O0 only dead globals and declarations go away,
    // so the output stays readable next to the input.
    MPM.addPass(GlobalDCEPass());
    MPM.addPass(StripDeadPrototypesPass());
  }
  MPM.run(M, MAM);
}

bool rewrite(const Options &O, std::string &Err) {
  std::vector<KernelSpec> Kernels;
  if (!parseSpec(O.SpecPath, Kernels, Err))
    return false;

  auto Buffer = MemoryBuffer::getFile(O.Input);
  if (!Buffer)
    return Err = "cannot read '" + O.Input + "': " + Buffer.getError().message(), false;

  LLVMContext Ctx;
  auto ModuleOrErr = parseBitcodeFile(Buffer.get()->getMemBufferRef(), Ctx);
  if (!ModuleOrErr)
    return Err = "cannot parse bitcode '" + O.Input + "': " +
                 toString(ModuleOrErr.takeError()),
           false;
  std::unique_ptr<Module> M = std::move(*ModuleOrErr);

  if (!checkKernelSignatures(*M, Kernels, Err))
    return false;

  // The stand-in target machine has to be built from the *input* triple, before the
  // module is retargeted to AIR (an air64_v28 triple has no LLVM target at all).
  std::unique_ptr<TargetMachine> TM = makeStandInTargetMachine(*M);

  std::string TripleText = airTriple(O);
  // AIR: triple and data layout of every reference module (vadd.ll:2-3).
  M->setTargetTriple(Triple(TripleText));
  M->setDataLayout(AirDataLayout);
  if (M->getTargetTriple().str() != TripleText)
    return Err = "LLVM rewrote the AIR triple '" + TripleText + "' to '" +
                 M->getTargetTriple().str() + "'",
           false;

  sanitize(*M, Kernels);
  attachArgElTypes(*M, Kernels);
  attachMetadata(*M, Kernels, O);
  runMiddleEnd(*M, O.Optimize, TM.get());
  // Step 5c: O3 infers parameter attributes of its own (writeonly/nofree/captures on the
  // output buffers), so the kernel signatures are normalized from the spec once more, and
  // the buffer GEPs are given the spec's element type.
  retypeBufferGEPs(*M, Kernels);
  for (const KernelSpec &K : Kernels) {
    Function *F = M->getFunction(K.Name);
    F->addFnAttr(Attribute::Convergent);
    F->addFnAttr(Attribute::NoUnwind);
    F->setUnnamedAddr(GlobalValue::UnnamedAddr::Local);
    setKernelParamAttrs(*F, K);
  }

  // Step 6: verify and write.
  if (verifyModule(*M, &errs()))
    return Err = "the rewritten module does not verify", false;

  std::error_code EC;
  raw_fd_ostream Out(O.Output, EC);
  if (EC)
    return Err = "cannot write '" + O.Output + "': " + EC.message(), false;
  WriteBitcodeToFile(*M, Out);
  Out.close();
  if (Out.has_error())
    return Err = "cannot write '" + O.Output + "': " + Out.error().message(), false;

  unsigned Definitions = 0;
  for (const Function &F : *M)
    Definitions += !F.isDeclaration();
  errs() << "air-rewrite: " << TripleText << ", " << Kernels.size() << " kernels (";
  for (unsigned I = 0; I < Kernels.size(); ++I)
    errs() << (I ? ", " : "") << Kernels[I].Name;
  errs() << "), " << Definitions << " defined functions, opt=" << (O.Optimize ? "O3" : "O0")
         << ", cost model=" << (TM ? TM->getTargetTriple().str() : "generic") << ", wrote "
         << O.Output << "\n";
  return true;
}

} // namespace

int main(int Argc, char **Argv) {
  Options O;
  std::string Err;
  if (!parseCommandLine(Argc, Argv, O, Err))
    usage(Err);
  if (!rewrite(O, Err)) {
    errs() << "air-rewrite: error: " << Err << "\n";
    return 1;
  }
  return 0;
}
