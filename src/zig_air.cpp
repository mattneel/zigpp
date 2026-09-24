/*
 * Copyright (c) Zig contributors
 *
 * This file is part of zig, which is MIT licensed.
 * See http://opensource.org/licenses/MIT
 */

// The AIR stage of the Metal backend: the rewrites of src/zig_air_rewrite.cpp, the LLVM
// middle end, and the bitcode downgrade to the format Apple's Metal library reader is built
// on (doc/proposals/metal.md sections 2, 4 and 6.2).
//
// The downgrade is the vendored llvm-downgrade, https://github.com/JuliaLLVM/llvm-downgrade
// (Apache-2.0 WITH LLVM-exception; the sources are LLVM's own BitcodeWriter/ValueEnumerator
// from the 5.0/7.0/14.0/15.0/18.1 releases, ported to LLVM 23's C++ API, plus the
// ModuleRewriter*/PointerRewriter passes that undo the opaque-pointer migration), at
// revision 4244e2a1ec56dd9c714755453ba840a48a0b5ee5 ("Target LLVM 23", 2026-09-11), vendored
// unmodified under src/llvm-downgrade/ (see src/llvm-downgrade/LICENSE.TXT for the license).
// Its translation units must be compiled with -DNDEBUG (they call Value::dump() inside
// #ifndef NDEBUG and a release LLVM has no definition of it), -fexceptions (writer errors
// unwind through them) and with the vendored include/ directory shadowing the system LLVM
// headers, whose BitcodeWriter.h/LLVMBitCodes.h they augment.
//
// The emitted bytes are the LLVM-14 bitcode itself; Apple's 20-byte module-section header
// (0x0b17c0de, u32 0, u32 0x14, u32 size, i32 -1), which the container the Metal runtime
// reads stores in front of it (doc section 5), is added by the metallib writer.

#include "zig_air.h"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/StripDeadPrototypes.h"

#include "llvm-downgrade/include/llvm-downgrade.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <string>

namespace {

char *duplicateString(const char *Text) {
    size_t Length = std::strlen(Text);
    char *Copy = (char *)std::malloc(Length + 1);
    if (!Copy) return nullptr;
    std::memcpy(Copy, Text, Length + 1);
    return Copy;
}

// Report a human-readable error the caller frees with ZigLLVMAirDisposeMessage. An
// allocation failure leaves *OutError null, which the C API documents as "no message".
int fail(char **OutError, const std::string &Message) {
    if (OutError) *OutError = duplicateString(Message.c_str());
    return 1;
}

// The standard module pipeline at the requested level. There is no TargetMachine: air64 has
// no LLVM backend (Apple's compiler does instruction selection, doc section 6.3), so the
// generic cost model is what runs; opt_level 0 keeps the module readable and only drops dead
// globals and declarations, which is what the spike measured with `--opt O0`.
bool runPipeline(llvm::Module &M, unsigned OptLevel, std::string &Err) {
    llvm::LoopAnalysisManager LAM;
    llvm::FunctionAnalysisManager FAM;
    llvm::CGSCCAnalysisManager CGAM;
    llvm::ModuleAnalysisManager MAM;
    llvm::PipelineTuningOptions PTO;
    llvm::PassBuilder PB(nullptr, PTO);
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    llvm::ModulePassManager MPM;
    switch (OptLevel) {
        case 0:
            MPM.addPass(llvm::GlobalDCEPass());
            MPM.addPass(llvm::StripDeadPrototypesPass());
            break;
        case 1:
            MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O1);
            break;
        case 2:
            MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O2);
            break;
        case 3:
            MPM = PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
            break;
        default:
            Err = "unsupported optimization level " + std::to_string(OptLevel) +
                  " (0 for GlobalDCE + StripDeadPrototypes, 1 to 3 for the standard "
                  "pipeline)";
            return false;
    }
    MPM.run(M, MAM);
    return true;
}

// The pipeline may attach an !llvm.ident of its own; the finished module has exactly the
// one the options name, or none.
void setIdent(llvm::Module &M, const char *Ident) {
    if (llvm::NamedMDNode *Existing = M.getNamedMetadata("llvm.ident"))
        M.eraseNamedMetadata(Existing);
    if (!Ident) return;
    M.getOrInsertNamedMetadata("llvm.ident")
        ->addOperand(llvm::MDNode::get(M.getContext(),
                                       {llvm::MDString::get(M.getContext(), Ident)}));
}

bool verify(const llvm::Module &M, std::string &Err) {
    std::string Message;
    llvm::raw_string_ostream OS(Message);
    if (!llvm::verifyModule(M, &OS)) return true;
    Err = "the optimized AIR module does not verify:\n" + OS.str();
    return false;
}

}  // namespace

extern "C" void ZigLLVMAirDisposeBytes(char *bytes) { std::free(bytes); }

extern "C" void ZigLLVMAirDisposeMessage(char *message) { std::free(message); }

extern "C" int ZigLLVMAirLower(LLVMModuleRef module, const ZigLLVMAirOptions *options,
                               char **out_bitcode, size_t *out_len, char **out_error) {
    if (out_bitcode) *out_bitcode = nullptr;
    if (out_len) *out_len = 0;
    if (out_error) *out_error = nullptr;
    if (!out_bitcode || !out_len)
        return fail(out_error, "the output bitcode pointer must not be null");
    if (!module) return fail(out_error, "the module is null");
    if (!options) return fail(out_error, "the options are null");

    try {
        llvm::Module &M = *llvm::unwrap(module);
        std::string Err;

        // The AIR conventions: kernel ABI, metadata, typed-pointer hints (steps 2 to 5).
        if (!zigAirRewrite(M, *options, Err)) return fail(out_error, Err);

        // The LLVM middle end (step 6).
        if (!runPipeline(M, options->opt_level, Err)) return fail(out_error, Err);

        // The pipeline's own inferrences are removed again (the AIR reader wants Apple's
        // kernel parameter shapes), and SROA has left GEPs on the buffer parameters that
        // want the buffer's element type (step 5b of the spike).
        zigAirRetypeBufferGEPs(M, Err);
        if (!Err.empty()) return fail(out_error, Err);
        setIdent(M, options->ident);
        if (!verify(M, Err)) return fail(out_error, Err);

        // Serialize the LLVM 23 module and downgrade it to the target bitcode format
        // (step 7).
        llvm::SmallVector<char, 0> Data;
        {
            llvm::raw_svector_ostream OS(Data);
            llvm::WriteBitcodeToFile(M, OS);
        }

        LLVMDGMemoryBufferRef Out = nullptr;
        char *Message = nullptr;
        int Result = LLVMDGDowngrade(Data.data(), Data.size(), options->downgrade_major,
                                     options->downgrade_minor, &Out, &Message);
        if (Result != 0) {
            std::string Text =
                Message ? std::string(Message) : std::string("llvm-downgrade failed");
            LLVMDGDisposeMessage(Message);
            if (Out) LLVMDGDisposeMemoryBuffer(Out);
            return fail(out_error, "cannot downgrade the AIR module to LLVM " +
                                       std::to_string(options->downgrade_major) + "." +
                                       std::to_string(options->downgrade_minor) +
                                       " bitcode: " + Text);
        }
        if (!Out) return fail(out_error, "llvm-downgrade returned no buffer");

        size_t Length = LLVMDGGetBufferSize(Out);
        const char *Start = LLVMDGGetBufferStart(Out);
        if (Length == 0 || !Start) {
            LLVMDGDisposeMemoryBuffer(Out);
            return fail(out_error, "llvm-downgrade produced an empty buffer");
        }
        char *Bytes = (char *)std::malloc(Length);
        if (!Bytes) {
            LLVMDGDisposeMemoryBuffer(Out);
            return fail(out_error, "out of memory copying the downgraded bitcode");
        }
        std::memcpy(Bytes, Start, Length);
        LLVMDGDisposeMemoryBuffer(Out);

        *out_bitcode = Bytes;
        *out_len = Length;
        return 0;
    } catch (const std::exception &Ex) {
        return fail(out_error, std::string("the AIR stage failed: ") + Ex.what());
    } catch (...) {
        return fail(out_error, "the AIR stage failed unexpectedly");
    }
}
