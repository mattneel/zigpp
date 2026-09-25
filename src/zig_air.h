/*
 * Copyright (c) Zig contributors
 *
 * This file is part of zig, which is MIT licensed.
 * See http://opensource.org/licenses/MIT
 */

#ifndef ZIG_ZIG_AIR_H
#define ZIG_ZIG_AIR_H

#include "zig_llvm.h"

#include <stddef.h>

// The AIR (Apple Intermediate Representation) stage of the Metal backend.
//
// The `air64` LLVM backend emits a module that is already in AIR conventions; what is
// left is (a) the rewrites that turn Zig's module into the shape Apple's reader accepts
// (kernel ABI, metadata, typed-pointer hints), (b) the LLVM middle end, and (c) the
// bitcode downgrade to the LLVM-14 format `Metal`'s library reader is built on
// (llvm-downgrade, vendored under src/llvm-downgrade/). This header is the C API the
// compiler calls in process; see doc/proposals/metal.md sections 2, 4 and 6.2.

struct ZigLLVMAirOptions {
    /* !air.source_file_name value; NULL = omit */
    const char *source_name;
    /* !llvm.ident value; NULL = omit */
    const char *ident;
    unsigned air_major, air_minor, air_patch;
    unsigned metal_major, metal_minor, metal_patch;
    unsigned sdk_major, sdk_minor, sdk_patch;
    /* 0 = only GlobalDCE + StripDeadPrototypes; 1..3 = the standard -O<level> pipeline. */
    unsigned opt_level;
    /* Target bitcode format for the downgraded output: 14, 0 for Apple's AIR reader. */
    unsigned downgrade_major, downgrade_minor;
};

/* Lower a module (LLVM 23 IR already in AIR conventions, as produced by the air64 LLVM
 * backend) into the LLVM-14-format bitcode Apple's Metal runtime reads: run the AIR
 * rewrites, the optimization pipeline, then llvm-downgrade, all in-process. Returns 0 on
 * success and sets *out_bitcode / *out_len (free with ZigLLVMAirDisposeBytes); otherwise
 * returns nonzero and sets *out_error when non-NULL (free with ZigLLVMAirDisposeMessage).
 * The module is modified in place. */
ZIG_EXTERN_C int ZigLLVMAirLower(LLVMModuleRef module, const ZigLLVMAirOptions *options,
                                 char **out_bitcode, size_t *out_len, char **out_error);
ZIG_EXTERN_C void ZigLLVMAirDisposeBytes(char *bytes);
ZIG_EXTERN_C void ZigLLVMAirDisposeMessage(char *message);

#ifdef __cplusplus

#include <string>

namespace llvm {
class Module;
}

/* The AIR rewrites (step 2 through step 5 of the Metal design; src/zig_air_rewrite.cpp).
 * Requires the module's `!air.kernel` named metadata, which the Zig emitter produces.
 * On failure returns false and sets `err`. */
bool zigAirRewrite(llvm::Module &module, const ZigLLVMAirOptions &options, std::string &err);

/* Retype the element-indexed GEPs on kernel buffer parameters to the buffer's element type
 * from `!air.kernel` (the type the host binds and the downgrader recovers pointer pointee
 * types from). Idempotent; run before and after the optimization pipeline, since the
 * pipeline's SROA is what turns Zig's staging allocas into GEPs on the parameters. */
void zigAirRetypeBufferGEPs(llvm::Module &module, std::string &err);

#endif // __cplusplus

#endif // ZIG_ZIG_AIR_H
