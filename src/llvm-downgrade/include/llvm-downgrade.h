/* C API for converting LLVM IR to legacy bitcode. */

#ifndef LLVM_DOWNGRADE_H
#define LLVM_DOWNGRADE_H

#include <stddef.h>

#if defined(_WIN32)
#if defined(LLVMDG_EXPORTS)
#define LLVMDG_API __declspec(dllexport)
#elif defined(LLVMDG_STATIC)
#define LLVMDG_API
#else
#define LLVMDG_API __declspec(dllimport)
#endif
#else
#define LLVMDG_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct LLVMDGOpaqueMemoryBuffer *LLVMDGMemoryBufferRef;

/* Downgrade a module (bitcode or textual IR, `Length` bytes at `Bitcode`) to
 * the bitcode format of LLVM `Major.Minor`, one of LLVMDGGetTargets. The input
 * is auto-upgraded to the host LLVM as it loads, so bitcode from any LLVM the
 * host can read is accepted. On success returns 0 and sets `*OutBitcode`; on
 * failure returns nonzero and sets `*OutMessage` when provided. Bitcode and
 * OutBitcode must be non-null; OutMessage is optional. Both outputs are cleared
 * before use. Dispose of results with LLVMDGDisposeMemoryBuffer and messages
 * with LLVMDGDisposeMessage. A message may be null if allocation fails.
 *
 * Calls are serialized. Unsupported constructs return errors; LLVM assertions
 * and internal fatal errors are not recoverable. */
LLVMDG_API int LLVMDGDowngrade(const char *Bitcode, size_t Length,
                               unsigned Major, unsigned Minor,
                               LLVMDGMemoryBufferRef *OutBitcode,
                               char **OutMessage);

/* The bitcode formats this build can emit, as "Major.Minor" strings
 * ("5.0", "7.0", "14.0", ...). Owned by the library. */
LLVMDG_API const char *const *LLVMDGGetTargets(size_t *Count);
/* The LLVM this library was built on, i.e. the newest bitcode it reads. */
LLVMDG_API void LLVMDGGetLLVMVersion(unsigned *Major, unsigned *Minor,
                                     unsigned *Patch);

/* Buffer access requires a non-null result. Dispose functions accept null.
 * Buffer data remains valid until disposal; callers synchronize access to it.
 */
LLVMDG_API const char *LLVMDGGetBufferStart(LLVMDGMemoryBufferRef Buffer);
LLVMDG_API size_t LLVMDGGetBufferSize(LLVMDGMemoryBufferRef Buffer);
LLVMDG_API void LLVMDGDisposeMemoryBuffer(LLVMDGMemoryBufferRef Buffer);
LLVMDG_API void LLVMDGDisposeMessage(char *Message);

#ifdef __cplusplus
}
#endif

#endif /* LLVM_DOWNGRADE_H */
