#include <elmruntime/ElmRuntimeError.hpp>

// Category message mapping (the .cpp half of the declare/define split --
// ProcessingManager.cpp:13 pattern). OUTCOME_CPP_DEFINE_CATEGORY_3 emits
// non-inline external definitions, so it lives in exactly one TU.

OUTCOME_CPP_DEFINE_CATEGORY_3( sgns::elmruntime, ElmRuntimeError, e )
{
    switch ( e )
    {
        case sgns::elmruntime::ElmRuntimeError::FETCH_FAILED:
            return "Fetch through the elmruntime fetch seam failed";
        case sgns::elmruntime::ElmRuntimeError::MANIFEST_FETCH_FAILED:
            return "Fetching the ELM model manifest failed";
        case sgns::elmruntime::ElmRuntimeError::MANIFEST_HASH_MISMATCH:
            return "Manifest sha256 does not match the declared model_manifest_hash";
        case sgns::elmruntime::ElmRuntimeError::MANIFEST_INVALID:
            return "Manifest failed the size/hash-format/parse/semantic gates";
        case sgns::elmruntime::ElmRuntimeError::ARTIFACT_FETCH_FAILED:
            return "Fetching a manifest artifact failed";
        case sgns::elmruntime::ElmRuntimeError::ARTIFACT_HASH_MISMATCH:
            return "Artifact sha256 does not match the manifest's declared hash";
        case sgns::elmruntime::ElmRuntimeError::ARTIFACT_SIZE_MISMATCH:
            return "Artifact size does not match the manifest's declared size_bytes";
        case sgns::elmruntime::ElmRuntimeError::CACHE_DIR_UNSET:
            return "ELM model cache directory is unset or empty";
        case sgns::elmruntime::ElmRuntimeError::CACHE_ENTRY_QUARANTINED:
            return "Cache entry quarantined after reuse-time verification failure";
        case sgns::elmruntime::ElmRuntimeError::SMOKE_CHECK_FAILED:
            return "ELM model bundle failed the loadability smoke check";
        case sgns::elmruntime::ElmRuntimeError::SMOKE_CHECK_UNAVAILABLE:
            return "ELM model smoke check could not run in this build";
    }
    return "Unknown error";
}
