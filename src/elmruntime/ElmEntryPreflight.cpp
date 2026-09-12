#include <elmruntime/ElmEntryPreflight.hpp>

#include <elmruntime/ElmManifest.hpp>
#include <elmruntime/ElmResourcePreflight.hpp>
#include <elmruntime/ElmRuntimeError.hpp>

#include <fstream>
#include <string>
#include <vector>

namespace sgns::elmruntime
{
    namespace
    {
        bool ReadFileBytes( const std::string &path, std::vector<uint8_t> &out )
        {
            std::ifstream file( path, std::ios::binary );
            if ( !file )
            {
                return false;
            }
            file.seekg( 0, std::ios::end );
            const std::streamoff size = file.tellg();
            if ( size < 0 )
            {
                return false;
            }
            file.seekg( 0, std::ios::beg );
            out.resize( static_cast<size_t>( size ) );
            if ( size > 0 )
            {
                file.read( reinterpret_cast<char *>( out.data() ), size );
            }
            return file.good() || file.eof();
        }
    } // namespace

    outcome::result<ElmEntryPreflightValues> PreflightPinnedEntry(
        const std::string &entryDir, const std::string &declaredManifestHash )
    {
        std::vector<uint8_t> manifestBytes;
        if ( !ReadFileBytes( entryDir + "elm_manifest.json", manifestBytes ) )
        {
            return outcome::failure( ElmRuntimeError::MANIFEST_INVALID );
        }

        // Re-verify against the work item's declared hash (fail-closed; the
        // pin's entry was verified at publish/hit time, but the re-check is
        // cheap and keeps this bridge self-contained).
        auto parsed = ParseAndVerifyManifest( manifestBytes, declaredManifestHash );
        if ( !parsed )
        {
            return outcome::failure( ElmRuntimeError::MANIFEST_HASH_MISMATCH );
        }

        ElmEntryPreflightValues values;
        const auto              reqs = ExtractElmResourceRequirements( parsed.value() );
        values.requiredMemoryBytes  = reqs.requiredMemoryBytes;
        values.totalArtifactBytes   = reqs.totalArtifactBytes;
        return outcome::success( values );
    }
} // namespace sgns::elmruntime
