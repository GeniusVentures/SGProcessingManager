/**
 * Implementation of the capture file binary format (see capture_file_format.hpp
 * for the full layout doc comment). Calls SerializeArtifact/SerializeManifest/
 * DeserializeArtifact/DeserializeManifest unmodified for the metadata+hash
 * portion; only the new raw-bytes section (per rawOutputCapture invocation)
 * and the outer magic/machine/fixture/combinedHash framing are implemented here.
 */

#include "tools/capture/capture_file_format.hpp"
#include "artifacts/artifact_serializer.hpp"
#include <cstring>

namespace sgns::sgproccapture
{

    namespace
    {
        constexpr char kMagic[4] = { 'S', 'G', 'C', '1' };

        /// Hard cap on any single declared length/count field before it is
        /// trusted to size an allocation or read (T-10-02). 1 GiB.
        constexpr uint64_t kMaxSectionBytes = 1073741824ULL;

        // ── Bounded-count guard ─────────────────────────────────────────
        // Rejects a declared item count BEFORE any per-item allocation
        // (reserve/push_back) if it could not possibly fit in the bytes
        // actually remaining in the input buffer, given a lower bound on
        // how many bytes each item must occupy on the wire. Closes the
        // "malformed count triggers unbounded allocation" risk (T-10-02)
        // for both the artifact count and each artifact's record count.
        bool CountFitsRemaining( uint64_t count, uint64_t minBytesPerItem, uint64_t remainingBytes )
        {
            if ( minBytesPerItem == 0 )
            {
                return true;
            }
            return count <= ( remainingBytes / minBytesPerItem );
        }

        // ── Write helpers ───────────────────────────────────────────────

        void AppendU32( std::vector<uint8_t> &out, uint32_t value )
        {
            uint8_t bytes[sizeof( uint32_t )];
            std::memcpy( bytes, &value, sizeof( uint32_t ) );
            out.insert( out.end(), bytes, bytes + sizeof( uint32_t ) );
        }

        void AppendU64( std::vector<uint8_t> &out, uint64_t value )
        {
            uint8_t bytes[sizeof( uint64_t )];
            std::memcpy( bytes, &value, sizeof( uint64_t ) );
            out.insert( out.end(), bytes, bytes + sizeof( uint64_t ) );
        }

        void AppendString( std::vector<uint8_t> &out, const std::string &value )
        {
            AppendU32( out, static_cast<uint32_t>( value.size() ) );
            out.insert( out.end(), value.begin(), value.end() );
        }

        /// Length-prefixed byte section with an 8-byte (uint64) length prefix --
        /// used for the per-record preQuantizeBytes/quantizedBytes sections.
        void AppendBytesU64( std::vector<uint8_t> &out, const std::vector<uint8_t> &value )
        {
            AppendU64( out, static_cast<uint64_t>( value.size() ) );
            out.insert( out.end(), value.begin(), value.end() );
        }

        /// Length-prefixed byte section with a 4-byte (uint32) length prefix --
        /// used for the trailing combinedHash section.
        void AppendBytesU32( std::vector<uint8_t> &out, const std::vector<uint8_t> &value )
        {
            AppendU32( out, static_cast<uint32_t>( value.size() ) );
            out.insert( out.end(), value.begin(), value.end() );
        }

        // ── Bounds-checked read helpers ─────────────────────────────────
        // Every helper validates `offset + needed <= bytes.size()` (and, for
        // length-prefixed sections, the declared length against
        // kMaxSectionBytes) BEFORE reading or allocating (T-10-02, T-10-01a).

        bool ReadU32( const std::vector<uint8_t> &bytes, size_t &offset, uint32_t &value )
        {
            if ( offset + sizeof( uint32_t ) > bytes.size() )
            {
                return false;
            }
            std::memcpy( &value, bytes.data() + offset, sizeof( uint32_t ) );
            offset += sizeof( uint32_t );
            return true;
        }

        bool ReadU64( const std::vector<uint8_t> &bytes, size_t &offset, uint64_t &value )
        {
            if ( offset + sizeof( uint64_t ) > bytes.size() )
            {
                return false;
            }
            std::memcpy( &value, bytes.data() + offset, sizeof( uint64_t ) );
            offset += sizeof( uint64_t );
            return true;
        }

        bool ReadString( const std::vector<uint8_t> &bytes, size_t &offset, std::string &value )
        {
            uint32_t len = 0;
            if ( !ReadU32( bytes, offset, len ) )
            {
                return false;
            }
            if ( len > kMaxSectionBytes || offset + len > bytes.size() )
            {
                return false;
            }
            value.assign( reinterpret_cast<const char *>( bytes.data() + offset ), len );
            offset += len;
            return true;
        }

        bool ReadBytesU64( const std::vector<uint8_t> &bytes, size_t &offset, std::vector<uint8_t> &value )
        {
            uint64_t len = 0;
            if ( !ReadU64( bytes, offset, len ) )
            {
                return false;
            }
            if ( len > kMaxSectionBytes || offset + len > bytes.size() )
            {
                return false;
            }
            value.assign( bytes.begin() + static_cast<ptrdiff_t>( offset ),
                          bytes.begin() + static_cast<ptrdiff_t>( offset + len ) );
            offset += len;
            return true;
        }

        bool ReadBytesU32( const std::vector<uint8_t> &bytes, size_t &offset, std::vector<uint8_t> &value )
        {
            uint32_t len = 0;
            if ( !ReadU32( bytes, offset, len ) )
            {
                return false;
            }
            if ( len > kMaxSectionBytes || offset + len > bytes.size() )
            {
                return false;
            }
            value.assign( bytes.begin() + static_cast<ptrdiff_t>( offset ),
                          bytes.begin() + static_cast<ptrdiff_t>( offset + len ) );
            offset += len;
            return true;
        }

        bool ReadFixedRegion( const std::vector<uint8_t> &bytes, size_t &offset, size_t regionSize,
                              std::vector<uint8_t> &region )
        {
            if ( offset + regionSize > bytes.size() )
            {
                return false;
            }
            region.assign( bytes.begin() + static_cast<ptrdiff_t>( offset ),
                          bytes.begin() + static_cast<ptrdiff_t>( offset + regionSize ) );
            offset += regionSize;
            return true;
        }

    }  // namespace

    std::vector<uint8_t> SerializeCaptureFile( const CaptureFile &capture )
    {
        std::vector<uint8_t> out;
        out.insert( out.end(), kMagic, kMagic + sizeof( kMagic ) );

        AppendString( out, capture.machineIdTag );
        AppendString( out, capture.fixtureLabel );

        AppendU32( out, static_cast<uint32_t>( capture.artifacts.size() ) );

        static const std::vector<CaptureRecord> kNoRecords;

        for ( size_t i = 0; i < capture.artifacts.size(); ++i )
        {
            const auto artifactBytes = sgns::sgprocessing::SerializeArtifact( capture.artifacts[i] );
            out.insert( out.end(), artifactBytes.begin(), artifactBytes.end() );

            const std::vector<CaptureRecord> &records =
                ( i < capture.rawRecordsPerArtifact.size() ) ? capture.rawRecordsPerArtifact[i] : kNoRecords;

            AppendU32( out, static_cast<uint32_t>( records.size() ) );
            for ( const auto &record : records )
            {
                AppendBytesU64( out, record.preQuantizeBytes );
                AppendBytesU64( out, record.quantizedBytes );
            }
        }

        const auto manifestBytes = sgns::sgprocessing::SerializeManifest( capture.manifest );
        out.insert( out.end(), manifestBytes.begin(), manifestBytes.end() );

        AppendBytesU32( out, capture.combinedHash );

        return out;
    }

    bool DeserializeCaptureFile( const std::vector<uint8_t> &bytes, CaptureFile &out )
    {
        size_t offset = 0;

        if ( bytes.size() < sizeof( kMagic ) || std::memcmp( bytes.data(), kMagic, sizeof( kMagic ) ) != 0 )
        {
            return false;
        }
        offset += sizeof( kMagic );

        CaptureFile parsed;

        if ( !ReadString( bytes, offset, parsed.machineIdTag ) )
        {
            return false;
        }
        if ( !ReadString( bytes, offset, parsed.fixtureLabel ) )
        {
            return false;
        }

        uint32_t artifactCount = 0;
        if ( !ReadU32( bytes, offset, artifactCount ) )
        {
            return false;
        }
        // Minimum wire size for one artifact entry: the fixed artifact region
        // plus its 4-byte recordCount field (records themselves are validated
        // individually below).
        const uint64_t kMinBytesPerArtifact = sgns::sgprocessing::ARTIFACT_SERIALIZED_SIZE + sizeof( uint32_t );
        if ( !CountFitsRemaining( artifactCount, kMinBytesPerArtifact, bytes.size() - offset ) )
        {
            return false;
        }

        parsed.artifacts.reserve( artifactCount );
        parsed.rawRecordsPerArtifact.reserve( artifactCount );

        for ( uint32_t i = 0; i < artifactCount; ++i )
        {
            std::vector<uint8_t> artifactRegion;
            if ( !ReadFixedRegion( bytes, offset, sgns::sgprocessing::ARTIFACT_SERIALIZED_SIZE, artifactRegion ) )
            {
                return false;
            }
            sgns::sgprocessing::Artifact artifact{};
            if ( !sgns::sgprocessing::DeserializeArtifact( artifactRegion, artifact ) )
            {
                return false;
            }

            uint32_t recordCount = 0;
            if ( !ReadU32( bytes, offset, recordCount ) )
            {
                return false;
            }
            // Minimum wire size for one record: two 8-byte length prefixes
            // (the byte payloads themselves are validated individually below).
            constexpr uint64_t kMinBytesPerRecord = 2 * sizeof( uint64_t );
            if ( !CountFitsRemaining( recordCount, kMinBytesPerRecord, bytes.size() - offset ) )
            {
                return false;
            }

            std::vector<CaptureRecord> records;
            records.reserve( recordCount );
            for ( uint32_t r = 0; r < recordCount; ++r )
            {
                CaptureRecord record;
                if ( !ReadBytesU64( bytes, offset, record.preQuantizeBytes ) )
                {
                    return false;
                }
                if ( !ReadBytesU64( bytes, offset, record.quantizedBytes ) )
                {
                    return false;
                }
                records.push_back( std::move( record ) );
            }

            parsed.artifacts.push_back( artifact );
            parsed.rawRecordsPerArtifact.push_back( std::move( records ) );
        }

        std::vector<uint8_t> manifestRegion;
        if ( !ReadFixedRegion( bytes, offset, sgns::sgprocessing::MANIFEST_V2_SERIALIZED_SIZE, manifestRegion ) )
        {
            return false;
        }
        if ( !sgns::sgprocessing::DeserializeManifest( manifestRegion, parsed.manifest ) )
        {
            return false;
        }

        if ( !ReadBytesU32( bytes, offset, parsed.combinedHash ) )
        {
            return false;
        }

        out = std::move( parsed );
        return true;
    }

}  // namespace sgns::sgproccapture
