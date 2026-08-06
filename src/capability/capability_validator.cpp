/**
 * CapabilityValidator implementation — BuildSnapshot and CanExecute.
 *
 * BuildSnapshot: queries Vulkan device properties, MNN executor registry,
 * disk space, and computes the deterministic executor identity hash.
 * CanExecute: validates jobs against the cached snapshot across all five
 * check categories (PassType, Vulkan, MNN, GPU memory, disk space).
 */

#include <capability/capability_validator.hpp>
#include <processingbase/vulkan_init_guard.hpp>
#include <util/sha256.hpp>
#include <ColorFormat.hpp>
#include <DepthFormat.hpp>
#include <ModelFormat.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <sstream>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/statvfs.h>
#endif

namespace sgns::sgprocessing
{

    // =========================================================================
    // PIMPL
    // =========================================================================

    struct CapabilityValidator::Impl
    {
        CapabilitySnapshot snapshot;
        bool               snapshotBuilt = false;
    };

    // =========================================================================
    // Anonymous-namespace helpers
    // =========================================================================

    namespace
    {

        std::string JoinStrings( const std::vector<std::string> &items, const std::string &sep )
        {
            std::ostringstream oss;
            for ( size_t i = 0; i < items.size(); ++i )
            {
                if ( i > 0 ) oss << sep;
                oss << items[i];
            }
            return oss.str();
        }

        std::string FormatBytes( uint64_t bytes )
        {
            const char *units[] = { "B", "KB", "MB", "GB", "TB" };
            int         unit    = 0;
            double      val     = static_cast<double>( bytes );
            while ( val >= 1024.0 && unit < 4 ) { val /= 1024.0; ++unit; }
            char tmp[64];
            if ( unit == 0 )
                std::snprintf( tmp, sizeof( tmp ), "%llu %s",
                               static_cast<unsigned long long>( bytes ), units[unit] );
            else
                std::snprintf( tmp, sizeof( tmp ), "%.1f %s", val, units[unit] );
            return tmp;
        }

        std::string ListAvailablePassTypes( const std::vector<ExecutorCapability> &caps )
        {
            std::vector<std::string> names;
            for ( const auto &cap : caps )
                names.push_back( std::to_string( static_cast<int>( cap.passType ) ) );
            if ( names.empty() ) return "";
            return JoinStrings( names, ", " );
        }

        uint64_t QueryAvailableDiskBytes( const std::string &path )
        {
#ifdef _WIN32
            ULARGE_INTEGER freeBytesAvailable;
            if ( GetDiskFreeSpaceExA( path.empty() ? "." : path.c_str(),
                                      &freeBytesAvailable, nullptr, nullptr ) )
                return freeBytesAvailable.QuadPart;
            return 0;
#else
            struct statvfs stat;
            if ( statvfs( path.empty() ? "." : path.c_str(), &stat ) == 0 )
                return static_cast<uint64_t>( stat.f_bavail ) * stat.f_frsize;
            return 0;
#endif
        }

        uint64_t BytesPerPixel( sgns::ColorFormat fmt )
        {
            switch ( fmt )
            {
                case sgns::ColorFormat::RGBA8: return 4;
                case sgns::ColorFormat::RGB8:  return 3;
                default:                       return 4;
            }
        }

        uint64_t BytesPerPixel( sgns::DepthFormat fmt )
        {
            switch ( fmt )
            {
                case sgns::DepthFormat::D32_SFLOAT:       return 4;
                case sgns::DepthFormat::D24_UNORM_S8_UINT: return 4;
                default:                                  return 4;
            }
        }

        std::vector<ExecutorCapability> CollectExecutorCapabilities(
            const std::unordered_map<PassType,
                std::function<std::unique_ptr<ProcessingProcessor>()>,
                PassTypeHash> &passFactories,
            size_t /*mnnProcessorCount*/ )
        {
            std::vector<ExecutorCapability> caps;
            for ( const auto &[passType, factory] : passFactories )
            {
                (void)factory;
                ExecutorCapability cap;
                cap.passType = passType;
                if ( passType == PassType::RENDER )
                {
                    cap.backend = "VULKAN";
                }
                else
                {
                    cap.backend                = "VULKAN";
                    cap.supportedModelFormats  = { "MNN" };
                    cap.supportedQuantizations = { "FP32", "FP16", "INT8" };
                }
                caps.push_back( std::move( cap ) );
            }
            return caps;
        }

        const ExecutorCapability *FindExecutorCap( const CapabilitySnapshot &snap,
                                                   PassType                  pt )
        {
            for ( const auto &cap : snap.executorCaps )
                if ( cap.passType == pt ) return &cap;
            return nullptr;
        }

        std::string DeriveExecutorId( const std::vector<uint8_t> &identityHash )
        {
            if ( identityHash.empty() ) return "sgproc-0000000000000000";
            std::ostringstream oss;
            oss << "sgproc-";
            size_t n = (std::min)( size_t( 8 ), identityHash.size() );
            for ( size_t i = 0; i < n; ++i )
            {
                char hex[3];
                std::snprintf( hex, sizeof( hex ), "%02x", identityHash[i] );
                oss << hex;
            }
            return oss.str();
        }

        std::string ModelFormatToString( sgns::ModelFormat fmt )
        {
            switch ( fmt )
            {
                case sgns::ModelFormat::MNN:         return "MNN";
                case sgns::ModelFormat::ONNX:        return "ONNX";
                case sgns::ModelFormat::PY_TORCH:    return "PY_TORCH";
                case sgns::ModelFormat::TENSOR_FLOW: return "TENSOR_FLOW";
                default:                             return "UNKNOWN";
            }
        }

    } // anonymous namespace

    // =========================================================================
    // Construction / destruction
    // =========================================================================

    CapabilityValidator::CapabilityValidator()
        : m_impl( std::make_unique<Impl>() ) {}

    CapabilityValidator::~CapabilityValidator() = default;

    // =========================================================================
    // BuildSnapshot (D-09, D-10, D-11, D-12, D-16)
    // =========================================================================

    void CapabilityValidator::BuildSnapshot(
        const std::unordered_map<PassType,
            std::function<std::unique_ptr<ProcessingProcessor>()>,
            PassTypeHash> &passFactories,
        size_t             mnnProcessorCount,
        std::function<VkPhysicalDevice()> ensureVulkanDevice )
    {
        CapabilitySnapshot snapshot;

        // Vulkan device query (D-10, D-14)
        // NOTE: ensureVulkanDevice() internally calls RenderProcessor::InitializeContext(),
        // which acquires VulkanInitMutex() itself via a double-check locking pattern.
        // Holding the mutex here while calling ensureVulkanDevice() would cause a
        // self-deadlock on the same thread. Only lock around the vkGetPhysicalDevice*
        // queries — the device is kept alive by the static RenderProcessor inside the
        // lambda, so it's safe to read its properties without the mutex.
        {
            VkPhysicalDevice device = ensureVulkanDevice();
            if ( device != VK_NULL_HANDLE )
            {
                std::lock_guard<std::mutex> lock( VulkanInitMutex() );
                vkGetPhysicalDeviceProperties( device, &snapshot.vulkanProps );
                vkGetPhysicalDeviceMemoryProperties( device, &snapshot.memProps );
            }
        }

        // MNN executor capability collection (D-11)
        snapshot.executorCaps = CollectExecutorCapabilities( passFactories, mnnProcessorCount );

        // Disk space query (D-16)
        snapshot.availableDiskBytes = QueryAvailableDiskBytes( "." );

        // Executor identity hash (D-08)
        {
            std::vector<uint8_t> hashInput;
            auto appendBytes = [&hashInput]( const void *data, size_t size )
            {
                const auto *bytes = static_cast<const uint8_t *>( data );
                hashInput.insert( hashInput.end(), bytes, bytes + size );
            };

            appendBytes( &snapshot.vulkanProps.deviceID,
                         sizeof( snapshot.vulkanProps.deviceID ) );
            appendBytes( &snapshot.vulkanProps.driverVersion,
                         sizeof( snapshot.vulkanProps.driverVersion ) );
            appendBytes( &snapshot.vulkanProps.vendorID,
                         sizeof( snapshot.vulkanProps.vendorID ) );
            appendBytes( snapshot.vulkanProps.deviceName,
                         std::strlen( snapshot.vulkanProps.deviceName ) );

            for ( uint32_t i = 0; i < snapshot.memProps.memoryHeapCount; ++i )
            {
                appendBytes( &snapshot.memProps.memoryHeaps[i].size,
                             sizeof( snapshot.memProps.memoryHeaps[i].size ) );
                appendBytes( &snapshot.memProps.memoryHeaps[i].flags,
                             sizeof( snapshot.memProps.memoryHeaps[i].flags ) );
            }

            for ( const auto &cap : snapshot.executorCaps )
            {
                auto pt = static_cast<int>( cap.passType );
                appendBytes( &pt, sizeof( pt ) );
                appendBytes( cap.backend.data(), cap.backend.size() );
                for ( const auto &fmt : cap.supportedModelFormats )
                    appendBytes( fmt.data(), fmt.size() );
                for ( const auto &q : cap.supportedQuantizations )
                    appendBytes( q.data(), q.size() );
            }

            snapshot.identityHash = sgns::sgprocmanagersha::sha256(
                hashInput.data(), hashInput.size() );
        }

        m_impl->snapshot      = std::move( snapshot );
        m_impl->snapshotBuilt = true;
    }

    const CapabilitySnapshot *CapabilityValidator::GetSnapshot() const
    {
        if ( !m_impl->snapshotBuilt ) return nullptr;
        return &m_impl->snapshot;
    }

    // =========================================================================
    // CanExecute — all five validation categories (CAP-02..05)
    // =========================================================================

    void CapabilityValidator::CanExecute( const sgns::Pass &pass,
                                          CanExecuteCallback callback )
    {
        CanExecuteResult              result;
        std::vector<UnmetRequirement> unmet;

        if ( !m_impl->snapshotBuilt )
        {
            result.executable = false;
            result.unmet.push_back(
                { UnmetRequirementCategory::RESOURCE,
                  "CapabilityValidator not initialized" } );
            callback( result );
            return;
        }

        const auto &snapshot = m_impl->snapshot;
        PassType    passType = pass.get_type();

        // —— Step 1: PassType registration check (CAP-04/D-04) ——
        const ExecutorCapability *executorCap = FindExecutorCap( snapshot, passType );
        if ( !executorCap )
        {
            unmet.push_back(
                { UnmetRequirementCategory::PASS_TYPE,
                  "No executor registered for PassType "
                      + std::to_string( static_cast<int>( passType ) )
                      + ". Available: ["
                      + ListAvailablePassTypes( snapshot.executorCaps ) + "]" } );
            result.executable = false;
            result.unmet      = std::move( unmet );
            callback( result );
            return;
        }

        // —— Step 2: Vulkan feature/limit check (CAP-02/D-14) ——
        if ( passType == PassType::RENDER )
        {
            const auto &limits = snapshot.vulkanProps.limits;

            if ( snapshot.vulkanProps.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
                 && snapshot.vulkanProps.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU )
            {
                unmet.push_back(
                    { UnmetRequirementCategory::VULKAN,
                      "Device type not acceptable (need DISCRETE_GPU or INTEGRATED_GPU)" } );
            }

            if ( auto rt = pass.get_render_target() )
            {
                uint32_t w = static_cast<uint32_t>( rt->get_width() );
                uint32_t h = static_cast<uint32_t>( rt->get_height() );

                if ( w > limits.maxImageDimension2D )
                    unmet.push_back(
                        { UnmetRequirementCategory::VULKAN,
                          "maxImageDimension2D: need " + std::to_string( w )
                              + ", have " + std::to_string( limits.maxImageDimension2D ) } );
                if ( h > limits.maxImageDimension2D )
                    unmet.push_back(
                        { UnmetRequirementCategory::VULKAN,
                          "maxImageDimension2D: need " + std::to_string( h )
                              + ", have " + std::to_string( limits.maxImageDimension2D ) } );
            }

            if ( limits.maxColorAttachments < 1 )
                unmet.push_back(
                    { UnmetRequirementCategory::VULKAN,
                      "maxColorAttachments: need 1, have "
                          + std::to_string( limits.maxColorAttachments ) } );

            if ( limits.maxMemoryAllocationCount < 4 )
                unmet.push_back(
                    { UnmetRequirementCategory::VULKAN,
                      "maxMemoryAllocationCount: need 4, have "
                          + std::to_string( limits.maxMemoryAllocationCount ) } );
        }

        if ( !unmet.empty() )
        {
            result.executable = false;
            result.unmet      = std::move( unmet );
            callback( result );
            return;
        }

        // —— Step 2b: MNN model format check (CAP-03/D-11) ——
        if ( passType == PassType::INFERENCE || passType == PassType::RETRAIN )
        {
            if ( auto model = pass.get_model() )
            {
                std::string fmtStr = ModelFormatToString( model->get_format() );
                bool        formatSupported = false;
                for ( const auto &sf : executorCap->supportedModelFormats )
                {
                    if ( sf == fmtStr ) { formatSupported = true; break; }
                }
                if ( !formatSupported )
                    unmet.push_back(
                        { UnmetRequirementCategory::MNN,
                          "Model format " + fmtStr + " not supported. Supported: ["
                              + JoinStrings( executorCap->supportedModelFormats, ", " )
                              + "]" } );
            }
        }

        if ( !unmet.empty() )
        {
            result.executable = false;
            result.unmet      = std::move( unmet );
            callback( result );
            return;
        }

        // —— Step 3: GPU memory estimation (CAP-05/D-15) ——
        if ( passType == PassType::RENDER )
        {
            uint64_t estimatedGpuMem = 0;

            if ( auto rt = pass.get_render_target() )
            {
                uint64_t w          = static_cast<uint64_t>( rt->get_width() );
                uint64_t h          = static_cast<uint64_t>( rt->get_height() );
                uint64_t colorBytes = BytesPerPixel( rt->get_color_format() );
                uint64_t depthBytes = BytesPerPixel( rt->get_depth_format() );
                estimatedGpuMem    += w * h * ( colorBytes + depthBytes );
            }

            estimatedGpuMem += 64ULL * 1024 * 1024; // pipeline overhead

            uint64_t largestHeap = 0;
            for ( uint32_t i = 0; i < snapshot.memProps.memoryHeapCount; ++i )
            {
                if ( snapshot.memProps.memoryHeaps[i].flags
                     & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT )
                {
                    largestHeap = (std::max)( largestHeap,
                                              snapshot.memProps.memoryHeaps[i].size );
                }
            }

            if ( largestHeap > 0 && estimatedGpuMem > largestHeap )
                unmet.push_back(
                    { UnmetRequirementCategory::RESOURCE,
                      "Estimated GPU memory " + FormatBytes( estimatedGpuMem )
                          + " exceeds largest device-local heap "
                          + FormatBytes( largestHeap ) } );
        }

        // —— Step 4: Disk space check (CAP-05/D-16) ——
        if ( snapshot.availableDiskBytes > 0 )
        {
            uint64_t estimatedOutputSize = 0;

            if ( passType == PassType::RENDER )
            {
                if ( auto rt = pass.get_render_target() )
                {
                    uint64_t w = static_cast<uint64_t>( rt->get_width() );
                    uint64_t h = static_cast<uint64_t>( rt->get_height() );
                    estimatedOutputSize = w * h
                                          * BytesPerPixel( rt->get_color_format() );
                }
            }

            if ( estimatedOutputSize > snapshot.availableDiskBytes )
                unmet.push_back(
                    { UnmetRequirementCategory::RESOURCE,
                      "Estimated output size " + FormatBytes( estimatedOutputSize )
                          + " exceeds available disk space "
                          + FormatBytes( snapshot.availableDiskBytes ) } );
        }

        // —— Build final result ——
        if ( !unmet.empty() )
        {
            result.executable = false;
            result.unmet      = std::move( unmet );
        }
        else
        {
            result.executable = true;
            result.executorId = DeriveExecutorId( snapshot.identityHash );
        }

        callback( result );
    }

#ifdef SGPROCMGR_TEST_FRIEND
    void CapabilityValidator::SetSnapshotForTest( CapabilitySnapshot snap )
    {
        m_impl->snapshot      = std::move( snap );
        m_impl->snapshotBuilt = true;
    }
#endif

} // namespace sgns::sgprocessing
