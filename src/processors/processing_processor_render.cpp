#include "processors/processing_processor_render.hpp"
#include "processingbase/vulkan_init_guard.hpp"
#include <VkBootstrap.h>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <ColorFormat.hpp>
#include <DepthFormat.hpp>
#include <Topology.hpp>
#include <CullMode.hpp>
#include <FrontFace.hpp>
#include <DepthTest.hpp>
#include <VertexLayoutFormat.hpp>

namespace sgns::sgprocessing
{

    bool RenderProcessor::IsAcceptable( VkPhysicalDeviceType type )
    {
        return type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
            || type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    }

    VkDeviceSize RenderProcessor::LargestDeviceLocalHeap( VkPhysicalDevice device )
    {
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties( device, &memProps );
        VkDeviceSize largest = 0;
        for ( uint32_t i = 0; i < memProps.memoryHeapCount; ++i )
        {
            if ( memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT )
                largest = (std::max)( largest, memProps.memoryHeaps[i].size );
        }
        return largest;
    }

    bool RenderProcessor::InitializeContext()
    {
        if ( m_contextInitialized )
            return true;

        std::lock_guard<std::mutex> lock( sgns::sgprocessing::VulkanInitMutex() );

        if ( m_contextInitialized )
            return true;

        vkb::InstanceBuilder instance_builder;
        auto inst_ret = instance_builder
                            .set_app_name( "SGProcessingManager RenderProcessor" )
                            .set_app_version( 1, 0, 0 )
                            .request_validation_layers( false )
                            .build();
        if ( !inst_ret )
        {
            m_logger->error( "RenderProcessor: failed to create Vulkan instance: {}",
                             inst_ret.error().message() );
            return false;
        }
        auto vkb_instance = inst_ret.value();

        vkb::PhysicalDeviceSelector selector( vkb_instance );
        auto devices_ret = selector.select_devices();
        if ( !devices_ret )
        {
            m_logger->error( "RenderProcessor: failed to enumerate physical devices: {}",
                             devices_ret.error().message() );
            vkb::destroy_instance( vkb_instance );
            return false;
        }

        auto devices = devices_ret.value();

        devices.erase(
            std::remove_if( devices.begin(), devices.end(),
                []( const vkb::PhysicalDevice &d ) {
                    return !IsAcceptable( d.properties.deviceType );
                } ),
            devices.end() );

        if ( devices.empty() )
        {
            m_logger->error( "RenderProcessor: no acceptable physical device found "
                             "(none with device type DISCRETE_GPU or INTEGRATED_GPU)" );
            vkb::destroy_instance( vkb_instance );
            return false;
        }

        std::sort( devices.begin(), devices.end(),
            []( const vkb::PhysicalDevice &a, const vkb::PhysicalDevice &b ) {
                int rank_a = ( a.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ) ? 2 : 1;
                int rank_b = ( b.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ) ? 2 : 1;
                if ( rank_a != rank_b )
                    return rank_a > rank_b;
                return LargestDeviceLocalHeap( a.physical_device )
                     > LargestDeviceLocalHeap( b.physical_device );
            } );

        vkb::DeviceBuilder device_builder( devices[0] );
        auto dev_ret = device_builder.build();
        if ( !dev_ret )
        {
            m_logger->error( "RenderProcessor: failed to create Vulkan device: {}",
                             dev_ret.error().message() );
            vkb::destroy_instance( vkb_instance );
            return false;
        }

        auto vkb_device = dev_ret.value();
        auto queue_ret = vkb_device.get_queue( vkb::QueueType::graphics );
        if ( !queue_ret )
        {
            m_logger->error( "RenderProcessor: failed to get graphics queue: {}",
                             queue_ret.error().message() );
            vkb::destroy_device( vkb_device );
            vkb::destroy_instance( vkb_instance );
            return false;
        }

        m_instance = vkb_instance.instance;
        m_physicalDevice = vkb_device.physical_device;
        m_device = vkb_device.device;
        m_queue = queue_ret.value();
        m_contextInitialized = true;

        return true;
    }

    ProcessingResult RenderProcessor::MakeError( sgns::sgprocessing::ProcessingErrorStage stage,
                                                  const std::string                       &message )
    {
        ProcessingResult result;
        result.hash = std::vector<uint8_t>( 32, 0 );
        ProcessingError error;
        error.stage = stage;
        error.message = message;
        result.error = error;
        return result;
    }

    void RenderProcessor::PushTeardown( std::function<void()> fn )
    {
        m_teardown.push_back( std::move( fn ) );
    }

    void RenderProcessor::RunTeardown()
    {
        for ( auto it = m_teardown.rbegin(); it != m_teardown.rend(); ++it )
        {
            ( *it )();
        }
        m_teardown.clear();
    }

    namespace
    {
        /// Bounds-checked little-endian primitive readers over a raw byte
        /// buffer. Every read advances `offset`; callers must check the
        /// return value before trusting `out`. Never reads past `size`.
        bool ReadU32( const char *data, size_t size, size_t &offset, uint32_t &out )
        {
            if ( offset + sizeof( uint32_t ) > size )
            {
                return false;
            }
            std::memcpy( &out, data + offset, sizeof( uint32_t ) );
            offset += sizeof( uint32_t );
            return true;
        }

        bool ReadU8( const char *data, size_t size, size_t &offset, uint8_t &out )
        {
            if ( offset + sizeof( uint8_t ) > size )
            {
                return false;
            }
            out = static_cast<uint8_t>( data[offset] );
            offset += sizeof( uint8_t );
            return true;
        }

        bool ReadF32( const char *data, size_t size, size_t &offset, float &out )
        {
            if ( offset + sizeof( float ) > size )
            {
                return false;
            }
            std::memcpy( &out, data + offset, sizeof( float ) );
            offset += sizeof( float );
            return true;
        }

        bool ReadBytes( const char *data, size_t size, size_t &offset, size_t count, const char *&outPtr )
        {
            if ( offset + count > size )
            {
                return false;
            }
            outPtr = data + offset;
            offset += count;
            return true;
        }

        bool ReadString( const char *data, size_t size, size_t &offset, std::string &out )
        {
            uint32_t len = 0;
            if ( !ReadU32( data, size, offset, len ) )
            {
                return false;
            }
            if ( len == 0 )
            {
                out.clear();
                return true;
            }
            const char *bytes = nullptr;
            if ( !ReadBytes( data, size, offset, len, bytes ) )
            {
                return false;
            }
            out.assign( bytes, len );
            return true;
        }
    }

    bool RenderProcessor::ParseCompiledStages( const std::vector<char>  &modelFile,
                                                std::vector<ParsedStage> &outStages,
                                                ProcessingResult         &errorOut )
    {
        outStages.clear();
        const char  *data   = modelFile.data();
        const size_t size   = modelFile.size();
        size_t       offset = 0;

        uint32_t stageCount = 0;
        if ( !ReadU32( data, size, offset, stageCount ) )
        {
            errorOut = MakeError( ProcessingErrorStage::RESOURCE_RESOLUTION,
                                  "ParseCompiledStages: truncated buffer reading stage_count" );
            return false;
        }

        outStages.reserve( stageCount );
        for ( uint32_t i = 0; i < stageCount; ++i )
        {
            ParsedStage stage;

            uint32_t stageTag = 0;
            if ( !ReadU32( data, size, offset, stageTag ) )
            {
                errorOut = MakeError( ProcessingErrorStage::RESOURCE_RESOLUTION,
                                      "ParseCompiledStages: truncated buffer reading stage_tag" );
                return false;
            }
            stage.stage = static_cast<sgns::Stage>( stageTag );

            uint32_t entryPointLen = 0;
            if ( !ReadU32( data, size, offset, entryPointLen ) )
            {
                errorOut = MakeError( ProcessingErrorStage::RESOURCE_RESOLUTION,
                                      "ParseCompiledStages: truncated buffer reading entry_point_len" );
                return false;
            }
            if ( entryPointLen > 0 )
            {
                const char *bytes = nullptr;
                if ( !ReadBytes( data, size, offset, entryPointLen, bytes ) )
                {
                    errorOut = MakeError( ProcessingErrorStage::RESOURCE_RESOLUTION,
                                          "ParseCompiledStages: truncated buffer reading entry_point bytes" );
                    return false;
                }
                stage.entry_point.assign( bytes, entryPointLen );
            }

            uint32_t wordCount = 0;
            if ( !ReadU32( data, size, offset, wordCount ) )
            {
                errorOut = MakeError( ProcessingErrorStage::RESOURCE_RESOLUTION,
                                      "ParseCompiledStages: truncated buffer reading word_count" );
                return false;
            }
            if ( wordCount > 0 )
            {
                size_t       byteCount = static_cast<size_t>( wordCount ) * sizeof( uint32_t );
                const char  *bytes     = nullptr;
                if ( !ReadBytes( data, size, offset, byteCount, bytes ) )
                {
                    errorOut = MakeError( ProcessingErrorStage::RESOURCE_RESOLUTION,
                                          "ParseCompiledStages: truncated buffer reading spirv_words" );
                    return false;
                }
                stage.spirv.resize( wordCount );
                std::memcpy( stage.spirv.data(), bytes, byteCount );
            }

            outStages.push_back( std::move( stage ) );
        }

        return true;
    }

    bool RenderProcessor::ParseRenderPassConfig(
        const std::vector<char>                                            &imageData,
        sgns::RenderTarget                                                 &outTarget,
        boost::optional<sgns::PipelineState>                               &outPipelineState,
        std::vector<sgns::VertexLayoutEntry>                               &outVertexLayout,
        boost::optional<std::map<std::string, sgns::RenderShaderUniform>>  &outUniforms,
        std::vector<uint8_t>                                               &outVertexBytes,
        bool                                                                &outHasIndex,
        sgns::IndexType                                                    &outIndexType,
        std::vector<uint8_t>                                               &outIndexBytes,
        uint32_t                                                           &outDataTransformCount,
        ProcessingResult                                                   &errorOut )
    // Function-try-block: several generated setters below (set_width/set_height/
    // set_clear_depth/set_offset, etc.) enforce schema-level constraints and throw
    // on violation. A malformed/truncated wire-format buffer must never crash the
    // process -- convert any such exception into a structured RESOURCE_RESOLUTION
    // error instead, per this task's "no crash/UB on out-of-bounds/malformed data"
    // requirement.
    try
    {
        outPipelineState = boost::none;
        outVertexLayout.clear();
        outUniforms = boost::none;
        outVertexBytes.clear();
        outHasIndex = false;
        outIndexBytes.clear();
        outDataTransformCount = 0;

        const char  *data   = imageData.data();
        const size_t size   = imageData.size();
        size_t       offset = 0;

        auto fail = [&]( const std::string &message ) -> bool
        {
            errorOut = MakeError( ProcessingErrorStage::RESOURCE_RESOLUTION, message );
            return false;
        };

        uint32_t width = 0, height = 0, colorFormatTag = 0, depthFormatTag = 0;
        if ( !ReadU32( data, size, offset, width ) )
        {
            return fail( "ParseRenderPassConfig: truncated buffer reading width" );
        }
        if ( !ReadU32( data, size, offset, height ) )
        {
            return fail( "ParseRenderPassConfig: truncated buffer reading height" );
        }
        if ( !ReadU32( data, size, offset, colorFormatTag ) )
        {
            return fail( "ParseRenderPassConfig: truncated buffer reading color_format_tag" );
        }
        if ( !ReadU32( data, size, offset, depthFormatTag ) )
        {
            return fail( "ParseRenderPassConfig: truncated buffer reading depth_format_tag" );
        }

        outTarget.set_width( static_cast<int64_t>( width ) );
        outTarget.set_height( static_cast<int64_t>( height ) );
        outTarget.set_color_format( static_cast<sgns::ColorFormat>( colorFormatTag ) );
        outTarget.set_depth_format( static_cast<sgns::DepthFormat>( depthFormatTag ) );

        std::vector<double> clearColor( 4, 0.0 );
        for ( size_t i = 0; i < 4; ++i )
        {
            float v = 0.0f;
            if ( !ReadF32( data, size, offset, v ) )
            {
                return fail( "ParseRenderPassConfig: truncated buffer reading clear_color" );
            }
            clearColor[i] = static_cast<double>( v );
        }
        outTarget.set_clear_color( clearColor );

        float clearDepth = 0.0f;
        if ( !ReadF32( data, size, offset, clearDepth ) )
        {
            return fail( "ParseRenderPassConfig: truncated buffer reading clear_depth" );
        }
        outTarget.set_clear_depth( static_cast<double>( clearDepth ) );

        uint8_t hasPipelineState = 0;
        if ( !ReadU8( data, size, offset, hasPipelineState ) )
        {
            return fail( "ParseRenderPassConfig: truncated buffer reading has_pipeline_state" );
        }
        if ( hasPipelineState )
        {
            sgns::PipelineState ps;

            uint8_t hasTopology = 0;
            if ( !ReadU8( data, size, offset, hasTopology ) )
            {
                return fail( "ParseRenderPassConfig: truncated buffer reading has_topology" );
            }
            if ( hasTopology )
            {
                uint32_t tag = 0;
                if ( !ReadU32( data, size, offset, tag ) )
                {
                    return fail( "ParseRenderPassConfig: truncated buffer reading topology_tag" );
                }
                ps.set_topology( static_cast<sgns::Topology>( tag ) );
            }

            uint8_t hasCullMode = 0;
            if ( !ReadU8( data, size, offset, hasCullMode ) )
            {
                return fail( "ParseRenderPassConfig: truncated buffer reading has_cull_mode" );
            }
            if ( hasCullMode )
            {
                uint32_t tag = 0;
                if ( !ReadU32( data, size, offset, tag ) )
                {
                    return fail( "ParseRenderPassConfig: truncated buffer reading cull_mode_tag" );
                }
                ps.set_cull_mode( static_cast<sgns::CullMode>( tag ) );
            }

            uint8_t hasFrontFace = 0;
            if ( !ReadU8( data, size, offset, hasFrontFace ) )
            {
                return fail( "ParseRenderPassConfig: truncated buffer reading has_front_face" );
            }
            if ( hasFrontFace )
            {
                uint32_t tag = 0;
                if ( !ReadU32( data, size, offset, tag ) )
                {
                    return fail( "ParseRenderPassConfig: truncated buffer reading front_face_tag" );
                }
                ps.set_front_face( static_cast<sgns::FrontFace>( tag ) );
            }

            uint8_t hasDepthTest = 0;
            if ( !ReadU8( data, size, offset, hasDepthTest ) )
            {
                return fail( "ParseRenderPassConfig: truncated buffer reading has_depth_test" );
            }
            if ( hasDepthTest )
            {
                uint32_t tag = 0;
                if ( !ReadU32( data, size, offset, tag ) )
                {
                    return fail( "ParseRenderPassConfig: truncated buffer reading depth_test_tag" );
                }
                ps.set_depth_test( static_cast<sgns::DepthTest>( tag ) );
            }

            outPipelineState = ps;
        }

        uint32_t vertexLayoutCount = 0;
        if ( !ReadU32( data, size, offset, vertexLayoutCount ) )
        {
            return fail( "ParseRenderPassConfig: truncated buffer reading vertex_layout_count" );
        }
        outVertexLayout.reserve( vertexLayoutCount );
        for ( uint32_t i = 0; i < vertexLayoutCount; ++i )
        {
            std::string name;
            if ( !ReadString( data, size, offset, name ) )
            {
                return fail( "ParseRenderPassConfig: truncated buffer reading vertex_layout name" );
            }
            uint32_t formatTag = 0;
            if ( !ReadU32( data, size, offset, formatTag ) )
            {
                return fail( "ParseRenderPassConfig: truncated buffer reading vertex_layout format_tag" );
            }
            uint32_t entryOffset = 0;
            if ( !ReadU32( data, size, offset, entryOffset ) )
            {
                return fail( "ParseRenderPassConfig: truncated buffer reading vertex_layout offset" );
            }

            sgns::VertexLayoutEntry entry;
            entry.set_name( name );
            entry.set_format( static_cast<sgns::VertexLayoutFormat>( formatTag ) );
            entry.set_offset( static_cast<int64_t>( entryOffset ) );
            outVertexLayout.push_back( std::move( entry ) );
        }

        uint8_t hasUniforms = 0;
        if ( !ReadU8( data, size, offset, hasUniforms ) )
        {
            return fail( "ParseRenderPassConfig: truncated buffer reading has_uniforms" );
        }
        if ( hasUniforms )
        {
            uint32_t uniformCount = 0;
            if ( !ReadU32( data, size, offset, uniformCount ) )
            {
                return fail( "ParseRenderPassConfig: truncated buffer reading uniform_count" );
            }

            std::map<std::string, sgns::RenderShaderUniform> uniformMap;
            for ( uint32_t i = 0; i < uniformCount; ++i )
            {
                std::string name;
                if ( !ReadString( data, size, offset, name ) )
                {
                    return fail( "ParseRenderPassConfig: truncated buffer reading uniform name" );
                }

                sgns::RenderShaderUniform uniform;

                uint8_t hasSource = 0;
                if ( !ReadU8( data, size, offset, hasSource ) )
                {
                    return fail( "ParseRenderPassConfig: truncated buffer reading uniform has_source" );
                }
                if ( hasSource )
                {
                    std::string source;
                    if ( !ReadString( data, size, offset, source ) )
                    {
                        return fail( "ParseRenderPassConfig: truncated buffer reading uniform source" );
                    }
                    uniform.set_source( source );
                }

                uint8_t hasType = 0;
                if ( !ReadU8( data, size, offset, hasType ) )
                {
                    return fail( "ParseRenderPassConfig: truncated buffer reading uniform has_type" );
                }
                if ( hasType )
                {
                    uint32_t typeTag = 0;
                    if ( !ReadU32( data, size, offset, typeTag ) )
                    {
                        return fail( "ParseRenderPassConfig: truncated buffer reading uniform type_tag" );
                    }
                    uniform.set_type( static_cast<sgns::DataType>( typeTag ) );
                }

                std::string valueJson;
                if ( !ReadString( data, size, offset, valueJson ) )
                {
                    return fail( "ParseRenderPassConfig: truncated buffer reading uniform value" );
                }
                if ( !valueJson.empty() )
                {
                    try
                    {
                        uniform.set_value( nlohmann::json::parse( valueJson ) );
                    }
                    catch ( const std::exception &e )
                    {
                        return fail( std::string( "ParseRenderPassConfig: malformed uniform value JSON: " ) +
                                     e.what() );
                    }
                }

                uniformMap[name] = std::move( uniform );
            }

            outUniforms = std::move( uniformMap );
        }

        uint32_t vertexLen = 0;
        if ( !ReadU32( data, size, offset, vertexLen ) )
        {
            return fail( "ParseRenderPassConfig: truncated buffer reading vertex_len" );
        }
        if ( vertexLen > 0 )
        {
            const char *bytes = nullptr;
            if ( !ReadBytes( data, size, offset, vertexLen, bytes ) )
            {
                return fail( "ParseRenderPassConfig: truncated buffer reading vertex bytes" );
            }
            outVertexBytes.assign( bytes, bytes + vertexLen );
        }

        uint8_t hasIndex = 0;
        if ( !ReadU8( data, size, offset, hasIndex ) )
        {
            return fail( "ParseRenderPassConfig: truncated buffer reading has_index" );
        }
        if ( hasIndex )
        {
            uint32_t indexTypeTag = 0;
            if ( !ReadU32( data, size, offset, indexTypeTag ) )
            {
                return fail( "ParseRenderPassConfig: truncated buffer reading index_type_tag" );
            }
            uint32_t indexLen = 0;
            if ( !ReadU32( data, size, offset, indexLen ) )
            {
                return fail( "ParseRenderPassConfig: truncated buffer reading index_len" );
            }
            if ( indexLen > 0 )
            {
                const char *bytes = nullptr;
                if ( !ReadBytes( data, size, offset, indexLen, bytes ) )
                {
                    return fail( "ParseRenderPassConfig: truncated buffer reading index bytes" );
                }
                outIndexBytes.assign( bytes, bytes + indexLen );
            }
            outHasIndex  = true;
            outIndexType = static_cast<sgns::IndexType>( indexTypeTag );
        }
        else
        {
            outHasIndex = false;
        }

        uint32_t dataTransformCount = 0;
        if ( !ReadU32( data, size, offset, dataTransformCount ) )
        {
            return fail( "ParseRenderPassConfig: truncated buffer reading data_transform_count" );
        }
        outDataTransformCount = dataTransformCount;

        return true;
    }
    catch ( const std::exception &e )
    {
        errorOut = MakeError( ProcessingErrorStage::RESOURCE_RESOLUTION,
                              std::string( "ParseRenderPassConfig: exception while parsing: " ) + e.what() );
        return false;
    }

    namespace
    {
        /// Appends `value` to `bytes`, then pads `bytes` up to the next
        /// 16-byte-aligned boundary (this plan's std430-avoidance strategy --
        /// see 03-03-PLAN.md's objective). Every uniform's packed region gets
        /// its own 16-byte-aligned slot regardless of its natural size.
        void AppendPadded16( std::vector<uint8_t> &bytes, const uint8_t *data, size_t size )
        {
            bytes.insert( bytes.end(), data, data + size );
            size_t remainder = bytes.size() % 16;
            if ( remainder != 0 )
            {
                bytes.resize( bytes.size() + ( 16 - remainder ), 0 );
            }
        }

        /// Converts a resolved nlohmann::json uniform value into raw bytes
        /// according to its declared DataType. Returns false (never
        /// crashes/UB) for a DataType this phase does not support as a
        /// uniform (STRING/TENSOR/TEXTURE*/BUFFER).
        bool PackUniformValue( sgns::DataType dataType, const nlohmann::json &value, std::vector<uint8_t> &out )
        {
            auto appendFloat = [&out]( double v )
            {
                float           f     = static_cast<float>( v );
                const uint8_t  *bytes = reinterpret_cast<const uint8_t *>( &f );
                out.insert( out.end(), bytes, bytes + sizeof( float ) );
            };

            // Reads up to `count` numeric components from a JSON array (missing/
            // absent entries default to 0.0) -- never throws on a short/malformed
            // array; a value that isn't an array at all yields an all-zero vector.
            auto readVec = []( const nlohmann::json &v, size_t componentCount ) -> std::vector<double>
            {
                std::vector<double> result( componentCount, 0.0 );
                if ( v.is_array() )
                {
                    for ( size_t i = 0; i < componentCount && i < v.size(); ++i )
                    {
                        if ( v[i].is_number() )
                        {
                            result[i] = v[i].get<double>();
                        }
                    }
                }
                return result;
            };

            try
            {
                switch ( dataType )
                {
                    case sgns::DataType::FLOAT:
                    {
                        appendFloat( value.is_number() ? value.get<double>() : 0.0 );
                        return true;
                    }
                    case sgns::DataType::INT:
                    {
                        int32_t        i     = value.is_number() ? static_cast<int32_t>( value.get<int64_t>() ) : 0;
                        const uint8_t *bytes = reinterpret_cast<const uint8_t *>( &i );
                        out.insert( out.end(), bytes, bytes + sizeof( int32_t ) );
                        return true;
                    }
                    case sgns::DataType::BOOL:
                    {
                        int32_t        b     = ( value.is_boolean() && value.get<bool>() ) ? 1 : 0;
                        const uint8_t *bytes = reinterpret_cast<const uint8_t *>( &b );
                        out.insert( out.end(), bytes, bytes + sizeof( int32_t ) );
                        return true;
                    }
                    case sgns::DataType::VEC2:
                    {
                        for ( double d : readVec( value, 2 ) )
                        {
                            appendFloat( d );
                        }
                        return true;
                    }
                    case sgns::DataType::VEC3:
                    {
                        for ( double d : readVec( value, 3 ) )
                        {
                            appendFloat( d );
                        }
                        return true;
                    }
                    case sgns::DataType::VEC4:
                    {
                        for ( double d : readVec( value, 4 ) )
                        {
                            appendFloat( d );
                        }
                        return true;
                    }
                    case sgns::DataType::MAT2:
                    {
                        for ( double d : readVec( value, 4 ) )
                        {
                            appendFloat( d );
                        }
                        return true;
                    }
                    case sgns::DataType::MAT3:
                    {
                        for ( double d : readVec( value, 9 ) )
                        {
                            appendFloat( d );
                        }
                        return true;
                    }
                    case sgns::DataType::MAT4:
                    {
                        for ( double d : readVec( value, 16 ) )
                        {
                            appendFloat( d );
                        }
                        return true;
                    }
                    case sgns::DataType::STRING:
                    case sgns::DataType::TENSOR:
                    case sgns::DataType::TEXTURE1_D:
                    case sgns::DataType::TEXTURE2_D:
                    case sgns::DataType::TEXTURE3_D:
                    case sgns::DataType::TEXTURE_CUBE:
                    case sgns::DataType::BUFFER:
                    default:
                        return false;
                }
            }
            catch ( const std::exception & )
            {
                return false;
            }
        }
    }

    bool RenderProcessor::ResolveUniforms(
        const boost::optional<std::map<std::string, sgns::RenderShaderUniform>> &uniforms,
        const std::vector<sgns::Parameter>                                      *parameters,
        ResolvedUniforms                                                        &outResolved,
        ProcessingResult                                                        &errorOut )
    {
        outResolved.packedBytes.clear();
        outResolved.pushConstant = true;

        if ( !uniforms )
        {
            return true;
        }

        // std::map's natural key-sorted iteration order -- deterministic,
        // satisfies DETV-01, matches SerializeRenderPassConfig()'s own
        // iteration order (03-02-SUMMARY.md).
        for ( const auto &entry : uniforms.value() )
        {
            const std::string               &name    = entry.first;
            const sgns::RenderShaderUniform &uniform = entry.second;

            nlohmann::json resolvedValue;

            if ( uniform.get_source() )
            {
                const std::string &source = uniform.get_source().value();
                static const std::string kParameterPrefix = "parameter:";
                if ( source.rfind( kParameterPrefix, 0 ) != 0 )
                {
                    errorOut = MakeError( ProcessingErrorStage::RESOURCE_RESOLUTION,
                                          "ResolveUniforms: unsupported uniform source prefix for '" + name + "'" );
                    return false;
                }
                std::string paramName = source.substr( kParameterPrefix.size() );

                const sgns::Parameter *found = nullptr;
                if ( parameters )
                {
                    for ( const auto &param : *parameters )
                    {
                        if ( param.get_name() == paramName )
                        {
                            found = &param;
                            break;
                        }
                    }
                }
                if ( !found )
                {
                    errorOut = MakeError( ProcessingErrorStage::RESOURCE_RESOLUTION,
                                          "ResolveUniforms: unresolvable parameter '" + paramName +
                                              "' for uniform '" + name + "'" );
                    return false;
                }
                resolvedValue = found->get_parameter_default();
            }
            else
            {
                resolvedValue = uniform.get_value();
            }

            if ( !uniform.get_type() )
            {
                errorOut = MakeError( ProcessingErrorStage::RESOURCE_RESOLUTION,
                                      "ResolveUniforms: uniform '" + name + "' has no declared DataType" );
                return false;
            }

            std::vector<uint8_t> packed;
            if ( !PackUniformValue( uniform.get_type().value(), resolvedValue, packed ) )
            {
                errorOut = MakeError( ProcessingErrorStage::RESOURCE_RESOLUTION,
                                      "ResolveUniforms: unsupported DataType for uniform '" + name + "'" );
                return false;
            }

            AppendPadded16( outResolved.packedBytes, packed.data(), packed.size() );
        }

        outResolved.pushConstant = ( outResolved.packedBytes.size() <= 128 );

        return true;
    }

    bool RenderProcessor::CheckFormatSupport( VkFormat                format,
                                               VkFormatFeatureFlagBits requiredFeature,
                                               ProcessingResult        &errorOut )
    {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties( m_physicalDevice, format, &props );

        if ( !( props.optimalTilingFeatures & requiredFeature ) )
        {
            errorOut = MakeError( ProcessingErrorStage::FORMAT_UNSUPPORTED,
                                  "CheckFormatSupport: VkFormat " + std::to_string( static_cast<int>( format ) ) +
                                      " does not support required feature " +
                                      std::to_string( static_cast<uint32_t>( requiredFeature ) ) +
                                      " for optimal tiling" );
            return false;
        }

        return true;
    }

    namespace
    {
        /// Linear scan over VkPhysicalDeviceMemoryProperties::memoryTypes for
        /// an index whose bit is set in `typeBits` and whose propertyFlags
        /// contain all of `properties` -- mirrors LargestDeviceLocalHeap's
        /// existing enumeration style.
        bool FindMemoryTypeIndex( const VkPhysicalDeviceMemoryProperties &memProps,
                                   uint32_t                                typeBits,
                                   VkMemoryPropertyFlags                  properties,
                                   uint32_t                               &outIndex )
        {
            for ( uint32_t i = 0; i < memProps.memoryTypeCount; ++i )
            {
                if ( ( typeBits & ( 1u << i ) ) &&
                     ( memProps.memoryTypes[i].propertyFlags & properties ) == properties )
                {
                    outIndex = i;
                    return true;
                }
            }
            return false;
        }
    }

    bool RenderProcessor::CreateBufferDedicated( VkDeviceSize          size,
                                                  VkBufferUsageFlags    usage,
                                                  VkMemoryPropertyFlags properties,
                                                  VkBuffer              &outBuffer,
                                                  VkDeviceMemory        &outMemory,
                                                  ProcessingResult      &errorOut )
    {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size        = size;
        bufferInfo.usage       = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buffer = VK_NULL_HANDLE;
        VkResult result = vkCreateBuffer( m_device, &bufferInfo, nullptr, &buffer );
        if ( result != VK_SUCCESS )
        {
            errorOut = MakeError( ProcessingErrorStage::BUFFER_ALLOCATION,
                                  "vkCreateBuffer failed: VkResult=" + std::to_string( result ) );
            return false;
        }

        VkMemoryRequirements memRequirements{};
        vkGetBufferMemoryRequirements( m_device, buffer, &memRequirements );

        VkPhysicalDeviceMemoryProperties memProps{};
        vkGetPhysicalDeviceMemoryProperties( m_physicalDevice, &memProps );

        uint32_t memTypeIndex = 0;
        if ( !FindMemoryTypeIndex( memProps, memRequirements.memoryTypeBits, properties, memTypeIndex ) )
        {
            vkDestroyBuffer( m_device, buffer, nullptr );
            errorOut = MakeError( ProcessingErrorStage::BUFFER_ALLOCATION,
                                  "CreateBufferDedicated: no suitable memory type found" );
            return false;
        }

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize  = memRequirements.size;
        allocInfo.memoryTypeIndex = memTypeIndex;

        VkDeviceMemory memory = VK_NULL_HANDLE;
        result                = vkAllocateMemory( m_device, &allocInfo, nullptr, &memory );
        if ( result != VK_SUCCESS )
        {
            vkDestroyBuffer( m_device, buffer, nullptr );
            errorOut = MakeError( ProcessingErrorStage::BUFFER_ALLOCATION,
                                  "CreateBufferDedicated: dedicated memory allocation failed: VkResult=" +
                                      std::to_string( result ) );
            return false;
        }

        result = vkBindBufferMemory( m_device, buffer, memory, 0 );
        if ( result != VK_SUCCESS )
        {
            vkFreeMemory( m_device, memory, nullptr );
            vkDestroyBuffer( m_device, buffer, nullptr );
            errorOut = MakeError( ProcessingErrorStage::BUFFER_ALLOCATION,
                                  "vkBindBufferMemory failed: VkResult=" + std::to_string( result ) );
            return false;
        }

        outBuffer = buffer;
        outMemory = memory;

        VkDevice device = m_device;
        PushTeardown( [device, buffer, memory]() {
            vkDestroyBuffer( device, buffer, nullptr );
            vkFreeMemory( device, memory, nullptr );
        } );

        return true;
    }

    bool RenderProcessor::CreateImageDedicated( const VkImageCreateInfo &imageInfo,
                                                 VkMemoryPropertyFlags   properties,
                                                 VkImage                 &outImage,
                                                 VkDeviceMemory          &outMemory,
                                                 ProcessingResult        &errorOut )
    {
        VkImage  image  = VK_NULL_HANDLE;
        VkResult result = vkCreateImage( m_device, &imageInfo, nullptr, &image );
        if ( result != VK_SUCCESS )
        {
            errorOut = MakeError( ProcessingErrorStage::IMAGE_ALLOCATION,
                                  "vkCreateImage failed: VkResult=" + std::to_string( result ) );
            return false;
        }

        VkMemoryRequirements memRequirements{};
        vkGetImageMemoryRequirements( m_device, image, &memRequirements );

        VkPhysicalDeviceMemoryProperties memProps{};
        vkGetPhysicalDeviceMemoryProperties( m_physicalDevice, &memProps );

        uint32_t memTypeIndex = 0;
        if ( !FindMemoryTypeIndex( memProps, memRequirements.memoryTypeBits, properties, memTypeIndex ) )
        {
            vkDestroyImage( m_device, image, nullptr );
            errorOut = MakeError( ProcessingErrorStage::IMAGE_ALLOCATION,
                                  "CreateImageDedicated: no suitable memory type found" );
            return false;
        }

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize  = memRequirements.size;
        allocInfo.memoryTypeIndex = memTypeIndex;

        VkDeviceMemory memory = VK_NULL_HANDLE;
        result                = vkAllocateMemory( m_device, &allocInfo, nullptr, &memory );
        if ( result != VK_SUCCESS )
        {
            vkDestroyImage( m_device, image, nullptr );
            errorOut = MakeError( ProcessingErrorStage::IMAGE_ALLOCATION,
                                  "CreateImageDedicated: dedicated memory allocation failed: VkResult=" +
                                      std::to_string( result ) );
            return false;
        }

        result = vkBindImageMemory( m_device, image, memory, 0 );
        if ( result != VK_SUCCESS )
        {
            vkFreeMemory( m_device, memory, nullptr );
            vkDestroyImage( m_device, image, nullptr );
            errorOut = MakeError( ProcessingErrorStage::IMAGE_ALLOCATION,
                                  "vkBindImageMemory failed: VkResult=" + std::to_string( result ) );
            return false;
        }

        outImage  = image;
        outMemory = memory;

        VkDevice device = m_device;
        PushTeardown( [device, image, memory]() {
            vkDestroyImage( device, image, nullptr );
            vkFreeMemory( device, memory, nullptr );
        } );

        return true;
    }

    VkFormat RenderProcessor::ToVkFormat( sgns::ColorFormat fmt )
    {
        switch ( fmt )
        {
            case sgns::ColorFormat::RGBA8:
                return VK_FORMAT_R8G8B8A8_UNORM;
            case sgns::ColorFormat::RGB8:
                return VK_FORMAT_R8G8B8_UNORM;
        }
        return VK_FORMAT_R8G8B8A8_UNORM;
    }

    VkFormat RenderProcessor::ToVkFormat( sgns::DepthFormat fmt )
    {
        switch ( fmt )
        {
            case sgns::DepthFormat::D32_SFLOAT:
                return VK_FORMAT_D32_SFLOAT;
            case sgns::DepthFormat::D24_UNORM_S8_UINT:
                return VK_FORMAT_D24_UNORM_S8_UINT;
        }
        return VK_FORMAT_D32_SFLOAT;
    }

    bool RenderProcessor::BuildRenderPass( const sgns::RenderTarget &target, ProcessingResult &errorOut )
    {
        if ( target.get_width() < 1 || target.get_width() > static_cast<int64_t>( kMaxRenderDimension ) ||
             target.get_height() < 1 || target.get_height() > static_cast<int64_t>( kMaxRenderDimension ) )
        {
            errorOut = MakeError( ProcessingErrorStage::IMAGE_ALLOCATION,
                                  "BuildRenderPass: render_target width/height out of bounds" );
            return false;
        }

        VkFormat colorFormat = ToVkFormat( target.get_color_format() );
        VkFormat depthFormat = ToVkFormat( target.get_depth_format() );

        if ( !CheckFormatSupport( colorFormat, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT, errorOut ) )
        {
            return false;
        }
        if ( !CheckFormatSupport( depthFormat, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT, errorOut ) )
        {
            return false;
        }

        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = colorFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = depthFormat;
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription attachments[2] = { colorAttachment, depthAttachment };

        VkAttachmentReference colorRef{};
        colorRef.attachment = 0;
        colorRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthRef{};
        depthRef.attachment = 1;
        depthRef.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount    = 1;
        subpass.pColorAttachments       = &colorRef;
        subpass.pDepthStencilAttachment = &depthRef;

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 2;
        renderPassInfo.pAttachments    = attachments;
        renderPassInfo.subpassCount    = 1;
        renderPassInfo.pSubpasses      = &subpass;

        VkRenderPass renderPass = VK_NULL_HANDLE;
        VkResult     result     = vkCreateRenderPass( m_device, &renderPassInfo, nullptr, &renderPass );
        if ( result != VK_SUCCESS )
        {
            errorOut = MakeError( ProcessingErrorStage::RENDER_PASS_CREATION,
                                  "vkCreateRenderPass failed: VkResult=" + std::to_string( result ) );
            return false;
        }

        m_renderPass   = renderPass;
        m_renderWidth  = static_cast<uint32_t>( target.get_width() );
        m_renderHeight = static_cast<uint32_t>( target.get_height() );

        VkDevice device = m_device;
        PushTeardown( [device, renderPass]() { vkDestroyRenderPass( device, renderPass, nullptr ); } );

        return true;
    }

    bool RenderProcessor::BuildFramebuffer( const sgns::RenderTarget &target, ProcessingResult &errorOut )
    {
        uint32_t width  = static_cast<uint32_t>( target.get_width() );
        uint32_t height = static_cast<uint32_t>( target.get_height() );

        VkFormat colorFormat = ToVkFormat( target.get_color_format() );
        VkFormat depthFormat = ToVkFormat( target.get_depth_format() );

        VkImageCreateInfo colorImageInfo{};
        colorImageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        colorImageInfo.imageType     = VK_IMAGE_TYPE_2D;
        colorImageInfo.format        = colorFormat;
        colorImageInfo.extent        = { width, height, 1 };
        colorImageInfo.mipLevels     = 1;
        colorImageInfo.arrayLayers   = 1;
        colorImageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
        colorImageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
        colorImageInfo.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        colorImageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        colorImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        if ( !CreateImageDedicated( colorImageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_colorImage, m_colorMemory,
                                     errorOut ) )
        {
            return false;
        }

        VkImageCreateInfo depthImageInfo{};
        depthImageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        depthImageInfo.imageType     = VK_IMAGE_TYPE_2D;
        depthImageInfo.format        = depthFormat;
        depthImageInfo.extent        = { width, height, 1 };
        depthImageInfo.mipLevels     = 1;
        depthImageInfo.arrayLayers   = 1;
        depthImageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
        depthImageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
        depthImageInfo.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        depthImageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
        depthImageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        if ( !CreateImageDedicated( depthImageInfo, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_depthImage, m_depthMemory,
                                     errorOut ) )
        {
            return false;
        }

        VkImageViewCreateInfo colorViewInfo{};
        colorViewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        colorViewInfo.image                           = m_colorImage;
        colorViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        colorViewInfo.format                          = colorFormat;
        colorViewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        colorViewInfo.subresourceRange.baseMipLevel   = 0;
        colorViewInfo.subresourceRange.levelCount     = 1;
        colorViewInfo.subresourceRange.baseArrayLayer = 0;
        colorViewInfo.subresourceRange.layerCount     = 1;

        VkResult result = vkCreateImageView( m_device, &colorViewInfo, nullptr, &m_colorView );
        if ( result != VK_SUCCESS )
        {
            errorOut = MakeError( ProcessingErrorStage::IMAGE_ALLOCATION,
                                  "vkCreateImageView (color) failed: VkResult=" + std::to_string( result ) );
            return false;
        }
        {
            VkDevice    device = m_device;
            VkImageView view   = m_colorView;
            PushTeardown( [device, view]() { vkDestroyImageView( device, view, nullptr ); } );
        }

        // D24_UNORM_S8_UINT has a stencil component the schema never exposes/uses;
        // the image view's aspectMask must still include it when present, per
        // Vulkan's depth-stencil-attachment image-view rules.
        VkImageAspectFlags depthAspect = VK_IMAGE_ASPECT_DEPTH_BIT;
        if ( target.get_depth_format() == sgns::DepthFormat::D24_UNORM_S8_UINT )
        {
            depthAspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }

        VkImageViewCreateInfo depthViewInfo{};
        depthViewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        depthViewInfo.image                           = m_depthImage;
        depthViewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        depthViewInfo.format                          = depthFormat;
        depthViewInfo.subresourceRange.aspectMask     = depthAspect;
        depthViewInfo.subresourceRange.baseMipLevel   = 0;
        depthViewInfo.subresourceRange.levelCount     = 1;
        depthViewInfo.subresourceRange.baseArrayLayer = 0;
        depthViewInfo.subresourceRange.layerCount     = 1;

        result = vkCreateImageView( m_device, &depthViewInfo, nullptr, &m_depthView );
        if ( result != VK_SUCCESS )
        {
            errorOut = MakeError( ProcessingErrorStage::IMAGE_ALLOCATION,
                                  "vkCreateImageView (depth) failed: VkResult=" + std::to_string( result ) );
            return false;
        }
        {
            VkDevice    device = m_device;
            VkImageView view   = m_depthView;
            PushTeardown( [device, view]() { vkDestroyImageView( device, view, nullptr ); } );
        }

        VkImageView attachments[2] = { m_colorView, m_depthView };

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass      = m_renderPass;
        framebufferInfo.attachmentCount = 2;
        framebufferInfo.pAttachments    = attachments;
        framebufferInfo.width           = width;
        framebufferInfo.height          = height;
        framebufferInfo.layers          = 1;

        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        result                    = vkCreateFramebuffer( m_device, &framebufferInfo, nullptr, &framebuffer );
        if ( result != VK_SUCCESS )
        {
            errorOut = MakeError( ProcessingErrorStage::IMAGE_ALLOCATION,
                                  "vkCreateFramebuffer failed: VkResult=" + std::to_string( result ) );
            return false;
        }

        m_framebuffer = framebuffer;

        VkDevice device = m_device;
        PushTeardown( [device, framebuffer]() { vkDestroyFramebuffer( device, framebuffer, nullptr ); } );

        return true;
    }

    VkFormat RenderProcessor::ToVkFormat( sgns::VertexLayoutFormat fmt )
    {
        switch ( fmt )
        {
            case sgns::VertexLayoutFormat::FLOAT32:
                return VK_FORMAT_R32_SFLOAT;
            case sgns::VertexLayoutFormat::FLOAT16:
                return VK_FORMAT_R16_SFLOAT;
            case sgns::VertexLayoutFormat::INT32:
                return VK_FORMAT_R32_SINT;
        }
        return VK_FORMAT_R32_SFLOAT;
    }

    VkPrimitiveTopology RenderProcessor::ToVkTopology( sgns::Topology t )
    {
        switch ( t )
        {
            case sgns::Topology::TRIANGLE_LIST:
                return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            case sgns::Topology::LINE_LIST:
                return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
            case sgns::Topology::POINT_LIST:
                return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        }
        return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    }

    VkCullModeFlags RenderProcessor::ToVkCullMode( sgns::CullMode c )
    {
        switch ( c )
        {
            case sgns::CullMode::NONE:
                return VK_CULL_MODE_NONE;
            case sgns::CullMode::FRONT:
                return VK_CULL_MODE_FRONT_BIT;
            case sgns::CullMode::BACK:
                return VK_CULL_MODE_BACK_BIT;
        }
        return VK_CULL_MODE_BACK_BIT;
    }

    VkFrontFace RenderProcessor::ToVkFrontFace( sgns::FrontFace f )
    {
        switch ( f )
        {
            case sgns::FrontFace::CCW:
                return VK_FRONT_FACE_COUNTER_CLOCKWISE;
            case sgns::FrontFace::CW:
                return VK_FRONT_FACE_CLOCKWISE;
        }
        return VK_FRONT_FACE_COUNTER_CLOCKWISE;
    }

    VkBool32 RenderProcessor::ToVkBool( sgns::DepthTest d )
    {
        return ( d == sgns::DepthTest::ENABLED ) ? VK_TRUE : VK_FALSE;
    }

    uint32_t RenderProcessor::VertexFormatByteSize( sgns::VertexLayoutFormat f )
    {
        switch ( f )
        {
            case sgns::VertexLayoutFormat::FLOAT32:
                return 4;
            case sgns::VertexLayoutFormat::INT32:
                return 4;
            case sgns::VertexLayoutFormat::FLOAT16:
                return 2;
        }
        return 4;
    }

    bool RenderProcessor::BuildPipeline( const std::vector<ParsedStage>             &stages,
                                          const std::vector<sgns::VertexLayoutEntry> &vertexLayout,
                                          const boost::optional<sgns::PipelineState> &pipelineState,
                                          const ResolvedUniforms                     &uniforms,
                                          ProcessingResult                           &errorOut )
    {
        std::vector<VkPipelineShaderStageCreateInfo> shaderStages;
        shaderStages.reserve( stages.size() );

        for ( const auto &s : stages )
        {
            VkShaderModuleCreateInfo moduleInfo{};
            moduleInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            moduleInfo.codeSize = s.spirv.size() * sizeof( uint32_t );
            moduleInfo.pCode    = s.spirv.data();

            VkShaderModule module = VK_NULL_HANDLE;
            VkResult       result = vkCreateShaderModule( m_device, &moduleInfo, nullptr, &module );
            if ( result != VK_SUCCESS )
            {
                errorOut = MakeError( ProcessingErrorStage::SHADER_MODULE_CREATION,
                                      "vkCreateShaderModule failed: VkResult=" + std::to_string( result ) );
                return false;
            }

            VkDevice device = m_device;
            PushTeardown( [device, module]() { vkDestroyShaderModule( device, module, nullptr ); } );

            VkPipelineShaderStageCreateInfo stageInfo{};
            stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stageInfo.stage = ( s.stage == sgns::Stage::VERTEX ) ? VK_SHADER_STAGE_VERTEX_BIT
                                                                  : VK_SHADER_STAGE_FRAGMENT_BIT;
            stageInfo.module = module;
            stageInfo.pName = s.entry_point.c_str();
            shaderStages.push_back( stageInfo );
        }

        uint32_t stride = 0;
        for ( const auto &entry : vertexLayout )
        {
            stride += VertexFormatByteSize( entry.get_format() );
        }

        VkVertexInputBindingDescription bindingDesc{};
        bindingDesc.binding   = 0;
        bindingDesc.stride    = stride;
        bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        std::vector<VkVertexInputAttributeDescription> attributeDescs;
        attributeDescs.reserve( vertexLayout.size() );
        for ( size_t i = 0; i < vertexLayout.size(); ++i )
        {
            VkVertexInputAttributeDescription attr{};
            attr.location = static_cast<uint32_t>( i );
            attr.binding  = 0;
            attr.format   = ToVkFormat( vertexLayout[i].get_format() );
            attr.offset   = static_cast<uint32_t>( vertexLayout[i].get_offset() );
            attributeDescs.push_back( attr );
        }

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount   = vertexLayout.empty() ? 0 : 1;
        vertexInputInfo.pVertexBindingDescriptions      = vertexLayout.empty() ? nullptr : &bindingDesc;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>( attributeDescs.size() );
        vertexInputInfo.pVertexAttributeDescriptions    = attributeDescs.empty() ? nullptr : attributeDescs.data();

        sgns::Topology  topology  = sgns::Topology::TRIANGLE_LIST;
        sgns::CullMode  cullMode  = sgns::CullMode::BACK;
        sgns::FrontFace frontFace = sgns::FrontFace::CCW;
        sgns::DepthTest depthTest = sgns::DepthTest::ENABLED;
        if ( pipelineState )
        {
            if ( pipelineState->get_topology() )
            {
                topology = pipelineState->get_topology().value();
            }
            if ( pipelineState->get_cull_mode() )
            {
                cullMode = pipelineState->get_cull_mode().value();
            }
            if ( pipelineState->get_front_face() )
            {
                frontFace = pipelineState->get_front_face().value();
            }
            if ( pipelineState->get_depth_test() )
            {
                depthTest = pipelineState->get_depth_test().value();
            }
        }

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology               = ToVkTopology( topology );
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.cullMode    = ToVkCullMode( cullMode );
        rasterizer.frontFace   = ToVkFrontFace( frontFace );
        rasterizer.lineWidth   = 1.0f;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable  = ToVkBool( depthTest );
        depthStencil.depthWriteEnable = depthStencil.depthTestEnable; // [ASSUMED] tied to depthTestEnable -- no
                                                                       // separate schema field exists (RESEARCH.md A1)
        depthStencil.depthCompareOp   = VK_COMPARE_OP_LESS;           // fixed per D-14, never schema-configurable

        VkPipelineMultisampleStateCreateInfo multisample{};
        multisample.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT; // ALWAYS -- DETV-02, never configurable

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments    = &colorBlendAttachment;

        // Fixed (never a runtime-settable pipeline attribute, per D-22) viewport/
        // scissor sized to BuildRenderPass()'s already-validated render target
        // dimensions.
        VkViewport viewport{};
        viewport.x        = 0.0f;
        viewport.y        = 0.0f;
        viewport.width    = static_cast<float>( m_renderWidth );
        viewport.height   = static_cast<float>( m_renderHeight );
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = { m_renderWidth, m_renderHeight };

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports    = &viewport;
        viewportState.scissorCount  = 1;
        viewportState.pScissors     = &scissor;

        // D-29/D-30: fixed 128-byte push-constant threshold, all-or-nothing.
        bool usePushConstant  = uniforms.pushConstant && !uniforms.packedBytes.empty();
        bool useDescriptorSet = !uniforms.pushConstant && !uniforms.packedBytes.empty();

        VkPushConstantRange pushConstantRange{};
        if ( usePushConstant )
        {
            pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            pushConstantRange.offset     = 0;
            pushConstantRange.size       = static_cast<uint32_t>( uniforms.packedBytes.size() );
        }

        VkResult result = VK_SUCCESS;

        if ( useDescriptorSet )
        {
            VkDescriptorSetLayoutBinding binding{};
            binding.binding         = 0;
            binding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            binding.descriptorCount = 1;
            binding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

            VkDescriptorSetLayoutCreateInfo layoutInfo{};
            layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layoutInfo.bindingCount = 1;
            layoutInfo.pBindings    = &binding;

            result = vkCreateDescriptorSetLayout( m_device, &layoutInfo, nullptr, &m_descriptorSetLayout );
            if ( result != VK_SUCCESS )
            {
                errorOut = MakeError( ProcessingErrorStage::PIPELINE_CREATION,
                                      "vkCreateDescriptorSetLayout failed: VkResult=" + std::to_string( result ) );
                return false;
            }
            {
                VkDevice              device = m_device;
                VkDescriptorSetLayout layout  = m_descriptorSetLayout;
                PushTeardown( [device, layout]() { vkDestroyDescriptorSetLayout( device, layout, nullptr ); } );
            }

            VkDescriptorPoolSize poolSize{};
            poolSize.type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            poolSize.descriptorCount = 1;

            VkDescriptorPoolCreateInfo poolInfo{};
            poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.poolSizeCount = 1;
            poolInfo.pPoolSizes    = &poolSize;
            poolInfo.maxSets       = 1; // matches D-22's per-job-only lifetime

            result = vkCreateDescriptorPool( m_device, &poolInfo, nullptr, &m_descriptorPool );
            if ( result != VK_SUCCESS )
            {
                errorOut = MakeError( ProcessingErrorStage::PIPELINE_CREATION,
                                      "vkCreateDescriptorPool failed: VkResult=" + std::to_string( result ) );
                return false;
            }
            {
                VkDevice         device = m_device;
                VkDescriptorPool pool   = m_descriptorPool;
                PushTeardown( [device, pool]() { vkDestroyDescriptorPool( device, pool, nullptr ); } );
            }

            VkDescriptorSetAllocateInfo allocInfo{};
            allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            allocInfo.descriptorPool     = m_descriptorPool;
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts        = &m_descriptorSetLayout;

            result = vkAllocateDescriptorSets( m_device, &allocInfo, &m_descriptorSet );
            if ( result != VK_SUCCESS )
            {
                errorOut = MakeError( ProcessingErrorStage::PIPELINE_CREATION,
                                      "vkAllocateDescriptorSets failed: VkResult=" + std::to_string( result ) );
                return false;
            }
            // m_descriptorSet is freed automatically when m_descriptorPool is
            // destroyed -- no separate PushTeardown needed for the set itself.
        }

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        if ( usePushConstant )
        {
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges    = &pushConstantRange;
        }
        if ( useDescriptorSet )
        {
            pipelineLayoutInfo.setLayoutCount = 1;
            pipelineLayoutInfo.pSetLayouts    = &m_descriptorSetLayout;
        }
        // If uniforms.packedBytes is empty (no uniforms declared at all), neither
        // branch above ran -- pipelineLayoutInfo keeps zero push-constant ranges
        // and zero descriptor sets, exactly as required.

        result = vkCreatePipelineLayout( m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout );
        if ( result != VK_SUCCESS )
        {
            errorOut = MakeError( ProcessingErrorStage::PIPELINE_CREATION,
                                  "vkCreatePipelineLayout failed: VkResult=" + std::to_string( result ) );
            return false;
        }
        {
            VkDevice         device = m_device;
            VkPipelineLayout layout = m_pipelineLayout;
            PushTeardown( [device, layout]() { vkDestroyPipelineLayout( device, layout, nullptr ); } );
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount          = static_cast<uint32_t>( shaderStages.size() );
        pipelineInfo.pStages             = shaderStages.data();
        pipelineInfo.pVertexInputState   = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState      = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState   = &multisample;
        pipelineInfo.pDepthStencilState  = &depthStencil;
        pipelineInfo.pColorBlendState    = &colorBlending;
        pipelineInfo.layout              = m_pipelineLayout;
        pipelineInfo.renderPass          = m_renderPass;
        pipelineInfo.subpass             = 0;

        result = vkCreateGraphicsPipelines( m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline );
        if ( result != VK_SUCCESS )
        {
            errorOut = MakeError( ProcessingErrorStage::PIPELINE_CREATION,
                                  "vkCreateGraphicsPipelines failed: VkResult=" + std::to_string( result ) );
            return false;
        }
        {
            VkDevice   device   = m_device;
            VkPipeline pipeline = m_pipeline;
            PushTeardown( [device, pipeline]() { vkDestroyPipeline( device, pipeline, nullptr ); } );
        }

        return true;
    }

    ProcessingResult RenderProcessor::StartProcessing(
        std::vector<std::vector<uint8_t>> &chunkhashes,
        const sgns::IoDeclaration         &proc,
        std::vector<char>                 &imageData,
        std::vector<char>                 &modelFile,
        const std::vector<sgns::Parameter> *parameters )
    {
        (void)proc;
        (void)imageData;
        (void)modelFile;
        (void)parameters;

        if ( !InitializeContext() )
        {
            ProcessingResult result;
            result.hash = std::vector<uint8_t>( 32, 0 );
            return result;
        }

        ProcessingResult result;
        result.hash = std::vector<uint8_t>( 32, 0 );
        m_progress = 100.0f;
        return result;
    }

}
