#include "processors/processing_processor_render.hpp"
#include "processingbase/vulkan_init_guard.hpp"
#include <VkBootstrap.h>
#include <algorithm>
#include <cstring>
#include <mutex>

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
