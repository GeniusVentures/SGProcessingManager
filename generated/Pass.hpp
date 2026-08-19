//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     Pass.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

#include "DataTransform.hpp"
#include "IndexBuffer.hpp"
#include "PassIoBinding.hpp"
#include "ModelConfig.hpp"
#include "PipelineState.hpp"
#include "RenderShaderConfig.hpp"
#include "RenderTarget.hpp"
#include "ShaderConfig.hpp"
#include "VertexBuffer.hpp"
#include "VertexLayoutEntry.hpp"

namespace sgns {
    enum class PassType : int;
}

namespace sgns {
    using nlohmann::json;

    class Pass {
        public:
        Pass() :
            estimated_gpu_memory_bytes_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none),
            max_output_artifact_bytes_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none),
            name_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, std::string("^[a-zA-Z][a-zA-Z0-9_]*$")),
            per_pass_deadline_ms_constraint(boost::none, boost::none, boost::none, boost::none, boost::none, boost::none, boost::none)
        {}
        virtual ~Pass() = default;

        private:
        boost::optional<std::vector<DataTransform>> data_transforms;
        boost::optional<std::string> description;
        boost::optional<bool> enabled;
        boost::optional<int64_t> estimated_gpu_memory_bytes;
        ClassMemberConstraints estimated_gpu_memory_bytes_constraint;
        boost::optional<IndexBuffer> index_buffer;
        boost::optional<std::vector<PassIoBinding>> inputs;
        boost::optional<int64_t> max_output_artifact_bytes;
        ClassMemberConstraints max_output_artifact_bytes_constraint;
        boost::optional<ModelConfig> model;
        std::string name;
        ClassMemberConstraints name_constraint;
        boost::optional<std::vector<PassIoBinding>> outputs;
        boost::optional<int64_t> per_pass_deadline_ms;
        ClassMemberConstraints per_pass_deadline_ms_constraint;
        boost::optional<PipelineState> pipeline_state;
        boost::optional<RenderShaderConfig> render_shader;
        boost::optional<RenderTarget> render_target;
        boost::optional<ShaderConfig> shader;
        PassType type;
        boost::optional<VertexBuffer> vertex_buffer;
        boost::optional<std::vector<VertexLayoutEntry>> vertex_layout;

        public:
        /**
         * Data transformation pipeline
         */
        boost::optional<std::vector<DataTransform>> get_data_transforms() const { return data_transforms; }
        void set_data_transforms(boost::optional<std::vector<DataTransform>> value) { this->data_transforms = value; }

        boost::optional<std::string> get_description() const { return description; }
        void set_description(boost::optional<std::string> value) { this->description = value; }

        /**
         * Whether this pass is enabled by default
         */
        boost::optional<bool> get_enabled() const { return enabled; }
        void set_enabled(boost::optional<bool> value) { this->enabled = value; }

        /**
         * Estimated GPU memory needed for this pass in bytes. 0 means no estimate provided.
         */
        boost::optional<int64_t> get_estimated_gpu_memory_bytes() const { return estimated_gpu_memory_bytes; }
        void set_estimated_gpu_memory_bytes(boost::optional<int64_t> value) { if (value) CheckConstraint("estimated_gpu_memory_bytes", estimated_gpu_memory_bytes_constraint, *value); this->estimated_gpu_memory_bytes = value; }

        /**
         * Index buffer binding + index type for render passes
         */
        boost::optional<IndexBuffer> get_index_buffer() const { return index_buffer; }
        void set_index_buffer(boost::optional<IndexBuffer> value) { this->index_buffer = value; }

        /**
         * Input bindings for non-model passes
         */
        boost::optional<std::vector<PassIoBinding>> get_inputs() const { return inputs; }
        void set_inputs(boost::optional<std::vector<PassIoBinding>> value) { this->inputs = value; }

        /**
         * Maximum output artifact size in bytes before the pass is considered budget-exceeded. 0
         * means no budget.
         */
        boost::optional<int64_t> get_max_output_artifact_bytes() const { return max_output_artifact_bytes; }
        void set_max_output_artifact_bytes(boost::optional<int64_t> value) { if (value) CheckConstraint("max_output_artifact_bytes", max_output_artifact_bytes_constraint, *value); this->max_output_artifact_bytes = value; }

        /**
         * Model configuration for inference/retrain passes
         */
        boost::optional<ModelConfig> get_model() const { return model; }
        void set_model(boost::optional<ModelConfig> value) { this->model = value; }

        /**
         * Unique name for this pass
         */
        const std::string & get_name() const { return name; }
        std::string & get_mutable_name() { return name; }
        void set_name(const std::string & value) { CheckConstraint("name", name_constraint, value); this->name = value; }

        /**
         * Output bindings for non-model passes
         */
        boost::optional<std::vector<PassIoBinding>> get_outputs() const { return outputs; }
        void set_outputs(boost::optional<std::vector<PassIoBinding>> value) { this->outputs = value; }

        /**
         * Per-pass wall-clock deadline in milliseconds. 0 means no deadline.
         */
        boost::optional<int64_t> get_per_pass_deadline_ms() const { return per_pass_deadline_ms; }
        void set_per_pass_deadline_ms(boost::optional<int64_t> value) { if (value) CheckConstraint("per_pass_deadline_ms", per_pass_deadline_ms_constraint, *value); this->per_pass_deadline_ms = value; }

        /**
         * Fixed-function pipeline state for render passes
         */
        boost::optional<PipelineState> get_pipeline_state() const { return pipeline_state; }
        void set_pipeline_state(boost::optional<PipelineState> value) { this->pipeline_state = value; }

        /**
         * Multi-stage (vertex+fragment) shader configuration for render passes
         */
        boost::optional<RenderShaderConfig> get_render_shader() const { return render_shader; }
        void set_render_shader(boost::optional<RenderShaderConfig> value) { this->render_shader = value; }

        /**
         * Offscreen framebuffer (color+depth) config for render passes
         */
        boost::optional<RenderTarget> get_render_target() const { return render_target; }
        void set_render_target(boost::optional<RenderTarget> value) { this->render_target = value; }

        /**
         * Shader configuration for compute passes
         */
        boost::optional<ShaderConfig> get_shader() const { return shader; }
        void set_shader(boost::optional<ShaderConfig> value) { this->shader = value; }

        /**
         * Type of processing pass
         */
        const PassType & get_type() const { return type; }
        PassType & get_mutable_type() { return type; }
        void set_type(const PassType & value) { this->type = value; }

        /**
         * Buffer binding supplying vertex attribute data referenced by vertex_layout (D-16
         * Amendment)
         */
        boost::optional<VertexBuffer> get_vertex_buffer() const { return vertex_buffer; }
        void set_vertex_buffer(boost::optional<VertexBuffer> value) { this->vertex_buffer = value; }

        /**
         * Vertex attribute layout for render passes
         */
        boost::optional<std::vector<VertexLayoutEntry>> get_vertex_layout() const { return vertex_layout; }
        void set_vertex_layout(boost::optional<std::vector<VertexLayoutEntry>> value) { this->vertex_layout = value; }
    };
}
