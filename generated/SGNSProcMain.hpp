//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     SGNSProcMain.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

#include "ElmType.hpp"
#include "ElmGeneration.hpp"
#include "Elm.hpp"
#include "ElmFunding.hpp"
#include "Dimensions.hpp"
#include "InputFormat.hpp"
#include "DataType.hpp"
#include "IoDeclaration.hpp"
#include "JobType.hpp"
#include "Constraints.hpp"
#include "ParameterType.hpp"
#include "Parameter.hpp"
#include "Params.hpp"
#include "DataTransformType.hpp"
#include "DataTransform.hpp"
#include "IndexType.hpp"
#include "IndexBuffer.hpp"
#include "PassIoBinding.hpp"
#include "ModelFormat.hpp"
#include "ModelNode.hpp"
#include "LossFunction.hpp"
#include "OptimizerType.hpp"
#include "OptimizerConfig.hpp"
#include "ModelConfig.hpp"
#include "BlendFactor.hpp"
#include "CullMode.hpp"
#include "DepthTest.hpp"
#include "FrontFace.hpp"
#include "Topology.hpp"
#include "PipelineState.hpp"
#include "Stage.hpp"
#include "ShaderSourceType.hpp"
#include "ShaderStage.hpp"
#include "RenderShaderUniform.hpp"
#include "RenderShaderConfig.hpp"
#include "ColorFormat.hpp"
#include "DepthFormat.hpp"
#include "RenderTarget.hpp"
#include "ShaderUniform.hpp"
#include "ShaderConfig.hpp"
#include "TextureFilter.hpp"
#include "TextureBuffer.hpp"
#include "PassType.hpp"
#include "VertexBuffer.hpp"
#include "VertexLayoutFormat.hpp"
#include "VertexLayoutEntry.hpp"
#include "Pass.hpp"
#include "Validation.hpp"
#include "SgnsProcessing.hpp"
namespace sgns {
}
