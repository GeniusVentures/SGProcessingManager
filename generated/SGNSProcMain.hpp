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

#include "Dimensions.hpp"
#include "InputFormat.hpp"
#include "DataType.hpp"
#include "IoDeclaration.hpp"
#include "Constraints.hpp"
#include "ParameterType.hpp"
#include "Parameter.hpp"
#include "Params.hpp"
#include "DataTransformType.hpp"
#include "DataTransform.hpp"
#include "PassIoBinding.hpp"
#include "ModelFormat.hpp"
#include "ModelNode.hpp"
#include "LossFunction.hpp"
#include "OptimizerType.hpp"
#include "OptimizerConfig.hpp"
#include "ModelConfig.hpp"
#include "ShaderType.hpp"
#include "Uniform.hpp"
#include "ShaderConfig.hpp"
#include "PassType.hpp"
#include "Pass.hpp"
#include "SgnsProcessing.hpp"
namespace sgns {
}
