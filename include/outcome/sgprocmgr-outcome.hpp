/**
 * Based on libp2p outcome.hpp
 * Copyright Soramitsu Co., Ltd. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Adapted for SGNS project
 */

#ifndef SGNS_OUTCOME_HPP
#define SGNS_OUTCOME_HPP

#include <boost/outcome/result.hpp>
#include <boost/outcome/success_failure.hpp>
#include <boost/outcome/try.hpp>

#define OUTCOME_TRY(...) BOOST_OUTCOME_TRY(__VA_ARGS__)

#include "sgprocmgr-outcome-register.hpp"

/**
 * __cpp_sized_deallocation macro interferes with protobuf generated files
 * and makes them uncompilable by means of clang
 * since it is not necessary outside of outcome internals
 * it can be safely undefined
 */
#ifdef __cpp_sized_deallocation
#undef __cpp_sized_deallocation
#endif

namespace sgns::outcome {
  using namespace BOOST_OUTCOME_V2_NAMESPACE;  // NOLINT

  template <class R, class S = std::error_code,
            class NoValuePolicy = policy::default_policy<R, S, void>>  //
  using result = basic_result<R, S, NoValuePolicy>;

}  // namespace sgns::outcome

// @see /docs/result.md

#endif  // SGNS_OUTCOME_HPP