#ifndef SUPERGENIUS_SHA256_HPP
#define SUPERGENIUS_SHA256_HPP

#include <string_view>
#include <vector>
#include <gsl/span>

namespace sgns::sgprocmanagersha
{
  std::vector<uint8_t> sha256(const void* data, size_t dataSize);
}

#endif
