

#include "util/sha256.hpp"

#include <openssl/evp.h>

namespace sgns::sgprocmanagersha
{
    std::vector<uint8_t> sha256(const void* data, size_t dataSize) {
        std::vector<uint8_t> hash(EVP_MAX_MD_SIZE);
        unsigned int hashLen;

        EVP_MD_CTX* ctx = EVP_MD_CTX_new();
        EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr);
        EVP_DigestUpdate(ctx, data, dataSize);
        EVP_DigestFinal_ex(ctx, hash.data(), &hashLen);
        EVP_MD_CTX_free(ctx);

        hash.resize(hashLen);
        return hash;
    }
} // namepace sgns::crypto
