#ifndef SGPROCMGR_QUANTIZATION_HPP
#define SGPROCMGR_QUANTIZATION_HPP

#include <cstddef>
#include <cstdint>

namespace sgns::sgprocmanagerquant
{
    /// Phase 10 no-op/identity stub. Phase 12 replaces this body with real
    /// IEEE-754 canonicalization (NaN/Inf/denormal/signed-zero normalization)
    /// plus fixed-precision scale-round-cast, once Phase 11's empirical
    /// cross-machine capture data justifies a real constant.
    ///
    /// @param data  Pointer to a float buffer to (eventually) quantize in place.
    /// @param count Number of float elements in the buffer.
    void QuantizeFloatBuffer( float *data, size_t count );

    /// Phase 10 no-op/identity stub. Phase 12 replaces this body with real
    /// integer tolerance-banding for the byte path, once Phase 11's empirical
    /// cross-machine capture data justifies a real constant.
    ///
    /// @param data  Pointer to a byte buffer to (eventually) quantize in place.
    /// @param count Number of bytes in the buffer.
    void QuantizeByteBuffer( uint8_t *data, size_t count );
}

#endif
