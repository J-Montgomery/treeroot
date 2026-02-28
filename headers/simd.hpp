#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace simd {

using mask_t = std::vector<uint64_t>;
inline size_t num_words(size_t bits) { return (bits + 63) / 64; }

inline void band(uint64_t *a, const uint64_t *b, size_t n) {
    for (size_t i = 0; i < n; ++i)
        a[i] &= b[i];
}
inline void bor(uint64_t *a, const uint64_t *b, size_t n) {
    for (size_t i = 0; i < n; ++i)
        a[i] |= b[i];
}
inline void bnot(uint64_t *a, size_t n) {
    for (size_t i = 0; i < n; ++i)
        a[i] = ~a[i];
}
inline void bandnot(uint64_t *a, const uint64_t *b, size_t n) {
    for (size_t i = 0; i < n; ++i)
        a[i] &= ~b[i];
}

inline void clear_tail(uint64_t *a, size_t nw, size_t nb) {
    if (nb % 64 && nw)
        a[nw - 1] &= (1ULL << (nb % 64)) - 1;
}

inline bool test(const uint64_t *a, size_t i) {
    return a[i / 64] & (1ULL << (i % 64));
}
inline void set(uint64_t *a, size_t i) { a[i / 64] |= (1ULL << (i % 64)); }

inline size_t popcount(const uint64_t *a, size_t n) {
    size_t c = 0;
    for (size_t i = 0; i < n; ++i)
        c += std::popcount(a[i]);
    return c;
}
inline bool any(const uint64_t *a, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (a[i])
            return true;
    return false;
}

template <typename T, typename V, typename Pred>
void cmp_fill(uint64_t *m, const T *d, V v, size_t n, Pred pred) {
    const T val = static_cast<T>(v);
    const size_t nw = n / 64;

    for (size_t w = 0; w < nw; ++w) {
        uint64_t word_bits = 0;
        const T *chunk = d + (w * 64);

        for (size_t b = 0; b < 64; ++b) {
            word_bits |= (static_cast<uint64_t>(pred(chunk[b], val)) << b);
        }
        m[w] = word_bits;
    }

    if (size_t tail_count = n % 64) {
        uint64_t tail_bits = 0;
        const T *chunk = d + (nw * 64);
        for (size_t b = 0; b < tail_count; ++b) {
            tail_bits |= (static_cast<uint64_t>(pred(chunk[b], val)) << b);
        }
        m[nw] = tail_bits;
    }
}

template <typename T, typename V>
void cmp_eq(uint64_t *m, const T *d, V v, size_t n) {
    cmp_fill(m, d, v, n, [](auto a, auto b) { return a == b; });
}
template <typename T, typename V>
void cmp_ne(uint64_t *m, const T *d, V v, size_t n) {
    cmp_fill(m, d, v, n, [](auto a, auto b) { return a != b; });
}
template <typename T, typename V>
void cmp_lt(uint64_t *m, const T *d, V v, size_t n) {
    cmp_fill(m, d, v, n, [](auto a, auto b) { return a < b; });
}
template <typename T, typename V>
void cmp_ge(uint64_t *m, const T *d, V v, size_t n) {
    cmp_fill(m, d, v, n, [](auto a, auto b) { return a >= b; });
}

} // namespace simd