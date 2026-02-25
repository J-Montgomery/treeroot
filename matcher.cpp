#include <algorithm>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <numeric>

#include <gtest/gtest.h>
#include <benchmark/benchmark.h>

// ═══════════════════════════════════════════════════════════════════════════════
// Fixed-Size String for NTTPs
// ═══════════════════════════════════════════════════════════════════════════════

template<size_t N>
struct fs {
    char data[N]{};
    constexpr fs(const char (&s)[N]) { std::copy_n(s, N, data); }
    template<size_t M>
    constexpr bool operator==(const fs<M>& o) const {
        if constexpr (N != M) return false;
        else { for (size_t i = 0; i < N; ++i) if (data[i] != o.data[i]) return false; return true; }
    }
    constexpr std::string_view view() const { return {data, N - 1}; }
};

// ═══════════════════════════════════════════════════════════════════════════════
// SIMD Abstraction Layer
// ═══════════════════════════════════════════════════════════════════════════════

namespace simd {

using mask_t = std::vector<uint64_t>;
inline size_t num_words(size_t bits) { return (bits + 63) / 64; }

inline void band   (uint64_t* a, const uint64_t* b, size_t n) { for (size_t i=0;i<n;++i) a[i] &=  b[i]; }
inline void bor    (uint64_t* a, const uint64_t* b, size_t n) { for (size_t i=0;i<n;++i) a[i] |=  b[i]; }
inline void bnot   (uint64_t* a, size_t n)                     { for (size_t i=0;i<n;++i) a[i] = ~a[i]; }
inline void bandnot(uint64_t* a, const uint64_t* b, size_t n) { for (size_t i=0;i<n;++i) a[i] &= ~b[i]; }

inline void clear_tail(uint64_t* a, size_t nw, size_t nb) {
    if (nb % 64 && nw) a[nw-1] &= (1ULL << (nb % 64)) - 1;
}

inline bool test(const uint64_t* a, size_t i) { return a[i/64] & (1ULL << (i%64)); }
inline void set (uint64_t* a, size_t i)       { a[i/64] |= (1ULL << (i%64)); }

inline size_t popcount(const uint64_t* a, size_t n) {
    size_t c = 0; for (size_t i = 0; i < n; ++i) c += std::popcount(a[i]); return c;
}
inline bool any(const uint64_t* a, size_t n) {
    for (size_t i = 0; i < n; ++i) if (a[i]) return true; return false;
}

// Word-building: accumulates into a register, assigns whole words.
// Handles partial last word (bits start at 0), so no pre-zeroing or
// clear_tail needed by callers.
template<typename T, typename V, typename Pred>
void cmp_fill(uint64_t* m, const T* d, V v, size_t n, Pred pred) {
    size_t nw = num_words(n);
    for (size_t w = 0; w < nw; ++w) {
        uint64_t bits = 0;
        size_t base = w * 64;
        size_t cnt = std::min<size_t>(64, n - base);
        for (size_t b = 0; b < cnt; ++b)
            if (pred(d[base + b], static_cast<T>(v))) bits |= 1ULL << b;
        m[w] = bits;
    }
}

template<typename T, typename V> void cmp_eq(uint64_t* m, const T* d, V v, size_t n) { cmp_fill(m,d,v,n,[](auto a, auto b){return a==b;}); }
template<typename T, typename V> void cmp_ne(uint64_t* m, const T* d, V v, size_t n) { cmp_fill(m,d,v,n,[](auto a, auto b){return a!=b;}); }
template<typename T, typename V> void cmp_lt(uint64_t* m, const T* d, V v, size_t n) { cmp_fill(m,d,v,n,[](auto a, auto b){return a< b;}); }
template<typename T, typename V> void cmp_ge(uint64_t* m, const T* d, V v, size_t n) { cmp_fill(m,d,v,n,[](auto a, auto b){return a>=b;}); }

} // namespace simd

// ═══════════════════════════════════════════════════════════════════════════════
// Algebraic Engine (constexpr)
// ═══════════════════════════════════════════════════════════════════════════════

namespace match {

template<typename T>
concept matcher = requires { typename std::remove_cvref_t<T>::is_matcher; };

struct always_t { using is_matcher = void; };
struct never_t  { using is_matcher = void; };
inline constexpr always_t always;
inline constexpr never_t  never;

template<matcher M> struct not_t { using is_matcher = void; M m; };
template<matcher M> not_t(M) -> not_t<M>;

template<matcher L, matcher R> struct and_t {
    using is_matcher = void; using lhs_t = L; using rhs_t = R;
    L lhs; R rhs;
};
template<matcher L, matcher R> and_t(L, R) -> and_t<L, R>;

template<matcher L, matcher R> struct or_t {
    using is_matcher = void; using lhs_t = L; using rhs_t = R;
    L lhs; R rhs;
};
template<matcher L, matcher R> or_t(L, R) -> or_t<L, R>;

template<typename>   struct is_not_i : std::false_type {};
template<matcher M>  struct is_not_i<not_t<M>> : std::true_type {};
template<typename>   struct is_and_i : std::false_type {};
template<matcher L, matcher R> struct is_and_i<and_t<L,R>> : std::true_type {};
template<typename>   struct is_or_i  : std::false_type {};
template<matcher L, matcher R> struct is_or_i<or_t<L,R>>  : std::true_type {};

template<typename T> inline constexpr bool is_not_v = is_not_i<std::remove_cvref_t<T>>::value;
template<typename T> inline constexpr bool is_and_v = is_and_i<std::remove_cvref_t<T>>::value;
template<typename T> inline constexpr bool is_or_v  = is_or_i <std::remove_cvref_t<T>>::value;

struct field_in {
    using is_matcher = void;
    std::string_view field;
    std::shared_ptr<std::vector<uint64_t>> bitmap;
};

template<typename T>             inline constexpr bool is_static_v = true;
template<>                       inline constexpr bool is_static_v<field_in> = false;
template<typename M>             inline constexpr bool is_static_v<not_t<M>>    = is_static_v<M>;
template<typename L, typename R> inline constexpr bool is_static_v<and_t<L,R>> = is_static_v<L> && is_static_v<R>;
template<typename L, typename R> inline constexpr bool is_static_v<or_t<L,R>>  = is_static_v<L> && is_static_v<R>;

constexpr inline class implies_t {
public:
    template<typename L, typename R>
    constexpr bool operator()(L const& l, R const& r) const {
        using UL = std::remove_cvref_t<L>;
        using UR = std::remove_cvref_t<R>;
        if constexpr      (std::is_same_v<UL, never_t>)  return true;
        else if constexpr (std::is_same_v<UR, always_t>)  return true;
        else if constexpr (std::is_same_v<UL, always_t>)  return false;
        else if constexpr (is_and_v<UL>)  return (*this)(l.lhs, r) || (*this)(l.rhs, r);
        else if constexpr (is_or_v<UR>)   return (*this)(l, r.lhs) || (*this)(l, r.rhs);
        else if constexpr (is_or_v<UL>)   return (*this)(l.lhs, r) && (*this)(l.rhs, r);
        else if constexpr (is_not_v<UL> && is_not_v<UR>) return (*this)(r.m, l.m);
        else if constexpr (requires { tag_invoke(*this, l, r); })
            return tag_invoke(*this, l, r);
        else return std::is_same_v<UL, UR>;
    }
} implies{};

constexpr inline class negate_t {
public:
    template<typename M>
    constexpr auto operator()(M const& m) const {
        using UM = std::remove_cvref_t<M>;
        if constexpr      (std::is_same_v<UM, always_t>) return never;
        else if constexpr (std::is_same_v<UM, never_t>)  return always;
        else if constexpr (is_not_v<UM>) return m.m;
        else if constexpr (requires { tag_invoke(*this, m); }) return tag_invoke(*this, m);
        else return not_t{m};
    }
} negate{};

template<matcher L, matcher R> constexpr auto operator&(L l, R r);
template<matcher L, matcher R> constexpr auto operator|(L l, R r);

constexpr inline class simplify_t {
public:
    template<typename M>
    constexpr auto operator()(M const& m) const {
        using UM = std::remove_cvref_t<M>;
        if constexpr (std::is_same_v<UM, always_t> || std::is_same_v<UM, never_t>)
            return m;
        else if constexpr (is_not_v<UM>)  return negate((*this)(m.m));
        else if constexpr (is_and_v<UM>)  return (*this)(m.lhs) & (*this)(m.rhs);
        else if constexpr (is_or_v<UM>)   return (*this)(m.lhs) | (*this)(m.rhs);
        else if constexpr (requires { tag_invoke(*this, m); }) return tag_invoke(*this, m);
        else return m;
    }
} simplify{};

constexpr inline class sum_of_products_t {
public:
    template<typename M>
    constexpr auto operator()(M const& m) const {
        auto s = simplify(m);
        using S = decltype(s);
        if constexpr (is_not_v<S>) {
            using Inner = std::remove_cvref_t<decltype(s.m)>;
            if constexpr (is_and_v<Inner>)
                return or_t{(*this)(negate(s.m.lhs)), (*this)(negate(s.m.rhs))};
            else if constexpr (is_or_v<Inner>)
                return (*this)(and_t{(*this)(negate(s.m.lhs)), (*this)(negate(s.m.rhs))});
            else return s;
        }
        else if constexpr (is_or_v<S>)
            return or_t{(*this)(s.lhs), (*this)(s.rhs)};
        else if constexpr (is_and_v<S>) {
            auto l = (*this)(s.lhs);
            auto r = (*this)(s.rhs);
            using LS = decltype(l); using RS = decltype(r);
            if constexpr (is_or_v<LS>)
                return or_t{(*this)(and_t<typename LS::lhs_t, RS>{l.lhs, r}),
                             (*this)(and_t<typename LS::rhs_t, RS>{l.rhs, r})};
            else if constexpr (is_or_v<RS>)
                return or_t{(*this)(and_t<LS, typename RS::lhs_t>{l, r.lhs}),
                             (*this)(and_t<LS, typename RS::rhs_t>{l, r.rhs})};
            else return and_t<LS, RS>{l, r};
        }
        else return s;
    }
} sum_of_products{};

template<matcher L, matcher R>
constexpr auto operator&(L l, R r) {
    if constexpr (!is_static_v<L> || !is_static_v<R>) {
        if constexpr      (std::is_same_v<L, never_t> || std::is_same_v<R, never_t>) return never;
        else if constexpr (std::is_same_v<L, always_t>) return r;
        else if constexpr (std::is_same_v<R, always_t>) return l;
        else return and_t<L,R>{l, r};
    } else {
        auto sl = simplify(l); auto sr = simplify(r);
        using TL = decltype(sl); using TR = decltype(sr);
        if constexpr      (std::is_same_v<TL, never_t> || std::is_same_v<TR, never_t>) return never;
        else if constexpr (std::is_same_v<TL, always_t>) return sr;
        else if constexpr (std::is_same_v<TR, always_t>) return sl;
        else if constexpr (implies(sl, sr)) return sl;
        else if constexpr (implies(sr, sl)) return sr;
        else if constexpr (implies(sl, negate(sr)) || implies(sr, negate(sl))) return never;
        else return and_t<TL,TR>{sl, sr};
    }
}

template<matcher L, matcher R>
constexpr auto operator|(L l, R r) {
    if constexpr (!is_static_v<L> || !is_static_v<R>) {
        if constexpr      (std::is_same_v<L, always_t> || std::is_same_v<R, always_t>) return always;
        else if constexpr (std::is_same_v<L, never_t>) return r;
        else if constexpr (std::is_same_v<R, never_t>) return l;
        else return or_t<L,R>{l, r};
    } else {
        auto sl = simplify(l); auto sr = simplify(r);
        using TL = decltype(sl); using TR = decltype(sr);
        if constexpr      (std::is_same_v<TL, always_t> || std::is_same_v<TR, always_t>) return always;
        else if constexpr (std::is_same_v<TL, never_t>) return sr;
        else if constexpr (std::is_same_v<TR, never_t>) return sl;
        else if constexpr (implies(sl, sr)) return sr;
        else if constexpr (implies(sr, sl)) return sl;
        else if constexpr (implies(negate(sl), sr) || implies(negate(sr), sl)) return always;
        else return or_t<TL,TR>{sl, sr};
    }
}

template<matcher M>
constexpr auto operator!(M m) { return negate(simplify(m)); }

} // namespace match

// ═══════════════════════════════════════════════════════════════════════════════
// Field Matchers
// ═══════════════════════════════════════════════════════════════════════════════

enum class op { eq, ne, lt, ge };

template<op O, fs Field, auto Val>
struct field_matcher {
    using is_matcher = void;

    template<op O2, fs F2, auto V2>
    friend constexpr bool tag_invoke(match::implies_t, field_matcher const&,
                                     field_matcher<O2,F2,V2> const&) {
        if constexpr (!(Field == F2)) return false;
        else if constexpr (O == op::eq) {
            if constexpr      (O2==op::eq) return Val==V2;
            else if constexpr (O2==op::ne) return Val!=V2;
            else if constexpr (O2==op::lt) return Val< V2;
            else                           return Val>=V2;
        }
        else if constexpr (O == op::lt) {
            if constexpr      (O2==op::lt) return Val<=V2;
            else if constexpr (O2==op::ne) return Val<=V2;
            else return false;
        }
        else if constexpr (O == op::ge) {
            if constexpr      (O2==op::ge) return Val>=V2;
            else if constexpr (O2==op::ne) return Val> V2;
            else return false;
        }
        else if constexpr (O == op::ne) {
            if constexpr (O2==op::ne) return Val==V2;
            else return false;
        }
        else return false;
    }

    friend constexpr auto tag_invoke(match::negate_t, field_matcher const&) {
        if constexpr (O==op::eq) return field_matcher<op::ne,Field,Val>{};
        if constexpr (O==op::ne) return field_matcher<op::eq,Field,Val>{};
        if constexpr (O==op::lt) return field_matcher<op::ge,Field,Val>{};
        if constexpr (O==op::ge) return field_matcher<op::lt,Field,Val>{};
    }
};

template<fs Field, fs Path>
struct field_hst {
    using is_matcher = void;
    template<fs F2, fs P2>
    friend constexpr bool tag_invoke(match::implies_t, field_hst const&,
                                     field_hst<F2,P2> const&) {
        if constexpr (!(Field == F2)) return false;
        else return Path.view().starts_with(P2.view());
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// Columnar Table
// ═══════════════════════════════════════════════════════════════════════════════

struct table {
    size_t rows = 0;
    size_t chunk_size = 1024;

    struct column {
        std::string_view name;
        std::vector<uint32_t>    u32;
        std::vector<float>       f32;
        std::vector<uint64_t>    u64;
        std::vector<std::string> str;
    };

    std::vector<column>   cols;
    std::vector<uint64_t> chunk_summaries;

    explicit table(size_t n = 0, size_t cs = 1024) : rows(n), chunk_size(cs) {
        assert(cs % 64 == 0);
        std::vector<uint32_t> ids(n);
        std::iota(ids.begin(), ids.end(), 0u);
        cols.push_back({"id", std::move(ids), {}, {}, {}});
        cols.push_back({"mask", {}, {}, std::vector<uint64_t>(n, 0), {}});
        chunk_summaries.resize(num_chunks(), 0);
    }

    size_t num_chunks() const { return rows ? (rows + chunk_size - 1) / chunk_size : 0; }
    column&       get_col(std::string_view n)       { for (auto& c:cols) if (c.name==n) return c; throw std::runtime_error("no col"); }
    const column& get_col(std::string_view n) const { for (auto& c:cols) if (c.name==n) return c; throw std::runtime_error("no col"); }

    void add_column_u32(std::string_view n, uint32_t d=0) { cols.push_back({n, std::vector<uint32_t>(rows,d), {}, {}, {}}); }
    void add_column_f32(std::string_view n, float d=0.f)  { cols.push_back({n, {}, std::vector<float>(rows,d), {}, {}}); }
    void add_column_str(std::string_view n)                { cols.push_back({n, {}, {}, {}, std::vector<std::string>(rows)}); }

    void set_tag(size_t eid, int tag) {
        uint64_t bit = 1ULL << (tag % 64);
        get_col("mask").u64[eid] |= bit;
        chunk_summaries[eid / chunk_size] |= bit;
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// Execution Engine
// ═══════════════════════════════════════════════════════════════════════════════

namespace detail {
// True when a matcher's eval_into needs the scratch buffer internally
// (i.e. it is a binary op, or a not_t wrapping one).
template<typename T> inline constexpr bool needs_scratch = false;
template<typename L, typename R> inline constexpr bool needs_scratch<match::and_t<L,R>> = true;
template<typename L, typename R> inline constexpr bool needs_scratch<match::or_t<L,R>> = true;
template<typename M> inline constexpr bool needs_scratch<match::not_t<M>> = needs_scratch<M>;
}

struct engine {
    using mask_t = simd::mask_t;

    static mask_t execute(const table& t, match::matcher auto const& m) {
        auto sop = match::sum_of_products(m);
        uint64_t req = extract_bits(sop);
        size_t nw = simd::num_words(t.rows);
        mask_t result(nw, 0);

        if (req == 0 || t.chunk_summaries.empty()) {
            mask_t scratch(nw);
            eval_into(t, sop, 0, t.rows, result.data(), scratch.data(), nw);
        } else {
            size_t max_cnw = simd::num_words(t.chunk_size);
            mask_t chunk_buf(max_cnw);
            mask_t scratch(max_cnw);
            for (size_t c = 0; c < t.num_chunks(); ++c) {
                if ((t.chunk_summaries[c] & req) != req) continue;
                size_t start = c * t.chunk_size;
                size_t count = std::min(t.chunk_size, t.rows - start);
                size_t cnw = simd::num_words(count);
                eval_into(t, sop, start, count, chunk_buf.data(), scratch.data(), cnw);
                simd::bor(result.data() + start / 64, chunk_buf.data(), cnw);
            }
        }
        simd::clear_tail(result.data(), nw, t.rows);
        return result;
    }

    static match::field_in semi_join(const table& t, match::matcher auto const& q,
                                     std::string_view fk_field) {
        return {fk_field, std::make_shared<mask_t>(execute(t, q))};
    }

    static match::field_in semi_naive_join(const table& t,
                                           match::matcher auto const& seed,
                                           std::string_view fk_field,
                                           std::string_view id_field = "id") {
        auto delta = execute(t, seed);
        auto total = delta;
        size_t nw = delta.size();
        while (simd::any(delta.data(), nw)) {
            match::field_in frontier{fk_field, std::make_shared<mask_t>(delta)};
            auto reachable = execute(t, frontier);
            delta = reachable;
            simd::bandnot(delta.data(), total.data(), nw);
            simd::bor(total.data(), delta.data(), nw);
        }
        return {id_field, std::make_shared<mask_t>(std::move(total))};
    }

    enum class agg { sum, max, min, mean, count };

    static double aggregate(const table& t, match::matcher auto const& q,
                            std::string_view col, agg op) {
        auto mask = execute(t, q);
        auto& c = t.get_col(col);
        if (!c.f32.empty()) return agg_impl(mask, c.f32, op, t.rows);
        if (!c.u32.empty()) return agg_impl(mask, c.u32, op, t.rows);
        if (!c.u64.empty()) return agg_impl(mask, c.u64, op, t.rows);
        return 0.0;
    }

private:
    // ── extract_bits ──

    template<typename M>
    static uint64_t extract_bits(M const&) { return 0; }

    template<op O, fs Field, auto Val>
    static uint64_t extract_bits(field_matcher<O, Field, Val> const&) {
        if constexpr (O == op::eq && Field == fs("mask"))
            return static_cast<uint64_t>(Val);
        else return 0;
    }
    template<match::matcher L, match::matcher R>
    static uint64_t extract_bits(match::and_t<L,R> const& a) {
        return extract_bits(a.lhs) | extract_bits(a.rhs);
    }
    template<match::matcher L, match::matcher R>
    static uint64_t extract_bits(match::or_t<L,R> const& o) {
        auto l = extract_bits(o.lhs), r = extract_bits(o.rhs);
        return (l && r) ? (l & r) : 0;
    }

    // ── agg_impl ──

    template<typename T>
    static double agg_impl(const mask_t& mask, const std::vector<T>& data,
                           agg op, size_t rows) {
        if (op == agg::count)
            return (double)simd::popcount(mask.data(), mask.size());
        double r = 0; size_t n = 0; bool first = true;
        for (size_t i = 0; i < rows; ++i) {
            if (!simd::test(mask.data(), i)) continue;
            double v = (double)data[i];
            if      (op==agg::sum || op==agg::mean) r += v;
            else if (op==agg::max) { if (first || v > r) r = v; }
            else if (op==agg::min) { if (first || v < r) r = v; }
            first = false; ++n;
        }
        return (op == agg::mean && n) ? r / n : r;
    }

    // ── eval_into: writes result into out[0..nw) ──
    // scratch[0..nw) is temp space for binary ops.  Leaves ignore it.
    // For left-deep trees (the shape operator& builds) zero extra
    // allocations occur; a fallback alloc fires only when the RIGHT
    // child of a binary node is itself compound.

    static void eval_into(const table&, match::always_t, size_t, size_t count,
                          uint64_t* out, uint64_t*, size_t nw) {
        std::fill_n(out, nw, ~0ULL);
        simd::clear_tail(out, nw, count);
    }
    static void eval_into(const table&, match::never_t, size_t, size_t,
                          uint64_t* out, uint64_t*, size_t nw) {
        std::fill_n(out, nw, 0);
    }

    template<match::matcher L, match::matcher R>
    static void eval_into(const table& t, match::and_t<L,R> const& a,
                          size_t s, size_t c, uint64_t* out, uint64_t* scratch, size_t nw) {
        eval_into(t, a.lhs, s, c, out, scratch, nw);
        if constexpr (detail::needs_scratch<R>) {
            mask_t tmp(nw);
            eval_into(t, a.rhs, s, c, scratch, tmp.data(), nw);
        } else {
            eval_into(t, a.rhs, s, c, scratch, nullptr, nw);
        }
        simd::band(out, scratch, nw);
    }

    template<match::matcher L, match::matcher R>
    static void eval_into(const table& t, match::or_t<L,R> const& o,
                          size_t s, size_t c, uint64_t* out, uint64_t* scratch, size_t nw) {
        eval_into(t, o.lhs, s, c, out, scratch, nw);
        if constexpr (detail::needs_scratch<R>) {
            mask_t tmp(nw);
            eval_into(t, o.rhs, s, c, scratch, tmp.data(), nw);
        } else {
            eval_into(t, o.rhs, s, c, scratch, nullptr, nw);
        }
        simd::bor(out, scratch, nw);
    }

    template<match::matcher M>
    static void eval_into(const table& t, match::not_t<M> const& n,
                          size_t s, size_t c, uint64_t* out, uint64_t* scratch, size_t nw) {
        eval_into(t, n.m, s, c, out, scratch, nw);
        simd::bnot(out, nw);
        simd::clear_tail(out, nw, c);
    }

    template<op O, fs Field, auto Val>
    static void eval_into(const table& t, field_matcher<O,Field,Val> const&,
                          size_t start, size_t count, uint64_t* out, uint64_t*, size_t nw) {
        auto& col = t.get_col(Field.view());
        auto do_cmp = [&]<typename T>(const std::vector<T>& d) {
            const T* p = d.data() + start;
            if constexpr (O==op::eq) simd::cmp_eq(out, p, (T)Val, count);
            if constexpr (O==op::ne) simd::cmp_ne(out, p, (T)Val, count);
            if constexpr (O==op::lt) simd::cmp_lt(out, p, (T)Val, count);
            if constexpr (O==op::ge) simd::cmp_ge(out, p, (T)Val, count);
        };
        if      (!col.u32.empty()) do_cmp(col.u32);
        else if (!col.f32.empty()) do_cmp(col.f32);
        else if (!col.u64.empty()) do_cmp(col.u64);
        else std::fill_n(out, nw, 0);
    }

    template<fs Field, fs Path>
    static void eval_into(const table& t, field_hst<Field,Path> const&,
                          size_t start, size_t count, uint64_t* out, uint64_t*, size_t nw) {
        std::fill_n(out, nw, 0);
        auto& d = t.get_col(Field.view()).str;
        for (size_t i = 0; i < count; ++i)
            if (d[start+i].starts_with(Path.view())) simd::set(out, i);
    }

    static void eval_into(const table& t, match::field_in const& f,
                          size_t start, size_t count, uint64_t* out, uint64_t*, size_t nw) {
        std::fill_n(out, nw, 0);
        auto& d = t.get_col(f.field).u32;
        auto& bm = *f.bitmap;
        for (size_t i = 0; i < count; ++i) {
            uint32_t v = d[start + i];
            if (v / 64 < bm.size() && simd::test(bm.data(), v))
                simd::set(out, i);
        }
    }
};
namespace datalog {

using mask_t = engine::mask_t;

namespace bits {
    inline size_t words(size_t n) { return (n + 63) / 64; }
    inline bool test(const mask_t& m, size_t i) {
        return i / 64 < m.size() && (m[i / 64] & (1ULL << (i % 64)));
    }
    inline void set(mask_t& m, size_t i) {
        if (i / 64 < m.size()) m[i / 64] |= (1ULL << (i % 64));
    }
    inline void clear_tail(mask_t& m, size_t n) {
        if (n % 64 && !m.empty()) m.back() &= (1ULL << (n % 64)) - 1;
    }
    inline bool any(const mask_t& m) {
        for (auto w : m) if (w) return true;
        return false;
    }
}

struct atom { std::string rel; std::vector<std::string> vars; bool negated = false; };

struct rule_def {
    std::string head;
    std::vector<std::string> head_vars;
    std::vector<atom> body;
    std::function<mask_t(const table&)> filter;
    size_t filter_body = 0;
};

class program {
    // ── storage ──────────────────────────────────────────────────────────────
    struct edb_entry { std::string name; const table* tbl; std::vector<std::string> cols; };

    struct tuple_hash {
        size_t operator()(const std::vector<uint32_t>& v) const {
            size_t h = v.size();
            for (auto x : v) h ^= std::hash<uint32_t>{}(x) + 0x9e3779b9 + (h<<6) + (h>>2);
            return h;
        }
    };

    struct idb_entry {
        std::string name;
        bool unary;
        size_t universe;
        mask_t mask;
        std::vector<std::string> col_names;
        std::vector<std::vector<uint32_t>> data;
        size_t nrows;
        std::unordered_set<std::vector<uint32_t>, tuple_hash> seen;

        bool insert(const std::vector<uint32_t>& t) {
            if (!seen.insert(t).second) return false;
            for (size_t c = 0; c < t.size(); ++c) data[c].push_back(t[c]);
            ++nrows;
            return true;
        }
    };

    std::vector<edb_entry> edbs_;
    std::vector<idb_entry> idbs_;
    std::vector<rule_def> rules_;

    const edb_entry* edb(const std::string& n) const {
        for (auto& e : edbs_) if (e.name == n) return &e;
        return nullptr;
    }
    idb_entry* idb(const std::string& n) {
        for (auto& e : idbs_) if (e.name == n) return &e;
        return nullptr;
    }
    const idb_entry* idb(const std::string& n) const {
        for (auto& e : idbs_) if (e.name == n) return &e;
        return nullptr;
    }

    // ── variable environment ─────────────────────────────────────────────────
    using env_t = std::vector<std::pair<std::string, uint32_t>>;

    static const uint32_t* lookup(const env_t& e, const std::string& v) {
        for (auto& [k, val] : e) if (k == v) return &val;
        return nullptr;
    }

    // find column in ea that shares a variable name with ba
    static int shared_edb_col(const atom& ea, const atom& ba) {
        for (size_t i = 0; i < ba.vars.size(); ++i)
            for (size_t j = 0; j < ea.vars.size(); ++j)
                if (ba.vars[i] == ea.vars[j]) return (int)j;
        return -1;
    }

    // ── bitmask fast path ────────────────────────────────────────────────────
    // eligible: unary head, exactly 1 EDB body, all other body atoms unary IDB
    bool can_fast(const rule_def& r) const {
        auto* h = idb(r.head);
        if (!h || !h->unary) return false;
        int ec = 0;
        for (auto& a : r.body) {
            if (edb(a.rel)) ++ec;
            else { auto* p = idb(a.rel); if (!p || !p->unary) return false; }
        }
        return ec == 1;
    }

    void fire_fast(const rule_def& rule, const std::string* dr,
                   const mask_t* db, mask_t& out) {
        size_t ei = SIZE_MAX;
        for (size_t i = 0; i < rule.body.size(); ++i)
            if (edb(rule.body[i].rel)) { ei = i; break; }
        if (ei == SIZE_MAX) return;

        auto& ea = rule.body[ei];
        auto* ed = edb(ea.rel);
        const table& t = *ed->tbl;
        size_t n = t.rows, w = bits::words(n);

        mask_t rm(w, ~0ULL);
        bits::clear_tail(rm, n);

        if (rule.filter && rule.filter_body == ei) {
            auto f = rule.filter(t);
            for (size_t i = 0; i < w; ++i) rm[i] &= f[i];
        }

        for (size_t i = 0; i < rule.body.size(); ++i) {
            if (i == ei) continue;
            auto& ba = rule.body[i];
            auto* p = idb(ba.rel);
            if (!p || !p->unary) continue;

            const mask_t& b = (dr && ba.rel == *dr && db) ? *db : p->mask;
            int ec = shared_edb_col(ea, ba);
            if (ec < 0) continue;

            auto& col = t.get_col(ed->cols[ec]).u32;
            mask_t sj(w, 0);
            for (size_t r = 0; r < n; ++r)
                if (bits::test(b, col[r])) sj[r / 64] |= 1ULL << (r % 64);

            if (ba.negated) {
                for (size_t j = 0; j < w; ++j) sj[j] = ~sj[j];
                bits::clear_tail(sj, n);
            }
            for (size_t j = 0; j < w; ++j) rm[j] &= sj[j];
        }

        int pc = -1;
        for (size_t j = 0; j < ea.vars.size(); ++j)
            if (ea.vars[j] == rule.head_vars[0]) { pc = (int)j; break; }
        if (pc < 0) return;

        auto& pcol = t.get_col(ed->cols[pc]).u32;
        for (size_t r = 0; r < n; ++r)
            if (rm[r / 64] & (1ULL << (r % 64)))
                bits::set(out, pcol[r]);
    }

    // ── general nested-loop join ─────────────────────────────────────────────
    struct output {
        mask_t* ubits = nullptr;
        std::vector<std::vector<uint32_t>>* cols = nullptr;
        size_t* nrows = nullptr;
        std::unordered_set<std::vector<uint32_t>, tuple_hash>* seen = nullptr;
        const std::vector<std::string>* hvars = nullptr;
    };

    void emit(const env_t& env, const output& o) {
        if (o.ubits) {
            auto* p = lookup(env, (*o.hvars)[0]);
            if (p) bits::set(*o.ubits, *p);
        } else if (o.cols) {
            std::vector<uint32_t> t(o.hvars->size());
            for (size_t i = 0; i < t.size(); ++i) {
                auto* p = lookup(env, (*o.hvars)[i]);
                t[i] = p ? *p : 0;
            }
            if (o.seen->insert(t).second) {
                for (size_t c = 0; c < t.size(); ++c) (*o.cols)[c].push_back(t[c]);
                ++(*o.nrows);
            }
        }
    }

    void njoin(const rule_def& rule, const std::vector<size_t>& order, size_t pos,
               env_t& env, const std::string* dr, const mask_t* db,
               const std::vector<std::vector<uint32_t>>* dc, size_t dnr,
               const output& out) {
        if (pos == order.size()) { emit(env, out); return; }

        size_t ai = order[pos];
        auto& a = rule.body[ai];
        bool use_d = dr && a.rel == *dr && !a.negated;

        auto try_row = [&](auto get_val, size_t nc) {
            size_t orig = env.size();
            bool ok = true;
            for (size_t c = 0; c < nc && ok; ++c) {
                uint32_t v = get_val(c);
                auto* p = lookup(env, a.vars[c]);
                if (p) { if (*p != v) ok = false; }
                else env.emplace_back(a.vars[c], v);
            }
            if (ok) njoin(rule, order, pos + 1, env, dr, db, dc, dnr, out);
            env.resize(orig);
        };

        if (auto* ed = edb(a.rel)) {
            const table& t = *ed->tbl;
            size_t n = t.rows;
            int bc = -1; uint32_t bv = 0;
            for (size_t c = 0; c < a.vars.size(); ++c) {
                auto* p = lookup(env, a.vars[c]);
                if (p) { bc = (int)c; bv = *p; break; }
            }
            mask_t fm;
            if (rule.filter && rule.filter_body == ai) fm = rule.filter(t);

            for (size_t r = 0; r < n; ++r) {
                if (!fm.empty() && !(fm[r / 64] & (1ULL << (r % 64)))) continue;
                if (bc >= 0 && t.get_col(ed->cols[bc]).u32[r] != bv) continue;
                try_row([&](size_t c) { return t.get_col(ed->cols[c]).u32[r]; },
                        a.vars.size());
            }
        } else if (auto* p = idb(a.rel)) {
            if (p->unary) {
                const mask_t& b = (use_d && db) ? *db : p->mask;
                auto* bound = lookup(env, a.vars[0]);
                if (a.negated) {
                    if (bound && !bits::test(b, *bound))
                        njoin(rule, order, pos + 1, env, dr, db, dc, dnr, out);
                } else if (bound) {
                    if (bits::test(b, *bound))
                        njoin(rule, order, pos + 1, env, dr, db, dc, dnr, out);
                } else {
                    for (size_t v = 0; v < p->universe; ++v) {
                        if (!bits::test(b, v)) continue;
                        size_t orig = env.size();
                        env.emplace_back(a.vars[0], (uint32_t)v);
                        njoin(rule, order, pos + 1, env, dr, db, dc, dnr, out);
                        env.resize(orig);
                    }
                }
            } else {
                auto& cs = (use_d && dc) ? *dc : p->data;
                size_t nr = (use_d && dc) ? dnr : p->nrows;
                int bc = -1; uint32_t bv = 0;
                for (size_t c = 0; c < a.vars.size(); ++c) {
                    auto* q = lookup(env, a.vars[c]);
                    if (q) { bc = (int)c; bv = *q; break; }
                }
                for (size_t r = 0; r < nr; ++r) {
                    if (bc >= 0 && cs[bc][r] != bv) continue;
                    try_row([&](size_t c) { return cs[c][r]; }, a.vars.size());
                }
            }
        }
    }

    void fire_general(const rule_def& rule, const std::string* dr,
                      const mask_t* db,
                      const std::vector<std::vector<uint32_t>>* dc, size_t dnr,
                      const output& out) {
        std::vector<size_t> order;
        if (dr)
            for (size_t i = 0; i < rule.body.size(); ++i)
                if (rule.body[i].rel == *dr && !rule.body[i].negated)
                    { order.push_back(i); break; }
        for (size_t i = 0; i < rule.body.size(); ++i)
            if (std::find(order.begin(), order.end(), i) == order.end()
                && edb(rule.body[i].rel))
                order.push_back(i);
        for (size_t i = 0; i < rule.body.size(); ++i)
            if (std::find(order.begin(), order.end(), i) == order.end())
                order.push_back(i);
        env_t env;
        njoin(rule, order, 0, env, dr, db, dc, dnr, out);
    }

    // ── stratification ───────────────────────────────────────────────────────
    struct stratum_t { std::vector<size_t> rule_ids; };

    std::vector<stratum_t> stratify() {
        std::unordered_map<std::string, int> sn;
        for (auto& e : edbs_) sn[e.name] = -1;
        for (auto& e : idbs_) sn[e.name] = 0;
        for (bool chg = true; chg;) {
            chg = false;
            for (auto& r : rules_) {
                int req = 0;
                for (auto& a : r.body) {
                    auto it = sn.find(a.rel);
                    if (it == sn.end() || it->second < 0) continue;
                    req = std::max(req, a.negated ? it->second + 1 : it->second);
                }
                if (req > sn[r.head]) { sn[r.head] = req; chg = true; }
            }
        }
        int mx = 0;
        for (auto& [_, s] : sn) mx = std::max(mx, s);
        std::vector<stratum_t> res(mx + 1);
        for (size_t i = 0; i < rules_.size(); ++i) {
            int s = sn[rules_[i].head];
            if (s >= 0) res[s].rule_ids.push_back(i);
        }
        return res;
    }

    bool is_recursive(const rule_def& r) const {
        for (auto& a : r.body) if (!a.negated && idb(a.rel)) return true;
        return false;
    }

    output make_output(const rule_def& r, idb_entry& h) {
        output o; o.hvars = &r.head_vars;
        if (h.unary) o.ubits = &h.mask;
        else { o.cols = &h.data; o.nrows = &h.nrows; o.seen = &h.seen; }
        return o;
    }

    // ── stratum evaluation ───────────────────────────────────────────────────
    void eval_stratum(stratum_t& s) {
        std::vector<size_t> base, rec;
        for (size_t ri : s.rule_ids)
            (is_recursive(rules_[ri]) ? rec : base).push_back(ri);

        for (size_t ri : base) {
            auto& r = rules_[ri]; auto* h = idb(r.head);
            if (can_fast(r)) fire_fast(r, nullptr, nullptr, h->mask);
            else fire_general(r, nullptr, nullptr, nullptr, 0, make_output(r, *h));
        }
        if (rec.empty()) return;

        std::unordered_set<std::string> inv;
        for (size_t ri : rec) {
            inv.insert(rules_[ri].head);
            for (auto& a : rules_[ri].body)
                if (!a.negated && idb(a.rel)) inv.insert(a.rel);
        }

        struct delta_t { mask_t bits; std::vector<std::vector<uint32_t>> cols; size_t nrows = 0; };
        std::unordered_map<std::string, delta_t> deltas;
        for (auto& nm : inv) {
            auto* p = idb(nm); delta_t d;
            if (p->unary) d.bits = p->mask;
            else { d.cols = p->data; d.nrows = p->nrows; }
            deltas[nm] = std::move(d);
        }

        for (bool any_new = true; any_new;) {
            any_new = false;

            std::unordered_map<std::string, delta_t> accum;
            std::unordered_map<std::string,
                std::unordered_set<std::vector<uint32_t>, tuple_hash>> acc_seen;
            for (auto& nm : inv) {
                auto* p = idb(nm); delta_t a;
                if (p->unary) a.bits.resize(p->mask.size(), 0);
                else a.cols.resize(p->col_names.size());
                accum[nm] = std::move(a);
                if (!p->unary) acc_seen[nm];
            }

            for (size_t ri : rec) {
                auto& rule = rules_[ri];
                for (auto& ba : rule.body) {
                    if (ba.negated || !idb(ba.rel)) continue;
                    std::string dn = ba.rel;
                    auto& d = deltas[dn];
                    auto* hi = idb(rule.head);
                    auto* di = idb(dn);

                    if (can_fast(rule) && hi->unary && di->unary) {
                        fire_fast(rule, &dn, &d.bits, accum[rule.head].bits);
                    } else {
                        auto& ac = accum[rule.head];
                        output o; o.hvars = &rule.head_vars;
                        if (hi->unary) {
                            o.ubits = &ac.bits;
                        } else {
                            o.cols = &ac.cols; o.nrows = &ac.nrows;
                            o.seen = &acc_seen[rule.head];
                        }
                        fire_general(rule, &dn,
                                     di->unary ? &d.bits : nullptr,
                                     di->unary ? nullptr : &d.cols, d.nrows, o);
                    }
                    break;
                }
            }

            for (auto& nm : inv) {
                auto* p = idb(nm); auto& ac = accum[nm];
                if (p->unary) {
                    for (size_t w = 0; w < ac.bits.size(); ++w) ac.bits[w] &= ~p->mask[w];
                    if (bits::any(ac.bits)) {
                        any_new = true;
                        for (size_t w = 0; w < ac.bits.size(); ++w) p->mask[w] |= ac.bits[w];
                    }
                    deltas[nm].bits = std::move(ac.bits);
                } else {
                    delta_t nd; nd.cols.resize(p->col_names.size());
                    for (size_t r = 0; r < ac.nrows; ++r) {
                        std::vector<uint32_t> t(p->col_names.size());
                        for (size_t c = 0; c < t.size(); ++c) t[c] = ac.cols[c][r];
                        if (p->insert(t)) {
                            for (size_t c = 0; c < t.size(); ++c) nd.cols[c].push_back(t[c]);
                            ++nd.nrows; any_new = true;
                        }
                    }
                    deltas[nm] = std::move(nd);
                }
            }
        }
    }

public:
    void add_edb(std::string name, const table& t, std::vector<std::string> cols) {
        edbs_.push_back({std::move(name), &t, std::move(cols)});
    }
    void add_idb(std::string name, size_t universe) {
        idb_entry e;
        e.name = std::move(name); e.unary = true; e.universe = universe;
        e.mask.resize(bits::words(universe), 0); e.nrows = 0;
        idbs_.push_back(std::move(e));
    }
    void add_idb_table(std::string name, std::vector<std::string> cols) {
        idb_entry e;
        e.name = std::move(name); e.unary = false; e.universe = 0;
        e.col_names = std::move(cols); e.data.resize(e.col_names.size()); e.nrows = 0;
        idbs_.push_back(std::move(e));
    }
    void add_rule(rule_def r) { rules_.push_back(std::move(r)); }
    void evaluate() { auto ss = stratify(); for (auto& s : ss) eval_stratum(s); }

    const mask_t& get_bits(const std::string& n) const { return idb(n)->mask; }
    struct store_view {
        size_t nrows, arity;
        const std::vector<std::vector<uint32_t>>& cols;
    };
    store_view get_store(const std::string& n) const {
        auto* p = idb(n);
        return {p->nrows, p->col_names.size(), p->data};
    }
};

} // namespace datalog

namespace {

table make_nodes(size_t n) {
    return table(n); // constructor creates "id" = {0..n-1} and "mask"
}

table make_edges(std::vector<std::pair<uint32_t, uint32_t>> es) {
    size_t n = es.size();
    table t(n);
    t.add_column_u32("src");
    t.add_column_u32("dst");
    auto& src = t.get_col("src").u32;
    auto& dst = t.get_col("dst").u32;
    for (size_t i = 0; i < n; ++i) { src[i] = es[i].first; dst[i] = es[i].second; }
    return t;
}

table make_table(size_t n,
                 std::initializer_list<std::pair<const char*, std::vector<uint32_t>>> extra) {
    table t(n);
    for (auto& [name, vals] : extra) {
        t.add_column_u32(name);
        t.get_col(name).u32 = vals;
    }
    return t;
}

bool bt(const engine::mask_t& m, size_t i) {
    return i / 64 < m.size() && (m[i / 64] & (1ULL << (i % 64)));
}
size_t popcnt(const engine::mask_t& m, size_t n) {
    size_t c = 0; for (size_t i = 0; i < n; ++i) c += bt(m, i); return c;
}
bool has_fact(const datalog::program::store_view& s, std::initializer_list<uint32_t> v) {
    std::vector<uint32_t> t(v);
    for (size_t r = 0; r < s.nrows; ++r) {
        bool ok = true;
        for (size_t c = 0; c < s.arity && ok; ++c)
            if (s.cols[c][r] != t[c]) ok = false;
        if (ok) return true;
    }
    return false;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

using namespace match;

TEST(Algebra, SubsumptionAndRedundancy) {
    using lt10 = field_matcher<op::lt, "x", 10>;
    using lt20 = field_matcher<op::lt, "x", 20>;
    static_assert(std::is_same_v<decltype(lt10{} & lt20{}), lt10>);
    static_assert(std::is_same_v<decltype(lt10{} | lt20{}), lt20>);
}

TEST(Algebra, ContradictionAndTautology) {
    using eq10 = field_matcher<op::eq, "x", 10>;
    using ne10 = field_matcher<op::ne, "x", 10>;
    static_assert(std::is_same_v<decltype(eq10{} & ne10{}), never_t>);
    static_assert(std::is_same_v<decltype(eq10{} | ne10{}), always_t>);

    using lt10 = field_matcher<op::lt, "x", 10>;
    using ge20 = field_matcher<op::ge, "x", 20>;
    static_assert(std::is_same_v<decltype(lt10{} & ge20{}), never_t>);
}

TEST(Algebra, FullImplicationTable) {
    using lt10 = field_matcher<op::lt, "x", 10>;
    using ne40 = field_matcher<op::ne, "x", 40>;
    static_assert(std::is_same_v<decltype(lt10{} & ne40{}), lt10>);

    using ge50 = field_matcher<op::ge, "x", 50>;
    using ne10 = field_matcher<op::ne, "x", 10>;
    static_assert(std::is_same_v<decltype(ge50{} & ne10{}), ge50>);

    using ne5a = field_matcher<op::ne, "x", 5>;
    using ne5b = field_matcher<op::ne, "x", 5>;
    static_assert(std::is_same_v<decltype(ne5a{} & ne5b{}), ne5a>);

    using eq15 = field_matcher<op::eq, "x", 15>;
    using lt20 = field_matcher<op::lt, "x", 20>;
    static_assert(std::is_same_v<decltype(eq15{} & lt20{}), eq15>);

    using ge10 = field_matcher<op::ge, "x", 10>;
    static_assert(std::is_same_v<decltype(lt20{} | ge10{}), always_t>);
    static_assert(is_and_v<decltype(lt20{} & ge10{})>);
}

TEST(Algebra, OrImpliesM) {
    using eq5  = field_matcher<op::eq, "x", 5>;
    using eq3  = field_matcher<op::eq, "x", 3>;
    using lt10 = field_matcher<op::lt, "x", 10>;

    // Both eq5 and eq3 imply lt10 → (eq5|eq3) & lt10 simplifies to (eq5|eq3)
    auto result = (eq5{} | eq3{}) & lt10{};
    static_assert(std::is_same_v<decltype(result), or_t<eq5, eq3>>);
}

TEST(Algebra, Contrapositive) {
    using parent = field_hst<"type", "sensor">;
    using child  = field_hst<"type", "sensor/temp">;

    auto neg_parent = !parent{};
    auto neg_child  = !child{};
    static_assert(is_not_v<decltype(neg_parent)>);

    // child ⟹ parent, so by contrapositive ¬parent ⟹ ¬child
    static_assert( implies(neg_parent, neg_child));
    static_assert(!implies(neg_child, neg_parent));

    // Subsumption via contrapositive: ¬parent & ¬child → ¬parent
    auto result = neg_parent & neg_child;
    static_assert(std::is_same_v<decltype(result), decltype(neg_parent)>);
}

TEST(Algebra, SimplifyRecursion) {
    using eq5  = field_matcher<op::eq, "x", 5>;
    using lt10 = field_matcher<op::lt, "x", 10>;

    // Raw and_t/or_t (not built through operators) must still simplify
    auto raw_and = and_t{eq5{}, lt10{}};
    static_assert(is_and_v<decltype(raw_and)>);
    static_assert(std::is_same_v<decltype(simplify(raw_and)), eq5>);

    auto raw_or = or_t{eq5{}, lt10{}};
    static_assert(std::is_same_v<decltype(simplify(raw_or)), lt10>);
}

TEST(Algebra, SumOfProducts) {
    using A = field_matcher<op::eq, "x", 1>;
    using B = field_matcher<op::eq, "y", 2>;
    using C = field_matcher<op::eq, "z", 3>;

    auto expr = and_t{or_t{A{}, B{}}, C{}};
    auto dnf = sum_of_products(expr);
    static_assert(is_or_v<decltype(dnf)>);

    auto absorb = and_t{A{}, or_t{A{}, B{}}};
    static_assert(std::is_same_v<decltype(sum_of_products(absorb)), A>);
}

TEST(Algebra, HST) {
    using child  = field_hst<"type","sensor/temp">;
    using parent = field_hst<"type","sensor">;
    using other  = field_hst<"type","motor">;
    static_assert(std::is_same_v<decltype(child{} & parent{}), child>);
    static_assert(is_and_v<decltype(child{} & other{})>);
}

// ── Runtime Engine ──

TEST(Engine, BasicExecution) {
    table t(5);
    t.add_column_u32("s"); t.get_col("s").u32 = {1,2,1,2,1};
    auto r = engine::execute(t, field_matcher<op::eq,"s",2>{});
    EXPECT_FALSE(simd::test(r.data(),0));
    EXPECT_TRUE (simd::test(r.data(),1));
    EXPECT_TRUE (simd::test(r.data(),3));
}

TEST(Engine, CompoundQuery) {
    table t(5);
    t.get_col("id").u32 = {10,11,12,13,14};
    t.add_column_u32("s"); t.get_col("s").u32 = {1,2,1,2,1};
    auto r = engine::execute(t, field_matcher<op::lt,"id",13>{} & field_matcher<op::eq,"s",2>{});
    EXPECT_TRUE (simd::test(r.data(),1));
    EXPECT_FALSE(simd::test(r.data(),3));
}

TEST(Engine, SemiJoin) {
    table users(3);
    table logs(4); logs.add_column_u32("uid"); logs.get_col("uid").u32 = {0,0,2,5};
    auto sj = engine::semi_join(users, field_matcher<op::lt,"id",2>{}, "uid");
    auto r  = engine::execute(logs, sj);
    EXPECT_TRUE (simd::test(r.data(),0));
    EXPECT_TRUE (simd::test(r.data(),1));
    EXPECT_FALSE(simd::test(r.data(),2));
    EXPECT_FALSE(simd::test(r.data(),3));
}

TEST(Engine, SemiJoinCompound) {
    table players(100);
    players.add_column_u32("team");
    for (size_t i=0;i<100;++i) players.get_col("team").u32[i] = (i%3)+1;

    table pets(100);
    pets.add_column_u32("oid"); pets.add_column_u32("sp");
    for (size_t i=0;i<100;++i) {
        pets.get_col("oid").u32[i] = i % 100;
        pets.get_col("sp").u32[i]  = (i%5)+1;
    }
    auto join  = engine::semi_join(players, field_matcher<op::eq,"team",2>{}, "oid");
    auto query = field_matcher<op::eq,"sp",3>{} & join;
    auto r = engine::execute(pets, query);
    for (size_t i=0;i<100;++i) {
        bool expect = (pets.get_col("sp").u32[i]==3)
                   && (players.get_col("team").u32[pets.get_col("oid").u32[i]]==2);
        EXPECT_EQ(simd::test(r.data(),i), expect);
    }
}

TEST(Engine, SemiNaiveChain) {
    table t(100); t.add_column_u32("pid",999);
    for (uint32_t i=1;i<5;++i) t.get_col("pid").u32[i] = i-1;
    auto rm = engine::semi_naive_join(t, field_matcher<op::eq,"id",0>{}, "pid");
    auto r  = engine::execute(t, rm);
    EXPECT_EQ(simd::popcount(r.data(), r.size()), 5u);
}

TEST(Engine, SemiNaiveCycle) {
    table t(100); t.add_column_u32("pid",999);
    t.get_col("pid").u32[0] = 1;
    t.get_col("pid").u32[1] = 0;
    auto rm = engine::semi_naive_join(t, field_matcher<op::eq,"id",0>{}, "pid");
    auto r  = engine::execute(t, rm);
    EXPECT_EQ(simd::popcount(r.data(), r.size()), 2u);
}

TEST(Engine, Aggregate) {
    table t(1000);
    t.add_column_u32("team"); t.add_column_f32("hp");
    for (size_t i=0;i<500;++i)  { t.get_col("team").u32[i]=1; t.get_col("hp").f32[i]=100.f; }
    for (size_t i=500;i<1000;++i){ t.get_col("team").u32[i]=2; t.get_col("hp").f32[i]=50.f; }
    auto q = field_matcher<op::eq,"team",1>{};
    EXPECT_DOUBLE_EQ(engine::aggregate(t, q, "hp", engine::agg::sum),   50000.0);
    EXPECT_DOUBLE_EQ(engine::aggregate(t, q, "hp", engine::agg::mean),  100.0);
    EXPECT_DOUBLE_EQ(engine::aggregate(t, q, "hp", engine::agg::count), 500.0);
    EXPECT_DOUBLE_EQ(engine::aggregate(t, q, "hp", engine::agg::max),   100.0);
    EXPECT_DOUBLE_EQ(engine::aggregate(t, q, "hp", engine::agg::min),   100.0);
}

TEST(Engine, BloomFilterSkip) {
    table t(10240, 1024);
    t.add_column_u32("dur", 100);
    for (int i=0;i<5;++i) { t.set_tag(i,0); t.set_tag(i,1); }
    for (int i=0;i<5;++i) { t.set_tag(5*1024+i,0); t.set_tag(5*1024+i,1); }
    auto q = field_matcher<op::eq,"mask",3>{} & field_matcher<op::lt,"dur",150>{};
    auto r = engine::execute(t, q);
    EXPECT_EQ(simd::popcount(r.data(), r.size()), 10u);
}

TEST(Engine, BloomFilterDNF) {
    // After DNF, top-level or_t — bloom skip should still work via
    // extract_bits intersecting both branches' required mask bits.
    table t(10240, 1024);
    t.add_column_u32("dur", 100); t.add_column_u32("sp", 0);
    for (int i = 0; i < 5; ++i) { t.set_tag(i, 0); t.set_tag(i, 1); }
    auto q = or_t{
        and_t{field_matcher<op::eq,"mask",3>{}, field_matcher<op::lt,"dur",150>{}},
        and_t{field_matcher<op::eq,"mask",3>{}, field_matcher<op::eq,"sp",0>{}}
    };
    auto r = engine::execute(t, q);
    EXPECT_EQ(simd::popcount(r.data(), r.size()), 5u);
}

TEST(Engine, Contradiction) {
    table t(1000); t.add_column_u32("v",10);
    auto q = field_matcher<op::eq,"v",10>{} & field_matcher<op::ne,"v",10>{};
    static_assert(std::is_same_v<decltype(q), never_t>);
    EXPECT_EQ(simd::popcount(engine::execute(t,q).data(), simd::num_words(1000)), 0u);
}

TEST(Engine, Tautology) {
    table t(1000); t.add_column_f32("hp",50.f);
    auto q = field_matcher<op::lt,"hp",100>{} | field_matcher<op::ge,"hp",100>{};
    static_assert(std::is_same_v<decltype(q), always_t>);
    EXPECT_EQ(simd::popcount(engine::execute(t,q).data(), simd::num_words(1000)), 1000u);
}

TEST(Engine, EmptyJoin) {
    table p(100); p.add_column_u32("team",0);
    table pets(100); pets.add_column_u32("oid",0);
    auto sj = engine::semi_join(p, field_matcher<op::eq,"team",999>{}, "oid");
    EXPECT_EQ(simd::popcount(engine::execute(pets,sj).data(), simd::num_words(100)), 0u);
}

TEST(Engine, NegativeJoin) {
    table p(10); p.add_column_u32("team");
    for (size_t i=0;i<10;++i) p.get_col("team").u32[i] = (i<5)?1:2;
    auto nj = engine::semi_join(p, !field_matcher<op::eq,"team",2>{}, "oid");
    table pets(10); pets.add_column_u32("oid");
    for (size_t i=0;i<10;++i) pets.get_col("oid").u32[i] = i;
    auto r = engine::execute(pets, nj);
    for (size_t i=0;i<5;++i)  EXPECT_TRUE (simd::test(r.data(),i));
    for (size_t i=5;i<10;++i) EXPECT_FALSE(simd::test(r.data(),i));
}


// ═══════════════════════════════════════════════════════════════════════════════
// Datalog Unit Tests
// ═══════════════════════════════════════════════════════════════════════════════

// ── Unary base cases ─────────────────────────────────────────────────────────

TEST(Datalog, UnaryBaseNoFilter) {
    auto edges = make_edges({{0,1},{1,2},{2,3}});
    datalog::program p;
    p.add_edb("edge", edges, {"src","dst"});
    p.add_idb("r", 4);
    p.add_rule({"r", {"X"}, {{"edge", {"X","D"}}}});
    p.evaluate();
    EXPECT_TRUE(bt(p.get_bits("r"), 0));
    EXPECT_TRUE(bt(p.get_bits("r"), 1));
    EXPECT_TRUE(bt(p.get_bits("r"), 2));
    EXPECT_FALSE(bt(p.get_bits("r"), 3));
}

TEST(Datalog, UnaryBaseWithFilter) {
    auto nodes = make_nodes(5);
    datalog::program p;
    p.add_edb("nodes", nodes, {"id"});
    p.add_idb("seed", 5);
    p.add_rule({"seed", {"X"}, {{"nodes", {"X"}}},
        [](const table& t) -> engine::mask_t {
            return engine::execute(t, field_matcher<op::lt, "id", 3>{});
        }, 0});
    p.evaluate();
    EXPECT_EQ(popcnt(p.get_bits("seed"), 5), 3u);
    EXPECT_TRUE(bt(p.get_bits("seed"), 0));
    EXPECT_TRUE(bt(p.get_bits("seed"), 2));
    EXPECT_FALSE(bt(p.get_bits("seed"), 3));
}

TEST(Datalog, UnaryBaseEmpty) {
    auto nodes = make_nodes(0);
    datalog::program p;
    p.add_edb("n", nodes, {"id"});
    p.add_idb("s", 10);
    p.add_rule({"s", {"X"}, {{"n", {"X"}}}});
    p.evaluate();
    EXPECT_EQ(popcnt(p.get_bits("s"), 10), 0u);
}

TEST(Datalog, UnaryBaseNoMatches) {
    auto nodes = make_nodes(3);
    datalog::program p;
    p.add_edb("n", nodes, {"id"});
    p.add_idb("s", 3);
    p.add_rule({"s", {"X"}, {{"n", {"X"}}},
        [](const table& t) -> engine::mask_t {
            return engine::execute(t, field_matcher<op::ge, "id", 100>{});
        }, 0});
    p.evaluate();
    EXPECT_EQ(popcnt(p.get_bits("s"), 3), 0u);
}

TEST(Datalog, MultipleRulesSameHead) {
    auto nodes = make_nodes(5);
    auto edges = make_edges({{3,4},{4,0}});
    datalog::program p;
    p.add_edb("nodes", nodes, {"id"});
    p.add_edb("edge", edges, {"src","dst"});
    p.add_idb("r", 5);
    p.add_rule({"r", {"X"}, {{"nodes", {"X"}}},
        [](const table& t) -> engine::mask_t {
            return engine::execute(t, field_matcher<op::lt, "id", 2>{});
        }, 0});
    p.add_rule({"r", {"X"}, {{"edge", {"X","D"}}}});
    p.evaluate();
    EXPECT_TRUE(bt(p.get_bits("r"), 0));
    EXPECT_TRUE(bt(p.get_bits("r"), 1));
    EXPECT_FALSE(bt(p.get_bits("r"), 2));
    EXPECT_TRUE(bt(p.get_bits("r"), 3));
    EXPECT_TRUE(bt(p.get_bits("r"), 4));
}

TEST(Datalog, TCLinearChain) {
    auto edges = make_edges({{0,1},{1,2},{2,3},{3,4}});
    datalog::program p;
    p.add_edb("edge", edges, {"src","dst"});
    p.add_idb("reach", 5);
    p.add_rule({"reach", {"X"}, {{"edge", {"X","D"}}}});
    p.add_rule({"reach", {"X"}, {{"reach", {"Y"}}, {"edge", {"Y","X"}}}});
    p.evaluate();
    for (int i = 0; i <= 4; ++i)
        EXPECT_TRUE(bt(p.get_bits("reach"), i)) << "missing " << i;
}

TEST(Datalog, TCCycle) {
    auto edges = make_edges({{0,1},{1,2},{2,0}});
    datalog::program p;
    p.add_edb("edge", edges, {"src","dst"});
    p.add_idb("reach", 3);
    p.add_rule({"reach", {"X"}, {{"edge", {"X","D"}}}});
    p.add_rule({"reach", {"X"}, {{"reach", {"Y"}}, {"edge", {"Y","X"}}}});
    p.evaluate();
    EXPECT_EQ(popcnt(p.get_bits("reach"), 3), 3u);
}

TEST(Datalog, TCDiamond) {
    auto edges = make_edges({{0,1},{0,2},{1,3},{2,3}});
    datalog::program p;
    p.add_edb("edge", edges, {"src","dst"});
    p.add_idb("reach", 4);
    p.add_rule({"reach", {"X"}, {{"edge", {"X","D"}}}});
    p.add_rule({"reach", {"X"}, {{"reach", {"Y"}}, {"edge", {"Y","X"}}}});
    p.evaluate();
    EXPECT_EQ(popcnt(p.get_bits("reach"), 4), 4u);
}

TEST(Datalog, TCDisconnected) {
    auto edges = make_edges({{0,1},{1,2},{5,6},{6,7}});
    datalog::program p;
    p.add_edb("edge", edges, {"src","dst"});
    p.add_idb("reach", 8);
    p.add_rule({"reach", {"X"}, {{"edge", {"X","D"}}}});
    p.add_rule({"reach", {"X"}, {{"reach", {"Y"}}, {"edge", {"Y","X"}}}});
    p.evaluate();
    for (int i : {0,1,2,5,6,7}) EXPECT_TRUE(bt(p.get_bits("reach"), i)) << i;
    for (int i : {3,4})         EXPECT_FALSE(bt(p.get_bits("reach"), i)) << i;
}

TEST(Datalog, TCDeep) {
    std::vector<std::pair<uint32_t,uint32_t>> es;
    for (uint32_t i = 0; i < 100; ++i) es.push_back({i, i + 1});
    auto edges = make_edges(std::move(es));
    datalog::program p;
    p.add_edb("edge", edges, {"src","dst"});
    p.add_idb("reach", 101);
    p.add_rule({"reach", {"X"}, {{"edge", {"X","D"}}}});
    p.add_rule({"reach", {"X"}, {{"reach", {"Y"}}, {"edge", {"Y","X"}}}});
    p.evaluate();
    EXPECT_EQ(popcnt(p.get_bits("reach"), 101), 101u);
}

TEST(Datalog, FixpointSaturates) {
    auto edges = make_edges({{0,1},{1,0}});
    datalog::program p;
    p.add_edb("edge", edges, {"src","dst"});
    p.add_idb("reach", 2);
    p.add_rule({"reach", {"X"}, {{"edge", {"X","D"}}}});
    p.add_rule({"reach", {"X"}, {{"reach", {"Y"}}, {"edge", {"Y","X"}}}});
    p.evaluate();
    EXPECT_EQ(popcnt(p.get_bits("reach"), 2), 2u);
}

TEST(Datalog, SimpleNegation) {
    auto nodes = make_nodes(5);
    auto edges = make_edges({{0,1},{1,2}});
    datalog::program p;
    p.add_edb("nodes", nodes, {"id"});
    p.add_edb("edge", edges, {"src","dst"});
    p.add_idb("reach", 5);
    p.add_idb("unreach", 5);
    p.add_rule({"reach", {"X"}, {{"edge", {"X","D"}}}});
    p.add_rule({"reach", {"X"}, {{"reach", {"Y"}}, {"edge", {"Y","X"}}}});
    p.add_rule({"unreach", {"X"}, {{"nodes", {"X"}}, {"reach", {"X"}, true}}});
    p.evaluate();
    for (int i : {0,1,2}) EXPECT_TRUE(bt(p.get_bits("reach"), i));
    EXPECT_FALSE(bt(p.get_bits("reach"), 3));
    for (int i : {0,1,2}) EXPECT_FALSE(bt(p.get_bits("unreach"), i));
    EXPECT_TRUE(bt(p.get_bits("unreach"), 3));
    EXPECT_TRUE(bt(p.get_bits("unreach"), 4));
}

TEST(Datalog, NegationAllReachable) {
    auto nodes = make_nodes(3);
    auto edges = make_edges({{0,1},{1,2},{2,0}});
    datalog::program p;
    p.add_edb("nodes", nodes, {"id"});
    p.add_edb("edge", edges, {"src","dst"});
    p.add_idb("reach", 3);
    p.add_idb("unreach", 3);
    p.add_rule({"reach", {"X"}, {{"edge", {"X","D"}}}});
    p.add_rule({"reach", {"X"}, {{"reach", {"Y"}}, {"edge", {"Y","X"}}}});
    p.add_rule({"unreach", {"X"}, {{"nodes", {"X"}}, {"reach", {"X"}, true}}});
    p.evaluate();
    EXPECT_EQ(popcnt(p.get_bits("reach"), 3), 3u);
    EXPECT_EQ(popcnt(p.get_bits("unreach"), 3), 0u);
}

TEST(Datalog, ThreeStrata) {
    auto nodes = make_nodes(10);
    datalog::program p;
    p.add_edb("nodes", nodes, {"id"});
    p.add_idb("base", 10);
    p.add_idb("derived", 10);
    p.add_idb("fin", 10);
    p.add_rule({"base", {"X"}, {{"nodes", {"X"}}},
        [](const table& t) -> engine::mask_t {
            return engine::execute(t, field_matcher<op::lt, "id", 5>{});
        }, 0});
    p.add_rule({"derived", {"X"}, {{"nodes", {"X"}}, {"base", {"X"}, true}}});
    p.add_rule({"fin", {"X"}, {{"nodes", {"X"}}, {"derived", {"X"}, true}}});
    p.evaluate();
    EXPECT_EQ(popcnt(p.get_bits("base"), 10), 5u);
    EXPECT_EQ(popcnt(p.get_bits("derived"), 10), 5u);
    EXPECT_EQ(popcnt(p.get_bits("fin"), 10), 5u);
    for (int i = 0; i < 5; ++i)  EXPECT_TRUE(bt(p.get_bits("fin"), i));
    for (int i = 5; i < 10; ++i) EXPECT_FALSE(bt(p.get_bits("fin"), i));
}

TEST(Datalog, NaryBinaryPath) {
    auto edges = make_edges({{0,1},{1,2},{2,3}});
    datalog::program p;
    p.add_edb("edge", edges, {"src","dst"});
    p.add_idb_table("path", {"src","dst"});
    p.add_rule({"path", {"X","Y"}, {{"edge", {"X","Y"}}}});
    p.add_rule({"path", {"X","Y"}, {{"path", {"X","Z"}}, {"edge", {"Z","Y"}}}});
    p.evaluate();
    auto s = p.get_store("path");
    EXPECT_EQ(s.nrows, 6u);
    EXPECT_TRUE(has_fact(s, {0,1}));
    EXPECT_TRUE(has_fact(s, {0,2}));
    EXPECT_TRUE(has_fact(s, {0,3}));
    EXPECT_TRUE(has_fact(s, {1,2}));
    EXPECT_TRUE(has_fact(s, {1,3}));
    EXPECT_TRUE(has_fact(s, {2,3}));
}

TEST(Datalog, NaryPathCycle) {
    auto edges = make_edges({{0,1},{1,2},{2,0}});
    datalog::program p;
    p.add_edb("edge", edges, {"src","dst"});
    p.add_idb_table("path", {"src","dst"});
    p.add_rule({"path", {"X","Y"}, {{"edge", {"X","Y"}}}});
    p.add_rule({"path", {"X","Y"}, {{"path", {"X","Z"}}, {"edge", {"Z","Y"}}}});
    p.evaluate();
    auto s = p.get_store("path");
    EXPECT_EQ(s.nrows, 9u);
    for (uint32_t i = 0; i < 3; ++i)
        for (uint32_t j = 0; j < 3; ++j)
            EXPECT_TRUE(has_fact(s, {i, j})) << "(" << i << "," << j << ")";
}

TEST(Datalog, NaryDedup) {
    auto edges = make_edges({{0,1},{0,1},{0,1}});
    datalog::program p;
    p.add_edb("edge", edges, {"src","dst"});
    p.add_idb_table("u", {"a","b"});
    p.add_rule({"u", {"X","Y"}, {{"edge", {"X","Y"}}}});
    p.evaluate();
    EXPECT_EQ(p.get_store("u").nrows, 1u);
}

TEST(Datalog, Projection) {
    auto edges = make_edges({{0,1},{0,2},{1,3}});
    datalog::program p;
    p.add_edb("edge", edges, {"src","dst"});
    p.add_idb_table("proj", {"x"});
    p.add_rule({"proj", {"X"}, {{"edge", {"X","Y"}}}});
    p.evaluate();
    auto s = p.get_store("proj");
    EXPECT_EQ(s.nrows, 2u);
    EXPECT_TRUE(has_fact(s, {0}));
    EXPECT_TRUE(has_fact(s, {1}));
}

TEST(Datalog, TwoTableJoin) {
    auto r1 = make_table(3, {{"a", {0,1,2}}, {"b", {10,20,30}}});
    auto r2 = make_table(3, {{"b", {10,20,30}}, {"c", {100,200,300}}});
    datalog::program p;
    p.add_edb("R", r1, {"a","b"});
    p.add_edb("S", r2, {"b","c"});
    p.add_idb_table("T", {"x","y"});
    p.add_rule({"T", {"A","C"}, {{"R",{"A","B"}}, {"S",{"B","C"}}}});
    p.evaluate();
    auto s = p.get_store("T");
    EXPECT_EQ(s.nrows, 3u);
    EXPECT_TRUE(has_fact(s, {0, 100}));
    EXPECT_TRUE(has_fact(s, {1, 200}));
    EXPECT_TRUE(has_fact(s, {2, 300}));
}

TEST(Datalog, TwoTablePartialJoin) {
    auto r1 = make_table(3, {{"a", {0,1,2}}, {"b", {10,20,30}}});
    auto r2 = make_table(2, {{"b", {20,40}}, {"c", {200,400}}});
    datalog::program p;
    p.add_edb("R", r1, {"a","b"});
    p.add_edb("S", r2, {"b","c"});
    p.add_idb_table("T", {"x","y"});
    p.add_rule({"T", {"A","C"}, {{"R",{"A","B"}}, {"S",{"B","C"}}}});
    p.evaluate();
    auto s = p.get_store("T");
    EXPECT_EQ(s.nrows, 1u);
    EXPECT_TRUE(has_fact(s, {1, 200}));
}

TEST(Datalog, SelfJoin) {
    auto edges = make_edges({{0,1},{1,2},{2,3}});
    datalog::program p;
    p.add_edb("edge", edges, {"src","dst"});
    p.add_idb_table("p2", {"a","b"});
    p.add_rule({"p2", {"A","C"}, {{"edge",{"A","B"}}, {"edge",{"B","C"}}}});
    p.evaluate();
    auto s = p.get_store("p2");
    EXPECT_EQ(s.nrows, 2u);
    EXPECT_TRUE(has_fact(s, {0, 2}));
    EXPECT_TRUE(has_fact(s, {1, 3}));
}

TEST(Datalog, ThreeWayJoin) {
    auto r1 = make_table(2, {{"a", {1,2}}, {"b", {10,20}}});
    auto r2 = make_table(3, {{"b", {10,20,30}}, {"c", {100,200,300}}});
    auto r3 = make_table(2, {{"c", {100,300}}, {"d", {1000,3000}}});
    datalog::program p;
    p.add_edb("R", r1, {"a","b"});
    p.add_edb("S", r2, {"b","c"});
    p.add_edb("T", r3, {"c","d"});
    p.add_idb_table("result", {"x","y"});
    p.add_rule({"result", {"A","D"}, {{"R",{"A","B"}}, {"S",{"B","C"}}, {"T",{"C","D"}}}});
    p.evaluate();
    auto s = p.get_store("result");
    EXPECT_EQ(s.nrows, 1u);
    EXPECT_TRUE(has_fact(s, {1, 1000}));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Benchmarks
// ═══════════════════════════════════════════════════════════════════════════════

static table bench_table(size_t n) {
    table t(n); t.add_column_u32("s");
    for (size_t i=0;i<n;++i) t.get_col("s").u32[i] = i%3;
    return t;
}

static void BM_Execute(benchmark::State& st) {
    auto t = bench_table(st.range(0));
    auto q = field_matcher<op::lt,"id",100>{} & field_matcher<op::eq,"s",1>{};
    for (auto _:st) benchmark::DoNotOptimize(engine::execute(t,q));
    st.SetItemsProcessed(st.range(0)*st.iterations());
}
BENCHMARK(BM_Execute)->RangeMultiplier(10)->Range(1000,1000000);

static void BM_SemiJoin(benchmark::State& st) {
    auto t = bench_table(st.range(0));
    for (auto _:st) benchmark::DoNotOptimize(engine::semi_join(t,field_matcher<op::lt,"id",500>{},"id"));
}
BENCHMARK(BM_SemiJoin)->Arg(100000);

static void BM_Aggregate(benchmark::State& st) {
    table t(st.range(0)); t.add_column_f32("v",42.f);
    for (auto _:st) benchmark::DoNotOptimize(engine::aggregate(t,always,"v",engine::agg::sum));
}
BENCHMARK(BM_Aggregate)->Arg(1000000);

static void BM_SemiNaive(benchmark::State& st) {
    table t(10000); t.add_column_u32("pid", 999999);
    for (uint32_t i=1;i<100;++i) t.get_col("pid").u32[i] = i-1;
    for (auto _:st) {
        auto r = engine::semi_naive_join(t, field_matcher<op::eq,"id",0>{}, "pid");
        benchmark::DoNotOptimize(r);
    }
}
BENCHMARK(BM_SemiNaive);

// ═══════════════════════════════════════════════════════════════════════════════
// Datalog Benchmarks
// ═══════════════════════════════════════════════════════════════════════════════
static void BM_Datalog_BaseCase(benchmark::State& st) {
    size_t n = st.range(0);
    auto nodes = make_nodes(n);
    for (auto _ : st) {
        datalog::program p;
        p.add_edb("n", nodes, {"id"}); p.add_idb("s", n);
        p.add_rule({"s", {"X"}, {{"n", {"X"}}}});
        p.evaluate();
        benchmark::DoNotOptimize(p.get_bits("s").data());
    }
    st.SetItemsProcessed(st.iterations() * (int64_t)n);
}
BENCHMARK(BM_Datalog_BaseCase)->RangeMultiplier(10)->Range(1000, 1'000'000);

static void BM_Datalog_UnaryTC(benchmark::State& st) {
    size_t n = st.range(0);
    std::vector<std::pair<uint32_t,uint32_t>> es;
    for (uint32_t i = 0; i < n; ++i) es.push_back({i, i + 1});
    auto edges = make_edges(std::move(es));
    for (auto _ : st) {
        datalog::program p;
        p.add_edb("edge", edges, {"src","dst"}); p.add_idb("reach", n + 1);
        p.add_rule({"reach", {"X"}, {{"edge", {"X","D"}}}});
        p.add_rule({"reach", {"X"}, {{"reach", {"Y"}}, {"edge", {"Y","X"}}}});
        p.evaluate();
        benchmark::DoNotOptimize(p.get_bits("reach").data());
    }
}
BENCHMARK(BM_Datalog_UnaryTC)->RangeMultiplier(10)->Range(100, 10'000);

static void BM_Datalog_DenseTC(benchmark::State& st) {
    std::vector<std::pair<uint32_t,uint32_t>> es;
    for (uint32_t i = 0; i < 32; ++i)
        for (uint32_t j = 0; j < 32; ++j)
            if (i != j) es.push_back({i, j});
    auto edges = make_edges(std::move(es));
    for (auto _ : st) {
        datalog::program p;
        p.add_edb("edge", edges, {"src","dst"}); p.add_idb("reach", 32);
        p.add_rule({"reach", {"X"}, {{"edge", {"X","D"}}}});
        p.add_rule({"reach", {"X"}, {{"reach", {"Y"}}, {"edge", {"Y","X"}}}});
        p.evaluate();
        benchmark::DoNotOptimize(p.get_bits("reach").data());
    }
}
BENCHMARK(BM_Datalog_DenseTC);

static void BM_Datalog_NaryPath(benchmark::State& st) {
    size_t n = st.range(0);
    std::vector<std::pair<uint32_t,uint32_t>> es;
    for (uint32_t i = 0; i < n; ++i) es.push_back({i, i + 1});
    auto edges = make_edges(std::move(es));
    for (auto _ : st) {
        datalog::program p;
        p.add_edb("edge", edges, {"src","dst"});
        p.add_idb_table("path", {"src","dst"});
        p.add_rule({"path", {"X","Y"}, {{"edge", {"X","Y"}}}});
        p.add_rule({"path", {"X","Y"}, {{"path", {"X","Z"}}, {"edge", {"Z","Y"}}}});
        p.evaluate();
        benchmark::DoNotOptimize(p.get_store("path").nrows);
    }
}
BENCHMARK(BM_Datalog_NaryPath)->Arg(10)->Arg(20)->Arg(50)->Arg(100);

static void BM_Datalog_Negation(benchmark::State& st) {
    size_t n = st.range(0);
    auto nodes = make_nodes(n);
    std::vector<std::pair<uint32_t,uint32_t>> es;
    for (uint32_t i = 0; i < n / 10; ++i) es.push_back({i, i + 1});
    auto edges = make_edges(std::move(es));
    for (auto _ : st) {
        datalog::program p;
        p.add_edb("nodes", nodes, {"id"});
        p.add_edb("edge", edges, {"src","dst"});
        p.add_idb("reach", n); p.add_idb("unreach", n);
        p.add_rule({"reach", {"X"}, {{"edge", {"X","D"}}}});
        p.add_rule({"reach", {"X"}, {{"reach", {"Y"}}, {"edge", {"Y","X"}}}});
        p.add_rule({"unreach", {"X"}, {{"nodes", {"X"}}, {"reach", {"X"}, true}}});
        p.evaluate();
        benchmark::DoNotOptimize(p.get_bits("unreach").data());
    }
    st.SetItemsProcessed(st.iterations() * (int64_t)n);
}
BENCHMARK(BM_Datalog_Negation)->RangeMultiplier(10)->Range(1000, 100'000);

static void BM_Datalog_MultiJoin(benchmark::State& st) {
    size_t n = st.range(0);
    std::vector<uint32_t> a(n), b(n), c(n);
    for (uint32_t i = 0; i < n; ++i) { a[i] = i; b[i] = i + (uint32_t)n; c[i] = i + 2*(uint32_t)n; }
    auto r1 = make_table(n, {{"a", a}, {"b", b}});
    auto r2 = make_table(n, {{"b", b}, {"c", c}});
    for (auto _ : st) {
        datalog::program p;
        p.add_edb("R", r1, {"a","b"}); p.add_edb("S", r2, {"b","c"});
        p.add_idb_table("T", {"x","y"});
        p.add_rule({"T", {"A","C"}, {{"R",{"A","B"}}, {"S",{"B","C"}}}});
        p.evaluate();
        benchmark::DoNotOptimize(p.get_store("T").nrows);
    }
    st.SetItemsProcessed(st.iterations() * (int64_t)n);
}
BENCHMARK(BM_Datalog_MultiJoin)->RangeMultiplier(10)->Range(100, 10'000);

#define RUN_BENCHMARKS 1
#if RUN_BENCHMARKS
BENCHMARK_MAIN();
#else
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
