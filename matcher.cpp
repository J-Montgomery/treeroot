#include <algorithm>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include <gtest/gtest.h>
#include <benchmark/benchmark.h>

// ═══════════════════════════════════════════════════════════════════════════════
// Fixed-Size String for Non-Type Template Parameters
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
// SIMD Abstraction Layer — swap per-architecture
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

template<typename T, typename V> void cmp_eq(uint64_t* m, const T* d, V v, size_t n) { for (size_t i=0;i<n;++i) if (d[i]==(T)v) set(m,i); }
template<typename T, typename V> void cmp_ne(uint64_t* m, const T* d, V v, size_t n) { for (size_t i=0;i<n;++i) if (d[i]!=(T)v) set(m,i); }
template<typename T, typename V> void cmp_lt(uint64_t* m, const T* d, V v, size_t n) { for (size_t i=0;i<n;++i) if (d[i]< (T)v) set(m,i); }
template<typename T, typename V> void cmp_ge(uint64_t* m, const T* d, V v, size_t n) { for (size_t i=0;i<n;++i) if (d[i]>=(T)v) set(m,i); }

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

// --- Type traits ---
template<typename>   struct is_or_i  : std::false_type {};
template<typename>   struct is_and_i : std::false_type {};
template<matcher L, matcher R> struct is_or_i <or_t<L,R>>  : std::true_type {};
template<matcher L, matcher R> struct is_and_i<and_t<L,R>> : std::true_type {};
template<typename T> inline constexpr bool is_or_v  = is_or_i <std::remove_cvref_t<T>>::value;
template<typename T> inline constexpr bool is_and_v = is_and_i<std::remove_cvref_t<T>>::value;

// ---- CPOs ----

struct implies_t {
    template<typename L, typename R>
    constexpr bool operator()(L const& l, R const& r) const {
        if constexpr (requires { tag_invoke(*this, l, r); })
            return tag_invoke(*this, l, r);
        else
            return std::is_same_v<std::remove_cvref_t<L>, std::remove_cvref_t<R>>;
    }
};
inline constexpr implies_t implies;

struct negate_t {
    template<typename M>
    constexpr auto operator()(M const& m) const {
        if constexpr (requires { tag_invoke(*this, m); })
            return tag_invoke(*this, m);
        else
            return not_t{m};
    }
};
inline constexpr negate_t negate;

struct simplify_t {
    template<typename M>
    constexpr auto operator()(M const& m) const {
        if constexpr (requires { tag_invoke(*this, m); })
            return tag_invoke(*this, m);
        else
            return m;
    }
};
inline constexpr simplify_t simplify;

struct sum_of_products_t {
    template<typename M>
    constexpr auto operator()(M const& m) const {
        if constexpr (requires { tag_invoke(*this, m); })
            return tag_invoke(*this, m);
        else
            return m;
    }
};
inline constexpr sum_of_products_t sum_of_products;

// ---- Structural rules: implies ----
constexpr bool tag_invoke(implies_t, never_t, auto const&)  { return true; }
constexpr bool tag_invoke(implies_t, auto const&, always_t) { return true; }
constexpr bool tag_invoke(implies_t, always_t, never_t)     { return false; }

template<matcher L, matcher R, matcher M>
constexpr bool tag_invoke(implies_t, and_t<L,R> const& a, M const& m) {
    return implies(a.lhs, m) || implies(a.rhs, m);
}
template<matcher M, matcher L, matcher R>
constexpr bool tag_invoke(implies_t, M const& m, or_t<L,R> const& o) {
    return implies(m, o.lhs) || implies(m, o.rhs);
}

// ---- Structural rules: negate ----
constexpr never_t  tag_invoke(negate_t, always_t) { return {}; }
constexpr always_t tag_invoke(negate_t, never_t)  { return {}; }
template<matcher M> constexpr M tag_invoke(negate_t, not_t<M> const& n) { return n.m; }

// ---- Structural rules: simplify ----
template<matcher M>
constexpr auto tag_invoke(simplify_t, not_t<M> const& n) { return negate(simplify(n.m)); }

// ---- Structural rules: sum_of_products (DNF) ----
template<matcher L, matcher R>
constexpr auto tag_invoke(sum_of_products_t, or_t<L,R> const& m) {
    return or_t{sum_of_products(m.lhs), sum_of_products(m.rhs)};
}

template<matcher L, matcher R>
constexpr auto tag_invoke(sum_of_products_t, and_t<L,R> const& m) {
    auto l = sum_of_products(m.lhs);
    auto r = sum_of_products(m.rhs);
    using LS = decltype(l); using RS = decltype(r);
    if constexpr (is_or_v<LS>) {
        return or_t{sum_of_products(and_t<typename LS::lhs_t, RS>{l.lhs, r}),
                     sum_of_products(and_t<typename LS::rhs_t, RS>{l.rhs, r})};
    } else if constexpr (is_or_v<RS>) {
        return or_t{sum_of_products(and_t<LS, typename RS::lhs_t>{l, r.lhs}),
                     sum_of_products(and_t<LS, typename RS::rhs_t>{l, r.rhs})};
    } else {
        return and_t<LS, RS>{l, r};
    }
}

template<matcher M>
constexpr auto tag_invoke(sum_of_products_t, not_t<M> const& n) {
    if constexpr (is_and_v<M>)
        return or_t{sum_of_products(negate(n.m.lhs)), sum_of_products(negate(n.m.rhs))};
    else if constexpr (is_or_v<M>)
        return sum_of_products(and_t{sum_of_products(negate(n.m.lhs)),
                                      sum_of_products(negate(n.m.rhs))});
    else return n;
}

// ---- Runtime bitmap matcher ----
struct field_in {
    using is_matcher = void;
    std::string_view field;
    std::shared_ptr<std::vector<uint64_t>> bitmap;
};

// ---- Static-matcher trait (gates compile-time simplification in operators) ----
template<typename T>           inline constexpr bool is_static_v = true;
template<>                     inline constexpr bool is_static_v<field_in> = false;
template<typename M>           inline constexpr bool is_static_v<not_t<M>>    = is_static_v<M>;
template<typename L,typename R>inline constexpr bool is_static_v<and_t<L,R>> = is_static_v<L> && is_static_v<R>;
template<typename L,typename R>inline constexpr bool is_static_v<or_t<L,R>>  = is_static_v<L> && is_static_v<R>;

// ---- Expression operators ----

template<matcher L, matcher R>
constexpr auto operator&(L l, R r) {
    if constexpr (!is_static_v<L> || !is_static_v<R>) {
        if constexpr (std::is_same_v<L, never_t> || std::is_same_v<R, never_t>) return never;
        else if constexpr (std::is_same_v<L, always_t>) return r;
        else if constexpr (std::is_same_v<R, always_t>) return l;
        else return and_t<L,R>{l, r};
    } else {
        auto sl = simplify(l); auto sr = simplify(r);
        using TL = decltype(sl); using TR = decltype(sr);
        if constexpr (std::is_same_v<TL,never_t>||std::is_same_v<TR,never_t>) return never;
        else if constexpr (std::is_same_v<TL,always_t>) return sr;
        else if constexpr (std::is_same_v<TR,always_t>) return sl;
        else if constexpr (implies(sl, sr)) return sl;
        else if constexpr (implies(sr, sl)) return sr;
        else if constexpr (implies(sl,negate(sr))||implies(sr,negate(sl))) return never;
        else return and_t<TL,TR>{sl, sr};
    }
}

template<matcher L, matcher R>
constexpr auto operator|(L l, R r) {
    if constexpr (!is_static_v<L> || !is_static_v<R>) {
        if constexpr (std::is_same_v<L,always_t>||std::is_same_v<R,always_t>) return always;
        else if constexpr (std::is_same_v<L,never_t>) return r;
        else if constexpr (std::is_same_v<R,never_t>) return l;
        else return or_t<L,R>{l, r};
    } else {
        auto sl = simplify(l); auto sr = simplify(r);
        using TL = decltype(sl); using TR = decltype(sr);
        if constexpr (std::is_same_v<TL,always_t>||std::is_same_v<TR,always_t>) return always;
        else if constexpr (std::is_same_v<TL,never_t>) return sr;
        else if constexpr (std::is_same_v<TR,never_t>) return sl;
        else if constexpr (implies(sl, sr)) return sr;
        else if constexpr (implies(sr, sl)) return sl;
        else if constexpr (implies(negate(sl),sr)||implies(negate(sr),sl)) return always;
        else return or_t<TL,TR>{sl, sr};
    }
}

template<matcher M>
constexpr auto operator!(M m) { return simplify(not_t{m}); }

} // namespace match

// ═══════════════════════════════════════════════════════════════════════════════
// Field Matchers
// ═══════════════════════════════════════════════════════════════════════════════

enum class op { eq, ne, lt, ge };

template<op O, fs Field, auto Val>
struct field_matcher {
    using is_matcher = void;

    // Complete implication table: Eq→*, Lt→Lt/Ne, Ge→Ge/Ne, Ne→Ne
    template<op O2, fs F2, auto V2>
    friend constexpr bool tag_invoke(match::implies_t, field_matcher const&,
                                     field_matcher<O2,F2,V2> const&) {
        if constexpr (!(Field == F2)) return false;
        else if constexpr (O == op::eq) {
            if constexpr (O2==op::eq) return Val==V2;
            if constexpr (O2==op::ne) return Val!=V2;
            if constexpr (O2==op::lt) return Val< V2;
            if constexpr (O2==op::ge) return Val>=V2;
        }
        else if constexpr (O == op::lt) {
            if constexpr (O2==op::lt) return Val<=V2;
            if constexpr (O2==op::ne) return Val<=V2;
            else return false;
        }
        else if constexpr (O == op::ge) {
            if constexpr (O2==op::ge) return Val>=V2;
            if constexpr (O2==op::ne) return Val> V2;
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
// Columnar Table with Bloom-Filter Chunk Summaries
// ═══════════════════════════════════════════════════════════════════════════════

struct table {
    size_t rows = 0;
    size_t chunk_size = 1024; // must be multiple of 64 for aligned chunk merge

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

struct engine {
    using mask_t = simd::mask_t;

    // Execute with bloom-filter chunk skipping
    static mask_t execute(const table& t, match::matcher auto const& m) {
        auto sop = match::sum_of_products(m);
        uint64_t req = extract_bits(sop);
        mask_t result(simd::num_words(t.rows), 0);

        if (req == 0 || t.chunk_summaries.empty()) {
            result = eval(t, sop, 0, t.rows);
        } else {
            for (size_t c = 0; c < t.num_chunks(); ++c) {
                if ((t.chunk_summaries[c] & req) != req) continue;
                size_t start = c * t.chunk_size;
                size_t count = std::min(t.chunk_size, t.rows - start);
                auto chunk = eval(t, sop, start, count);
                simd::bor(result.data() + start / 64, chunk.data(), chunk.size());
            }
        }
        simd::clear_tail(result.data(), result.size(), t.rows);
        return result;
    }

    // Semi-join: project query results into a bitmap matcher on fk_field
    static match::field_in semi_join(const table& t, match::matcher auto const& q,
                                     std::string_view fk_field) {
        return {fk_field, std::make_shared<mask_t>(execute(t, q))};
    }

    // Semi-naive join: transitive closure via iterated frontier expansion
    static match::field_in semi_naive_join(const table& t,
                                           match::matcher auto const& seed,
                                           std::string_view fk_field,
                                           std::string_view id_field = "id") {
        auto delta = execute(t, seed);
        auto total = delta;
        while (simd::any(delta.data(), delta.size())) {
            match::field_in frontier{fk_field, std::make_shared<mask_t>(delta)};
            auto reachable = execute(t, frontier);
            delta = reachable;
            simd::bandnot(delta.data(), total.data(), delta.size());
            simd::bor(total.data(), delta.data(), total.size());
        }
        return {id_field, std::make_shared<mask_t>(std::move(total))};
    }

    // Aggregate reductions over query results
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
    // ---- Extract required mask bits from AND chains ----
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

    // ---- Aggregate impl ----
    template<typename T>
    static double agg_impl(const mask_t& mask, const std::vector<T>& data,
                           agg op, size_t rows) {
        if (op == agg::count)
            return (double)simd::popcount(mask.data(), mask.size());
        double r = 0; size_t n = 0; bool first = true;
        for (size_t i = 0; i < rows; ++i) {
            if (!simd::test(mask.data(), i)) continue;
            double v = (double)data[i];
            if      (op==agg::sum||op==agg::mean) r += v;
            else if (op==agg::max) { if (first||v>r) r=v; }
            else if (op==agg::min) { if (first||v<r) r=v; }
            first = false; ++n;
        }
        return (op == agg::mean && n) ? r / n : r;
    }

    // ---- Eval: returns local mask for rows [start .. start+count) ----

    static mask_t eval(const table&, match::always_t, size_t, size_t count) {
        mask_t m(simd::num_words(count), ~0ULL);
        simd::clear_tail(m.data(), m.size(), count);
        return m;
    }
    static mask_t eval(const table&, match::never_t, size_t, size_t count) {
        return mask_t(simd::num_words(count), 0);
    }

    template<match::matcher L, match::matcher R>
    static mask_t eval(const table& t, match::and_t<L,R> const& a, size_t s, size_t c) {
        auto l = eval(t, a.lhs, s, c);
        auto r = eval(t, a.rhs, s, c);
        simd::band(l.data(), r.data(), l.size());
        return l;
    }
    template<match::matcher L, match::matcher R>
    static mask_t eval(const table& t, match::or_t<L,R> const& o, size_t s, size_t c) {
        auto l = eval(t, o.lhs, s, c);
        auto r = eval(t, o.rhs, s, c);
        simd::bor(l.data(), r.data(), l.size());
        return l;
    }
    template<match::matcher M>
    static mask_t eval(const table& t, match::not_t<M> const& n, size_t s, size_t c) {
        auto m = eval(t, n.m, s, c);
        simd::bnot(m.data(), m.size());
        simd::clear_tail(m.data(), m.size(), c);
        return m;
    }

    template<op O, fs Field, auto Val>
    static mask_t eval(const table& t, field_matcher<O,Field,Val> const&,
                       size_t start, size_t count) {
        auto& col = t.get_col(Field.view());
        mask_t m(simd::num_words(count), 0);
        auto cmp = [&]<typename T>(const std::vector<T>& d) {
            const T* p = d.data() + start;
            if constexpr (O==op::eq) simd::cmp_eq(m.data(), p, (T)Val, count);
            if constexpr (O==op::ne) simd::cmp_ne(m.data(), p, (T)Val, count);
            if constexpr (O==op::lt) simd::cmp_lt(m.data(), p, (T)Val, count);
            if constexpr (O==op::ge) simd::cmp_ge(m.data(), p, (T)Val, count);
        };
        if      (!col.u32.empty()) cmp(col.u32);
        else if (!col.f32.empty()) cmp(col.f32);
        else if (!col.u64.empty()) cmp(col.u64);
        return m;
    }

    template<fs Field, fs Path>
    static mask_t eval(const table& t, field_hst<Field,Path> const&,
                       size_t start, size_t count) {
        auto& d = t.get_col(Field.view()).str;
        mask_t m(simd::num_words(count), 0);
        for (size_t i = 0; i < count; ++i)
            if (d[start+i].starts_with(Path.view())) simd::set(m.data(), i);
        return m;
    }

    static mask_t eval(const table& t, match::field_in const& f,
                       size_t start, size_t count) {
        auto& d = t.get_col(f.field).u32;
        auto& bm = *f.bitmap;
        mask_t m(simd::num_words(count), 0);
        for (size_t i = 0; i < count; ++i) {
            uint32_t v = d[start+i];
            if (v/64 < bm.size() && simd::test(bm.data(), v))
                simd::set(m.data(), i);
        }
        return m;
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

using namespace match;

// ---- Compile-Time Algebra ----

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
    using ge50 = field_matcher<op::ge, "x", 50>;
    using ne10 = field_matcher<op::ne, "x", 10>;
    using ge30 = field_matcher<op::ge, "x", 30>;
    using eq15 = field_matcher<op::eq, "x", 15>;
    using lt20 = field_matcher<op::lt, "x", 20>;
    using ge10 = field_matcher<op::ge, "x", 10>;
    using ge5  = field_matcher<op::ge, "x", 5>;

    // Lt → Ne: x<10 ⟹ x≠40
    static_assert(std::is_same_v<decltype(lt10{} & ne40{}), lt10>);
    // Ge → Ne: x≥50 ⟹ x≠10
    static_assert(std::is_same_v<decltype(ge50{} & ne10{}), ge50>);
    // OR merging: (x≥50)|(x≥30) → x≥30
    static_assert(std::is_same_v<decltype(ge50{} | ge30{}), ge30>);
    // Eq subsumption
    static_assert(std::is_same_v<decltype(eq15{} & lt20{}), eq15>);
    // Range tautology: negate(lt20)=ge20, ge20⟹ge10
    static_assert(std::is_same_v<decltype(lt20{} | ge10{}), always_t>);
    // Not-pushdown: !(x<10) & (x≥5) → ge10 & ge5 → ge10
    static_assert(std::is_same_v<decltype(!lt10{} & ge5{}), field_matcher<op::ge,"x",10>>);
    // Overlapping ranges stay as And
    static_assert(is_and_v<decltype(lt20{} & ge10{})>);
}

TEST(Algebra, SumOfProducts) {
    using A = field_matcher<op::eq, "x", 1>;
    using B = field_matcher<op::eq, "y", 2>;
    using C = field_matcher<op::eq, "z", 3>;
    // (A|B) & C distributes to (A&C)|(B&C)
    auto expr = (A{}|B{}) & C{};
    static_assert(is_and_v<decltype(expr)>);
    auto dnf = sum_of_products(expr);
    static_assert(is_or_v<decltype(dnf)>);
}

TEST(Algebra, HST) {
    using child  = field_hst<"type","sensor/temp">;
    using parent = field_hst<"type","sensor">;
    using other  = field_hst<"type","motor">;
    static_assert(std::is_same_v<decltype(child{}&parent{}), child>);
    static_assert(is_and_v<decltype(child{}&other{})>);
}

// ---- Runtime Engine ----

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
    table t(10000, 1024);
    t.add_column_u32("dur", 100);
    for (int i=0;i<5;++i) { t.set_tag(i,0); t.set_tag(i,1); }
    for (int i=0;i<5;++i) { t.set_tag(5*1024+i,0); t.set_tag(5*1024+i,1); }
    auto q = field_matcher<op::eq,"mask",3>{} & field_matcher<op::lt,"dur",150>{};
    auto r = engine::execute(t, q);
    EXPECT_EQ(simd::popcount(r.data(), r.size()), 10u);
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

#define RUN_BENCHMARKS 0
#if RUN_BENCHMARKS
BENCHMARK_MAIN();
#else
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
