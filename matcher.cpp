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
#include <future>

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
// Lexicographical Hierarchy Manager
// ═══════════════════════════════════════════════════════════════════════════════

class hierarchy_dict {
    std::vector<std::string> strings_;
    bool frozen_ = false;
public:
    hierarchy_dict() = default;
    template<typename It>
    hierarchy_dict(It begin, It end) : strings_(begin, end) { freeze(); }

    void insert(std::string s) {
        if (!frozen_) strings_.push_back(std::move(s));
    }

    void freeze() {
        if (frozen_) return;
        std::sort(strings_.begin(), strings_.end());
        strings_.erase(std::unique(strings_.begin(), strings_.end()), strings_.end());
        frozen_ = true;
    }

    uint32_t get_id(std::string_view s) const {
        auto it = std::lower_bound(strings_.begin(), strings_.end(), s);
        if (it != strings_.end() && *it == s) return std::distance(strings_.begin(), it);
        return 0; 
    }

    std::pair<uint32_t, uint32_t> get_prefix_range(std::string_view prefix) const {
        auto it_start = std::lower_bound(strings_.begin(), strings_.end(), prefix);
        std::string upper = std::string(prefix);
        if (!upper.empty()) {
            upper.back()++; 
        }
        auto it_end = std::lower_bound(strings_.begin(), strings_.end(), upper);
        return {
            std::distance(strings_.begin(), it_start),
            std::distance(strings_.begin(), it_end)
        };
    }
};

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

struct column_view {
    const float* f32 = nullptr;
    const uint32_t* u32 = nullptr;
    const uint64_t* u64 = nullptr;
    const uint64_t* chunk_summaries = nullptr;
};

template <size_t CS = 1024>
struct table_view {
    size_t rows = 0;
    size_t chunk_size = CS;
    std::unordered_map<std::string_view, column_view> cols;
    const std::unordered_map<std::string, hierarchy_dict>* dicts = nullptr;

    size_t num_chunks() const { return rows ? (rows + chunk_size - 1) / chunk_size : 0; }
    const column_view& get_col(std::string_view name) const {
        auto it = cols.find(name);
        if (it == cols.end()) throw std::runtime_error("no col: " + std::string(name));
        return it->second;
    }
    bool chunk_summaries_empty() const {
        for (auto& [k, c] : cols) if (c.chunk_summaries) return false;
        return true;
    }
};

struct table {
    size_t rows = 0;
    size_t chunk_size = 1024;

    struct column {
        std::string_view name;
        std::vector<uint32_t> u32;
        std::vector<float>    f32;
        std::vector<uint64_t> u64;
    };

    std::vector<column>   cols;
    std::unordered_map<std::string, hierarchy_dict> dicts;
    std::vector<uint64_t> chunk_summaries;

    explicit table(size_t n = 0, size_t cs = 1024) : rows(n), chunk_size(cs) {
        assert(cs % 64 == 0);
        std::vector<uint32_t> ids(n);
        std::iota(ids.begin(), ids.end(), 0u);
        cols.push_back({"id", std::move(ids), {}, {}});
        cols.push_back({"mask", {}, {}, std::vector<uint64_t>(n, 0)});
        chunk_summaries.resize(num_chunks(), 0);
    }

    size_t num_chunks() const { return rows ? (rows + chunk_size - 1) / chunk_size : 0; }
    column&       get_col(std::string_view n)       { for (auto& c:cols) if (c.name==n) return c; throw std::runtime_error(std::string("no col: ") + std::string(n)); }
    const column& get_col(std::string_view n) const { for (auto& c:cols) if (c.name==n) return c; throw std::runtime_error(std::string("no col: ") + std::string(n)); }

    void add_column_u32(std::string_view n, uint32_t d=0) { cols.push_back({n, std::vector<uint32_t>(rows,d), {}, {}}); }
    void add_column_f32(std::string_view n, float d=0.f)  { cols.push_back({n, {}, std::vector<float>(rows,d), {}}); }

    void add_column_hst(std::string_view n, const std::vector<std::string>& vals) {
        hierarchy_dict dict(vals.begin(), vals.end());
        std::vector<uint32_t> ids(rows);
        for (size_t i = 0; i < rows && i < vals.size(); ++i) ids[i] = dict.get_id(vals[i]);
        cols.push_back({n, std::move(ids), {}, {}});
        dicts[std::string(n)] = std::move(dict);
    }

    void set_tag(size_t eid, int tag) {
        uint64_t bit = 1ULL << (tag % 64);
        get_col("mask").u64[eid] |= bit;
        chunk_summaries[eid / chunk_size] |= bit;
    }

    bool chunk_summaries_empty() const { return chunk_summaries.empty(); }

    template<size_t CS = 1024>
    table_view<CS> to_view() const {
        table_view<CS> tv;
        tv.rows = rows;
        tv.chunk_size = chunk_size;
        for (const auto& c : cols) {
            column_view cv;
            cv.f32 = c.f32.empty() ? nullptr : c.f32.data();
            cv.u32 = c.u32.empty() ? nullptr : c.u32.data();
            cv.u64 = c.u64.empty() ? nullptr : c.u64.data();
            cv.chunk_summaries = chunk_summaries.empty() ? nullptr : chunk_summaries.data();
            tv.cols[c.name] = cv;
        }
        tv.dicts = &dicts;
        return tv;
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// Execution Engine
// ═══════════════════════════════════════════════════════════════════════════════

namespace detail {
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

    template<size_t CS>
    static mask_t execute(const table_view<CS>& t, match::matcher auto const& m) {
        auto sop = match::sum_of_products(m);
        uint64_t req = extract_bits(sop);
        size_t nw = simd::num_words(t.rows);
        mask_t result(nw, 0);

        if (req == 0 || t.chunk_summaries_empty()) {
            mask_t scratch(nw);
            eval_into_view(t, sop, 0, t.rows, result.data(), scratch.data(), nw);
        } else {
            size_t max_cnw = simd::num_words(t.chunk_size);
            mask_t chunk_buf(max_cnw);
            mask_t scratch(max_cnw);
            for (size_t c = 0; c < t.num_chunks(); ++c) {
                auto& col = t.get_col("mask");
                uint64_t summary = col.chunk_summaries ? col.chunk_summaries[c] : 0;
                if ((summary & req) != req) continue;
                size_t start = c * t.chunk_size;
                size_t count = std::min(t.chunk_size, t.rows - start);
                size_t cnw = simd::num_words(count);
                eval_into_view(t, sop, start, count, chunk_buf.data(), scratch.data(), cnw);
                simd::bor(result.data() + start / 64, chunk_buf.data(), cnw);
            }
        }
        simd::clear_tail(result.data(), nw, t.rows);
        return result;
    }

    template<size_t CS>
    static match::field_in semi_join(const table_view<CS>& t, match::matcher auto const& q,
                                     std::string_view fk_field) {
        return {fk_field, std::make_shared<mask_t>(execute(t, q))};
    }

    template<size_t CS>
    static double aggregate(const table_view<CS>& t, match::matcher auto const& q,
                            std::string_view col, agg op) {
        auto mask = execute(t, q);
        auto& c = t.get_col(col);
        if (c.f32) return agg_impl_view(mask, c.f32, op, t.rows);
        if (c.u32) return agg_impl_view(mask, c.u32, op, t.rows);
        if (c.u64) return agg_impl_view(mask, c.u64, op, t.rows);
        return 0.0;
    }

private:
    template<typename M>
    static uint64_t extract_bits(M const&) { return 0; }

    template<op O, fs Field, auto Val>
    static uint64_t extract_bits(field_matcher<O, Field, Val> const&) {
        if constexpr (O == op::eq && Field == fs("mask")) return static_cast<uint64_t>(Val);
        else return 0;
    }
    template<match::matcher L, match::matcher R>
    static uint64_t extract_bits(match::and_t<L,R> const& a) { return extract_bits(a.lhs) | extract_bits(a.rhs); }
    template<match::matcher L, match::matcher R>
    static uint64_t extract_bits(match::or_t<L,R> const& o) {
        auto l = extract_bits(o.lhs), r = extract_bits(o.rhs);
        return (l && r) ? (l & r) : 0;
    }

    template<typename T>
    static double agg_impl(const mask_t& mask, const std::vector<T>& data,
                           agg op, size_t rows) {
        if (op == agg::count) return (double)simd::popcount(mask.data(), mask.size());
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

    template<typename T>
    static double agg_impl_view(const mask_t& mask, const T* data,
                                agg op, size_t rows) {
        if (op == agg::count) return (double)simd::popcount(mask.data(), mask.size());
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

    static void eval_into(const table&, match::always_t, size_t, size_t count,
                          uint64_t* out, uint64_t*, size_t nw) {
        std::fill_n(out, nw, ~0ULL);
        simd::clear_tail(out, nw, count);
    }
    static void eval_into(const table&, match::never_t, size_t, size_t,
                          uint64_t* out, uint64_t*, size_t nw) { std::fill_n(out, nw, 0); }

    template<match::matcher L, match::matcher R>
    static void eval_into(const table& t, match::and_t<L,R> const& a,
                          size_t s, size_t c, uint64_t* out, uint64_t* scratch, size_t nw) {
        eval_into(t, a.lhs, s, c, out, scratch, nw);
        if constexpr (detail::needs_scratch<R>) {
            mask_t tmp(nw); eval_into(t, a.rhs, s, c, scratch, tmp.data(), nw);
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
            mask_t tmp(nw); eval_into(t, o.rhs, s, c, scratch, tmp.data(), nw);
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
        auto it = t.dicts.find(std::string(Field.view()));
        if (it == t.dicts.end()) { std::fill_n(out, nw, 0); return; }

        auto [low, high] = it->second.get_prefix_range(Path.view());
        if (low == high) { std::fill_n(out, nw, 0); return; }

        auto& col = t.get_col(Field.view()).u32;
        const uint32_t* p = col.data() + start;
        simd::cmp_fill(out, p, 0, count, [=](uint32_t v, auto) {
            return v >= low && v < high;
        });
    }

    static void eval_into(const table& t, match::field_in const& f,
                          size_t start, size_t count, uint64_t* out, uint64_t*, size_t nw) {
        std::fill_n(out, nw, 0);
        auto& d = t.get_col(f.field).u32;
        auto& bm = *f.bitmap;
        for (size_t i = 0; i < count; ++i) {
            uint32_t v = d[start + i];
            if (v / 64 < bm.size() && simd::test(bm.data(), v)) simd::set(out, i);
        }
    }

    template<size_t CS>
    static void eval_into_view(const table_view<CS>&, match::always_t, size_t, size_t count,
                               uint64_t* out, uint64_t*, size_t nw) {
        std::fill_n(out, nw, ~0ULL);
        simd::clear_tail(out, nw, count);
    }

    template<size_t CS>
    static void eval_into_view(const table_view<CS>&, match::never_t, size_t, size_t,
                               uint64_t* out, uint64_t*, size_t nw) { 
        std::fill_n(out, nw, 0); 
    }

    template<size_t CS, match::matcher L, match::matcher R>
    static void eval_into_view(const table_view<CS>& t, match::and_t<L,R> const& a,
                               size_t s, size_t c, uint64_t* out, uint64_t* scratch, size_t nw) {
        eval_into_view(t, a.lhs, s, c, out, scratch, nw);
        if constexpr (detail::needs_scratch<R>) {
            mask_t tmp(nw); eval_into_view(t, a.rhs, s, c, scratch, tmp.data(), nw);
        } else {
            eval_into_view(t, a.rhs, s, c, scratch, nullptr, nw);
        }
        simd::band(out, scratch, nw);
    }

    template<size_t CS, match::matcher L, match::matcher R>
    static void eval_into_view(const table_view<CS>& t, match::or_t<L,R> const& o,
                               size_t s, size_t c, uint64_t* out, uint64_t* scratch, size_t nw) {
        eval_into_view(t, o.lhs, s, c, out, scratch, nw);
        if constexpr (detail::needs_scratch<R>) {
            mask_t tmp(nw); eval_into_view(t, o.rhs, s, c, scratch, tmp.data(), nw);
        } else {
            eval_into_view(t, o.rhs, s, c, scratch, nullptr, nw);
        }
        simd::bor(out, scratch, nw);
    }

    template<size_t CS, match::matcher M>
    static void eval_into_view(const table_view<CS>& t, match::not_t<M> const& n,
                               size_t s, size_t c, uint64_t* out, uint64_t* scratch, size_t nw) {
        eval_into_view(t, n.m, s, c, out, scratch, nw);
        simd::bnot(out, nw);
        simd::clear_tail(out, nw, c);
    }

    template<size_t CS, op O, fs Field, auto Val>
    static void eval_into_view(const table_view<CS>& t, field_matcher<O,Field,Val> const&,
                               size_t start, size_t count, uint64_t* out, uint64_t*, size_t nw) {
        auto& col = t.get_col(Field.view());
        if (col.u32) {
            if constexpr (O==op::eq) simd::cmp_eq(out, col.u32 + start, (uint32_t)Val, count);
            if constexpr (O==op::ne) simd::cmp_ne(out, col.u32 + start, (uint32_t)Val, count);
            if constexpr (O==op::lt) simd::cmp_lt(out, col.u32 + start, (uint32_t)Val, count);
            if constexpr (O==op::ge) simd::cmp_ge(out, col.u32 + start, (uint32_t)Val, count);
        } else if (col.f32) {
            if constexpr (O==op::eq) simd::cmp_eq(out, col.f32 + start, (float)Val, count);
            if constexpr (O==op::ne) simd::cmp_ne(out, col.f32 + start, (float)Val, count);
            if constexpr (O==op::lt) simd::cmp_lt(out, col.f32 + start, (float)Val, count);
            if constexpr (O==op::ge) simd::cmp_ge(out, col.f32 + start, (float)Val, count);
        } else if (col.u64) {
            if constexpr (O==op::eq) simd::cmp_eq(out, col.u64 + start, (uint64_t)Val, count);
            if constexpr (O==op::ne) simd::cmp_ne(out, col.u64 + start, (uint64_t)Val, count);
            if constexpr (O==op::lt) simd::cmp_lt(out, col.u64 + start, (uint64_t)Val, count);
            if constexpr (O==op::ge) simd::cmp_ge(out, col.u64 + start, (uint64_t)Val, count);
        } else {
            std::fill_n(out, nw, 0);
        }
    }

    template<size_t CS, fs Field, fs Path>
    static void eval_into_view(const table_view<CS>& t, field_hst<Field,Path> const&,
                               size_t start, size_t count, uint64_t* out, uint64_t*, size_t nw) {
        if (!t.dicts) { std::fill_n(out, nw, 0); return; }
        auto it = t.dicts->find(std::string(Field.view()));
        if (it == t.dicts->end()) { std::fill_n(out, nw, 0); return; }

        auto [low, high] = it->second.get_prefix_range(Path.view());
        if (low == high) { std::fill_n(out, nw, 0); return; }

        auto& col = t.get_col(Field.view());
        if (!col.u32) { std::fill_n(out, nw, 0); return; }

        simd::cmp_fill(out, col.u32 + start, 0, count, [=](uint32_t v, auto) {
            return v >= low && v < high;
        });
    }

    template<size_t CS>
    static void eval_into_view(const table_view<CS>& t, match::field_in const& f,
                               size_t start, size_t count, uint64_t* out, uint64_t*, size_t nw) {
        std::fill_n(out, nw, 0);
        auto& col = t.get_col(f.field);
        if (!col.u32) return;
        auto& bm = *f.bitmap;
        for (size_t i = 0; i < count; ++i) {
            uint32_t v = col.u32[start + i];
            if (v / 64 < bm.size() && simd::test(bm.data(), v)) simd::set(out, i);
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// Monadic Datalog Engine
// ═══════════════════════════════════════════════════════════════════════════════

namespace datalog {

using mask_t = engine::mask_t;

namespace bits {
    inline size_t words(size_t n) { return (n + 63) / 64; }
    inline bool test(const mask_t& m, size_t i) { return i / 64 < m.size() && (m[i / 64] & (1ULL << (i % 64))); }
    inline void set(mask_t& m, size_t i) { if (i / 64 < m.size()) m[i / 64] |= (1ULL << (i % 64)); }
    inline void clear_tail(mask_t& m, size_t n) { if (n % 64 && !m.empty()) m.back() &= (1ULL << (n % 64)) - 1; }
    inline bool any(const mask_t& m) { for (auto w : m) if (w) return true; return false; }
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
    struct edb_entry { std::string name; const table* tbl; std::vector<std::string> cols; };
    struct idb_entry { std::string name; size_t universe; mask_t mask; };

    friend class batch_program;
    std::vector<edb_entry> edbs_;
    std::vector<idb_entry> idbs_;
    std::vector<rule_def> rules_;

    const edb_entry* edb(const std::string& n) const {
        for (auto& e : edbs_) if (e.name == n) return &e; return nullptr;
    }
    idb_entry* idb(const std::string& n) {
        for (auto& e : idbs_) if (e.name == n) return &e; return nullptr;
    }
    const idb_entry* idb(const std::string& n) const {
        for (auto& e : idbs_) if (e.name == n) return &e; return nullptr;
    }

    static int shared_edb_col(const atom& ea, const atom& ba) {
        for (size_t i = 0; i < ba.vars.size(); ++i)
            for (size_t j = 0; j < ea.vars.size(); ++j)
                if (ba.vars[i] == ea.vars[j]) return (int)j;
        return -1;
    }

    void fire(const rule_def& rule, const std::string* dr, const mask_t* db, mask_t& out) {
        size_t ei = SIZE_MAX;
        for (size_t i = 0; i < rule.body.size(); ++i)
            if (edb(rule.body[i].rel)) { ei = i; break; }

        if (ei == SIZE_MAX) {
            if (rule.body.empty()) return;
            size_t w = out.size();
            mask_t res(w, ~0ULL);
            for (auto& ba : rule.body) {
                auto* p = idb(ba.rel);
                if (!p) continue;
                const mask_t& b = (dr && ba.rel == *dr && db) ? *db : p->mask;
                if (ba.negated) {
                    for (size_t i = 0; i < w; ++i) res[i] &= ~b[i];
                } else {
                    for (size_t i = 0; i < w; ++i) res[i] &= b[i];
                }
            }
            for (size_t i = 0; i < w; ++i) out[i] |= res[i];
            return;
        }

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
            if (!p) continue;

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
        for (auto& [k, s] : sn) mx = std::max(mx, s);
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

    void eval_stratum(stratum_t& s) {
        std::vector<size_t> base, rec;
        for (size_t ri : s.rule_ids)
            (is_recursive(rules_[ri]) ? rec : base).push_back(ri);

        for (size_t ri : base) {
            auto& r = rules_[ri]; auto* h = idb(r.head);
            fire(r, nullptr, nullptr, h->mask);
        }
        if (rec.empty()) return;

        std::unordered_set<std::string> inv;
        for (size_t ri : rec) {
            inv.insert(rules_[ri].head);
            for (auto& a : rules_[ri].body)
                if (!a.negated && idb(a.rel)) inv.insert(a.rel);
        }

        std::unordered_map<std::string, mask_t> deltas;
        for (auto& nm : inv) deltas[nm] = idb(nm)->mask;

        for (bool any_new = true; any_new;) {
            any_new = false;
            std::unordered_map<std::string, mask_t> accum;
            for (auto& nm : inv) accum[nm].resize(idb(nm)->mask.size(), 0);

            for (size_t ri : rec) {
                auto& rule = rules_[ri];
                for (auto& ba : rule.body) {
                    if (ba.negated || !idb(ba.rel)) continue;
                    std::string dn = ba.rel;
                    auto& d = deltas[dn];
                    fire(rule, &dn, &d, accum[rule.head]);
                    break;
                }
            }

            for (auto& nm : inv) {
                auto* p = idb(nm); auto& ac = accum[nm];
                for (size_t w = 0; w < ac.size(); ++w) ac[w] &= ~p->mask[w];
                if (bits::any(ac)) {
                    any_new = true;
                    for (size_t w = 0; w < ac.size(); ++w) p->mask[w] |= ac[w];
                }
                deltas[nm] = std::move(ac);
            }
        }
    }

public:
    void add_edb(std::string name, const table& t, std::vector<std::string> cols) {
        edbs_.push_back({std::move(name), &t, std::move(cols)});
    }
    void add_idb(std::string name, size_t universe) {
        idb_entry e; e.name = std::move(name); e.universe = universe;
        e.mask.resize(bits::words(universe), 0);
        idbs_.push_back(std::move(e));
    }
    void add_rule(rule_def r) { rules_.push_back(std::move(r)); }
    void evaluate() { auto ss = stratify(); for (auto& s : ss) eval_stratum(s); }

    mask_t& get_bits(const std::string& n) { return idb(n)->mask; }
    const mask_t& get_bits(const std::string& n) const { return idb(n)->mask; }
};

} // namespace datalog

namespace {
table make_nodes(size_t n) { return table(n); }
table make_edges(std::vector<std::pair<uint32_t, uint32_t>> es) {
    size_t n = es.size(); table t(n);
    t.add_column_u32("src"); t.add_column_u32("dst");
    auto& src = t.get_col("src").u32; auto& dst = t.get_col("dst").u32;
    for (size_t i = 0; i < n; ++i) { src[i] = es[i].first; dst[i] = es[i].second; }
    return t;
}
bool bt(const engine::mask_t& m, size_t i) {
    return i / 64 < m.size() && (m[i / 64] & (1ULL << (i % 64)));
}
size_t popcnt(const engine::mask_t& m, size_t n) {
    size_t c = 0; for (size_t i = 0; i < n; ++i) c += bt(m, i); return c;
}
}

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

    auto result = (eq5{} | eq3{}) & lt10{};
    static_assert(std::is_same_v<decltype(result), or_t<eq5, eq3>>);
}

TEST(Algebra, Contrapositive) {
    using parent = field_hst<"type", "sensor">;
    using child  = field_hst<"type", "sensor/temp">;

    auto neg_parent = !parent{};
    auto neg_child  = !child{};
    static_assert(is_not_v<decltype(neg_parent)>);

    static_assert( implies(neg_parent, neg_child));
    static_assert(!implies(neg_child, neg_parent));

    auto result = neg_parent & neg_child;
    static_assert(std::is_same_v<decltype(result), decltype(neg_parent)>);
}

TEST(Algebra, SimplifyRecursion) {
    using eq5  = field_matcher<op::eq, "x", 5>;
    using lt10 = field_matcher<op::lt, "x", 10>;

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

TEST(Engine, TableViewIntegration) {
    std::vector<uint32_t> ids = {10, 20, 30, 40, 50};
    std::vector<uint32_t> scores = {1, 2, 1, 2, 1};

    table_view<> tv;
    tv.rows = 5;
    tv.chunk_size = 1024;
    tv.cols["id"] = {nullptr, ids.data(), nullptr, nullptr};
    tv.cols["s"] = {nullptr, scores.data(), nullptr, nullptr};

    auto r = engine::execute(tv, field_matcher<op::eq,"s",2>{});
    EXPECT_FALSE(simd::test(r.data(),0));
    EXPECT_TRUE (simd::test(r.data(),1));
    EXPECT_TRUE (simd::test(r.data(),3));
    EXPECT_FALSE(simd::test(r.data(),4));
}

TEST(Engine, TableViewCompound) {
    std::vector<uint32_t> ids = {10, 20, 30, 40, 50};
    std::vector<float> values = {1.0f, 25.0f, 30.0f, 15.0f, 50.0f};

    table_view<> tv;
    tv.rows = 5;
    tv.cols["id"] = {nullptr, ids.data(), nullptr, nullptr};
    tv.cols["val"] = {values.data(), nullptr, nullptr, nullptr};

    auto q = field_matcher<op::lt,"id",40>{} & field_matcher<op::ge,"val",20.0f>{};
    auto r = engine::execute(tv, q);
    EXPECT_TRUE (simd::test(r.data(),1));
    EXPECT_TRUE (simd::test(r.data(),2));
    EXPECT_FALSE(simd::test(r.data(),3));
}

TEST(Engine, TableToViewConversion) {
    table t(5);
    t.add_column_u32("s"); t.get_col("s").u32 = {1,2,1,2,1};

    auto tv = t.to_view();
    auto r = engine::execute(tv, field_matcher<op::eq,"s",2>{});

    EXPECT_TRUE (simd::test(r.data(),1));
    EXPECT_TRUE (simd::test(r.data(),3));
    EXPECT_FALSE(simd::test(r.data(),0));
}

TEST(Engine, TableViewAggregate) {
    std::vector<uint32_t> team = std::vector<uint32_t>(1000, 1);
    std::vector<float> hp(1000, 100.0f);
    for (size_t i=500;i<1000;++i) { team[i] = 2; hp[i] = 50.0f; }

    table_view<> tv;
    tv.rows = 1000;
    tv.cols["team"] = {nullptr, team.data(), nullptr, nullptr};
    tv.cols["hp"] = {hp.data(), nullptr, nullptr, nullptr};

    auto q = field_matcher<op::eq,"team",1>{};
    EXPECT_DOUBLE_EQ(engine::aggregate(tv, q, "hp", engine::agg::sum), 50000.0);
    EXPECT_DOUBLE_EQ(engine::aggregate(tv, q, "hp", engine::agg::mean), 100.0);
}

TEST(Engine, TableViewSemiJoin) {
    std::vector<uint32_t> user_ids = {0, 1, 2};
    std::vector<uint32_t> log_uids = {0, 0, 2, 5};

    table_view<> users;
    users.rows = 3;
    users.cols["id"] = {nullptr, user_ids.data(), nullptr, nullptr};

    table_view<> logs;
    logs.rows = 4;
    logs.cols["uid"] = {nullptr, log_uids.data(), nullptr, nullptr};

    auto sj = engine::semi_join(users, field_matcher<op::lt,"id",2>{}, "uid");
    auto r  = engine::execute(logs, sj);
    EXPECT_TRUE (simd::test(r.data(),0));
    EXPECT_TRUE (simd::test(r.data(),1));
    EXPECT_FALSE(simd::test(r.data(),2));
}

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

TEST(Engine, HierarchyDict) {
    table t(4);
    t.add_column_hst("path", {"sensor/temp/1", "sensor/voltage", "motor/rpm", "sensor/temp/2"});

    auto q1 = field_hst<"path", "sensor">();
    auto r1 = engine::execute(t, q1);
    EXPECT_FALSE(simd::test(r1.data(), 2));
    EXPECT_TRUE(simd::test(r1.data(), 0));
    EXPECT_TRUE(simd::test(r1.data(), 3));
    EXPECT_TRUE(simd::test(r1.data(), 1));

    auto q2 = field_hst<"path", "sensor/temp">();
    auto r2 = engine::execute(t, q2);
    EXPECT_TRUE(simd::test(r2.data(), 0));
    EXPECT_TRUE(simd::test(r2.data(), 3));
    EXPECT_FALSE(simd::test(r2.data(), 1));
}

TEST(Datalog, PureIDBIntersection) {
    auto nodes = make_nodes(5);
    datalog::program p;
    p.add_edb("nodes", nodes, {"id"});
    p.add_idb("A", 5); p.add_idb("B", 5); p.add_idb("C", 5);
    p.add_rule({"A", {"X"}, {{"nodes", {"X"}}}, [](const table& t){ return engine::execute(t, field_matcher<op::lt, "id", 4>{}); }, 0});
    p.add_rule({"B", {"X"}, {{"nodes", {"X"}}}, [](const table& t){ return engine::execute(t, field_matcher<op::ge, "id", 2>{}); }, 0});
    p.add_rule({"C", {"X"}, {{"A", {"X"}}, {"B", {"X"}}}});
    p.evaluate();

    EXPECT_FALSE(bt(p.get_bits("C"), 1));
    EXPECT_TRUE(bt(p.get_bits("C"), 2));
    EXPECT_TRUE(bt(p.get_bits("C"), 3));
    EXPECT_FALSE(bt(p.get_bits("C"), 4));
}

TEST(GameECS, ComponentFilter) {
    const size_t entity_count = 128;
    table entities(entity_count);
    entities.add_column_u32("level");
    entities.add_column_f32("health");

    entities.get_col("level").u32[42] = 10;
    entities.get_col("health").f32[42] = 25.5f;
    entities.set_tag(42, 3);

    entities.get_col("level").u32[43] = 5;
    entities.get_col("health").f32[43] = 25.5f;
    entities.set_tag(43, 3);

    using namespace match;
    auto query = field_matcher<op::ge, fs("level"), 10u>{} &
                 field_matcher<op::lt, fs("health"), 50.0f>{} &
                 field_matcher<op::eq, fs("mask"), 8ULL>{};

    auto result = engine::execute(entities, query);

    EXPECT_TRUE(simd::test(result.data(), 42));
    EXPECT_FALSE(simd::test(result.data(), 43));
}

TEST(QuestSystem, QuestChainLogic) {
    const size_t world_size = 100;
    table players(world_size);
    players.add_column_u32("level");
    players.get_col("level").u32[10] = 15;

    datalog::program p;
    p.add_edb("player_data", players, {"id", "level"});

    p.add_idb("quest_tutorial_done", world_size);
    p.add_idb("quest_boss_unlocked", world_size);

    p.add_rule({
        "quest_boss_unlocked", {"X"}, 
        {{"player_data", {"X"}}, {"quest_tutorial_done", {"X"}}},
        [](const table& t) { 
            return engine::execute(t, field_matcher<op::ge, fs("level"), 10u>{}); 
        }, 
        0
    });

    p.evaluate();
    EXPECT_FALSE(datalog::bits::test(p.get_bits("quest_boss_unlocked"), 10));

    datalog::bits::set(p.get_bits("quest_tutorial_done"), 10);
    p.evaluate();

    EXPECT_TRUE(datalog::bits::test(p.get_bits("quest_boss_unlocked"), 10));
}

TEST(SpatialEngine, BoundingBoxQuery) {
    const size_t n = 100;
    table world(n);
    world.add_column_f32("x");
    world.add_column_f32("y");

    world.get_col("x").f32[7] = 15.0f;
    world.get_col("y").f32[7] = 35.0f;

    world.get_col("x").f32[8] = 25.0f;
    world.get_col("y").f32[8] = 35.0f;

    using namespace match;
    auto box_query = field_matcher<op::ge, fs("x"), 10.0f>{} &
                     field_matcher<op::lt, fs("x"), 20.0f>{} &
                     field_matcher<op::ge, fs("y"), 30.0f>{} &
                     field_matcher<op::lt, fs("y"), 40.0f>{};

    auto result = engine::execute(world, box_query);

    EXPECT_TRUE(simd::test(result.data(), 7));
    EXPECT_FALSE(simd::test(result.data(), 8));
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

static void BM_ExecuteTableView(benchmark::State& st) {
    auto owned = bench_table(st.range(0));
    auto t = owned.to_view();
    auto q = field_matcher<op::lt,"id",100>{} & field_matcher<op::eq,"s",1>{};
    for (auto _:st) benchmark::DoNotOptimize(engine::execute(t,q));
    st.SetItemsProcessed(st.range(0)*st.iterations());
}
BENCHMARK(BM_ExecuteTableView)->RangeMultiplier(10)->Range(1000,1000000);

static void BM_ExecuteExternalMemory(benchmark::State& st) {
    size_t n = st.range(0);
    std::vector<uint32_t> ids(n);
    std::vector<uint32_t> scores(n);
    for (size_t i=0;i<n;++i) { ids[i] = i; scores[i] = i%3; }

    table_view<> tv;
    tv.rows = n;
    tv.cols["id"] = {nullptr, ids.data(), nullptr, nullptr};
    tv.cols["s"] = {nullptr, scores.data(), nullptr, nullptr};

    auto q = field_matcher<op::lt,"id",100>{} & field_matcher<op::eq,"s",1>{};
    for (auto _:st) benchmark::DoNotOptimize(engine::execute(tv,q));
    st.SetItemsProcessed(st.range(0)*st.iterations());
}
BENCHMARK(BM_ExecuteExternalMemory)->RangeMultiplier(10)->Range(1000,1000000);

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

static void BM_Execute_HST(benchmark::State& st) {
    table t(st.range(0));
    std::vector<std::string> paths(st.range(0));
    for (size_t i=0;i<st.range(0);++i) {
        paths[i] = (i % 2 == 0) ? "sensor/temp/a" : "motor/rpm/b";
    }
    t.add_column_hst("path", paths);
    auto q = field_hst<"path", "sensor">{};
    for (auto _:st) benchmark::DoNotOptimize(engine::execute(t,q));
    st.SetItemsProcessed(st.range(0)*st.iterations());
}
BENCHMARK(BM_Execute_HST)->RangeMultiplier(10)->Range(1000,1000000);

#define RUN_BENCHMARKS 1
#if RUN_BENCHMARKS
BENCHMARK_MAIN();
#else
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif
