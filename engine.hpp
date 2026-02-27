#pragma once

#include <algorithm>
#include <concepts>
#include <cassert>
#include <cstddef>
#include <cstdint>

#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <unordered_map>

#include "simd.hpp"

template<size_t N>
struct FixedString {
    char data[N]{};
    constexpr FixedString(const char (&s)[N]) { std::copy_n(s, N, data); }
    template<size_t M>
    constexpr bool operator==(const FixedString<M>& o) const {
        if constexpr (N != M) return false;
        else { for (size_t i = 0; i < N; ++i) if (data[i] != o.data[i]) return false; return true; }
    }
    constexpr std::string_view view() const { return {data, N - 1}; }
};

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

namespace expr {

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

constexpr inline class to_dnf_t {
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
} to_dnf{};

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

} // namespace expr


enum class op { eq, ne, lt, ge };

template<op O, FixedString Field, auto Val>
struct field_matcher {
    using is_matcher = void;

    template<op O2, FixedString F2, auto V2>
    friend constexpr bool tag_invoke(expr::implies_t, field_matcher const&,
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

    friend constexpr auto tag_invoke(expr::negate_t, field_matcher const&) {
        if constexpr (O==op::eq) return field_matcher<op::ne,Field,Val>{};
        if constexpr (O==op::ne) return field_matcher<op::eq,Field,Val>{};
        if constexpr (O==op::lt) return field_matcher<op::ge,Field,Val>{};
        if constexpr (O==op::ge) return field_matcher<op::lt,Field,Val>{};
    }
};

template<FixedString Field, FixedString Path>
struct prefix_matcher {
    using is_matcher = void;
    template<FixedString F2, FixedString P2>
    friend constexpr bool tag_invoke(expr::implies_t, prefix_matcher const&,
                                     prefix_matcher<F2,P2> const&) {
        if constexpr (!(Field == F2)) return false;
        else return Path.view().starts_with(P2.view());
    }
};

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


namespace detail {
template<typename T> inline constexpr bool needs_scratch = false;
template<typename L, typename R> inline constexpr bool needs_scratch<expr::and_t<L,R>> = true;
template<typename L, typename R> inline constexpr bool needs_scratch<expr::or_t<L,R>> = true;
template<typename M> inline constexpr bool needs_scratch<expr::not_t<M>> = needs_scratch<M>;
}

struct engine {
    using mask_t = simd::mask_t;

    static mask_t execute(const table& t, expr::matcher auto const& m) {
        const table_view<> tv = t.to_view<>();
       return execute(tv, m);
    }

    static expr::field_in semi_join(const table& t, expr::matcher auto const& q,
                                     std::string_view fk_field) {
        return {fk_field, std::make_shared<mask_t>(execute(t, q))};
    }

    static expr::field_in semi_naive_join(const table& t,
                                           expr::matcher auto const& seed,
                                           std::string_view fk_field,
                                           std::string_view id_field = "id") {
        auto delta = execute(t, seed);
        auto total = delta;
        size_t num_words = delta.size();
        while (simd::any(delta.data(), num_words)) {
            expr::field_in frontier{fk_field, std::make_shared<mask_t>(delta)};
            auto reachable = execute(t, frontier);
            delta = reachable;
            simd::bandnot(delta.data(), total.data(), num_words);
            simd::bor(total.data(), delta.data(), num_words);
        }
        return {id_field, std::make_shared<mask_t>(std::move(total))};
    }

    enum class agg { sum, max, min, mean, count };

    static double aggregate(const table& t, expr::matcher auto const& q,
                            std::string_view col, agg op) {
        auto mask = execute(t, q);
        auto& c = t.get_col(col);
        if (!c.f32.empty()) return agg_impl(mask, c.f32, op, t.rows);
        if (!c.u32.empty()) return agg_impl(mask, c.u32, op, t.rows);
        if (!c.u64.empty()) return agg_impl(mask, c.u64, op, t.rows);
        return 0.0;
    }

    template<size_t CS>
    static mask_t execute(const table_view<CS>& t, expr::matcher auto const& matcher) {
        auto query = expr::to_dnf(matcher);
        uint64_t req = extract_bits(query);
        size_t num_words = simd::num_words(t.rows);
        mask_t result(num_words, 0);

        if (req == 0 || t.chunk_summaries_empty()) {
            mask_t scratch(num_words);
            eval_from_view(t, query, 0, t.rows, result.data(), scratch.data(), num_words);
        } else {
            size_t max_cnum_words = simd::num_words(t.chunk_size);
            mask_t chunk_buf(max_cnum_words);
            mask_t scratch(max_cnum_words);
            for (size_t c = 0; c < t.num_chunks(); ++c) {
                auto& col = t.get_col("mask");
                uint64_t summary = col.chunk_summaries ? col.chunk_summaries[c] : 0;
                if ((summary & req) != req) continue;
                size_t start = c * t.chunk_size;
                size_t count = std::min(t.chunk_size, t.rows - start);
                size_t cnum_words = simd::num_words(count);
                eval_from_view(t, query, start, count, chunk_buf.data(), scratch.data(), cnum_words);
                simd::bor(result.data() + start / 64, chunk_buf.data(), cnum_words);
            }
        }
        simd::clear_tail(result.data(), num_words, t.rows);
        return result;
    }

    template<size_t CS>
    static expr::field_in semi_join(const table_view<CS>& t, expr::matcher auto const& q,
                                     std::string_view fk_field) {
        return {fk_field, std::make_shared<mask_t>(execute(t, q))};
    }

    template<size_t CS>
    static double aggregate(const table_view<CS>& t, expr::matcher auto const& q,
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

    template<op O, FixedString Field, auto Val>
    static uint64_t extract_bits(field_matcher<O, Field, Val> const&) {
        if constexpr (O == op::eq && Field == FixedString("mask")) return static_cast<uint64_t>(Val);
        else return 0;
    }
    template<expr::matcher L, expr::matcher R>
    static uint64_t extract_bits(expr::and_t<L,R> const& a) { return extract_bits(a.lhs) | extract_bits(a.rhs); }
    template<expr::matcher L, expr::matcher R>
    static uint64_t extract_bits(expr::or_t<L,R> const& o) {
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

    template<size_t CS>
    static void eval_from_view(const table_view<CS>&, expr::always_t, size_t, size_t count,
                               uint64_t* out, uint64_t*, size_t num_words) {
        std::fill_n(out, num_words, ~0ULL);
        simd::clear_tail(out, num_words, count);
    }

    template<size_t CS>
    static void eval_from_view(const table_view<CS>&, expr::never_t, size_t, size_t,
                               uint64_t* out, uint64_t*, size_t num_words) { 
        std::fill_n(out, num_words, 0); 
    }

    template<size_t CS, expr::matcher L, expr::matcher R>
    static void eval_from_view(const table_view<CS>& t, expr::and_t<L,R> const& a,
                               size_t s, size_t c, uint64_t* out, uint64_t* scratch, size_t num_words) {
        eval_from_view(t, a.lhs, s, c, out, scratch, num_words);
        if constexpr (detail::needs_scratch<R>) {
            mask_t tmp(num_words); eval_from_view(t, a.rhs, s, c, scratch, tmp.data(), num_words);
        } else {
            eval_from_view(t, a.rhs, s, c, scratch, nullptr, num_words);
        }
        simd::band(out, scratch, num_words);
    }

    template<size_t CS, expr::matcher L, expr::matcher R>
    static void eval_from_view(const table_view<CS>& t, expr::or_t<L,R> const& o,
                               size_t s, size_t c, uint64_t* out, uint64_t* scratch, size_t num_words) {
        eval_from_view(t, o.lhs, s, c, out, scratch, num_words);
        if constexpr (detail::needs_scratch<R>) {
            mask_t tmp(num_words); eval_from_view(t, o.rhs, s, c, scratch, tmp.data(), num_words);
        } else {
            eval_from_view(t, o.rhs, s, c, scratch, nullptr, num_words);
        }
        simd::bor(out, scratch, num_words);
    }

    template<size_t CS, expr::matcher M>
    static void eval_from_view(const table_view<CS>& t, expr::not_t<M> const& n,
                               size_t s, size_t c, uint64_t* out, uint64_t* scratch, size_t num_words) {
        eval_from_view(t, n.m, s, c, out, scratch, num_words);
        simd::bnot(out, num_words);
        simd::clear_tail(out, num_words, c);
    }

    template<size_t CS, op O, FixedString Field, auto Val>
    static void eval_from_view(const table_view<CS>& t, field_matcher<O,Field,Val> const&,
                               size_t start, size_t count, uint64_t* out, uint64_t*, size_t num_words) {
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
            std::fill_n(out, num_words, 0);
        }
    }

    template<size_t CS, FixedString Field, FixedString Path>
    static void eval_from_view(const table_view<CS>& t, prefix_matcher<Field,Path> const&,
                               size_t start, size_t count, uint64_t* out, uint64_t*, size_t num_words) {
        if (!t.dicts) { std::fill_n(out, num_words, 0); return; }
        auto it = t.dicts->find(std::string(Field.view()));
        if (it == t.dicts->end()) { std::fill_n(out, num_words, 0); return; }

        auto [low, high] = it->second.get_prefix_range(Path.view());
        if (low == high) { std::fill_n(out, num_words, 0); return; }

        auto& col = t.get_col(Field.view());
        if (!col.u32) { std::fill_n(out, num_words, 0); return; }

        simd::cmp_fill(out, col.u32 + start, 0, count, [=](uint32_t v, auto) {
            return v >= low && v < high;
        });
    }

    template<size_t CS>
    static void eval_from_view(const table_view<CS>& t, expr::field_in const& f,
                               size_t start, size_t count, uint64_t* out, uint64_t*, size_t num_words) {
        std::fill_n(out, num_words, 0);
        auto& col = t.get_col(f.field);
        if (!col.u32) return;
        auto& bm = *f.bitmap;
        for (size_t i = 0; i < count; ++i) {
            uint32_t v = col.u32[start + i];
            if (v / 64 < bm.size() && simd::test(bm.data(), v)) simd::set(out, i);
        }
    }
};
