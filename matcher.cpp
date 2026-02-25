#include <algorithm>
#include <concepts>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>
#include <memory>
#include <type_traits>
#include <gtest/gtest.h>
#include <benchmark/benchmark.h>

// --- C++20 Structural String for Template Parameters ---
template <size_t N>
struct fs {
    char data[N];
    constexpr fs(const char (&str)[N]) { std::copy_n(str, N, data); }

    // Fix: Allow comparison with fs of any size M
    template <size_t M>
    constexpr bool operator==(const fs<M>& other) const {
        if constexpr (N != M) return false;
        for(size_t i=0; i<N; ++i) if(data[i] != other.data[i]) return false;
        return true;
    }
    constexpr std::string_view view() const { return {data, N-1}; }
};

// --- Core Matcher Concept & AST Nodes ---
namespace match {
    template<typename T> concept matcher = requires { typename std::remove_cvref_t<T>::is_matcher; };

    struct always_t { using is_matcher = void; };
    struct never_t  { using is_matcher = void; };
    inline constexpr always_t always;
    inline constexpr never_t  never;

    template<matcher M> struct not_t { using is_matcher = void; M m; };
    template<matcher L, matcher R> struct and_t { using is_matcher = void; L lhs; R rhs; };
    template<matcher L, matcher R> struct or_t  { using is_matcher = void; L lhs; R rhs; };

    // --- CPOs for Algebraic Engine ---
    
    // Implication: A => B means A is more constrained (A <= B)
    constexpr inline class implies_t {
    public:
        template <typename L, typename R>
        constexpr bool operator()(L const& l, R const& r) const {
            if constexpr (requires { tag_invoke(*this, l, r); }) {
                return tag_invoke(*this, l, r);
            } else {
                return std::is_same_v<std::remove_cvref_t<L>, std::remove_cvref_t<R>>;
            }
        }
    } implies{};

    // Negation: !A
    constexpr inline class negate_t {
    public:
        template <typename M>
        constexpr auto operator()(M const& m) const {
            if constexpr (requires { tag_invoke(*this, m); }) {
                return tag_invoke(*this, m);
            } else {
                return not_t<std::remove_cvref_t<M>>{m};
            }
        }
    } negate{};

    // Simplify: Normalizes AST
    constexpr inline class simplify_t {
    public:
        template <typename M>
        constexpr auto operator()(M const& m) const {
            if constexpr (requires { tag_invoke(*this, m); }) {
                return tag_invoke(*this, m);
            } else {
                return m;
            }
        }
    } simplify{};

    // --- Base Algebraic Rules ---
    constexpr bool tag_invoke(implies_t, never_t, auto const&) { return true; }
    constexpr bool tag_invoke(implies_t, auto const&, always_t) { return true; }
    constexpr bool tag_invoke(implies_t, always_t, never_t)  { return false; }

    template<matcher L, matcher R, matcher M>
    constexpr bool tag_invoke(implies_t, and_t<L, R> const& a, M const& m) {
        return implies(a.lhs, m) || implies(a.rhs, m);
    }
    
    template<matcher M, matcher L, matcher R>
    constexpr bool tag_invoke(implies_t, M const& m, or_t<L, R> const& o) {
        return implies(m, o.lhs) || implies(m, o.rhs);
    }

    constexpr never_t  tag_invoke(negate_t, always_t) { return never; }
    constexpr always_t tag_invoke(negate_t, never_t)  { return always; }
    template<matcher M> constexpr auto tag_invoke(negate_t, not_t<M> const& n) { return n.m; }

    template<matcher M> constexpr auto tag_invoke(simplify_t, not_t<M> const& n) { return negate(simplify(n.m)); }

    // --- Expression Operators ---
    template<matcher L, matcher R> constexpr auto operator&(L l, R r) {
        auto sl = simplify(l); auto sr = simplify(r);
        using TL = decltype(sl); using TR = decltype(sr);
        if constexpr (std::is_same_v<TL, never_t> || std::is_same_v<TR, never_t>) return never;
        else if constexpr (std::is_same_v<TL, always_t>) return sr;
        else if constexpr (std::is_same_v<TR, always_t>) return sl;
        else if constexpr (implies(sl, sr)) return sl;
        else if constexpr (implies(sr, sl)) return sr;
        else if constexpr (implies(sl, negate(sr)) || implies(sr, negate(sl))) return never; // Contradiction
        else return and_t<TL, TR>{sl, sr};
    }

    template<matcher L, matcher R> constexpr auto operator|(L l, R r) {
        auto sl = simplify(l); auto sr = simplify(r);
        using TL = decltype(sl); using TR = decltype(sr);
        if constexpr (std::is_same_v<TL, always_t> || std::is_same_v<TR, always_t>) return always;
        else if constexpr (std::is_same_v<TL, never_t>) return sr;
        else if constexpr (std::is_same_v<TR, never_t>) return sl;
        else if constexpr (implies(sl, sr)) return sr;
        else if constexpr (implies(sr, sl)) return sl;
        else if constexpr (implies(negate(sl), sr) || implies(negate(sr), sl)) return always; // Tautology
        else return or_t<TL, TR>{sl, sr};
    }

    template<matcher M> constexpr auto operator!(M m) { return simplify(not_t<M>{m}); }
}

// --- Specific Matchers (NTTP driven) ---
enum class op { eq, ne, lt, ge };

template<op O, fs Field, auto Val> 
struct field_matcher {
    using is_matcher = void;
    static constexpr op operation = O;

    // Relational Implication Rules evaluated statically
    template<op O2, fs F2, auto V2>
    friend constexpr bool tag_invoke(match::implies_t, field_matcher const&, field_matcher<O2, F2, V2> const&) {
        if constexpr (!(Field == F2)) return false;
        else if constexpr (O == op::eq) {
            if constexpr (O2 == op::eq) return Val == V2;
            if constexpr (O2 == op::lt) return Val <  V2;
            if constexpr (O2 == op::ge) return Val >= V2;
            if constexpr (O2 == op::ne) return Val != V2;
        }
        else if constexpr (O == op::lt && O2 == op::lt) return Val <= V2;
        else if constexpr (O == op::ge && O2 == op::ge) return Val >= V2;
        else return false;
    }

    // Leaf Swap Negation
    friend constexpr auto tag_invoke(match::negate_t, field_matcher const&) {
        if constexpr (O == op::eq) return field_matcher<op::ne, Field, Val>{};
        if constexpr (O == op::ne) return field_matcher<op::eq, Field, Val>{};
        if constexpr (O == op::lt) return field_matcher<op::ge, Field, Val>{};
        if constexpr (O == op::ge) return field_matcher<op::lt, Field, Val>{};
    }
};

// Hierarchical String Tag Matcher
template<fs Field, fs Path>
struct field_hst {
    using is_matcher = void;
    
    template<fs F2, fs P2>
    friend constexpr bool tag_invoke(match::implies_t, field_hst const&, field_hst<F2, P2> const&) {
        if constexpr (!(Field == F2)) return false;
        else return Path.view().starts_with(P2.view()); 
    }
};

// Runtime Semi-Join Set Matcher
struct field_in {
    using is_matcher = void;
    std::string_view field;
    std::shared_ptr<std::vector<uint64_t>> bitmap; // SharedPtr makes copies cheap during simplification
};

// --- Execution Engine ---
struct table {
    size_t rows;
    struct column { std::string_view name; std::vector<uint32_t> data; std::vector<std::string> sdata; };
    std::vector<column> cols;
    const auto& get_col(std::string_view n) const { 
        for(auto& c : cols) if(c.name == n) return c; 
        throw std::runtime_error("Column not found"); 
    }
};

struct engine {
    using mask_t = std::vector<uint64_t>;

    static mask_t execute(const table& t, match::matcher auto const& m) {
        mask_t res((t.rows + 63) / 64, 0);
        eval(t, m, res);
        return res;
    }

    static auto semi_join(const table& t, match::matcher auto const& q, 
                     std::string_view source_field, std::string_view dest_field) {
        auto mask = execute(t, q);
        auto& col = t.get_col(source_field).data; 
        
        uint32_t max_v = 0;
        for(auto v : col) max_v = std::max(max_v, v);
        
        auto bits = std::make_shared<std::vector<uint64_t>>((max_v + 64) / 64, 0);
        for(size_t i=0; i < t.rows; ++i) 
            if(mask[i/64] & (1ULL << (i%64))) (*bits)[col[i]/64] |= (1ULL << (col[i]%64));
                
        return field_in{ dest_field, bits };
    }

private:
    static void eval(const table& t, match::always_t, mask_t& m) { std::fill(m.begin(), m.end(), ~0ULL); }
    static void eval(const table& t, match::never_t,  mask_t& m) { std::fill(m.begin(), m.end(), 0ULL); }

    template<typename L, typename R>
    static void eval(const table& t, match::and_t<L, R> const& a, mask_t& m) {
        eval(t, a.lhs, m);
        mask_t rhs(m.size(), 0); eval(t, a.rhs, rhs);
        for(size_t i=0; i<m.size(); ++i) m[i] &= rhs[i];
    }
    
    template<typename L, typename R>
    static void eval(const table& t, match::or_t<L, R> const& o, mask_t& m) {
        eval(t, o.lhs, m);
        mask_t rhs(m.size(), 0); eval(t, o.rhs, rhs);
        for(size_t i=0; i<m.size(); ++i) m[i] |= rhs[i];
    }

    template<typename M>
    static void eval(const table& t, match::not_t<M> const& n, mask_t& m) {
        eval(t, n.m, m);
        for(auto& block : m) block = ~block;
    }

    template<op O, fs Field, auto Val>
    static void eval(const table& t, field_matcher<O, Field, Val> const&, mask_t& m) {
        auto& data = t.get_col(Field.view()).data;
        for(size_t i=0; i<t.rows; ++i) {
            bool hit = false;
            if constexpr (O == op::eq) hit = (data[i] == Val);
            if constexpr (O == op::ne) hit = (data[i] != Val);
            if constexpr (O == op::lt) hit = (data[i] <  Val);
            if constexpr (O == op::ge) hit = (data[i] >= Val);
            if(hit) m[i/64] |= (1ULL << (i%64));
        }
    }

    template<fs Field, fs Path>
    static void eval(const table& t, field_hst<Field, Path> const&, mask_t& m) {
        auto& data = t.get_col(Field.view()).sdata;
        for(size_t i=0; i<t.rows; ++i) 
            if(data[i].starts_with(Path.view())) m[i/64] |= (1ULL << (i%64));
    }

    static void eval(const table& t, field_in const& f, mask_t& m) {
        auto& data = t.get_col(f.field).data;
        for(size_t i=0; i<t.rows; ++i) {
            uint32_t v = data[i];
            if(v/64 < f.bitmap->size() && ((*f.bitmap)[v/64] & (1ULL << (v%64))))
                m[i/64] |= (1ULL << (i%64));
        }
    }
};

// --- Exhaustive Test Suite ---
using namespace match;

TEST(AlgebraTest, PruningAndRedundancy) {
    using lt_10 = field_matcher<op::lt, "x", 10>;
    using lt_20 = field_matcher<op::lt, "x", 20>;

    // (x < 10) & (x < 20) -> (x < 10) because lt_10 implies lt_20
    auto a = lt_10{} & lt_20{}; 
    static_assert(std::is_same_v<decltype(a), lt_10>);
    
    // (x < 10) | (x < 20) -> (x < 20)
    auto o = lt_10{} | lt_20{}; 
    static_assert(std::is_same_v<decltype(o), lt_20>);
}

TEST(AlgebraTest, ContradictionsAndTautologies) {
    using eq_10 = field_matcher<op::eq, "x", 10>;
    using ne_10 = field_matcher<op::ne, "x", 10>;

    // Contradiction: (x == 10) & (x != 10) -> never
    auto contra = eq_10{} & ne_10{};
    static_assert(std::is_same_v<decltype(contra), never_t>);

    // Tautology: (x == 10) | (x != 10) -> always
    auto taut = eq_10{} | ne_10{};
    static_assert(std::is_same_v<decltype(taut), always_t>);
    
    // Range Contradiction: (x < 10) & (x >= 20) -> never
    using lt_10 = field_matcher<op::lt, "x", 10>;
    using ge_20 = field_matcher<op::ge, "x", 20>;
    auto range_contra = lt_10{} & ge_20{};
    static_assert(std::is_same_v<decltype(range_contra), never_t>);
}

TEST(HSTTest, HierarchicalMatching) {
    using child_t  = field_hst<"type", "sensor/temp">;
    using parent_t = field_hst<"type", "sensor">;
    using diff_t   = field_hst<"type", "motor">;

    // child implies parent, so child & parent == child
    auto result = child_t{} & parent_t{};
    static_assert(std::is_same_v<decltype(result), child_t>);
    
    // disjoint trees remain unchanged
    auto diff_result = child_t{} & diff_t{};
    static_assert(std::is_same_v<decltype(diff_result), and_t<child_t, diff_t>>);
}

TEST(EngineTest, SemiJoinEfficiency) {
    table users{ 3, {{"id", {1, 2, 3}, {}}} };
    table logs { 4, {{"user_id", {1, 1, 3, 5}, {}}} };

    auto sj = engine::semi_join(users, field_matcher<op::lt, "id", 3>{}, "id", "user_id");
    auto result = engine::execute(logs, sj);
    
    EXPECT_TRUE(result[0] & (1ULL << 0));  
    EXPECT_TRUE(result[0] & (1ULL << 1));  
    EXPECT_FALSE(result[0] & (1ULL << 2)); 
}


TEST(EngineTest, StructuralExecution) {
    table data{ 5, {
        {"id", {10, 11, 12, 13, 14}, {}},
        {"status", {1, 2, 1, 2, 1}, {}}
    }};
    
    // (id < 13) & (status == 2) -> Should only hit row 1 (id 11)
    auto query = field_matcher<op::lt, "id", 13>{} & field_matcher<op::eq, "status", 2>{};
    auto result = engine::execute(data, query);

    EXPECT_FALSE(result[0] & (1ULL << 0));
    EXPECT_TRUE(result[0] & (1ULL << 1));
    EXPECT_FALSE(result[0] & (1ULL << 2));
}

/**************************** Benchmarks **********************************/

table create_large_table(size_t rows) {
    table t;
    t.rows = rows;
    std::vector<uint32_t> ids(rows);
    std::vector<uint32_t> status(rows);
    for(size_t i = 0; i < rows; ++i) {
        ids[i] = i;
        status[i] = i % 3; 
    }
    t.cols.push_back({"id", ids, {}});
    t.cols.push_back({"status", status, {}});
    return t;
}

// --- Benchmark 1: Simplification Overhead ---
// Measures how fast the CPOs (simplify, implies) normalize the AST
static void BM_AlgebraicSimplification(benchmark::State& state) {
    using namespace match;
    for (auto _ : state) {
        // (x < 10) & (x < 20) & (x < 30) & (x < 5)
        auto query = field_matcher<op::lt, "id", 10>{} & 
                     field_matcher<op::lt, "id", 20>{} & 
                     field_matcher<op::lt, "id", 30>{} & 
                     field_matcher<op::lt, "id", 5>{};
        benchmark::DoNotOptimize(query);
    }
}
BENCHMARK(BM_AlgebraicSimplification);

// --- Benchmark 2: Execution Performance ---
// Measures the bit-parallel evaluation of the matcher
static void BM_EngineExecution(benchmark::State& state) {
    auto t = create_large_table(state.range(0));
    auto query = field_matcher<op::lt, "id", 100>{} & field_matcher<op::eq, "status", 1>{};
    
    for (auto _ : state) {
        auto result = engine::execute(t, query);
        benchmark::DoNotOptimize(result);
    }
    state.SetItemsProcessed(state.range(0) * state.iterations());
}
// Run for 1k to 1M rows
BENCHMARK(BM_EngineExecution)->RangeMultiplier(10)->Range(1000, 1000000);

// --- Benchmark 3: Semi-Join Bitmap Building ---
// Measures the cost of projection and bitmap population
static void BM_SemiJoinConstruction(benchmark::State& state) {
    auto users = create_large_table(state.range(0));
    auto query = field_matcher<op::lt, "id", 500>{};
    
    for (auto _ : state) {
        auto sj = engine::semi_join(users, query, "id", "user_id");
        benchmark::DoNotOptimize(sj);
    }
}
BENCHMARK(BM_SemiJoinConstruction)->Arg(100000);

#define RUN_BENCHMARKS 1
#ifdef RUN_BENCHMARKS
BENCHMARK_MAIN();
#else
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
#endif