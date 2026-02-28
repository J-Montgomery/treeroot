#include <benchmark/benchmark.h>

#include "engine.hpp"
#include "datalog.hpp"


using namespace expr;

namespace {
table make_nodes(size_t n) { return table(n); }
table make_edges(std::vector<std::pair<uint32_t, uint32_t>> es) {
    size_t n = es.size(); table t(n);
    t.add_column_u32("src"); t.add_column_u32("dst");
    auto& src = t.get_col("src").u32; auto& dst = t.get_col("dst").u32;
    for (size_t i = 0; i < n; ++i) { src[i] = es[i].first; dst[i] = es[i].second; }
    return t;
}

auto count_set_bits = [](const auto& mask, size_t n) {
    return simd::popcount(mask.data(), simd::num_words(n));
};

auto bit_at = [](const auto& mask, size_t i) {
    return simd::test(mask.data(), i);
};
}

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

static void BM_ExecuteExternalMemory_Incremental(benchmark::State& st) {
    size_t n = st.range(0);
    std::vector<uint32_t> ids(n);
    std::vector<uint32_t> scores(n);
    for (size_t i=0;i<n;++i) { ids[i] = i; scores[i] = i%3; }

    table_view<> tv;
    tv.rows = n;
    tv.cols["id"] = {nullptr, ids.data(), nullptr, nullptr};
    tv.cols["s"] = {nullptr, scores.data(), nullptr, nullptr};

    auto ir = engine::incremental_result{};
    engine::exec_ctx ctx{n};

    auto q = field_matcher<op::lt,"id",100>{} & field_matcher<op::eq,"s",1>{};
    for (auto _:st) benchmark::DoNotOptimize(engine::execute_incremental(ctx, ir, tv,q));
    st.SetItemsProcessed(st.range(0)*st.iterations());
}
BENCHMARK(BM_ExecuteExternalMemory_Incremental)->RangeMultiplier(10)->Range(1000,1000000);

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

static void BM_Datalog_DenseTC_Detailed(benchmark::State& st) {
    const uint32_t num_nodes = 32;
    std::vector<std::pair<uint32_t, uint32_t>> es;
    
    // Completely connected graph
    for (uint32_t i = 0; i < num_nodes; ++i) {
        for (uint32_t j = 0; j < num_nodes; ++j) {
            if (i != j) es.push_back({i, j});
        }
    }
    
    auto edges = make_edges(std::move(es));

    for (auto _ : st) {
        datalog::program p;
        p.add_edb("edge", edges, {"src", "dst"});
        p.add_idb("reach", num_nodes);

        // Rule 1: reach(X) :- edge(X, _)
        p.add_rule({"reach", {"X"}, {{"edge", {"X", "D"}}}});
        
        // Rule 2: reach(X) :- reach(Y), edge(Y, X)
        p.add_rule({"reach", {"X"}, {{"reach", {"Y"}}, {"edge", {"Y", "X"}}}});

        p.evaluate();
        
        benchmark::DoNotOptimize(p.get_bits("reach").data());
    }
    
    st.SetItemsProcessed(st.iterations() * num_nodes * num_nodes);
}
BENCHMARK(BM_Datalog_DenseTC_Detailed);

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
    auto q = prefix_matcher<"path", "sensor">{};
    for (auto _:st) benchmark::DoNotOptimize(engine::execute(t,q));
    st.SetItemsProcessed(st.range(0)*st.iterations());
}
BENCHMARK(BM_Execute_HST)->RangeMultiplier(10)->Range(1000,1000000);

static void BM_BoundingBox(benchmark::State& st) {
    size_t n = st.range(0);
    table world(n);
    world.add_column_f32("x");
    world.add_column_f32("y");

    // Populate with distributed coordinates
    for (size_t i = 0; i < n; ++i) {
        world.get_col("x").f32[i] = static_cast<float>(i % 1000) * 0.1f;
        world.get_col("y").f32[i] = static_cast<float>((i / 1000) % 1000) * 0.1f;
    }

    auto box_query = field_matcher<op::ge, FixedString("x"), 10.0f>{} &
                     field_matcher<op::lt, FixedString("x"), 20.0f>{} &
                     field_matcher<op::ge, FixedString("y"), 30.0f>{} &
                     field_matcher<op::lt, FixedString("y"), 40.0f>{};

    for (auto _ : st) {
        benchmark::DoNotOptimize(engine::execute(world, box_query));
    }
    st.SetItemsProcessed(st.range(0) * st.iterations());
}
BENCHMARK(BM_BoundingBox)->RangeMultiplier(10)->Range(1000, 1000000);

static void BM_BoundingBox_Bloom(benchmark::State& st) {
    size_t n = st.range(0);
    size_t chunk_size = 1024;
    table world(n, chunk_size);
    world.add_column_f32("x");
    world.add_column_f32("y");

    // Populate coordinates
    for (size_t i = 0; i < n; ++i) {
        world.get_col("x").f32[i] = static_cast<float>(i % 1000) * 0.1f;
        world.get_col("y").f32[i] = static_cast<float>((i / 1000) % 1000) * 0.1f;
    }

    // Tag entities in target region with bit 0
    // This allows bloom filter to skip chunks with no candidates
    for (size_t i = 0; i < n; ++i) {
        float x = world.get_col("x").f32[i];
        float y = world.get_col("y").f32[i];
        if (x >= 10.0f && x < 20.0f && y >= 30.0f && y < 40.0f) {
            world.set_tag(i, 0);
        }
    }

    // Query includes mask matcher to trigger bloom filter pruning
    auto box_query = field_matcher<op::eq, FixedString("mask"), 1ULL>{} &
                     field_matcher<op::ge, FixedString("x"), 10.0f>{} &
                     field_matcher<op::lt, FixedString("x"), 20.0f>{} &
                     field_matcher<op::ge, FixedString("y"), 30.0f>{} &
                     field_matcher<op::lt, FixedString("y"), 40.0f>{};

    for (auto _ : st) {
        benchmark::DoNotOptimize(engine::execute(world, box_query));
    }
    st.SetItemsProcessed(st.range(0) * st.iterations());
}
BENCHMARK(BM_BoundingBox_Bloom)->RangeMultiplier(10)->Range(1000, 1000000);

static void BM_Bloom_Crossover(benchmark::State& st) {
    const size_t n = 1'000'000;
    const size_t chunk_size = 1024;
    const double pollution_pct = static_cast<double>(st.range(0)) / 100.0;
    
    table world(n, chunk_size);
    world.add_column_f32("x");
    world.add_column_f32("y");

    // Space out active entities so they land in different chunks
    size_t num_polluted_chunks = static_cast<size_t>((n / chunk_size) * pollution_pct);
    size_t stride = (num_polluted_chunks > 0) ? (n / chunk_size) / num_polluted_chunks : n;

    for (size_t i = 0; i < num_polluted_chunks; ++i) {
        size_t entity_idx = i * stride * chunk_size;
        if (entity_idx < n) {
            world.get_col("x").f32[entity_idx] = 15.0f;
            world.get_col("y").f32[entity_idx] = 35.0f;
            world.set_tag(entity_idx, 0);
        }
    }

    auto q = field_matcher<op::eq, FixedString("mask"), 1ULL>{} &
             field_matcher<op::ge, FixedString("x"), 10.0f>{} &
             field_matcher<op::lt, FixedString("x"), 20.0f>{} &
             field_matcher<op::ge, FixedString("y"), 30.0f>{} &
             field_matcher<op::lt, FixedString("y"), 40.0f>{};

    for (auto _ : st) {
        benchmark::DoNotOptimize(engine::execute(world, q));
    }
    st.SetItemsProcessed(n * st.iterations());
}
BENCHMARK(BM_Bloom_Crossover)->DenseRange(0, 100, 25);

static void BM_Datalog_Pathfinding_Long(benchmark::State& st) {
    const uint32_t num_nodes = st.range(0);
    std::vector<std::pair<uint32_t, uint32_t>> es;
    
    // Create a chain: 0 -> 1 -> 2 -> ... -> N
    for (uint32_t i = 0; i < num_nodes - 1; ++i) {
        es.push_back({i, i + 1});
    }
    
    auto edges = make_edges(std::move(es));

    for (auto _ : st) {
        datalog::program p;
        p.add_edb("edge", edges, {"src", "dst"});
        p.add_idb("reachable", num_nodes);
        p.add_rule({"reachable", {"X"}, {{"edge", {"0", "X"}}}});
        
        // reachable(X) :- reachable(Y), edge(Y, X)
        p.add_rule({"reachable", {"X"}, {{"reachable", {"Y"}}, {"edge", {"Y", "X"}}}});

        p.evaluate();
        
        benchmark::DoNotOptimize(p.get_bits("reachable").data());
    }
    
    st.SetItemsProcessed(st.iterations() * num_nodes);
}
BENCHMARK(BM_Datalog_Pathfinding_Long)->Arg(100)->Arg(1000)->Arg(10000)->Arg(100000)->Arg(1000000);

static void BM_Datalog_Pathfinding_Bloom(benchmark::State& st) {
    const uint32_t cluster_size = 1024;
    const uint32_t num_clusters = st.range(0);
    const uint32_t total_nodes = cluster_size * num_clusters;
    
    std::vector<std::pair<uint32_t, uint32_t>> es;
    
    for (uint32_t c = 0; c < num_clusters; ++c) {
        uint32_t start = c * cluster_size;
        for (uint32_t i = 0; i < 100; ++i) {
            es.push_back({start + i, start + i + 1});
        }
        if (c < num_clusters - 1) {
            es.push_back({start + cluster_size - 1, start + cluster_size});
        }
    }
    
    auto world = make_edges(std::move(es));

    for (auto _ : st) {
        datalog::program p;
        p.add_edb("edge", world, {"src", "dst"});
        p.add_idb("reachable", total_nodes);

        // Seed with node 0
        p.add_rule({"reachable", {"X"}, {{"edge", {"0", "X"}}}});
        p.add_rule({"reachable", {"X"}, {{"reachable", {"Y"}}, {"edge", {"Y", "X"}}}});

        p.evaluate();
        benchmark::DoNotOptimize(p.get_bits("reachable").data());
    }
    
    st.SetItemsProcessed(st.iterations() * total_nodes);
}
BENCHMARK(BM_Datalog_Pathfinding_Bloom)->Arg(1000);

// static void BM_Datalog_Regex_Propagation(benchmark::State& st) {
//     const size_t n = st.range(0);
//     std::vector<std::pair<uint32_t, uint32_t>> es;
//     for (uint32_t i = 0; i < n - 1; ++i) es.push_back({i, i + 1});
//     auto edges = make_edges(std::move(es));

//     for (auto _ : st) {
//         datalog::program p;
//         p.add_edb("step", edges, {"src", "dst"});
        
//         p.add_idb("state_a", n);
//         p.add_idb("state_b", n);
//         p.add_idb("state_c", n);
//         p.add_idb("state_d", n);

//         datalog::bits::set(p.get_bits("state_a"), 0);

//         p.add_rule({"state_b", {"Next"}, {{"state_a", {"Curr"}}, {"step", {"Curr", "Next"}}}});
//         p.add_rule({"state_c", {"Next"}, {{"state_b", {"Curr"}}, {"step", {"Curr", "Next"}}}});
//         p.add_rule({"state_d", {"Next"}, {{"state_c", {"Curr"}}, {"step", {"Curr", "Next"}}}});

//         p.evaluate();
//         benchmark::DoNotOptimize(p.get_bits("state_d").data());
//     }
//     st.SetItemsProcessed(st.iterations() * n);
// }
// BENCHMARK(BM_Datalog_Regex_Propagation)->RangeMultiplier(10)->Range(1000, 100'000);

static void BM_Engine_Manual_DFA(benchmark::State& st) {
    const size_t n = st.range(0);
    std::vector<uint32_t> next_ids(n);
    for (uint32_t i = 0; i < n - 1; ++i) next_ids[i] = i + 1;
    next_ids[n - 1] = n - 1;

    table_view<> tv;
    tv.rows = n;
    tv.cols["next"] = {nullptr, next_ids.data(), nullptr, nullptr};

    auto s1 = simd::mask_t(n);
    auto s2 = simd::mask_t(n);
    auto s3 = simd::mask_t(n);

    for (auto _ : st) {
        std::fill(s1.begin(), s1.end(), 0);
        std::fill(s2.begin(), s2.end(), 0);
        std::fill(s3.begin(), s3.end(), 0);
        simd::set(s1.data(), 0);

        bool changed = true;
        while (changed) {
            changed = false;
            const uint32_t* next = tv.cols["next"].u32;
            
            for (size_t i = 0; i < n; ++i) {
                uint32_t target = next[i];
                
                // State 1 -> State 2
                if (simd::test(s1.data(), i) && !simd::test(s2.data(), target)) {
                    simd::set(s2.data(), target);
                    changed = true;
                }
                // State 2 -> State 3
                if (simd::test(s2.data(), i) && !simd::test(s3.data(), target)) {
                    simd::set(s3.data(), target);
                    changed = true;
                }
            }
        }
        benchmark::DoNotOptimize(s3.data());
    }
    st.SetItemsProcessed(st.iterations() * n);
}
BENCHMARK(BM_Engine_Manual_DFA)->Range(1000, 100000);

// static void BM_Engine_Regex_Optimized(benchmark::State& st) {
//     const uint32_t n = st.range(0);
//     table t(n);
//     t.add_column_u32("next");
//     for (uint32_t i = 0; i < n - 1; ++i) t.get_col("next").u32[i] = i + 1;

//     auto tv = t.to_view();
//     auto nw = simd::num_words(n);
    
//     engine::mask_t s1(nw, 0), s2(nw, 0), s3(nw, 0), scratch(nw, 0);
//     for (auto _ : st) {
//         simd::set(s1.data(), 0); 
//         bool changed = true;

//         while (changed) {
//             changed = false;
//             expr::field_in f1{"next", std::make_shared<engine::mask_t>(s1)};
//             size_t before = simd::popcount(s2.data(), nw);
//             engine::eval_into_view(tv, f1, 0, n, s2.data(), scratch.data(), nw);
            
//             size_t after = simd::popcount(s2.data(), nw);
//             if (after > before) changed = true;
            
//             expr::field_in f2{"next", std::make_shared<engine::mask_t>(s2)};
//             engine::eval_into_view(tv, f2, 0, n, s3.data(), scratch.data(), nw);
//         }
//         benchmark::DoNotOptimize(s3.data());
//     }
//     st.SetItemsProcessed(st.iterations() * n);
// }
// BENCHMARK(BM_Engine_Regex_Optimized)->Range(1000, 100000);

BENCHMARK_MAIN();