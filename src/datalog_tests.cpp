
#include <type_traits>

#include <gtest/gtest.h>

#include "engine.hpp"
#include "datalog.hpp"

// Tests

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
    using parent = prefix_matcher<"type", "sensor">;
    using child  = prefix_matcher<"type", "sensor/temp">;

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

TEST(Algebra, DNFTest) {
    using A = field_matcher<op::eq, "x", 1>;
    using B = field_matcher<op::eq, "y", 2>;
    using C = field_matcher<op::eq, "z", 3>;

    auto expr = and_t{or_t{A{}, B{}}, C{}};
    auto dnf = to_dnf(expr);
    static_assert(is_or_v<decltype(dnf)>);

    auto absorb = and_t{A{}, or_t{A{}, B{}}};
    static_assert(std::is_same_v<decltype(to_dnf(absorb)), A>);
}

TEST(Algebra, HST) {
    using child  = prefix_matcher<"type","sensor/temp">;
    using parent = prefix_matcher<"type","sensor">;
    using other  = prefix_matcher<"type","motor">;
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
    for (size_t i=0;i<5;++i)  EXPECT_TRUE (bit_at(r,i));
    for (size_t i=5;i<10;++i) EXPECT_FALSE(bit_at(r,i));
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
    EXPECT_FALSE(bit_at(r,0));
    EXPECT_TRUE (bit_at(r,1));
    EXPECT_TRUE (bit_at(r,3));
    EXPECT_FALSE(bit_at(r,4));
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
    EXPECT_TRUE (bit_at(r,1));
    EXPECT_TRUE (bit_at(r,2));
    EXPECT_FALSE(bit_at(r,3));
}

TEST(Engine, TableToViewConversion) {
    table t(5);
    t.add_column_u32("s"); t.get_col("s").u32 = {1,2,1,2,1};

    auto tv = t.to_view();
    auto r = engine::execute(tv, field_matcher<op::eq,"s",2>{});

    EXPECT_TRUE (bit_at(r,1));
    EXPECT_TRUE (bit_at(r,3));
    EXPECT_FALSE(bit_at(r,0));
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
    EXPECT_TRUE(bit_at(p.get_bits("r"), 0));
    EXPECT_TRUE(bit_at(p.get_bits("r"), 1));
    EXPECT_TRUE(bit_at(p.get_bits("r"), 2));
    EXPECT_FALSE(bit_at(p.get_bits("r"), 3));
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
    EXPECT_EQ(count_set_bits(p.get_bits("seed"), 5), 3u);
    EXPECT_TRUE(bit_at(p.get_bits("seed"), 0));
    EXPECT_TRUE(bit_at(p.get_bits("seed"), 2));
    EXPECT_FALSE(bit_at(p.get_bits("seed"), 3));
}

TEST(Datalog, UnaryBaseEmpty) {
    auto nodes = make_nodes(0);
    datalog::program p;
    p.add_edb("n", nodes, {"id"});
    p.add_idb("s", 10);
    p.add_rule({"s", {"X"}, {{"n", {"X"}}}});
    p.evaluate();
    EXPECT_EQ(count_set_bits(p.get_bits("s"), 10), 0u);
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
    EXPECT_EQ(count_set_bits(p.get_bits("s"), 3), 0u);
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
    EXPECT_TRUE(bit_at(p.get_bits("r"), 0));
    EXPECT_TRUE(bit_at(p.get_bits("r"), 1));
    EXPECT_FALSE(bit_at(p.get_bits("r"), 2));
    EXPECT_TRUE(bit_at(p.get_bits("r"), 3));
    EXPECT_TRUE(bit_at(p.get_bits("r"), 4));
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
        EXPECT_TRUE(bit_at(p.get_bits("reach"), i)) << "missing " << i;
}

TEST(Datalog, TCCycle) {
    auto edges = make_edges({{0,1},{1,2},{2,0}});
    datalog::program p;
    p.add_edb("edge", edges, {"src","dst"});
    p.add_idb("reach", 3);
    p.add_rule({"reach", {"X"}, {{"edge", {"X","D"}}}});
    p.add_rule({"reach", {"X"}, {{"reach", {"Y"}}, {"edge", {"Y","X"}}}});
    p.evaluate();
    EXPECT_EQ(count_set_bits(p.get_bits("reach"), 3), 3u);
}

TEST(Datalog, TCDiamond) {
    auto edges = make_edges({{0,1},{0,2},{1,3},{2,3}});
    datalog::program p;
    p.add_edb("edge", edges, {"src","dst"});
    p.add_idb("reach", 4);
    p.add_rule({"reach", {"X"}, {{"edge", {"X","D"}}}});
    p.add_rule({"reach", {"X"}, {{"reach", {"Y"}}, {"edge", {"Y","X"}}}});
    p.evaluate();
    EXPECT_EQ(count_set_bits(p.get_bits("reach"), 4), 4u);
}

TEST(Datalog, TCDisconnected) {
    auto edges = make_edges({{0,1},{1,2},{5,6},{6,7}});
    datalog::program p;
    p.add_edb("edge", edges, {"src","dst"});
    p.add_idb("reach", 8);
    p.add_rule({"reach", {"X"}, {{"edge", {"X","D"}}}});
    p.add_rule({"reach", {"X"}, {{"reach", {"Y"}}, {"edge", {"Y","X"}}}});
    p.evaluate();
    for (int i : {0,1,2,5,6,7}) EXPECT_TRUE(bit_at(p.get_bits("reach"), i)) << i;
    for (int i : {3,4})         EXPECT_FALSE(bit_at(p.get_bits("reach"), i)) << i;
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
    EXPECT_EQ(count_set_bits(p.get_bits("reach"), 101), 101u);
}

TEST(Datalog, FixpointSaturates) {
    auto edges = make_edges({{0,1},{1,0}});
    datalog::program p;
    p.add_edb("edge", edges, {"src","dst"});
    p.add_idb("reach", 2);
    p.add_rule({"reach", {"X"}, {{"edge", {"X","D"}}}});
    p.add_rule({"reach", {"X"}, {{"reach", {"Y"}}, {"edge", {"Y","X"}}}});
    p.evaluate();
    EXPECT_EQ(count_set_bits(p.get_bits("reach"), 2), 2u);
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
    for (int i : {0,1,2}) EXPECT_TRUE(bit_at(p.get_bits("reach"), i));
    EXPECT_FALSE(bit_at(p.get_bits("reach"), 3));
    for (int i : {0,1,2}) EXPECT_FALSE(bit_at(p.get_bits("unreach"), i));
    EXPECT_TRUE(bit_at(p.get_bits("unreach"), 3));
    EXPECT_TRUE(bit_at(p.get_bits("unreach"), 4));
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
    EXPECT_EQ(count_set_bits(p.get_bits("reach"), 3), 3u);
    EXPECT_EQ(count_set_bits(p.get_bits("unreach"), 3), 0u);
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
    EXPECT_EQ(count_set_bits(p.get_bits("base"), 10), 5u);
    EXPECT_EQ(count_set_bits(p.get_bits("derived"), 10), 5u);
    EXPECT_EQ(count_set_bits(p.get_bits("fin"), 10), 5u);
    for (int i = 0; i < 5; ++i)  EXPECT_TRUE(bit_at(p.get_bits("fin"), i));
    for (int i = 5; i < 10; ++i) EXPECT_FALSE(bit_at(p.get_bits("fin"), i));
}

TEST(Engine, HierarchyDict) {
    table t(4);
    t.add_column_hst("path", {"sensor/temp/1", "sensor/voltage", "motor/rpm", "sensor/temp/2"});

    auto q1 = prefix_matcher<"path", "sensor">();
    auto r1 = engine::execute(t, q1);
    EXPECT_FALSE(simd::test(r1.data(), 2));
    EXPECT_TRUE(simd::test(r1.data(), 0));
    EXPECT_TRUE(simd::test(r1.data(), 3));
    EXPECT_TRUE(simd::test(r1.data(), 1));

    auto q2 = prefix_matcher<"path", "sensor/temp">();
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

    EXPECT_FALSE(simd::test(p.get_bits("C").data(), 1));
    EXPECT_TRUE(simd::test(p.get_bits("C").data(), 2));
    EXPECT_TRUE(simd::test(p.get_bits("C").data(), 3));
    EXPECT_FALSE(simd::test(p.get_bits("C").data(), 4));
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

    auto query = field_matcher<op::ge, FixedString("level"), 10u>{} &
                 field_matcher<op::lt, FixedString("health"), 50.0f>{} &
                 field_matcher<op::eq, FixedString("mask"), 8ULL>{};

    auto result = engine::execute(entities, query);

    EXPECT_TRUE(simd::test(result.data(), 42));
    EXPECT_FALSE(simd::test(result.data(), 43));
}

// TEST(QuestSystem, QuestChainLogic) {
//     const size_t world_size = 100;
//     table players(world_size);
//     players.add_column_u32("level");
//     players.get_col("level").u32[10] = 15;

//     datalog::program p;
//     p.add_edb("player_data", players, {"id", "level"});

//     p.add_idb("quest_tutorial_done", world_size);
//     p.add_idb("quest_boss_unlocked", world_size);

//     p.add_rule({
//         "quest_boss_unlocked", {"X"}, 
//         {{"player_data", {"X"}}, {"quest_tutorial_done", {"X"}}},
//         [](const table& t) { 
//             return engine::execute(t, field_matcher<op::ge, FixedString("level"), 10u>{}); 
//         }, 
//         0
//     });

//     p.evaluate();
//     EXPECT_FALSE(datalog::bits::test(p.get_bits("quest_boss_unlocked"), 10));

//     datalog::bits::set(p.get_bits("quest_tutorial_done"), 10);
//     p.evaluate();

//     EXPECT_TRUE(datalog::bits::test(p.get_bits("quest_boss_unlocked"), 10));
// }

TEST(SpatialEngine, BoundingBoxQuery) {
    const size_t n = 100;
    table world(n);
    world.add_column_f32("x");
    world.add_column_f32("y");

    world.get_col("x").f32[7] = 15.0f;
    world.get_col("y").f32[7] = 35.0f;

    world.get_col("x").f32[8] = 25.0f;
    world.get_col("y").f32[8] = 35.0f;

    auto box_query = field_matcher<op::ge, FixedString("x"), 10.0f>{} &
                     field_matcher<op::lt, FixedString("x"), 20.0f>{} &
                     field_matcher<op::ge, FixedString("y"), 30.0f>{} &
                     field_matcher<op::lt, FixedString("y"), 40.0f>{};

    auto result = engine::execute(world, box_query);

    EXPECT_TRUE(simd::test(result.data(), 7));
    EXPECT_FALSE(simd::test(result.data(), 8));
}

TEST(Datalog, RegexPathMatcher) {
    // Alphabet: 'step' edges
    // Pattern: Find all entities 'X' that can reach a terminal node through 
    // the sequence: step -> step -> step
    // Regex equivalent: step{3}
    
    auto edges = make_edges({
        {0, 1}, {1, 2}, {2, 3}, // Match: 0 -> 1 -> 2 -> 3
        {10, 11}, {11, 12},     // No Match: too short
        {20, 21}, {21, 22}, {22, 23}, {23, 24} // Match: 20 -> 21 -> 22 -> 23
    });
    
    datalog::program p;
    p.add_edb("step", edges, {"src", "dst"});
    
    // States
    p.add_idb("S1", 30);
    p.add_idb("S2", 30);
    p.add_idb("S3", 30);

    // S1(Next) :- step(Start, Next)
    p.add_rule({"S1", {"Next"}, {{"step", {"Start", "Next"}}}});
    // S2(Next) :- S1(Curr), step(Curr, Next)
    p.add_rule({"S2", {"Next"}, {{"S1", {"Curr"}}, {"step", {"Curr", "Next"}}}});
    // S3(Next) :- S2(Curr), step(Curr, Next)
    p.add_rule({"S3", {"Next"}, {{"S2", {"Curr"}}, {"step", {"Curr", "Next"}}}});

    p.evaluate();

    EXPECT_TRUE(simd::test(p.get_bits("S3").data(), 3));  // 0 reached 3
    EXPECT_FALSE(simd::test(p.get_bits("S3").data(), 12)); // 10 only reached 12 (2 steps)
    EXPECT_TRUE(simd::test(p.get_bits("S3").data(), 23));  // 20 reached 23
    EXPECT_TRUE(simd::test(p.get_bits("S3").data(), 24));  // 20 reached 24 (it's also a valid 3-step end from 21)
}

TEST(Engine, ManualDFA) {
    const size_t n = 1024;
    table transitions(n);
    transitions.add_column_u32("next_id");
    
    for (uint32_t i = 0; i < n - 1; ++i) {
        transitions.get_col("next_id").u32[i] = i + 1;
    }

    engine::mask_t state_ready = simd::mask_t(n);
    engine::mask_t state_active = simd::mask_t(n);
    engine::mask_t state_done = simd::mask_t(n);

    simd::set(state_ready.data(), 0);

    auto step_dfa = [&](const table& t) {
        bool changed = false;
        for (size_t r = 0; r < t.rows; ++r) {
            uint32_t current_id = r; 
            uint32_t next_id = t.get_col("next_id").u32[r];

            // Ready -> Active
            if (simd::test(state_ready.data(), current_id)) {
                if (!simd::test(state_active.data(), next_id)) {
                    simd::set(state_active.data(), next_id);
                    changed = true;
                }
            }
            // Active -> Done
            if (simd::test(state_active.data(), current_id)) {
                if (!simd::test(state_done.data(), next_id)) {
                    simd::set(state_done.data(), next_id);
                    changed = true;
                }
            }
        }
        return changed;
    };

    while (step_dfa(transitions));

    EXPECT_TRUE(simd::test(state_done.data(), 2)); 
    EXPECT_FALSE(simd::test(state_done.data(), 1));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
