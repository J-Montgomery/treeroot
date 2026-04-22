#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "engine.hpp"
#include "simd.hpp"

namespace datalog {

using mask_t = engine::mask_t;

struct atom {
    std::string rel;
    std::vector<std::string> vars;
    bool negated = false;
};

struct rule_def {
    std::string head;
    std::vector<std::string> head_vars;
    std::vector<atom> body;
    std::function<mask_t(const table &)> filter;
    size_t filter_body = 0;
};

class program {
    struct edb_entry {
        std::string name;
        const table *tbl;
        std::vector<std::string> cols;
    };
    struct idb_entry {
        std::string name;
        size_t universe;
        mask_t mask;
    };

    std::vector<edb_entry> edbs_;
    std::vector<idb_entry> idbs_;
    std::vector<rule_def> rules_;
    std::vector<mask_t> filter_cache_;
    std::unordered_map<std::string, size_t> idb_index_;

    const edb_entry *edb(const std::string &n) const {
        for (auto &e : edbs_)
            if (e.name == n)
                return &e;
        return nullptr;
    }
    idb_entry *idb(const std::string &n) {
        auto it = idb_index_.find(n);
        return it != idb_index_.end() ? &idbs_[it->second] : nullptr;
    }
    const idb_entry *idb(const std::string &n) const {
        auto it = idb_index_.find(n);
        return it != idb_index_.end() ? &idbs_[it->second] : nullptr;
    }

    static int shared_edb_col(const atom &ea, const atom &ba) {
        for (size_t i = 0; i < ba.vars.size(); ++i)
            for (size_t j = 0; j < ea.vars.size(); ++j)
                if (ba.vars[i] == ea.vars[j])
                    return (int)j;
        return -1;
    }

    void precompute_filters() {
        filter_cache_.assign(rules_.size(), {});
        for (size_t i = 0; i < rules_.size(); ++i) {
            auto &r = rules_[i];
            if (!r.filter || r.filter_body >= r.body.size())
                continue;
            auto *ed = edb(r.body[r.filter_body].rel);
            if (ed)
                filter_cache_[i] = r.filter(*ed->tbl);
        }
    }

    void fire(size_t rule_idx, const std::string *dr, const mask_t *db,
              mask_t &out) {
        const rule_def &rule = rules_[rule_idx];

        size_t ei = SIZE_MAX;
        for (size_t i = 0; i < rule.body.size(); ++i)
            if (edb(rule.body[i].rel)) {
                ei = i;
                break;
            }

        if (ei == SIZE_MAX) {
            if (rule.body.empty())
                return;
            size_t w = out.size();
            mask_t res(w, ~0ULL);
            for (auto &ba : rule.body) {
                auto *p = idb(ba.rel);
                if (!p)
                    continue;
                const mask_t &b = (dr && ba.rel == *dr && db) ? *db : p->mask;
                if (ba.negated)
                    simd::bandnot(res.data(), b.data(), w);
                else
                    simd::band(res.data(), b.data(), w);
            }
            simd::bor(out.data(), res.data(), w);
            return;
        }

        auto &ea = rule.body[ei];
        auto *ed = edb(ea.rel);
        const table &t = *ed->tbl;
        size_t n = t.rows;
        size_t w = simd::num_words(n);

        mask_t rm(w, ~0ULL);
        simd::clear_tail(rm.data(), w, n);

        if (rule.filter && rule.filter_body == ei) {
            const mask_t &f = filter_cache_[rule_idx];
            if (!f.empty())
                simd::band(rm.data(), f.data(), w);
        }

        for (size_t i = 0; i < rule.body.size(); ++i) {
            if (i == ei)
                continue;
            auto &ba = rule.body[i];
            auto *p = idb(ba.rel);
            if (!p)
                continue;

            const mask_t &b = (dr && ba.rel == *dr && db) ? *db : p->mask;
            int ec = shared_edb_col(ea, ba);
            if (ec < 0)
                continue;

            const auto &col = t.get_col(ed->cols[static_cast<size_t>(ec)]).u32;
            mask_t sj(w, 0);
            for (size_t r = 0; r < n; ++r)
                if (simd::test(b.data(), col[r]))
                    sj[r / 64] |= 1ULL << (r % 64);

            if (ba.negated) {
                simd::bnot(sj.data(), w);
                simd::clear_tail(sj.data(), w, n);
            }
            simd::band(rm.data(), sj.data(), w);
        }

        int pc = -1;
        for (size_t j = 0; j < ea.vars.size(); ++j)
            if (ea.vars[j] == rule.head_vars[0]) {
                pc = (int)j;
                break;
            }
        if (pc < 0)
            return;

        const auto &pcol = t.get_col(ed->cols[static_cast<size_t>(pc)]).u32;
        for (size_t r = 0; r < n; ++r)
            if (rm[r / 64] & (1ULL << (r % 64)))
                simd::set(out.data(), pcol[r]);
    }

    struct stratum_t {
        std::vector<size_t> base_rules;
        std::vector<size_t> rec_rules;
    };

    std::vector<mask_t> accum_;
    std::vector<mask_t> deltas_;
    std::vector<size_t> involved_;
    std::vector<uint8_t> seen_idb_;

    std::vector<stratum_t> stratify() {
        std::unordered_map<std::string, int> sn;
        for (auto &e : edbs_)
            sn[e.name] = -1;
        for (auto &e : idbs_)
            sn[e.name] = 0;
        for (bool chg = true; chg;) {
            chg = false;
            for (auto &r : rules_) {
                int req = 0;
                for (auto &a : r.body) {
                    auto it = sn.find(a.rel);
                    if (it == sn.end() || it->second < 0)
                        continue;
                    req = std::max(req, a.negated ? it->second + 1 : it->second);
                }
                if (req > sn[r.head]) {
                    sn[r.head] = req;
                    chg = true;
                }
            }
        }
        int mx = 0;
        for (auto &[k, s] : sn)
            mx = std::max(mx, s);
        
        std::vector<stratum_t> res(mx + 1);
        for (size_t i = 0; i < rules_.size(); ++i) {
            int s = sn[rules_[i].head];
            if (s >= 0) {
                if (is_recursive(rules_[i]))
                    res[s].rec_rules.push_back(i);
                else
                    res[s].base_rules.push_back(i);
            }
        }
        return res;
    }

    bool is_recursive(const rule_def &r) const {
        for (auto &a : r.body)
            if (!a.negated && idb(a.rel))
                return true;
        return false;
    }

    void eval_stratum(const stratum_t &s) {
        for (size_t ri : s.base_rules) {
            auto *h = idb(rules_[ri].head);
            fire(ri, nullptr, nullptr, h->mask);
        }
        
        if (s.rec_rules.empty())
            return;

        involved_.clear();
        for (size_t ri : s.rec_rules) {
            size_t h_idx = idb_index_[rules_[ri].head];
            if (!seen_idb_[h_idx]) {
                seen_idb_[h_idx] = 1;
                involved_.push_back(h_idx);
            }
            for (auto &a : rules_[ri].body) {
                if (!a.negated) {
                    auto it = idb_index_.find(a.rel);
                    if (it != idb_index_.end() && !seen_idb_[it->second]) {
                        seen_idb_[it->second] = 1;
                        involved_.push_back(it->second);
                    }
                }
            }
        }

        for (size_t idx : involved_) {
            deltas_[idx] = idbs_[idx].mask; 
        }

        for (bool any_new = true; any_new;) {
            any_new = false;
            
            for (size_t idx : involved_) {
                std::fill(accum_[idx].begin(), accum_[idx].end(), 0);
            }

            for (size_t ri : s.rec_rules) {
                auto &rule = rules_[ri];
                size_t h_idx = idb_index_[rule.head];
                
                for (auto &ba : rule.body) {
                    if (ba.negated) continue;
                    
                    auto it = idb_index_.find(ba.rel);
                    if (it == idb_index_.end()) continue;
                    
                    fire(ri, &ba.rel, &deltas_[it->second], accum_[h_idx]);
                    break;
                }
            }

            for (size_t idx : involved_) {
                auto &mask = idbs_[idx].mask;
                auto &ac = accum_[idx];
                size_t w = ac.size();

                simd::bandnot(ac.data(), mask.data(), w);
                if (simd::any(ac.data(), w)) {
                    any_new = true;
                    simd::bor(mask.data(), ac.data(), w);
                }
                
                deltas_[idx].swap(ac); 
            }
        }
        
        for (size_t idx : involved_) {
            seen_idb_[idx] = 0;
        }
    }

  public:
    void add_edb(std::string name, const table &t,
                 std::vector<std::string> cols) {
        edbs_.push_back({std::move(name), &t, std::move(cols)});
    }

    void add_idb(std::string name, size_t universe) {
        idb_index_[name] = idbs_.size();
        idb_entry e;
        e.name = std::move(name);
        e.universe = universe;
        e.mask.assign(simd::num_words(universe), 0);
        idbs_.push_back(std::move(e));
    }

    void add_rule(rule_def r) { rules_.push_back(std::move(r)); }

    void evaluate() {
        precompute_filters();
        auto ss = stratify();

        size_t num_idbs = idbs_.size();
        accum_.resize(num_idbs);
        deltas_.resize(num_idbs);
        seen_idb_.assign(num_idbs, 0);
        involved_.reserve(num_idbs);

        for (size_t i = 0; i < num_idbs; ++i) {
            size_t words = simd::num_words(idbs_[i].universe);
            accum_[i].assign(words, 0);
            deltas_[i].assign(words, 0);
        }

        for (auto &s : ss)
            eval_stratum(s);
    }

    mask_t &get_bits(const std::string &n) { return idb(n)->mask; }
    const mask_t &get_bits(const std::string &n) const { return idb(n)->mask; }
};

} // namespace datalog
