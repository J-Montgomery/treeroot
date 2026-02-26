#pragma once

#include <cstdint>
#include <cstddef>

#include <functional>
#include <string>
#include <unordered_set>
#include <vector>
#include "engine.hpp"

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

// A basic monadic datalog impl
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
