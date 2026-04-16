#include "hybrid_checkpoint.hpp"

#include <algorithm>

bool hybrid_ckpt_prefix_matches(const std::vector<int32_t>& candidate,
                                const std::vector<int32_t>& full,
                                size_t len) {
    if(len > candidate.size() || len > full.size()) return false;
    for(size_t i = 0; i < len; ++i) {
        if(candidate[i] != full[i]) return false;
    }
    return true;
}

int hybrid_ckpt_pick_best(const std::vector<HybridCheckpoint>& pool,
                          const std::vector<int32_t>& new_tokens,
                          int fastforward_n_past) {
    int best = -1;
    llama_pos best_pos = -1;
    for(size_t i = 0; i < pool.size(); ++i) {
        const auto& c = pool[i];
        if(c.pos_max < 0) continue;
        if((int)(c.pos_max + 1) > fastforward_n_past) continue;
        size_t prefix_len = (size_t)(c.pos_max + 1);
        if(!hybrid_ckpt_prefix_matches(c.prefix_tokens, new_tokens, prefix_len)) continue;
        if(c.pos_max > best_pos) {
            best = (int)i;
            best_pos = c.pos_max;
        }
    }
    return best;
}

size_t hybrid_ckpt_evict_to(std::vector<HybridCheckpoint>& pool, size_t max_slots) {
    size_t non_tail = 0;
    for(const auto& c : pool) if(!c.is_tail) ++non_tail;
    if(non_tail <= max_slots) return 0;
    size_t to_evict = non_tail - max_slots;
    size_t evicted = 0;
    auto it = pool.begin();
    while(it != pool.end() && evicted < to_evict) {
        if(!it->is_tail) {
            it = pool.erase(it);
            ++evicted;
        } else {
            ++it;
        }
    }
    return evicted;
}

size_t hybrid_ckpt_erase_stale(std::vector<HybridCheckpoint>& pool,
                               int restored_n_tokens) {
    size_t before = pool.size();
    pool.erase(
        std::remove_if(pool.begin(), pool.end(),
            [restored_n_tokens](const HybridCheckpoint& c) {
                return (int)c.n_tokens > restored_n_tokens;
            }),
        pool.end());
    return before - pool.size();
}

bool hybrid_ckpt_crossed_boundary(int prev_pos, int curr_pos, int interval) {
    if(interval <= 0) return false;
    if(curr_pos <= prev_pos) return false;
    return (curr_pos / interval) > (prev_pos / interval);
}

std::vector<HybridCkptSubBatch> hybrid_ckpt_plan_sub_batches(
    int start_pos, int embd_size, int final_pos, int interval)
{
    std::vector<HybridCkptSubBatch> out;
    if(embd_size <= 0) return out;
    const int end_pos  = start_pos + embd_size;
    const bool has_final = (final_pos >= 0);
    const int tail_pos = (has_final && final_pos > 32) ? (final_pos - 32) : -1;

    std::vector<int> cuts;
    cuts.reserve(16);

    if(tail_pos > start_pos && tail_pos <= end_pos) {
        cuts.push_back(tail_pos);
    }

    if(interval > 0) {
        int first_k = ((start_pos / interval) + 1) * interval;
        for(int k = first_k; k > 0 && k <= end_pos; k += interval) {
            cuts.push_back(k);
        }
    }

    cuts.push_back(end_pos);

    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());

    int prev = start_pos;
    for(int c : cuts) {
        if(c <= prev) continue;
        HybridCkptSubBatch sb;
        sb.size = c - prev;
        HybridCkptSaveKind kind = HybridCkptSaveKind::None;
        if(tail_pos >= 0 && c == tail_pos) {
            kind = HybridCkptSaveKind::Tail;
        } else if(interval > 0 && c > 0 && (c % interval) == 0) {
            bool below_tail;
            if(has_final) {
                below_tail = (tail_pos >= 0 && c < tail_pos);
            } else {
                below_tail = true;
            }
            if(below_tail) kind = HybridCkptSaveKind::Periodic;
        }
        sb.save = kind;
        out.push_back(sb);
        prev = c;
    }
    return out;
}

size_t hybrid_ckpt_prune_off_layout(std::vector<HybridCheckpoint>& pool,
                                    int L, int interval)
{
    const size_t before = pool.size();
    const int tail_pos  = (L > 32) ? (L - 32) : -1;
    auto it = pool.begin();
    while(it != pool.end()) {
        const int pos = (int)it->n_tokens;
        const bool at_tail = (tail_pos >= 0 && pos == tail_pos);
        const bool on_grid = (!at_tail
                              && interval > 0 && pos > 0
                              && (pos % interval) == 0
                              && tail_pos >= 0 && pos < tail_pos);
        if(!at_tail && !on_grid) {
            it = pool.erase(it);
        } else {
            it->is_tail = at_tail;
            ++it;
        }
    }
    return before - pool.size();
}
