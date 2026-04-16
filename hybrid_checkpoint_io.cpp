#include "hybrid_checkpoint.hpp"
#include "otherarch/utils.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

static std::vector<HybridCheckpoint> g_pool;

void hybrid_ckpt_clear() {
    hybrid_ckpt_invalidate(g_pool);
}

extern "C" void kcpp_hybrid_checkpoints_invalidate() {
    hybrid_ckpt_invalidate(g_pool);
    printf("[hybrid-cpp] pool invalidated via API\n");
    fflush(stdout);
}

bool hybrid_ckpt_save(std::vector<HybridCheckpoint>& pool,
                      size_t max_slots,
                      llama_context * ctx,
                      const std::vector<int32_t>& ctx_tokens,
                      int n_past,
                      bool is_tail) {
    if(max_slots == 0 || ctx == nullptr) return false;

    auto * mem = llama_get_memory(ctx);
    llama_pos pos_min = llama_memory_seq_pos_min(mem, 0);
    llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
    if(pos_min < 0 || pos_max < 0) return false;
    if((int)(pos_max + 1) != n_past) return false;

    size_t sz = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if(sz == 0) return false;

    HybridCheckpoint cp;
    cp.pos_min = pos_min;
    cp.pos_max = pos_max;
    cp.n_tokens = n_past;
    cp.data.resize(sz);
    cp.prefix_tokens = ctx_tokens;
    cp.is_tail = is_tail;

    size_t n = llama_state_seq_get_data_ext(ctx, cp.data.data(), sz, 0,
                                            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    if(n != sz) return false;

    if(is_tail) {
        pool.erase(
            std::remove_if(pool.begin(), pool.end(),
                [](const HybridCheckpoint& c) { return c.is_tail; }),
            pool.end());
    }

    pool.push_back(std::move(cp));
    hybrid_ckpt_evict_to(pool, max_slots);

    size_t non_tail = 0;
    for(const auto& c : pool) if(!c.is_tail) ++non_tail;
    printf("[hybrid-cpp] saved slot at pos=%d, %zu bytes, pool=%zu/%zu%s [%s]\n",
           n_past, sz, non_tail, max_slots,
           is_tail ? " (+1 tail)" : "",
           is_tail ? "tail" : "interval");
    return true;
}

int hybrid_ckpt_try_restore(std::vector<HybridCheckpoint>& pool,
                            llama_context * ctx,
                            const std::vector<int32_t>& new_tokens,
                            int fastforward_n_past) {
    if(pool.empty() || ctx == nullptr) return -1;

    int idx = hybrid_ckpt_pick_best(pool, new_tokens, fastforward_n_past);
    if(idx < 0) return -1;

    const HybridCheckpoint& cp = pool[idx];
    const int         restored_n_tokens = (int)cp.n_tokens;
    const llama_pos   restored_pos_max  = cp.pos_max;
    const size_t      restored_data_sz  = cp.data.size();
    const uint8_t *   restored_data_ptr = cp.data.data();

    using Clock = std::chrono::steady_clock;
    const auto t0_set = Clock::now();
    size_t n = llama_state_seq_set_data_ext(ctx, restored_data_ptr, restored_data_sz, 0,
                                            LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    const auto t1_set = Clock::now();

    if(n != restored_data_sz) {
        printf("[hybrid-cpp] restore FAILED: set_data_ext returned %zu != %zu\n",
               n, restored_data_sz);
        return -1;
    }

    const auto t0_rm = Clock::now();
    llama_memory_seq_rm(llama_get_memory(ctx), 0, restored_pos_max + 1, -1);
    const auto t1_rm = Clock::now();

    {
        using us = std::chrono::microseconds;
        const long long set_us   = std::chrono::duration_cast<us>(t1_set - t0_set).count();
        const long long rm_us    = std::chrono::duration_cast<us>(t1_rm  - t0_rm ).count();
        const long long total_us = set_us + rm_us;
        printf("[hybrid-cpp] restore timing: set_data=%lld us, seq_rm=%lld us, total=%lld us\n",
               set_us, rm_us, total_us);
    }

    pool[idx].is_tail = false;

    size_t erased = hybrid_ckpt_erase_stale(pool, restored_n_tokens);

    printf("[hybrid-cpp] restored slot pos=%d (%zu bytes), ff_n_past=%d -> %d, erased %zu stale\n",
           restored_n_tokens, restored_data_sz, fastforward_n_past, restored_n_tokens, erased);
    return restored_n_tokens;
}

bool hybrid_ckpt_try_fast_forward(llama_context * ctx,
                                  const std::vector<int>& current_context_tokens,
                                  std::vector<int>& embd_inp,
                                  std::vector<int>& last_n_tokens,
                                  int& n_past,
                                  bool is_hybrid_model,
                                  int slots, int interval) {
    if(!is_hybrid_model || slots <= 0 || interval <= 0 || g_pool.empty()) {
        return false;
    }

    int common = 0;
    const int max_common = std::min<int>((int)current_context_tokens.size(),
                                         (int)embd_inp.size());
    while(common < max_common && current_context_tokens[common] == embd_inp[common]) {
        common++;
    }

    if(!(common > 0 && common < (int)current_context_tokens.size())) {
        return false;
    }

    std::vector<int32_t> new_tokens(embd_inp.begin(), embd_inp.end());
    const int restored_n_past = hybrid_ckpt_try_restore(
        g_pool, ctx, new_tokens, common);
    if(restored_n_past <= 0 || restored_n_past > (int)embd_inp.size()) {
        return false;
    }

    for(int i = 0; i < restored_n_past; ++i) {
        last_n_tokens.push_back(current_context_tokens[i]);
    }
    last_n_tokens.erase(last_n_tokens.begin(),
                        last_n_tokens.begin() + restored_n_past);

    embd_inp.erase(embd_inp.begin(), embd_inp.begin() + restored_n_past);
    n_past = restored_n_past;

    printf("[hybrid-cpp] restored at pos=%d, replaying %zu tail tokens\n",
           n_past, embd_inp.size());
    fflush(stdout);
    return true;
}

HybridCkptDecodeResult hybrid_ckpt_run_planned_decode(
    llama_context * ctx,
    const std::vector<int>& embd,
    const std::vector<int>& current_context_tokens,
    int n_past,
    int final_pos,
    int interval,
    int slots,
    bool use_mrope) {
    HybridCkptDecodeResult result{true, 0};

    auto plan = hybrid_ckpt_plan_sub_batches(
        n_past, (int)embd.size(), final_pos, interval);

    size_t offset = 0;
    for(const auto& chunk : plan) {
        if(chunk.size <= 0) continue;

        std::vector<llama_token> sub(embd.begin() + offset,
                                     embd.begin() + offset + (size_t)chunk.size);
        const int sub_past = n_past + (int)offset;
        kcpp_embd_batch sub_batch(sub, sub_past, use_mrope, false);

        const int32_t sub_status = llama_decode(ctx, sub_batch.batch);
        if(sub_status != 0) {
            result.ok = false;
            result.status = sub_status;
            return result;
        }

        offset += (size_t)chunk.size;
        const int end_pos = n_past + (int)offset;

        if(chunk.save == HybridCkptSaveKind::Periodic
           || chunk.save == HybridCkptSaveKind::Tail) {
            std::vector<int32_t> snap_tokens;
            snap_tokens.reserve((size_t)end_pos);
            if(n_past > 0 && (int)current_context_tokens.size() >= n_past) {
                snap_tokens.insert(snap_tokens.end(),
                                   current_context_tokens.begin(),
                                   current_context_tokens.begin() + n_past);
            }
            snap_tokens.insert(snap_tokens.end(),
                               embd.begin(),
                               embd.begin() + offset);
            hybrid_ckpt_save(g_pool, (size_t)slots, ctx, snap_tokens, end_pos,
                             chunk.save == HybridCkptSaveKind::Tail);
        }
    }

    return result;
}

void hybrid_ckpt_prune_end_of_pp(int final_n_past,
                                 int interval,
                                 int slots,
                                 bool is_hybrid_model,
                                 int debugmode,
                                 bool is_quiet) {
    if(!is_hybrid_model || slots <= 0 || interval <= 0) return;

    const size_t dropped = hybrid_ckpt_prune_off_layout(
        g_pool, final_n_past, interval);
    if(dropped > 0 && debugmode == 1 && !is_quiet) {
        printf("[hybrid-cpp] pruned %zu off-layout slots at PP end (L=%d, I=%d)\n",
               dropped, final_n_past, interval);
    }
}
