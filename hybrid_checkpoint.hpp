#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include "llama.h"

struct HybridCheckpoint {
    llama_pos pos_min = -1;
    llama_pos pos_max = -1;
    int64_t   n_tokens = 0;
    std::vector<uint8_t> data;
    std::vector<int32_t> prefix_tokens;
    bool is_tail = false;
};

bool hybrid_ckpt_prefix_matches(const std::vector<int32_t>& candidate,
                                const std::vector<int32_t>& full,
                                size_t len);

int hybrid_ckpt_pick_best(const std::vector<HybridCheckpoint>& pool,
                          const std::vector<int32_t>& new_tokens,
                          int fastforward_n_past);

size_t hybrid_ckpt_evict_to(std::vector<HybridCheckpoint>& pool, size_t max_slots);

size_t hybrid_ckpt_erase_stale(std::vector<HybridCheckpoint>& pool,
                               int restored_n_tokens);

bool hybrid_ckpt_crossed_boundary(int prev_pos, int curr_pos, int interval);

enum class HybridCkptSaveKind : uint8_t {
    None     = 0,
    Periodic = 1,
    Tail     = 2,
};

struct HybridCkptSubBatch {
    int                size;
    HybridCkptSaveKind save;
};

std::vector<HybridCkptSubBatch> hybrid_ckpt_plan_sub_batches(
    int start_pos, int embd_size, int final_pos, int interval);

size_t hybrid_ckpt_prune_off_layout(std::vector<HybridCheckpoint>& pool,
                                    int L, int interval);

bool hybrid_ckpt_save(std::vector<HybridCheckpoint>& pool,
                      size_t max_slots,
                      llama_context * ctx,
                      const std::vector<int32_t>& ctx_tokens,
                      int n_past,
                      bool is_tail = false);

int hybrid_ckpt_try_restore(std::vector<HybridCheckpoint>& pool,
                            llama_context * ctx,
                            const std::vector<int32_t>& new_tokens,
                            int fastforward_n_past);

inline void hybrid_ckpt_invalidate(std::vector<HybridCheckpoint>& pool) {
    pool.clear();
}

void hybrid_ckpt_clear();

bool hybrid_ckpt_try_fast_forward(llama_context * ctx,
                                  const std::vector<int>& current_context_tokens,
                                  std::vector<int>& embd_inp,
                                  std::vector<int>& last_n_tokens,
                                  int& n_past,
                                  bool is_hybrid_model,
                                  int slots, int interval);

struct HybridCkptDecodeResult {
    bool    ok;
    int32_t status;
};

HybridCkptDecodeResult hybrid_ckpt_run_planned_decode(
    llama_context * ctx,
    const std::vector<int>& embd,
    const std::vector<int>& current_context_tokens,
    int n_past,
    int final_pos,
    int interval,
    int slots,
    bool use_mrope);

void hybrid_ckpt_prune_end_of_pp(int final_n_past,
                                 int interval,
                                 int slots,
                                 bool is_hybrid_model,
                                 int debugmode,
                                 bool is_quiet);

struct NoCachePromptGuard {
    llama_context * ctx;
    std::vector<int> & ctx_tokens_ref;
    std::vector<int> & last_n_ref;
    std::vector<uint8_t> state_backup;
    std::vector<int> tokens_backup;
    std::vector<int> last_n_backup;
    bool active = false;

    NoCachePromptGuard(llama_context * c,
                       std::vector<int> & ct,
                       std::vector<int> & ln)
        : ctx(c), ctx_tokens_ref(ct), last_n_ref(ln) {}

    bool arm() {
        if(!ctx) return false;
        tokens_backup  = ctx_tokens_ref;
        last_n_backup  = last_n_ref;
        size_t sz = llama_state_get_size(ctx);
        try {
            state_backup.resize(sz);
        } catch(const std::bad_alloc&) {
            printf("[hybrid-cpp] cache_prompt=false: failed to allocate %zu bytes for state backup, falling back to normal mode\n", sz);
            fflush(stdout);
            return false;
        }
        size_t n = llama_state_get_data(ctx, state_backup.data(), sz);
        if(n == 0) {
            printf("[hybrid-cpp] cache_prompt=false: state backup failed, falling back to normal mode\n");
            fflush(stdout);
            state_backup.clear();
            return false;
        }
        active = true;
        printf("[hybrid-cpp] cache_prompt=false: state backed up (%zu MB), hybrid pool frozen\n", sz / (1024*1024));
        fflush(stdout);
        return true;
    }

    ~NoCachePromptGuard() {
        if(!active || state_backup.empty()) return;
        size_t n = llama_state_set_data(ctx, state_backup.data(), state_backup.size());
        if(n > 0) {
            ctx_tokens_ref = tokens_backup;
            last_n_ref     = last_n_backup;
            printf("[hybrid-cpp] cache_prompt=false: state restored (%zu MB)\n", state_backup.size() / (1024*1024));
        } else {
            printf("[hybrid-cpp] cache_prompt=false: WARNING state restore failed! KV state may be corrupted\n");
        }
        fflush(stdout);
    }

    NoCachePromptGuard(const NoCachePromptGuard&) = delete;
    NoCachePromptGuard& operator=(const NoCachePromptGuard&) = delete;
};
