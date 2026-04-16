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
