#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

// Internal adapter layout (for diagnostics)
#include "../../src/llama-adapter.h"

#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <filesystem>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

static int g_finetune_progress_every = 1;
static int g_finetune_step_save_every = 0;
static int g_finetune_exit_after_step = 0;
static int64_t g_finetune_last_saved_iter = 1;
static std::string g_finetune_step_save_work_file;
static std::vector<llama_adapter_lora *> g_finetune_step_save_loras;

static void maybe_save_step_checkpoint(bool train, ggml_opt_context_t opt_ctx);

static void ggml_opt_epoch_callback_progress_bar_throttled(
        bool               train,
        ggml_opt_context_t opt_ctx,
        ggml_opt_dataset_t dataset,
        ggml_opt_result_t  result,
        int64_t            ibatch,
        int64_t            ibatch_max,
        int64_t            t_start_us) {
    const int every = g_finetune_progress_every;
    if (every <= 1 || ibatch == 0 || ibatch >= ibatch_max || (ibatch % every) == 0) {
        ggml_opt_epoch_callback_progress_bar(train, opt_ctx, dataset, result, ibatch, ibatch_max, t_start_us);
    }

    maybe_save_step_checkpoint(train, opt_ctx);
}

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)  // possible loss of data
#endif

static bool opt_param_filter_lora_only(const struct ggml_tensor * tensor, void * userdata) {
    (void) userdata;
    const std::string_view name(tensor->name);
    const auto has_suffix = [](std::string_view s, std::string_view suffix) {
        return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    return has_suffix(name, ".lora_a") || has_suffix(name, ".lora_b");
}

static bool opt_param_filter_lora_selected(const struct ggml_tensor * tensor, void * userdata) {
    const auto * selected = static_cast<const std::vector<std::string> *>(userdata);
    if (selected == nullptr || selected->empty()) {
        return opt_param_filter_lora_only(tensor, nullptr);
    }

    const std::string_view name(tensor->name);
    const auto has_suffix = [](std::string_view s, std::string_view suffix) {
        return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    std::string_view base = name;
    if (has_suffix(base, ".lora_a")) {
        base = base.substr(0, base.size() - std::string_view(".lora_a").size());
    } else if (has_suffix(base, ".lora_b")) {
        base = base.substr(0, base.size() - std::string_view(".lora_b").size());
    } else {
        return false;
    }

    for (const std::string & s : *selected) {
        if (s.empty()) {
            continue;
        }
        std::string_view want = s;
        if (has_suffix(want, ".lora_a")) {
            want = want.substr(0, want.size() - std::string_view(".lora_a").size());
        } else if (has_suffix(want, ".lora_b")) {
            want = want.substr(0, want.size() - std::string_view(".lora_b").size());
        }
        // Support trailing '*' as prefix match
        if (!want.empty() && want.back() == '*') {
            std::string_view prefix = want.substr(0, want.size() - 1);
            if (base.size() >= prefix.size() &&
                base.compare(0, prefix.size(), prefix) == 0) {
                return true;
            }
        } else if (base == want) {
            return true;
        }
    }

    return false;
}

static void lora_prune_adapter_to_selected(llama_adapter_lora * adapter, const std::vector<std::string> & selected) {
    if (adapter == nullptr || selected.empty()) {
        return;
    }

    const auto has_suffix = [](std::string_view s, std::string_view suffix) {
        return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
    };

    const auto is_selected = [&](std::string_view base) {
        for (const std::string & s : selected) {
            if (s.empty()) {
                continue;
            }
            std::string_view want = s;
            if (has_suffix(want, ".lora_a")) {
                want = want.substr(0, want.size() - std::string_view(".lora_a").size());
            } else if (has_suffix(want, ".lora_b")) {
                want = want.substr(0, want.size() - std::string_view(".lora_b").size());
            }
            // Support trailing '*' as prefix match (e.g., "blk.4*" matches blk.40..blk.47)
            if (!want.empty() && want.back() == '*') {
                std::string_view prefix = want.substr(0, want.size() - 1);
                if (base.size() >= prefix.size() &&
                    base.compare(0, prefix.size(), prefix) == 0) {
                    return true;
                }
            } else if (base == want) {
                return true;
            }
        }
        return false;
    };

    const size_t before = adapter->ab_map.size();
    for (auto it = adapter->ab_map.begin(); it != adapter->ab_map.end(); ) {
        if (!is_selected(it->first)) {
            it = adapter->ab_map.erase(it);
        } else {
            ++it;
        }
    }
    const size_t after = adapter->ab_map.size();
    if (after != before) {
        LOG_INF("%s: pruned LoRA adapter tensors by --lora-train-base: %zu -> %zu base tensors (%zu -> %zu tensors)\n",
                __func__, before, after, before * 2, after * 2);
    }
}

template <typename TLoras>
static void log_lora_stats_if_requested(
    const common_params & params,
    const TLoras & loras,
    const char * base_name) {
    if (params.verbosity < 2 || loras.empty() || base_name == nullptr) {
        return;
    }

    llama_adapter_lora * adapter = loras[0].get();
    if (adapter == nullptr) {
        return;
    }

    const auto it = adapter->ab_map.find(base_name);
    if (it == adapter->ab_map.end() || it->second.a == nullptr || it->second.b == nullptr) {
        fprintf(stderr, "%s: LoRA stats: '%s' not found in adapter[0]\n", __func__, base_name);
        return;
    }

    const auto & w = it->second;

    auto summarize_f32 = [](const ggml_tensor * t) {
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));

        int64_t nnz = 0;
        double abs_sum = 0.0;
        float abs_max = 0.0f;
        for (float v : buf) {
            if (v != 0.0f) {
                ++nnz;
            }
            const float av = std::abs(v);
            abs_sum += av;
            abs_max = std::max(abs_max, av);
        }
        const int64_t n = (int64_t) buf.size();
        const double abs_mean = n > 0 ? abs_sum / (double) n : 0.0;
        return std::tuple<int64_t, double, float>(nnz, abs_mean, abs_max);
    };

    if (w.a->type == GGML_TYPE_F32) {
        auto [nnz, abs_mean, abs_max] = summarize_f32(w.a);
        fprintf(stderr, "%s: LoRA[%s].a: ne=(%lld,%lld) nnz=%lld/%lld abs_mean=%.6g abs_max=%.6g\n",
                __func__, base_name,
                (long long) w.a->ne[0], (long long) w.a->ne[1],
                (long long) nnz, (long long) ggml_nelements(w.a), abs_mean, abs_max);
    } else {
        fprintf(stderr, "%s: LoRA[%s].a: unsupported type=%d for stats (expected f32)\n",
                __func__, base_name, (int) w.a->type);
    }

    if (w.b->type == GGML_TYPE_F32) {
        auto [nnz, abs_mean, abs_max] = summarize_f32(w.b);
        fprintf(stderr, "%s: LoRA[%s].b: ne=(%lld,%lld) nnz=%lld/%lld abs_mean=%.6g abs_max=%.6g\n",
                __func__, base_name,
                (long long) w.b->ne[0], (long long) w.b->ne[1],
                (long long) nnz, (long long) ggml_nelements(w.b), abs_mean, abs_max);
    } else {
        fprintf(stderr, "%s: LoRA[%s].b: unsupported type=%d for stats (expected f32)\n",
                __func__, base_name, (int) w.b->type);
    }
}

static bool save_file_atomic(const std::string & tmp_path, const std::string & final_path) {
    try {
        namespace fs = std::filesystem;

        std::error_code ec;
        fs::remove(final_path, ec); // best-effort (Windows rename won't overwrite)
        fs::rename(tmp_path, final_path, ec);
        if (!ec) {
            return true;
        }

        // Fallback: copy + remove
        fs::copy_file(tmp_path, final_path, fs::copy_options::overwrite_existing, ec);
        if (ec) {
            return false;
        }
        fs::remove(tmp_path, ec);
        return true;
    } catch (...) {
        return false;
    }
}

static std::string make_work_path_from_out(const std::string & out_file) {
    // Example: out.gguf -> out_WORK.gguf
    const std::string ext = ".gguf";
    if (out_file.size() >= ext.size() && out_file.compare(out_file.size() - ext.size(), ext.size(), ext) == 0) {
        return out_file.substr(0, out_file.size() - ext.size()) + "_WORK" + ext;
    }
    return out_file + "_WORK.gguf";
}

static std::string make_work_path_indexed(const std::string & work_file, size_t i, size_t n) {
    if (n <= 1) {
        return work_file;
    }
    const std::string ext = ".gguf";
    if (work_file.size() >= ext.size() && work_file.compare(work_file.size() - ext.size(), ext.size(), ext) == 0) {
        return work_file.substr(0, work_file.size() - ext.size()) + "-" + std::to_string(i) + ext;
    }
    return work_file + "-" + std::to_string(i) + ext;
}

static void maybe_save_step_checkpoint(bool train, ggml_opt_context_t opt_ctx) {
    if (!train || opt_ctx == nullptr) {
        return;
    }
    if (g_finetune_step_save_every <= 0 || g_finetune_step_save_work_file.empty() || g_finetune_step_save_loras.empty()) {
        return;
    }

    const int64_t iter = ggml_opt_get_iter(opt_ctx);
    if (iter <= g_finetune_last_saved_iter) {
        return;
    }

    g_finetune_last_saved_iter = iter;

    const int64_t completed_step = iter - 1;
    if (completed_step <= 0 || (completed_step % g_finetune_step_save_every) != 0) {
        return;
    }

    const size_t n = g_finetune_step_save_loras.size();
    for (size_t i = 0; i < n; ++i) {
        llama_adapter_lora * adapter = g_finetune_step_save_loras[i];
        if (adapter == nullptr) {
            continue;
        }

        const std::string out_i = make_work_path_indexed(g_finetune_step_save_work_file, i, n);
        const std::string tmp_i = out_i + ".tmp";
        const bool ok = llama_adapter_lora_save(adapter, tmp_i.c_str()) && save_file_atomic(tmp_i, out_i);
        if (!ok) {
            fprintf(stderr, "%s: WARNING: failed to write step checkpoint '%s'\n", __func__, out_i.c_str());
        } else {
            fprintf(stderr, "%s: wrote checkpoint (step %lld) to '%s'\n",
                    __func__, (long long) completed_step, out_i.c_str());
        }
    }
    fflush(stderr);

    if (g_finetune_exit_after_step > 0 && completed_step >= g_finetune_exit_after_step) {
        fprintf(stderr,
                "%s: reached diagnostic stop step %lld (target=%d), exiting after checkpoint save\n",
                __func__, (long long) completed_step, g_finetune_exit_after_step);
        fflush(stderr);
        std::exit(0);
    }
}

static ggml_opt_dataset_t build_sft_dataset(
        struct llama_context * ctx,
    const struct llama_vocab * vocab,
        const std::string & text,
        int64_t n_ctx) {
    struct sft_example {
        std::string user;      // includes leading space if present after ':'
        std::string assistant; // includes leading space if present after ':'
    };

    auto starts_with = [](const std::string & s, const char * pref) {
        const size_t n = std::strlen(pref);
        return s.size() >= n && s.compare(0, n, pref) == 0;
    };

    std::vector<sft_example> examples;
    sft_example cur;
    enum { NONE, IN_USER, IN_ASSISTANT } state = NONE;

    std::istringstream iss(text);
    std::string line;
    while (std::getline(iss, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        // ignore comments
        if (!line.empty() && line[0] == '#') {
            continue;
        }

        // separators / rule blocks are treated as non-data by default
        if (line == "---" || line == "Rule:" || line == "") {
            continue;
        }

        if (starts_with(line, "User:")) {
            if (!cur.user.empty() && !cur.assistant.empty()) {
                examples.push_back(cur);
                cur = {};
            }
            cur.user = line.substr(std::strlen("User:"));
            state = IN_USER;
            continue;
        }

        if (starts_with(line, "Assistant:")) {
            cur.assistant = line.substr(std::strlen("Assistant:"));
            state = IN_ASSISTANT;
            continue;
        }

        // Multi-line continuation support (rare in our tiny dataset, but safe)
        if (state == IN_USER) {
            cur.user.append("\n");
            cur.user.append(line);
        } else if (state == IN_ASSISTANT) {
            cur.assistant.append("\n");
            cur.assistant.append(line);
        }
    }
    if (!cur.user.empty() && !cur.assistant.empty()) {
        examples.push_back(cur);
    }

    if (examples.empty()) {
        LOG_ERR("%s: --train-sft enabled but no User:/Assistant: pairs were found in the training text\n", __func__);
        return nullptr;
    }

    ggml_opt_dataset_t dataset = ggml_opt_dataset_init(
        GGML_TYPE_I32, GGML_TYPE_I32, n_ctx, n_ctx, (int64_t) examples.size(), /*ndata_shard=*/ 1);

    llama_token * data   = (llama_token *) ggml_opt_dataset_data(dataset)->data;
    llama_token * labels = (llama_token *) ggml_opt_dataset_labels(dataset)->data;

    const llama_token tok_eos = llama_vocab_eos(vocab);

    for (int64_t idata = 0; idata < (int64_t) examples.size(); ++idata) {
        const auto & ex = examples[idata];

        const std::string prompt = "User:" + ex.user + "\nAssistant:";
        std::string completion = ex.assistant;
        if (completion.empty() || completion.back() != '\n') {
            completion.push_back('\n');
        }

        std::vector<llama_token> tok_prompt = common_tokenize(ctx, prompt, /*add_special=*/ true);
        std::vector<llama_token> tok_comp   = common_tokenize(ctx, completion, /*add_special=*/ false);

        std::vector<llama_token> seq;
        seq.reserve(tok_prompt.size() + tok_comp.size());
        seq.insert(seq.end(), tok_prompt.begin(), tok_prompt.end());
        seq.insert(seq.end(), tok_comp.begin(), tok_comp.end());

        if ((int64_t) seq.size() > n_ctx) {
            LOG_ERR("%s: SFT example %lld is too long for --ctx-size (tokens=%lld, n_ctx=%lld)\n",
                    __func__, (long long) idata, (long long) seq.size(), (long long) n_ctx);
            ggml_opt_dataset_free(dataset);
            return nullptr;
        }

        // Fill data with EOS padding, labels masked with -1 by default.
        for (int64_t i = 0; i < n_ctx; ++i) {
            data  [idata*n_ctx + i] = tok_eos;
            labels[idata*n_ctx + i] = -1;
        }

        for (int64_t i = 0; i < (int64_t) seq.size(); ++i) {
            data[idata*n_ctx + i] = seq[i];
        }

        // Apply loss only to assistant completion tokens.
        // To predict the first completion token, include the last prompt token position.
        const int64_t start = (int64_t) tok_prompt.size() - 1;
        const int64_t end   = (int64_t) seq.size() - 1;
        for (int64_t i = std::max<int64_t>(0, start); i < end; ++i) {
            labels[idata*n_ctx + i] = seq[i + 1];
        }
    }

    return dataset;
}

int main(int argc, char ** argv) {
    common_params params;
    params.escape = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    if (const char * env = getenv("LLAMA_FINETUNE_PROGRESS_EVERY"); env != nullptr && *env != '\0') {
        char * end = nullptr;
        const long v = strtol(env, &end, 10);
        if (end != env && *end == '\0' && v > 0) {
            g_finetune_progress_every = (int) v;
            LOG_INF("%s: LLAMA_FINETUNE_PROGRESS_EVERY=%d\n", __func__, g_finetune_progress_every);
        } else {
            LOG_WRN("%s: invalid LLAMA_FINETUNE_PROGRESS_EVERY='%s' (expected positive integer), using default\n", __func__, env);
        }
    }

    // This example is for training; the generation warmup pass is not needed and can fail
    // for some model+adapter configurations.
    params.warmup = false;

    // Full-model training requires writable weights, so memory-mapping must be disabled.
    // For LoRA-only training (including --lora-create), weights are not updated and mmap is OK.
    if (!params.lora_create && params.lora_adapters.empty() && params.use_mmap) {
        LOG_INF("%s: force disabling memory mapping for full-model training (weights must be writable)\n", __func__);
        params.use_mmap = false;
    }
    if (params.cache_type_k != GGML_TYPE_F32) {
        LOG_INF("%s: force changing k cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
        params.cache_type_k = GGML_TYPE_F32;
    }
    if (params.cache_type_v != GGML_TYPE_F32) {
        LOG_INF("%s: force changing v cache type to f32 due to a lack of f16 support for OUT_PROD\n", __func__);
        params.cache_type_v = GGML_TYPE_F32;
    }

    // UNSLOTH/bitsandbytes style optimizations
    if (params.gradient_checkpointing) {
        LOG_INF("%s: gradient checkpointing enabled (reduces memory usage at cost of recomputation)\n", __func__);
        if (params.checkpoint_layers == -1) {
            LOG_INF("%s: using automatic layer selection for gradient checkpointing\n", __func__);
        } else if (params.checkpoint_layers > 0) {
            LOG_INF("%s: checkpointing %d transformer layers\n", __func__, params.checkpoint_layers);
        }
    }

    if (params.use_mixed_precision) {
        LOG_INF("%s: mixed precision training enabled (FP16/BF16 for speed and memory efficiency)\n", __func__);
        // Note: Actual FP16/BF16 compute depends on backend support
    }

    if (params.use_8bit_optimizer) {
        LOG_INF("%s: 8-bit quantized optimizer enabled (AdamW8bit - reduces optimizer memory by ~75%%)\n", __func__);
        if (params.lora_opt_offload) {
            LOG_WRN("%s: both --8bit-optimizer and --lora-opt-offload are enabled; they serve similar purposes\n", __func__);
            LOG_WRN("%s: consider using only one memory optimization strategy at a time\n", __func__);
        }
    }

    if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED) {
        LOG_INF("%s: Flash Attention enabled (reduces attention memory from O(n²) to O(n))\n", __func__);
    } else if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
        LOG_INF("%s: Flash Attention on auto (will be enabled if supported by backend)\n", __func__);
    }

    // Combined optimization summary
    if (params.gradient_checkpointing || params.use_mixed_precision || 
        params.use_8bit_optimizer || params.lora_opt_offload) {
        LOG_INF("\n%s: === Memory Optimization Summary ===\n", __func__);
        LOG_INF("%s: Gradient Checkpointing:  %s\n", __func__, params.gradient_checkpointing ? "ON" : "off");
        LOG_INF("%s: Mixed Precision:         %s\n", __func__, params.use_mixed_precision ? "ON" : "off");
        LOG_INF("%s: 8-bit Optimizer:         %s\n", __func__, params.use_8bit_optimizer ? "ON" : "off");
        LOG_INF("%s: LoRA Opt Offload:        %s\n", __func__, params.lora_opt_offload ? "ON" : "off");
        LOG_INF("%s: Flash Attention:         %s\n", __func__, 
                params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED ? "ON" :
                params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO ? "AUTO" : "off");
        LOG_INF("%s: ===================================\n\n", __func__);
    }

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    // Ensure n_ctx is a multiple of 256 (matches GGML_PAD in llama_context, see llama-context.cpp:179).
    // Then set n_batch = n_ubatch = n_ctx so that each training sample is processed as a single
    // ubatch.  Without this, GGML_PAD may silently enlarge n_ctx, and the resulting
    // ubatch_per_ctx > 1 produces zero-loss padding steps that corrupt AdamW momentum.
    {
        const int64_t padded = GGML_PAD(params.n_ctx, 256);
        if ((int64_t)params.n_ctx != padded) {
            LOG_INF("%s: adjusting n_ctx %d -> %lld (GGML_PAD to 256)\n", __func__, params.n_ctx, (long long)padded);
        }
        params.n_ctx    = (int32_t) padded;
        params.n_batch  = params.n_ctx;
        params.n_ubatch = params.n_ctx;
    }

    // load the model and apply lora adapter, if any
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == NULL) {
        LOG_ERR("%s: unable to load model\n", __func__);
        return 1;
    }

    // If the training run is limited to a subset of base tensors via --lora-train-base,
    // prune any user-provided adapters too. Otherwise, untrained/random LoRA tensors can
    // still be applied during forward passes and destabilize both training and inference.
    if (!params.lora_train_base.empty()) {
        auto & loras = llama_init->lora();
        if (!loras.empty()) {
            for (auto & lora : loras) {
                lora_prune_adapter_to_selected(lora.get(), params.lora_train_base);
            }

            // Re-apply adapter bindings after pruning.
            common_set_adapter_lora(ctx, params.lora_adapters);
        }
    }

    // Work/checkpoint path derived from -o (used for per-epoch checkpoints and resume).
    const std::string work_file = (!params.out_file.empty())
        ? make_work_path_from_out(params.out_file)
        : std::string();

    // If training a newly created adapter, automatically resume from the work checkpoint
    // when present (e.g. after an interrupted run).
    if (params.lora_create && params.lora_adapters.empty() && !work_file.empty()) {
        namespace fs = std::filesystem;
        std::error_code ec;
        if (fs::exists(work_file, ec) && !ec) {
            llama_adapter_lora_ptr lora_resume;
            lora_resume.reset(llama_adapter_lora_init(model, work_file.c_str()));
            if (lora_resume == nullptr) {
                LOG_ERR("%s: found checkpoint '%s' but failed to load it\n", __func__, work_file.c_str());
                llama_backend_free();
                return 1;
            }

            auto & loras = llama_init->lora();
            const float scale = 1.0f;

            // If resuming a training run that was limited to a subset of base tensors,
            // keep the checkpoint compact and avoid carrying around unused (all-zero) LoRA tensors.
            if (!params.lora_train_base.empty()) {
                lora_prune_adapter_to_selected(lora_resume.get(), params.lora_train_base);
            }

            params.lora_adapters.push_back({ work_file, scale, "", "", lora_resume.get() });
            loras.emplace_back(std::move(lora_resume));

            common_set_adapter_lora(ctx, params.lora_adapters);
            LOG_INF("%s: resumed LoRA adapter from checkpoint '%s'\n", __func__, work_file.c_str());
        }
    }

    // If requested, create a fresh LoRA adapter when none is provided.
    if (params.lora_create && params.lora_adapters.empty()) {
        if (params.out_file.empty()) {
            LOG_ERR("%s: --lora-create requires -o <output_adapter.gguf> to save the trained adapter\n", __func__);
            llama_backend_free();
            return 1;
        }

        const int32_t rank = params.lora_rank > 0 ? params.lora_rank : 8;
        const float alpha = (params.lora_alpha == 0.0f) ? (float) rank : params.lora_alpha;

        llama_adapter_lora_ptr lora_new;
        lora_new.reset(llama_adapter_lora_init_from_model(model, rank, alpha));
        if (lora_new == nullptr) {
            LOG_ERR("%s: failed to create a new LoRA adapter (rank=%d alpha=%.3f)\n", __func__, rank, (double) alpha);
            llama_backend_free();
            return 1;
        }

        // When the user explicitly limits training to specific base tensors (e.g. output.weight),
        // prune the freshly created adapter so the resulting GGUF contains only what was requested.
        if (!params.lora_train_base.empty()) {
            lora_prune_adapter_to_selected(lora_new.get(), params.lora_train_base);
        }

        // Keep ownership in common init result (same pattern as loaded adapters).
        auto & loras = llama_init->lora();
        const float scale = 1.0f;
        params.lora_adapters.push_back({ "<created>", scale, "", "", lora_new.get() });
        loras.emplace_back(std::move(lora_new));

        common_set_adapter_lora(ctx, params.lora_adapters);
        LOG_INF("%s: created a new LoRA adapter in memory (rank=%d alpha=%.3f) and applied it\n", __func__, rank, (double) alpha);
    }

    // Diagnostics: initial LoRA tensor stats (verbosity>=2)
    log_lora_stats_if_requested(params, llama_init->lora(), "output.weight");

    if (!params.lora_adapters.empty() && !work_file.empty()) {
        g_finetune_step_save_every = 0;
        g_finetune_exit_after_step = 0;
        g_finetune_last_saved_iter = 1;
        g_finetune_step_save_work_file = work_file;
        g_finetune_step_save_loras.clear();

        if (const char * env = getenv("LLAMA_FINETUNE_SAVE_EVERY_STEP"); env != nullptr && *env != '\0') {
            char * end = nullptr;
            const long v = strtol(env, &end, 10);
            if (end != env && *end == '\0' && v > 0) {
                g_finetune_step_save_every = (int) v;
            } else {
                LOG_WRN("%s: invalid LLAMA_FINETUNE_SAVE_EVERY_STEP='%s' (expected positive integer), step checkpointing disabled\n",
                        __func__, env);
            }
        }

        if (const char * env = getenv("LLAMA_FINETUNE_EXIT_AFTER_STEP"); env != nullptr && *env != '\0') {
            char * end = nullptr;
            const long v = strtol(env, &end, 10);
            if (end != env && *end == '\0' && v > 0) {
                g_finetune_exit_after_step = (int) v;
            } else {
                LOG_WRN("%s: invalid LLAMA_FINETUNE_EXIT_AFTER_STEP='%s' (expected positive integer), diagnostic stop disabled\n",
                        __func__, env);
            }
        }

        if (g_finetune_exit_after_step > 0 && g_finetune_step_save_every <= 0) {
            g_finetune_step_save_every = 1;
        }

        if ((g_finetune_step_save_every > 0 || g_finetune_exit_after_step > 0) && g_finetune_step_save_loras.empty()) {
            for (auto & lora : llama_init->lora()) {
                if (lora) {
                    g_finetune_step_save_loras.push_back(lora.get());
                }
            }
        }

        if (g_finetune_step_save_every > 0) {
            LOG_INF("%s: step checkpointing enabled every %d optimizer step(s) -> overwrite '%s'\n",
                    __func__, g_finetune_step_save_every, work_file.c_str());
        }

        if (g_finetune_exit_after_step > 0) {
            LOG_INF("%s: diagnostic stop will exit after optimizer step %d\n",
                    __func__, g_finetune_exit_after_step);
        }
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    }

    const int64_t n_ctx  = llama_n_ctx(ctx);

    ggml_opt_dataset_t dataset = nullptr;
    if (params.train_sft) {
        dataset = build_sft_dataset(ctx, llama_model_get_vocab(model), params.prompt, n_ctx);
        if (!dataset) {
            llama_backend_free();
            return 1;
        }
        LOG_INF("%s: --train-sft enabled -> building supervised dataset from User:/Assistant: pairs\n", __func__);
    } else {
        std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);

        const int64_t stride = std::max<int64_t>(1, n_ctx / 2);
        const int64_t n_tok  = (int64_t) tokens.size();
        const int64_t ndata  = (n_tok > n_ctx + 1) ? (n_tok - n_ctx - 1) / stride : 0;

        if (ndata <= 0) {
            LOG_ERR(
                "%s: training prompt is too short for the selected --ctx-size (n_ctx=%" PRId64 ")\n"
                "%s: tokens=%" PRId64 ", stride=%" PRId64 ", ndata=%" PRId64 "\n"
                "%s: fix by using a longer training prompt/file or reducing --ctx-size\n",
                __func__, n_ctx, __func__, n_tok, stride, ndata, __func__);
            llama_backend_free();
            return 1;
        }

        dataset = common_opt_dataset_init(ctx, tokens, stride);
    }

    struct lr_opt & lr = params.lr;
    LOG_INF("-optimizer %s -lr0 %.2g -wd %.2g -lr-min %.2g -min-epochs %.2g -epochs %d -period %.2g -val %.2g\n",
            ggml_opt_optimizer_name(params.optimizer), (double) lr.lr0, (double) lr.wd, (double) lr.lr_min, (double) lr.decay_epochs,
            (unsigned) lr.epochs, (double) params.n_batch / params.n_ubatch, (double) params.val_split);

    // Use flat LR so cosine warmup+decay schedule in ggml_opt_eval is the sole controller.
    // common_opt_lr_pars applies epoch-based exponential decay on top of cosine, causing double decay.
    auto finetune_opt_pars = [](void * userdata) -> ggml_opt_optimizer_params {
        ggml_opt_optimizer_params result = ggml_opt_get_default_optimizer_params(nullptr);
        const lr_opt & d = *(const lr_opt *) userdata;
        result.adamw.alpha = d.lr0;
        result.sgd.alpha   = d.lr0;
        result.sgd.wd = result.adamw.wd = d.wd;
        return result;
    };

    struct llama_opt_params lopt_params{
        /*n_ctx_train     =*/0,
        /*param_filter    =*/params.lora_adapters.empty() ? llama_opt_param_filter_all : (params.lora_train_base.empty() ? opt_param_filter_lora_only : opt_param_filter_lora_selected),
        /*param_filter_ud =*/params.lora_adapters.empty() ? nullptr : (params.lora_train_base.empty() ? nullptr : (void *) &params.lora_train_base),
        /*get_opt_pars    =*/finetune_opt_pars,
        /*get_opt_pars_ud =*/&params.lr,
        /*optimizer_type  =*/params.optimizer,
    };
    if (!params.lora_adapters.empty()) {
        LOG_INF("%s: LoRA adapters provided -> training LoRA-only (base model F32 weights will not be updated)\n", __func__);

        // If requested, allocate LoRA optimizer state on host (pinned when available)
        if (params.lora_opt_offload) {
            auto & loras = llama_init->lora();
            for (auto & la_ptr : loras) {
                if (!la_ptr) continue;
                llama_adapter_lora * adapter = la_ptr.get();
                llama_adapter_lora_set_opt_state_on_host(adapter, true);
                if (!llama_adapter_lora_alloc_opt_state(adapter, params.lora_opt_chunk_bytes)) {
                    LOG_ERR("%s: failed to allocate LoRA opt state on host (adapter)", __func__);
                    llama_backend_free();
                    return 1;
                }
            }
            LOG_INF("%s: LoRA optimizer state allocated on host (chunk_bytes=%zu)\n", __func__, params.lora_opt_chunk_bytes);
        }
    }
    llama_opt_init(ctx, model, lopt_params);

    // Offload AdamW first/second moment tensors to host RAM to reduce VRAM usage.
    if (params.lora_opt_offload) {
        ggml_opt_context_t opt_ctx = llama_opt_context_get(ctx);
        if (opt_ctx) {
            ggml_opt_set_momenta_on_host(opt_ctx, true);
            LOG_INF("%s: AdamW m/v tensors will be allocated on host RAM\n", __func__);
        }
    }

    // Configure gradient clipping.
    {
        ggml_opt_context_t opt_ctx = llama_opt_context_get(ctx);
        if (!opt_ctx) {
            fprintf(stderr, "%s: WARNING: opt_ctx is NULL, gradient clipping could not be set\n", __func__);
        } else {
            // Global gradient L2 norm clipping (preserves gradient direction).
            if (params.grad_norm_clip > 0.0f) {
                ggml_opt_set_grad_norm_clip(opt_ctx, params.grad_norm_clip);
                if (params.optimizer == GGML_OPT_OPTIMIZER_TYPE_ADAMW) {
                    fprintf(stderr, "%s: WARNING: --grad-norm-clip is skipped for AdamW (m/v ratio cancels global scaling).\n"
                                    "%s:          Use --grad-clip for per-element clipping instead.\n", __func__, __func__);
                } else {
                    fprintf(stderr, "%s: global gradient norm clipping enabled (max_norm=%.4g)\n", __func__, (double) params.grad_norm_clip);
                }
            }
            // Per-element gradient clipping (legacy, can destroy gradient direction).
            if (params.grad_clip > 0.0f) {
                ggml_opt_set_grad_clip(opt_ctx, params.grad_clip);
                fprintf(stderr, "%s: per-element gradient clipping enabled (max_grad=%.4g)\n", __func__, (double) params.grad_clip);
            }
        }
    }

    // Apply loss scaling to prevent FP32 overflow in backward pass.
    // This scales the initial loss gradient by 1/S, keeping all intermediate backward
    // values within safe FP32 range, then compensates by multiplying S back in AdamW.
    {
        ggml_opt_context_t opt_ctx = llama_opt_context_get(ctx);
        if (opt_ctx && params.loss_scale != 1.0f) {
            ggml_opt_set_loss_scale(opt_ctx, params.loss_scale);
            fprintf(stderr, "%s: loss scaling enabled (scale=%.1f, initial_grad=%.2e)\n",
                    __func__, (double) params.loss_scale, 1.0 / (double) params.loss_scale);
        }
    }

    // Configure learning rate schedule: linear warmup + cosine decay.
    // This is critical for AdamW stability — without warmup, large initial updates
    // can destabilize LoRA weights, causing loss reversal and NaN.
    {
        ggml_opt_context_t opt_ctx = llama_opt_context_get(ctx);
        if (opt_ctx) {
            const int64_t ndata_train  = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);
            const int32_t opt_period   = std::max(1, (int32_t)(params.n_batch / params.n_ubatch));
            // Each data entry produces n_ctx/n_ubatch ubatches; optimizer steps every opt_period ubatches.
            const int64_t n_ctx_eff    = params.n_ctx > 0 ? params.n_ctx : llama_n_ctx(ctx);
            const int64_t ub_per_data  = std::max((int64_t)1, n_ctx_eff / params.n_ubatch);
            const int64_t steps_per_ep = std::max((int64_t)1, ndata_train * ub_per_data / opt_period);
            const int64_t total_steps  = steps_per_ep * lr.epochs;
            const int64_t warmup_steps = std::max((int64_t)1, total_steps / 10); // 10% warmup

            ggml_opt_set_total_steps(opt_ctx, total_steps);
            ggml_opt_set_warmup_steps(opt_ctx, warmup_steps);
            ggml_opt_set_min_lr_ratio(opt_ctx, 0.1f);

            fprintf(stderr, "%s: lr schedule: warmup=%lld steps, cosine total=%lld steps, min_lr_ratio=0.1\n",
                    __func__, (long long)warmup_steps, (long long)total_steps);
        }
    }

    // Diagnostics: track whether LoRA A/B tensors get unexpectedly zeroed.
    log_lora_stats_if_requested(params, llama_init->lora(), "output.weight");

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);

    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    bool nan_abort = false;

    for (lr.epoch = 0; lr.epoch < lr.epochs; ++lr.epoch) {
        // Shuffle training data each epoch to prevent ordering bias
        {
            ggml_opt_context_t opt_ctx = llama_opt_context_get(ctx);
            if (opt_ctx && idata_split > 1) {
                ggml_opt_dataset_shuffle(opt_ctx, dataset, idata_split);
            }
        }
        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split,
                        ggml_opt_epoch_callback_progress_bar_throttled, ggml_opt_epoch_callback_progress_bar_throttled);
        fprintf(stderr, "\n");

        // === NaN/Inf early-stop check ===
        {
            double train_loss = 0.0;
            ggml_opt_result_loss(result_train, &train_loss, nullptr);
            const bool loss_nan   = !std::isfinite(train_loss);
            const bool result_nan = ggml_opt_result_nan_detected(result_train);
            const int  nan_grads  = ggml_opt_result_nan_grad_count(result_train);

            if (loss_nan || result_nan) {
                fprintf(stderr,
                    "\n"
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    "!!! NaN/Inf DETECTED — stopping training to prevent     !!!\n"
                    "!!! corrupted LoRA weights.                             !!!\n"
                    "!!!                                                     !!!\n"
                    "!!! epoch=%u  loss=%.6g  nan_grads=%d                   !!!\n"
                    "!!!                                                     !!!\n"
                    "!!! Suggestions:                                        !!!\n"
                    "!!!   - Lower learning rate (--lr0)                     !!!\n"
                    "!!!   - Enable gradient norm clipping (--grad-norm-clip)!!!\n"
                    "!!!   - Enable loss scaling (--loss-scale)              !!!\n"
                    "!!!   - Use a less aggressively quantized model         !!!\n"
                    "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                    "\n",
                    (unsigned)(lr.epoch + 1),
                    train_loss,
                    nan_grads);
                fflush(stderr);
                nan_abort = true;
                break;
            }
        }

        log_lora_stats_if_requested(params, llama_init->lora(), "output.weight");

        // NOTE: Host-side AdamW at epoch boundary is DISABLED to prevent double/triple
        // weight updates. The GPU-side optimizer in ggml_opt_eval() already updates LoRA
        // weights each step via the gb_opt graph. See llama-context.cpp for details.

        // Checkpoint: write current LoRA adapter(s) each epoch so we can safely resume.
        // Automatic resume: rerun with the same -o <out.gguf> and --lora-create.
        // Manual resume: pass --lora <..._WORK.gguf>.
        if (!params.lora_adapters.empty() && !work_file.empty()) {
            auto & loras = llama_init->lora();
            if (!loras.empty()) {
                const size_t n = loras.size();
                for (size_t i = 0; i < n; ++i) {
                    const std::string out_i = make_work_path_indexed(work_file, i, n);
                    const std::string tmp_i = out_i + ".tmp";
                    const bool ok = llama_adapter_lora_save(loras[i].get(), tmp_i.c_str()) && save_file_atomic(tmp_i, out_i);
                    if (!ok) {
                        fprintf(stderr, "%s: WARNING: failed to write checkpoint '%s'\n", __func__, out_i.c_str());
                    } else {
                        fprintf(stderr, "%s: wrote checkpoint (epoch %u) to '%s'\n", __func__, (unsigned) (lr.epoch + 1), out_i.c_str());
                    }
                }
            }
        }

        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }
    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    ggml_opt_dataset_free(dataset);

    if (nan_abort) {
        LOG_ERR("%s: training aborted due to NaN/Inf. Checkpoint (if any) was NOT updated.\n", __func__);
        llama_backend_free();
        return 2;
    }

    // Free LoRA optimizer state if allocated on host
    if (params.lora_opt_offload) {
        auto & loras = llama_init->lora();
        for (auto & la_ptr : loras) {
            if (!la_ptr) continue;
            llama_adapter_lora_free_opt_state(la_ptr.get());
        }
        LOG_INF("%s: freed host-side LoRA optimizer state\n", __func__);
    }

    // Writing a merged full model can be extremely large.
    // For LoRA-only training, save the trained LoRA adapter(s) as GGUF if -o is provided.
    if (!params.lora_adapters.empty()) {
        if (params.out_file.empty()) {
            LOG_INF("%s: LoRA-only training finished. Use -o <path.gguf> to save the trained LoRA adapter.\n", __func__);
        } else {
            auto & loras = llama_init->lora();
            if (loras.empty()) {
                LOG_ERR("%s: LoRA-only training requested, but no adapters are loaded in memory.\n", __func__);
                llama_backend_free();
                return 1;
            }

            auto make_out_name = [](const std::string & base, size_t i) {
                const std::string suf = "-" + std::to_string(i) + ".gguf";
                if (base.size() >= 5 && base.compare(base.size() - 5, 5, ".gguf") == 0) {
                    return base.substr(0, base.size() - 5) + suf;
                }
                return base + "-" + std::to_string(i) + ".gguf";
            };

            if (loras.size() == 1) {
                if (!params.lora_train_base.empty()) {
                    lora_prune_adapter_to_selected(loras[0].get(), params.lora_train_base);
                }
                const bool ok = llama_adapter_lora_save(loras[0].get(), params.out_file.c_str());
                if (!ok) {
                    LOG_ERR("%s: failed to save LoRA adapter to '%s'\n", __func__, params.out_file.c_str());
                    llama_backend_free();
                    return 1;
                }
                LOG_INF("%s: wrote trained LoRA adapter to '%s'\n", __func__, params.out_file.c_str());
            } else {
                for (size_t i = 0; i < loras.size(); ++i) {
                    if (!params.lora_train_base.empty()) {
                        lora_prune_adapter_to_selected(loras[i].get(), params.lora_train_base);
                    }
                    const std::string out_i = make_out_name(params.out_file, i);
                    const bool ok = llama_adapter_lora_save(loras[i].get(), out_i.c_str());
                    if (!ok) {
                        LOG_ERR("%s: failed to save LoRA adapter[%zu] to '%s'\n", __func__, i, out_i.c_str());
                        llama_backend_free();
                        return 1;
                    }
                    LOG_INF("%s: wrote trained LoRA adapter[%zu] to '%s'\n", __func__, i, out_i.c_str());
                }
            }
        }
    } else {
        llama_model_save_to_file(model, params.out_file.c_str());
    }

    llama_backend_free();

    return 0;
}
