// Perplexity of a GGUF on OUR windows, through llama.cpp's public API.
//
// Why not llama-perplexity: it tokenizes the text itself and scores only the
// second half of each window (`first = n_ctx/2`, tools/perplexity/perplexity.cpp),
// so its number is not ours. `bin/ppl` scores every next-token position of
// non-overlapping windows over the raw WikiText-2 test split: positions 0..ctx-2
// predict tokens 1..ctx-1, ctx-1 predictions a window, no BOS.
//
// This file does exactly that on the ids `ops/duel/inputs.py` wrote
// (ppl-tokens.u32, fingerprint 3f1baca9033bf251), so the two numbers describe
// the same tokens and the same predictions. It prints the per-window mean NLL
// at nine digits, as `bin/ppl` does, so two arms pair window by window.
//
// Build (after llama.cpp, same commit):
//   g++ -O2 -std=c++17 ppl_windows.cpp -I<llama.cpp>/include -I<llama.cpp>/ggml/include \
//       -L<build>/bin -lllama -lggml -lggml-base -Wl,-rpath,<build>/bin -o llama-ppl-windows
// Usage:
//   llama-ppl-windows <model.gguf> <tokens.u32> <ctx> <windows>

#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

static uint64_t fnv1a(const std::vector<llama_token> & ids, size_t n) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t v = (uint32_t) ids[i];
        for (int b = 0; b < 4; ++b) {
            h ^= (v >> (8 * b)) & 0xff;
            h *= 0x00000100000001b3ULL;
        }
    }
    return h;
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <model.gguf> <tokens.u32> <ctx> <windows>\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const int ctx = atoi(argv[3]);
    const int nwin = atoi(argv[4]);

    std::ifstream f(argv[2], std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    const size_t bytes = (size_t) f.tellg();
    f.seekg(0);
    std::vector<uint32_t> raw(bytes / 4);
    f.read((char *) raw.data(), (std::streamsize) bytes);
    std::vector<llama_token> ids(raw.begin(), raw.end());
    if (ids.size() < (size_t) ctx * nwin) {
        fprintf(stderr, "%zu tokens, %d x %d needed\n", ids.size(), nwin, ctx);
        return 1;
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "cannot load %s\n", model_path); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = ctx;
    cp.n_batch = ctx;
    cp.n_seq_max = 1;
    llama_context * lctx = llama_init_from_model(model, cp);
    if (!lctx) { fprintf(stderr, "cannot create a context\n"); return 1; }

    fprintf(stderr, "model %s: %llu params, %llu bytes, vocab %d, add_bos %d (no BOS is fed)\n", model_path,
            (unsigned long long) llama_model_n_params(model), (unsigned long long) llama_model_size(model), n_vocab,
            (int) llama_vocab_get_add_bos(vocab));
    fprintf(stderr, "tokens %zu, scoring %d windows of %d, fingerprint %016llx\n", ids.size(), nwin, ctx,
            (unsigned long long) fnv1a(ids, (size_t) ctx * nwin));

    llama_batch batch = llama_batch_init(ctx, 0, 1);
    double nll = 0.0;
    long count = 0;
    for (int w = 0; w < nwin; ++w) {
        llama_memory_clear(llama_get_memory(lctx), true);
        batch.n_tokens = ctx;
        for (int k = 0; k < ctx; ++k) {
            batch.token[k] = ids[(size_t) w * ctx + k];
            batch.pos[k] = k;
            batch.n_seq_id[k] = 1;
            batch.seq_id[k][0] = 0;
            batch.logits[k] = k < ctx - 1;
        }
        if (llama_decode(lctx, batch) != 0) { fprintf(stderr, "decode failed on window %d\n", w + 1); return 1; }
        double wn = 0.0;
        for (int k = 0; k < ctx - 1; ++k) {
            const float * lg = llama_get_logits_ith(lctx, k);
            double mx = lg[0];
            for (int v = 1; v < n_vocab; ++v) mx = lg[v] > mx ? lg[v] : mx;
            double se = 0.0;
            for (int v = 0; v < n_vocab; ++v) se += std::exp((double) lg[v] - mx);
            const llama_token next = ids[(size_t) w * ctx + k + 1];
            wn += (mx + std::log(se)) - (double) lg[next];
        }
        nll += wn;
        count += ctx - 1;
        fprintf(stderr, "  window %3d/%d  nll %.9f  n %d  running ppl %8.3f\n", w + 1, nwin, wn / (ctx - 1),
                ctx - 1, std::exp(nll / count));
    }
    printf("%s, ctx %d, %d windows, %ld predictions\nppl = %.4f\n", model_path, ctx, nwin, count,
           std::exp(nll / count));
    llama_batch_free(batch);
    llama_free(lctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
