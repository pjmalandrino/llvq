// attn_dec_batch.cu: the split-KV decode attention for B sequences in one
// launch, one decode row each. Concatenated AFTER attn_dec.cu into a module of
// its own, so the served attention module keeps its text and its sha256.
//
// Grid (splits, n_kv, B). CTA (s, h, z) is attn_dec_body's CTA (s, h) for
// sequence z, on pointers offset to z's query row, fresh row, cache slot,
// partials and tickets, with k = 1 and pos0 = pos[z]. The body's arithmetic
// depends on the row's length alone (attn_dec.cu, "THE ARITHMETIC IS
// CANONICAL"), and a split past a row's length draws its ticket and does
// nothing, so each sequence gets the bits of a one-row call of the served
// kernel, whatever the other sequences' lengths and whatever gridDim.x.
//
// Layout: q and out [B][n_kv*G][HD]; k_new, v_new [B][n_kv][HD]; kc, vc
// [B][n_kv][cap][HD]; part [B][n_kv][s_stride][KMAX][G][PSTRIDE]; ticket
// [B][n_kv], zero at rest; pos [B], the position of each fresh row (int32).

#define ATTN_BATCH_ENTRY(NAME, G, UNIT)                                                          \
    extern "C" __global__ void ATTN_BOUNDS NAME(                                                 \
        const unsigned short* q, const unsigned short* k_new, const unsigned short* v_new,      \
        unsigned short* kc, unsigned short* vc, const int* pos, int cap, float c, float* part,  \
        unsigned* ticket, unsigned short* out, int s_stride) {                                   \
        const long long z = (long long)blockIdx.z;                                               \
        const long long nkv = (long long)gridDim.y;                                              \
        const long long qz = z * nkv * G * ATTN_HD;                                              \
        const long long kz = z * nkv * ATTN_HD;                                                  \
        const long long cz = z * nkv * (long long)cap * ATTN_HD;                                 \
        const long long pz = z * nkv * (long long)s_stride * ATTN_KMAX * G * ATTN_PSTRIDE;       \
        attn_dec_body<G, UNIT>(q + qz, k_new + kz, v_new + kz, kc + cz, vc + cz, pos[z], 1, cap, \
                               c, part + pz, ticket + z * nkv, out + qz, s_stride);              \
    }

ATTN_BATCH_ENTRY(attn_batch_g4_u64, 4, 64)
ATTN_BATCH_ENTRY(attn_batch_g5_u64, 5, 64)

// The same launch over rows that live in any slots of a larger pool: row z
// reads and writes cache slot `slot[z]`, everything else is indexed by z.
// Continuous batching steps the sequences that are live, wherever they sit;
// the slots of the others are not touched. Row z's bits are those of a
// one-row call on its slot, as above.
#define ATTN_SLOT_ENTRY(NAME, G, UNIT)                                                           \
    extern "C" __global__ void ATTN_BOUNDS NAME(                                                 \
        const unsigned short* q, const unsigned short* k_new, const unsigned short* v_new,      \
        unsigned short* kc, unsigned short* vc, const int* pos, const int* slot, int cap,       \
        float c, float* part, unsigned* ticket, unsigned short* out, int s_stride) {            \
        const long long z = (long long)blockIdx.z;                                               \
        const long long nkv = (long long)gridDim.y;                                              \
        const long long qz = z * nkv * G * ATTN_HD;                                              \
        const long long kz = z * nkv * ATTN_HD;                                                  \
        const long long cz = (long long)slot[z] * nkv * (long long)cap * ATTN_HD;                \
        const long long pz = z * nkv * (long long)s_stride * ATTN_KMAX * G * ATTN_PSTRIDE;       \
        attn_dec_body<G, UNIT>(q + qz, k_new + kz, v_new + kz, kc + cz, vc + cz, pos[z], 1, cap, \
                               c, part + pz, ticket + z * nkv, out + qz, s_stride);              \
    }

ATTN_SLOT_ENTRY(attn_slot_g4_u64, 4, 64)
ATTN_SLOT_ENTRY(attn_slot_g5_u64, 5, 64)
