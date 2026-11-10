// Command 'Y' of host_pers.cpp (persistent side, PERS_4E_ROWS 7): the tree
// verify pass (pers_4et.cu) and its reference, 4E's token, on 'G's model.
//
// The model is host_pers_4e.inc's, drawn the same way from the seed: two
// layers at the 4B's widths (hidden 2,560, inter 9,728, 32 query heads, 8 KV
// heads of 128), layer 0 with a Tetra down, layer 1 an int4 down, a tied int4
// g64 table of `vocab` rows; variant 16 adds the 8B's Tetra o on layer 1 and
// an untied head. The caches hold random values at every position. One
// difference: the SiLU table is SiLU, not random draws.
//
// in : u32 form, grid, threads, cap, vocab, u64 seed, u32 variant (16: the
//      8B's parts; 64: every tree pass stamps its phases, and the command
//      fails if a CTA left an event unwritten; 32: the tree passes run the
//      tiled symbols on tree-tiled copies of every projection, built by
//      p4t_tile_word from the words 4E reads, as the card's install does;
//      128: the stream symbols of pers_4et_stream.cu on those same copies;
//      256: the a4 symbols of pers_4et_a4.cu on those same copies;
//      512: the a5 symbols of pers_4et_a5.cu on those same copies;
//      1024, 2048, 4096: a6, a7, a3f of pers_4et_a6.cu on those same copies;
//      8192, 16384, 32768: a3p, a3t, a8 of pers_4et_a8.cu on those same copies;
//      65536: a9 of pers_4et_a9.cu on those same copies, n_bar_a9 waits;
//      1048576: a11i (a11 with fine stamps, its 40 slots a CTA after the phase
//      events when stamped); 131072: a10 of pers_4et_a10.cu, a9's waits; 524288: a11 (a10 with the
//      L2 prefetch in its wait gaps), a9's waits),
//      three rotations (attention input at 2,560, MLP input at 2,560, down
//      input at 9,728), each u32 m, k, f32 inv, u32 signs[n/32], f32
//      small[1024], then with variant 16 the o rotation at 4,096,
//      u32 split[5]: the d_in slices of q/k/v, o, gate/up, down, head,
//      u32 ref_pos, ref_tok: one 4E token at ref_pos from ref_tok before the
//      passes (ref_pos 0xFFFFFFFF: none), its K and V appended to the cache,
//      u32 n_pass, then each pass: u32 pos0, R, u64 seed (the scheduler's),
//      u32 toks[R], u32 depth[R], u8 path[R][32], u32 n_move, move_pos0,
//      u32 mv[n_move]. The tree buffer persists from pass to pass.
// out: with a reference, u32 its token, u16 its logits[vocab] (4E's head rows
//      over its final residual, pers_q4g64_row, whose argmax must be 4E's
//      token or the command fails);
//      each pass: u32 ring[R], u16 logits[R][vocab], u16 x[R][hidden],
//      u16 tk[layers][R][n_kv x 128], u16 tv[layers][R][n_kv x 128];
//      then u32 bar, u32 the abort and NaN flags plus every ticket left.
{
    const u32 form = rd1<u32>(), grid = rd1<u32>(), threads = rd1<u32>(), cap = rd1<u32>(), vocab = rd1<u32>();
    const uint64_t seed = rd1<uint64_t>();
    const u32 variant = rd1<u32>();
    const bool ext = (variant & 16u) != 0u;
    const u32 H = 2560, I = 9728, NH = 32, NKV = 8, GRP = 4, NL = 2;
    const u32 dq = NH * 128, dkv = NKV * 128;
    struct Rot { u32 m, k; float inv; std::vector<u32> signs; std::vector<float> small; };
    auto read_rot = [](u32 width) {
        Rot r;
        r.m = rd1<u32>(); r.k = rd1<u32>(); r.inv = rd1<float>();
        r.signs = rd<u32>(width / 32); r.small = rd<float>(1024);
        return r;
    };
    const Rot ra = read_rot(H), rm = read_rot(H), rd_ = read_rot(I);
    Rot ro{};
    if (ext) ro = read_rot(dq);
    const auto split = rd<u32>(5);
    const u32 ref_pos = rd1<u32>(), ref_tok = rd1<u32>();
    if (form > 2u || threads % 32u != 0u || threads > 512u || grid == 0u || vocab % 16u != 0u) {
        std::fprintf(stderr, "host_pers: tree verify, bad form, grid, threads or vocab\n");
        std::exit(2);
    }
    const float eps = 1e-6f;
    // ---- the model: host_pers_4e.inc's draws, in its order ----
    Rng r(seed);
    const u32 egpr = H / 64u;
    std::vector<u32> ew((size_t)vocab * H / 8);
    for (auto& w : ew) w = (u32)r.next();
    std::vector<unsigned short> es((size_t)vocab * egpr), eb((size_t)vocab * egpr);
    for (auto& v : es) v = r.h16(5, 8);
    for (auto& v : eb) v = r.h16(6, 9);
    std::vector<u32> hw;
    std::vector<unsigned short> hs, hb;
    if (ext) {
        Rng rh(seed + 15);
        hw.resize(ew.size()); for (auto& w : hw) w = (u32)rh.next();
        hs.resize(es.size()); hb.resize(eb.size());
        for (auto& v : hs) v = rh.h16(5, 8);
        for (auto& v : hb) v = rh.h16(6, 9);
    }
    const std::vector<u32>& head_w = ext ? hw : ew;
    const std::vector<unsigned short>& head_s = ext ? hs : es;
    const std::vector<unsigned short>& head_b = ext ? hb : eb;
    const auto final_norm = act_h(H, seed + 11, 13, 16);
    const auto cosr = act_h(cap * 64, seed + 12, 4, 14), sinr = act_h(cap * 64, seed + 13, 4, 14);
    // SiLU itself, x / (1 + e^-x) of every binary16 pattern, where 'G' draws
    // a random table: a random table jumps at every ulp of the gate's output,
    // which the closeness gate would read as the arithmetic's error.
    std::vector<unsigned short> silu(65536);
    for (u32 b = 0; b < 65536u; ++b) {
        const double v = (double)h2f((unsigned short)b);
        const double s = (std::isinf(v) && v < 0.0) ? -0.0 : v / (1.0 + std::exp(-v));
        silu[b] = f2h((float)s);
    }
    struct Layer {
        std::vector<unsigned short> ln1, ln2, qn, kn, vs, vb, os, ob, dqs, dqb, kc, vc;
        TMat q, k, gate, up, down, ot;
        std::vector<u32> vw, ow, dqw;
        bool tetra_down, tetra_o;
    };
    std::vector<Layer> Ls(NL);
    for (u32 l = 0; l < NL; ++l) {
        Layer& Y = Ls[l];
        const uint64_t s0 = seed + 100 * (l + 1);
        Y.ln1 = act_h(H, s0 + 1, 13, 16); Y.ln2 = act_h(H, s0 + 2, 13, 16);
        Y.qn = act_h(128, s0 + 3, 13, 16); Y.kn = act_h(128, s0 + 4, 13, 16);
        Y.q = tmat(dq, H, s0 + 5); Y.k = tmat(dkv, H, s0 + 6);
        Y.gate = tmat(I, H, s0 + 7); Y.up = tmat(I, H, s0 + 8); Y.down = tmat(H, I, s0 + 9);
        Rng rv(s0 + 10);
        Y.vw.resize((size_t)dkv * H / 8); for (auto& w : Y.vw) w = (u32)rv.next();
        Y.vs.resize(dkv * 20); Y.vb.resize(dkv * 20);
        for (auto& e : Y.vs) e = rv.h16(6, 10);
        for (auto& e : Y.vb) e = rv.h16(8, 12);
        Y.ow.resize((size_t)H * dq / 8); for (auto& w : Y.ow) w = (u32)rv.next();
        Y.os.resize(H * 32); Y.ob.resize(H * 32);
        for (auto& e : Y.os) e = rv.h16(5, 9);
        for (auto& e : Y.ob) e = rv.h16(7, 11);
        Y.dqw.resize((size_t)H * I / 8); for (auto& w : Y.dqw) w = (u32)rv.next();
        Y.dqs.resize(H * (I / 128)); Y.dqb.resize(H * (I / 128));
        for (auto& e : Y.dqs) e = rv.h16(6, 10);
        for (auto& e : Y.dqb) e = rv.h16(8, 12);
        Y.tetra_down = l == 0;
        Y.tetra_o = ext && l == 1;
        if (Y.tetra_o) Y.ot = tmat(H, dq, s0 + 13);
        Y.kc = act_h(NKV * cap * 128, s0 + 11, 9, 15);
        Y.vc = act_h(NKV * cap * 128, s0 + 12, 9, 15);
    }
    // ---- 4E's arguments, as host_pers_4e.inc builds them for form 7 ----
    const F1rTables tb = tables();
    std::vector<unsigned short> x(H, 0u), qs(dq), ks(dkv), vs(dkv), x_b(H), q_out(dq), ctx(dq), act(I);
    std::vector<float> scratch((size_t)NKV * ((cap + 255) / 256) * GRP * 130);
    std::vector<u32> bar(1, 0u), ab(1, 0u), nan(1, 0u), ticket(NKV, 0u);
    std::vector<unsigned long long> slots(2, 0ull);
    const auto gu = [&] { std::vector<u32> v; for (u32 g = 0; g < grid; ++g) { v.push_back((u32)((uint64_t)I * g / grid)); v.push_back((u32)((uint64_t)I * (g + 1) / grid)); } return v; }();
    const auto dn = [&] { std::vector<u32> v; for (u32 g = 0; g < grid; ++g) { v.push_back((u32)((uint64_t)H * g / grid)); v.push_back((u32)((uint64_t)H * (g + 1) / grid)); } return v; }();
    auto segt = [](const TMat& m) { return SegTetra{m.words.data(), m.rscale.data(), m.tail.data(), m.gscale.data(), m.stride, m.nblocks, m.tail_w, m.d_out}; };
    std::vector<Pers4eLayer> PL(NL);
    std::vector<Pers4eOT> OT(NL);
    std::vector<std::vector<u32>> tiled;
    auto tile = [&](const std::vector<u32>& w, u32 d_out, u32 d_in) -> const u32* {
        const u32 nw = d_in / 8;
        std::vector<u32> t(w.size());
        for (u32 row = 0; row < d_out; ++row)
            for (u32 wi = 0; wi < nw; ++wi) t[(size_t)row * nw + pers_q4_tile_index(wi, nw)] = w[(size_t)row * nw + wi];
        tiled.push_back(std::move(t));
        return tiled.back().data();
    };
    tiled.reserve(3 * NL);
    for (u32 l = 0; l < NL; ++l) {
        Layer& Y = Ls[l];
        Pers4eLayer& P = PL[l];
        P = Pers4eLayer{};
        P.ln1 = Y.ln1.data(); P.signs_a = ra.signs.data(); P.small_a = ra.small.data(); P.m_a = ra.m; P.k_a = ra.k; P.inv_a = ra.inv;
        P.qkv = Pers4eQkv{Y.q.words.data(), Y.q.rscale.data(), Y.q.tail.data(), Y.q.gscale.data(), Y.q.stride, dq,
                          Y.k.words.data(), Y.k.rscale.data(), Y.k.tail.data(), Y.k.gscale.data(), Y.k.stride, dkv,
                          tile(Y.vw, dkv, H), Y.vs.data(), Y.vb.data(), dkv, 20u, Y.q.nblocks, Y.q.tail_w, H,
                          qs.data(), ks.data(), vs.data()};
        P.q_norm = Y.qn.data(); P.k_norm = Y.kn.data(); P.kc = Y.kc.data(); P.vc = Y.vc.data();
        P.ow = Y.tetra_o ? nullptr : tile(Y.ow, H, dq); P.osc = Y.tetra_o ? nullptr : Y.os.data();
        P.obi = Y.tetra_o ? nullptr : Y.ob.data(); P.o_gpr = dq / 128; P.tetra_down = Y.tetra_down ? 1u : 0u;
        SegArgs& M = P.mlp;
        M.alpha = Y.ln2.data(); M.signs_h = rm.signs.data(); M.small_h = rm.small.data(); M.silu = silu.data(); M.act = act.data();
        M.gate = segt(Y.gate); M.up = segt(Y.up); M.down = segt(Y.down);
        M.signs_i = rd_.signs.data(); M.small_i = rd_.small.data();
        M.q4w = Y.tetra_down ? Y.dqw.data() : tile(Y.dqw, H, I); M.q4s = Y.dqs.data(); M.q4b = Y.dqb.data();
        M.rows = tb.rows; M.pref = tb.prefixes; M.br = tb.branches; M.suf = tb.suffixes; M.invnorm = g_inv.data();
        M.slice_gu = gu.data(); M.slice_dn = dn.data(); M.bar = bar.data(); M.abort_flag = ab.data();
        M.hidden = H; M.inter = I; M.m_h = rm.m; M.k_h = rm.k; M.m_i = rd_.m; M.k_i = rd_.k;
        M.inv_h = rm.inv; M.inv_i = rd_.inv; M.eps = eps; M.gpr = I / 128;
        if (Y.tetra_o) {
            OT[l] = Pers4eOT{segt(Y.ot), ro.signs.data(), ro.small.data(), ro.m, ro.k, ro.inv, 0u};
            P.ot = &OT[l];
        }
    }
    Pers4eArgs A{};
    A.layers = PL.data(); A.emb_w = ew.data(); A.emb_s = es.data(); A.emb_b = eb.data(); A.final_norm = final_norm.data();
    A.cosr = cosr.data(); A.sinr = sinr.data(); A.x_a = x.data(); A.x_b = x_b.data(); A.q_out = q_out.data(); A.ctx = ctx.data();
    A.attn_scratch = scratch.data(); A.bar = bar.data(); A.abort_flag = ab.data(); A.slots = slots.data(); A.nan_flag = nan.data();
    A.rows = tb.rows; A.pref = tb.prefixes; A.br = tb.branches; A.suf = tb.suffixes; A.invnorm = g_inv.data();
    A.attn_ticket = ticket.data();
    A.n_layers = NL; A.vocab = vocab; A.hidden = H; A.emb_gpr = egpr; A.n_heads = NH; A.n_kv = NKV; A.groups = GRP; A.cap = cap;
    A.eps = eps; A.c = (float)((1.0 / std::sqrt(128.0)) * 1.4426950408889634);
    A.head_w = head_w.data(); A.head_s = head_s.data(); A.head_b = head_b.data();
    // ---- variant 32: the tree-tiled copies and the arguments that point at them ----
    const u32 a6k = (variant & 1024u) ? 1u : (variant & 2048u) ? 2u : (variant & 4096u) ? 3u : (variant & 8192u) ? 4u : (variant & 16384u) ? 5u : (variant & 32768u) ? 6u : (variant & 65536u) ? 7u : (variant & 131072u) ? 8u : (variant & 524288u) ? 9u : (variant & 1048576u) ? 10u : (variant & 2097152u) ? 11u : (variant & 4194304u) ? 12u : (variant & 8388608u) ? 13u : (variant & 16777216u) ? 14u : (variant & 33554432u) ? 15u : (variant & 67108864u) ? 16u : (variant & 134217728u) ? 17u : (variant & 268435456u) ? 18u : 0u;
    const bool a12v = (variant & 268435456u) != 0u;
    const bool a9v = (variant & (65536u | 131072u | 524288u | 1048576u | 2097152u | 4194304u | 8388608u | 16777216u | 33554432u | 67108864u | 134217728u | 268435456u)) != 0u;
    const bool a11iv = (variant & 1048576u) != 0u;
    const bool a5v = (variant & 512u) != 0u || a6k != 0u;
    const bool a4v = (variant & 256u) != 0u || a5v;
    const bool streamv = (variant & 128u) != 0u;
    const bool tiledv = (variant & 32u) != 0u || streamv || a4v;
    std::vector<std::vector<u32>> tcopies;
    auto tcopy = [&](const u32* src, u32 kind, u32 d_out, u32 stride, u32 d_in, u32 served) -> const u32* {
        const u32 nsub = (d_in + P4T_SUB - 1u) / P4T_SUB;
        const unsigned long long n = (unsigned long long)(d_out / 16u) * nsub * (kind == 0u ? P4T_TSW : P4T_TSQ);
        std::vector<u32> t(n);
        for (unsigned long long i = 0; i < n; ++i) t[i] = p4t_tile_word(src, kind, stride, nsub, served, i);
        tcopies.push_back(std::move(t));
        return tcopies.back().data();
    };
    auto tseg = [&](SegTetra t) {
        t.words = tcopy(t.words, 0u, t.d_out, t.stride_u32, t.nblocks * 24u + t.tail_w, 0u);
        return t;
    };
    std::vector<Pers4eLayer> PLt = PL;
    std::vector<Pers4eOT> OTt = OT;
    Pers4eArgs At = A;
    if (tiledv) {
        for (u32 l = 0; l < NL; ++l) {
            Pers4eLayer& P = PLt[l];
            Pers4eQkv& N = P.qkv;
            N.qw = tcopy(N.qw, 0u, N.q_dout, N.q_stride, N.nblocks * 24u + N.tail_w, 0u);
            N.kw = tcopy(N.kw, 0u, N.k_dout, N.k_stride, N.nblocks * 24u + N.tail_w, 0u);
            N.vw = tcopy(N.vw, 1u, N.v_dout, N.d_in / 8u, N.d_in, 1u);
            if (P.ot != nullptr) {
                OTt[l].t = tseg(OTt[l].t);
                P.ot = &OTt[l];
            } else {
                P.ow = tcopy(P.ow, 1u, H, dq / 8u, dq, 1u);
            }
            P.mlp.gate = tseg(P.mlp.gate);
            P.mlp.up = tseg(P.mlp.up);
            if (P.tetra_down) P.mlp.down = tseg(P.mlp.down);
            else P.mlp.q4w = tcopy(P.mlp.q4w, 1u, H, I / 8u, I, 1u);
        }
        At.layers = PLt.data();
        At.head_w = tcopy(A.head_w, 1u, vocab, H / 8u, H, 0u);
    }
    u32 host_base = 0u;
    // ---- the reference: one 4E token, then its logits ----
    if (ref_pos != 0xFFFFFFFFu) {
        typedef void (*Fn)(const Pers4eArgs*, const Pers4eLaunch);
        const Fn tok4e[3] = {pers_4e_tokens_f0_t512, pers_4e_tokens_f1_t512, pers_4e_tokens_f2_t512};
        std::vector<u32> ring4e(1, 0xFFFFFFFFu);
        const Pers4eLaunch L4{host_base, ref_pos, 1u, ref_tok, 0xFFFFFFFFu, 0u, ring4e.data(), nullptr};
        const size_t smem = PERS_SMEM_HEADER + 18816 + std::max(4 * (size_t)std::max(I, 2 * H), (size_t)PERS_ATTN_SMEM(GRP)) +
                            PERS_4E_SCRATCH;
        hostgrid::launch_coop(grid, threads, smem, seed + 0x4e, [&] { tok4e[form](&A, L4); });
        if (ab[0] != 0u || nan[0] != 0u) {
            std::fprintf(stderr, "host_pers: the reference 4E token aborted\n");
            std::exit(3);
        }
        host_base = bar[0];
        // 4E's head over its final residual: its norm (exact_rms_scale,
        // exact_rms_apply), then pers_q4g64_row a row, rounded to binary16
        std::vector<unsigned short> lg(vocab, 0xFFFFu);
        hostgrid::launch_coop(1, threads, PERS_SMEM_HEADER + 4 * H, 0, [&] {
            float* v = reinterpret_cast<float*>(PERS_SMEM_BASE() + PERS_SMEM_HEADER);
            const float sc = exact_rms_scale<0>(x.data(), H, eps);
            for (u32 i = threadIdx.x; i < H; i += blockDim.x) v[i] = h2f(exact_rms_apply(sc, x[i], final_norm[i]));
            __syncthreads();
            const u32 lane = threadIdx.x & 31u;
            for (u32 row = threadIdx.x >> 5; row < vocab; row += blockDim.x >> 5) {
                const float acc = pers_q4g64_row<false>(head_w.data(), head_s.data(), head_b.data(), v, row, H, egpr, lane);
                if (lane == 0u) lg[row] = f2h(acc);
            }
        });
        unsigned long long best = 0ull;
        u32 arg = 0xFFFFFFFFu;
        for (u32 i = 0; i < vocab; ++i)
            if (!pers_is_nan16(lg[i]) && pers_tie_key(lg[i], i) > best) { best = pers_tie_key(lg[i], i); arg = i; }
        if (arg != ring4e[0]) {
            std::fprintf(stderr, "host_pers: 4E's logits rebuilt give %u, its token is %u\n", arg, ring4e[0]);
            std::exit(3);
        }
        wr(ring4e);
        wr(lg);
    }
    // ---- the tree verify's buffers, sized for P4T_RMAX rows ----
    const u32 RM = P4T_RMAX, ldx = std::max(std::max(H, I), dq);
    const u32 pstride = NKV * ((cap + 255u) / 256u) * GRP * 130u;
    // x_a's buffer carries a12's taps past its RM rows: [RM][5][H]
    std::vector<unsigned short> txa((size_t)RM * H * 6), txb((size_t)RM * H), txin((size_t)RM * ldx), txn((size_t)RM * H),
        tq((size_t)RM * dq), tkr((size_t)RM * dkv), tvr((size_t)RM * dkv), tqo((size_t)RM * dq), tctx((size_t)RM * dq),
        tg((size_t)RM * I), tu((size_t)RM * I), tlog((size_t)RM * vocab), tk((size_t)NL * RM * dkv, 0u),
        tv((size_t)NL * RM * dkv, 0u);
    std::vector<float> tpart((size_t)RM * pstride);
    const size_t fp_need = std::max({(size_t)split[0] * (dq + 2 * dkv), (size_t)split[1] * H, (size_t)split[2] * 2 * I,
                                     (size_t)split[3] * H, (size_t)split[4] * vocab});
    std::vector<float> tfpart(fp_need * RM);
    std::vector<u32> ttk((size_t)RM * NKV, 3u), tft((dq + 2 * dkv + 2 * I + H + vocab) / 16, 0u);
    std::vector<unsigned long long> tslots(RM, 9ull);
    const u32 n_ot = ext ? 1u : 0u;
    const u32 area = p4t_area_floats(threads / 32u, 8u * ((RM + 7u) / 8u), ldx, GRP);
    const size_t tsmem = PERS_SMEM_HEADER + 18816 + 4 * (size_t)area;
    typedef void (*Ft)(const Pers4eArgs*, const Pers4tLaunch);
    const Ft tfn[3] = {pers_4et_rows_f0_t512, pers_4et_rows_f1_t512, pers_4et_rows_f2_t512};
    const Ft tfnt[3] = {pers_4et_tiled_f0_t512, pers_4et_tiled_f1_t512, pers_4et_tiled_f2_t512};
    const Ft tfns[3] = {pers_4et_stream_f0_t512, pers_4et_stream_f1_t512, pers_4et_stream_f2_t512};
    const Ft tfn4[3] = {pers_4et_a4_f0_t512, pers_4et_a4_f1_t512, pers_4et_a4_f2_t512};
    const Ft tfn5[3] = {pers_4et_a5_f0_t512, pers_4et_a5_f1_t512, pers_4et_a5_f2_t512};
    const Ft tfn6[18][3] = {{pers_4et_a6_f0_t512, pers_4et_a6_f1_t512, pers_4et_a6_f2_t512},
                           {pers_4et_a7_f0_t512, pers_4et_a7_f1_t512, pers_4et_a7_f2_t512},
                           {pers_4et_a3f_f0_t512, pers_4et_a3f_f1_t512, pers_4et_a3f_f2_t512},
                           {pers_4et_a3p_f0_t512, pers_4et_a3p_f1_t512, pers_4et_a3p_f2_t512},
                           {pers_4et_a3t_f0_t512, pers_4et_a3t_f1_t512, pers_4et_a3t_f2_t512},
                           {pers_4et_a8_f0_t512, pers_4et_a8_f1_t512, pers_4et_a8_f2_t512},
                           {pers_4et_a9_f0_t512, pers_4et_a9_f1_t512, pers_4et_a9_f2_t512},
                           {pers_4et_a10_f0_t512, pers_4et_a10_f1_t512, pers_4et_a10_f2_t512},
                           {pers_4et_a11_f0_t512, pers_4et_a11_f1_t512, pers_4et_a11_f2_t512},
                           {pers_4et_a11i_f0_t512, pers_4et_a11i_f1_t512, pers_4et_a11i_f2_t512},
                           {pers_4et_a10r8_f0_t512, pers_4et_a10r8_f1_t512, pers_4et_a10r8_f2_t512},
                           {pers_4et_a10r16_f0_t512, pers_4et_a10r16_f1_t512, pers_4et_a10r16_f2_t512},
                           {pers_4et_a10_f0_t384, pers_4et_a10_f1_t384, pers_4et_a10_f2_t384},
                           {pers_4et_a10_f0_t256, pers_4et_a10_f1_t256, pers_4et_a10_f2_t256},
                           {pers_4et_a10_f0_t448, pers_4et_a10_f1_t448, pers_4et_a10_f2_t448},
                           {pers_4et_a10_f0_t320, pers_4et_a10_f1_t320, pers_4et_a10_f2_t320},
                           {pers_4et_a10r16_f0_t384, pers_4et_a10r16_f1_t384, pers_4et_a10r16_f2_t384},
                           {pers_4et_a12_f0_t384, pers_4et_a12_f1_t384, pers_4et_a12_f2_t384}};
    const u32 n_pass = rd1<u32>();
    for (u32 pi = 0; pi < n_pass; ++pi) {
        const u32 pos0 = rd1<u32>(), R = rd1<u32>();
        const uint64_t pseed = rd1<uint64_t>();
        const auto toks = rd<u32>(R);
        const auto depth = rd<u32>(R);
        const auto path = rd<unsigned char>((size_t)R * RM);
        const u32 n_move = rd1<u32>(), move_pos0 = rd1<u32>();
        const auto mv = rd<u32>(n_move);
        std::vector<u32> ring(R, 0xFFFFFFFFu);
        std::fill(tlog.begin(), tlog.end(), (unsigned short)0xFFFFu);
        Pers4tLaunch T{};
        T.bar_base = host_base; T.pos0 = pos0; T.n_rows = R; T.area_floats = area;
        T.n_move = n_move; T.move_pos0 = move_pos0; T.part_stride = pstride; T.ldx = ldx;
        T.split_qkv = split[0]; T.split_o = split[1]; T.split_gu = split[2]; T.split_dn = split[3]; T.split_head = split[4];
        T.toks = toks.data(); T.depth = depth.data(); T.path = path.data(); T.mv = mv.data();
        T.ring = ring.data(); T.logits = tlog.data();
        T.x_a = txa.data(); T.x_b = txb.data(); T.xin = txin.data(); T.xn = txn.data();
        // a12: taps at layers 0, 1, ... (first 0, stride 1); the twin's last slot is the final residual
        T.pad = a12v ? (1u << 16) : 0u;
        if (a12v) std::fill(txa.begin() + (size_t)RM * H, txa.end(), (unsigned short)0);
        T.q_raw = tq.data(); T.k_raw = tkr.data(); T.v_raw = tvr.data(); T.q_out = tqo.data(); T.ctx = tctx.data();
        T.g_raw = tg.data(); T.u_raw = tu.data(); T.tk = tk.data(); T.tv = tv.data();
        T.part = tpart.data(); T.fpart = tfpart.data(); T.tickets = ttk.data(); T.ftickets = tft.data(); T.slots = tslots.data();
        // a4 adds its F3 slots after the phase events: 20 a CTA
        const u32 nbar = a9v ? n_bar_a9(NL, n_ot) : n_bar_4et(NL, n_ot);
        const size_t n_ev = (size_t)(1u + 2u * nbar) * grid;
        std::vector<unsigned long long> st((variant & 64u) ? n_ev + (a4v ? 20u * (size_t)grid : 0u) + (a11iv ? 40u * (size_t)grid : 0u) : 0u, 0ull);
        if (variant & 64u) T.stamps = st.data();
        hostgrid::launch_coop(grid, threads, tsmem, pseed, [&] { if (a6k != 0u) tfn6[a6k - 1u][form](&At, T); else if (a5v) tfn5[form](&At, T); else if (a4v) tfn4[form](&At, T); else if (streamv) tfns[form](&At, T); else if (tiledv) tfnt[form](&At, T); else tfn[form](&A, T); });
        for (size_t i = 0; i < std::min(st.size(), n_ev); ++i)
            if (st[i] == 0ull) {
                std::fprintf(stderr, "host_pers: tree pass %u left a phase stamp unwritten\n", pi);
                std::exit(3);
            }
        if (ab[0] != 0u || nan[0] != 0u) {
            std::fprintf(stderr, "host_pers: tree pass %u aborted (abort %u, NaN %u)\n", pi, ab[0], nan[0]);
            std::exit(3);
        }
        if (bar[0] != host_base + nbar * grid) {
            std::fprintf(stderr, "host_pers: tree pass %u: barrier count %u, expected %u\n", pi, bar[0],
                         host_base + nbar * grid);
            std::exit(3);
        }
        host_base = bar[0];
        wr(ring);
        wr(std::vector<unsigned short>(tlog.begin(), tlog.begin() + (size_t)R * vocab));
        wr(std::vector<unsigned short>(txa.begin(), txa.begin() + (size_t)R * H));
        if (a12v) wr(std::vector<unsigned short>(txa.begin() + (size_t)RM * H, txa.begin() + (size_t)RM * H + (size_t)R * 5 * H));
        for (const auto* buf : {&tk, &tv})
            for (u32 l = 0; l < NL; ++l)
                wr(std::vector<unsigned short>(buf->begin() + (size_t)l * RM * dkv, buf->begin() + ((size_t)l * RM + R) * dkv));
    }
    std::vector<u32> tail(2, 0u);
    tail[0] = bar[0];
    tail[1] = ab[0] + nan[0];
    for (u32 t : ticket) tail[1] += t;
    for (u32 t : ttk) tail[1] += t;
    for (u32 t : tft) tail[1] += t;
    wr(tail);
}
