// Command 'G' of host_pers.cpp: 4E's token loop against the served chain.
//
// A two-layer model at the 4B's widths (hidden 2,560, inter 9,728, 32 query
// heads, 8 KV heads of 128), layer 0 with a Tetra down, layer 1 an int4
// down, a tied int4 g64 table of `vocab` rows. The caches hold `pos0`
// random positions; the run produces up to `n` tokens from `tok0`.
//
// in : u32 form, grid, threads, n, pos0, cap, tok0, eos, vocab, u64 seed,
//      three rotations (attention input at 2,560, MLP input at 2,560, down
//      input at 9,728), each u32 m, k, f32 inv, u32 signs[n/32], f32 small[1024],
//      u32 n_more, then n_more launches (u32 pos0, n, tok0) run after the
//      first through the same state, as the runtime runs consecutive decodes
//      (a later launch may rewind to an earlier position), then u32 variant
//      of the persistent side (the served side ignores it): 0 the 768-thread
//      token symbols (rows without prefetch), 1 the 512-thread stamp symbols,
//      2 the 512-thread token symbols (rows with prefetch), 3 the 768-thread
//      stamp symbols
// out: u32 ring[n] then each later launch's ring, u16 x[hidden], u16 kc, vc of
//      each layer, u32 bar, abort + the tickets left; with a stamp variant (1
//      or 3) on the persistent side, then
//      u64 stamps[sum of n][3 + 15 x layers][grid] (the harness clock counts
//      calls: an order, not a duration)
{
    const u32 form = rd1<u32>(), grid = rd1<u32>(), threads = rd1<u32>(), n = rd1<u32>(), pos0 = rd1<u32>(),
              cap = rd1<u32>(), tok0 = rd1<u32>(), eos = rd1<u32>(), vocab = rd1<u32>();
    const uint64_t seed = rd1<uint64_t>();
    const u32 H = 2560, I = 9728, NH = 32, NKV = 8, GRP = 4, NL = 2;
    struct Rot { u32 m, k; float inv; std::vector<u32> signs; std::vector<float> small; };
    auto read_rot = [](u32 width) {
        Rot r;
        r.m = rd1<u32>(); r.k = rd1<u32>(); r.inv = rd1<float>();
        r.signs = rd<u32>(width / 32); r.small = rd<float>(1024);
        return r;
    };
    const Rot ra = read_rot(H), rm = read_rot(H), rd_ = read_rot(I);
    const u32 n_more = rd1<u32>();
    std::vector<u32> runs{pos0, n, tok0};
    { const auto m = rd<u32>(3 * (size_t)n_more); runs.insert(runs.end(), m.begin(), m.end()); }
    const u32 variant = rd1<u32>();
    const u32 stamp = (variant == 1u || variant == 3u) ? 1u : 0u;
    size_t ring_len = 0;
    for (size_t i = 0; i < runs.size(); i += 3) ring_len += runs[i + 1];
    const float eps = 1e-6f;
    Rng r(seed);
    // the tied table
    const u32 egpr = H / 64u;
    std::vector<u32> ew((size_t)vocab * H / 8);
    for (auto& w : ew) w = (u32)r.next();
    std::vector<unsigned short> es((size_t)vocab * egpr), eb((size_t)vocab * egpr);
    for (auto& v : es) v = r.h16(5, 8);
    for (auto& v : eb) v = r.h16(6, 9);
    const auto final_norm = act_h(H, seed + 11, 13, 16);
    const auto cosr = act_h(cap * 64, seed + 12, 4, 14), sinr = act_h(cap * 64, seed + 13, 4, 14);
    std::vector<unsigned short> silu(65536);
    { Rng rs(seed + 14); for (auto& e : silu) e = rs.h16(8, 15); }
    struct Layer {
        std::vector<unsigned short> ln1, ln2, qn, kn, vs, vb, os, ob, dqs, dqb, kc, vc;
        TMat q, k, gate, up, down;
        std::vector<u32> vw, ow, dqw;
        bool tetra_down;
    };
    std::vector<Layer> Ls(NL);
    for (u32 l = 0; l < NL; ++l) {
        Layer& Y = Ls[l];
        const uint64_t s0 = seed + 100 * (l + 1);
        Y.ln1 = act_h(H, s0 + 1, 13, 16); Y.ln2 = act_h(H, s0 + 2, 13, 16);
        Y.qn = act_h(128, s0 + 3, 13, 16); Y.kn = act_h(128, s0 + 4, 13, 16);
        Y.q = tmat(NH * 128, H, s0 + 5); Y.k = tmat(NKV * 128, H, s0 + 6);
        Y.gate = tmat(I, H, s0 + 7); Y.up = tmat(I, H, s0 + 8); Y.down = tmat(H, I, s0 + 9);
        Rng rv(s0 + 10);
        Y.vw.resize((size_t)NKV * 128 * H / 8); for (auto& w : Y.vw) w = (u32)rv.next();
        Y.vs.resize(NKV * 128 * 20); Y.vb.resize(NKV * 128 * 20);
        for (auto& e : Y.vs) e = rv.h16(6, 10);
        for (auto& e : Y.vb) e = rv.h16(8, 12);
        Y.ow.resize((size_t)H * NH * 128 / 8); for (auto& w : Y.ow) w = (u32)rv.next();
        Y.os.resize(H * 32); Y.ob.resize(H * 32);
        for (auto& e : Y.os) e = rv.h16(5, 9);
        for (auto& e : Y.ob) e = rv.h16(7, 11);
        Y.dqw.resize((size_t)H * I / 8); for (auto& w : Y.dqw) w = (u32)rv.next();
        Y.dqs.resize(H * (I / 128)); Y.dqb.resize(H * (I / 128));
        for (auto& e : Y.dqs) e = rv.h16(6, 10);
        for (auto& e : Y.dqb) e = rv.h16(8, 12);
        Y.tetra_down = l == 0;
        Y.kc = act_h(NKV * cap * 128, s0 + 11, 9, 15);
        Y.vc = act_h(NKV * cap * 128, s0 + 12, 9, 15);
    }
    // the score scale attn_ref::score_scale(128): (1/sqrt(128)) x log2(e), in f64 then f32
    const float scale = (float)((1.0 / std::sqrt(128.0)) * 1.4426950408889634);
    std::vector<u32> ring(ring_len, 0xFFFFFFFFu), tail(2, 0u);
    std::vector<unsigned short> x(H, 0u);
#if PERS_SIDE == 0
    (void)grid; (void)threads; (void)stamp; (void)variant;
    auto norm = [&](const std::vector<unsigned short>& v, const unsigned short* al, u32 w) {
        const float sc = candle_rms_scale(v.data(), w, eps, 0);
        std::vector<unsigned short> o(w);
        for (u32 i = 0; i < w; ++i) o[i] = f2h(__fmul_rn(__fmul_rn(sc, h2f(v[i])), h2f(al[i])));
        return o;
    };
    auto rot = [&](const std::vector<unsigned short>& in, const Rot& R, u32 w) {
        std::vector<float> o(w);
        hostgrid::launch_seq(1, 256, [&] { rot_apply(in.data(), R.signs.data(), R.small.data(), o.data(), w, R.m, R.k, R.inv, 0u); });
        return o;
    };
    auto q4 = [&](const std::vector<u32>& w, const std::vector<unsigned short>& sc, const std::vector<unsigned short>& bi,
                  const std::vector<unsigned short>& in, u32 dout, u32 din) {
        std::vector<unsigned short> o(dout);
        hostgrid::launch_seq(dout / 8, 256, [&] { tv_q4_hh(w.data(), sc.data(), bi.data(), in.data(), o.data(), din, din / 128); });
        return o;
    };
  size_t ring_at = 0;
  for (size_t run_i = 0; run_i < runs.size(); run_i += 3) {
    const u32 pos0 = runs[run_i], n = runs[run_i + 1], tok0 = runs[run_i + 2];
    u32* ring_run = ring.data() + ring_at;
    ring_at += n;
    u32 tok = tok0;
    for (u32 t = 0; t < n; ++t) {
        const u32 pos = pos0 + t;
        std::vector<u32> ids(1, tok);
        hostgrid::launch_seq(1, 256, [&] { emb_q4_gather(ew.data(), es.data(), eb.data(), ids.data(), x.data(), H, egpr, 0u); });
        for (u32 l = 0; l < NL; ++l) {
            Layer& Y = Ls[l];
            const auto h = norm(x, Y.ln1.data(), H);
            const auto xr = rot(h, ra, H);
            const auto qv = served_ref(Y.q, xr), kv = served_ref(Y.k, xr);
            const auto vv = q4(Y.vw, Y.vs, Y.vb, h, NKV * 128, H);
            std::vector<unsigned short> qo(NH * 128);
            auto head = [&](const unsigned short* src, const unsigned short* al, unsigned short* dst) {
                std::vector<unsigned short> sv(src, src + 128);
                const float sc = candle_rms_scale(sv.data(), 128, eps, 0);
                for (u32 i = 0; i < 64; ++i) {
                    const unsigned short a = f2h(__fmul_rn(__fmul_rn(sc, h2f(src[i])), h2f(al[i])));
                    const unsigned short b = f2h(__fmul_rn(__fmul_rn(sc, h2f(src[i + 64])), h2f(al[i + 64])));
                    ref_rope_pair((int)form, a, b, cosr[pos * 64 + i], sinr[pos * 64 + i], dst + i, dst + i + 64);
                }
            };
            for (u32 hh = 0; hh < NH; ++hh) head(qv.data() + hh * 128, Y.qn.data(), qo.data() + hh * 128);
            for (u32 hh = 0; hh < NKV; ++hh) {
                head(kv.data() + hh * 128, Y.kn.data(), Y.kc.data() + ((size_t)hh * cap + pos) * 128);
                for (u32 i = 0; i < 128; ++i) Y.vc[((size_t)hh * cap + pos) * 128 + i] = vv[hh * 128 + i];
            }
            std::vector<unsigned short> ctx(NH * 128);
            for (u32 hq = 0; hq < NH; ++hq)
                ref_attn_row(qo.data() + hq * 128, Y.kc.data() + (size_t)(hq / GRP) * cap * 128,
                             Y.vc.data() + (size_t)(hq / GRP) * cap * 128, pos + 1, scale, ctx.data() + hq * 128);
            const auto o = q4(Y.ow, Y.os, Y.ob, ctx, H, NH * 128);
            for (u32 i = 0; i < H; ++i) x[i] = f2h(__fadd_rn(h2f(x[i]), h2f(o[i])));
            // the MLP, as the served chain runs it (host_pers_segment.inc)
            const auto h2 = norm(x, Y.ln2.data(), H);
            const auto xr2 = rot(h2, rm, H);
            const auto g = served_ref(Y.gate, xr2), u = served_ref(Y.up, xr2);
            std::vector<unsigned short> act(I);
            for (u32 i = 0; i < I; ++i) act[i] = f2h(__fmul_rn(h2f(silu[g[i]]), h2f(u[i])));
            std::vector<unsigned short> dn;
            if (Y.tetra_down) dn = served_ref(Y.down, rot(act, rd_, I));
            else dn = q4(Y.dqw, Y.dqs, Y.dqb, act, H, I);
            for (u32 i = 0; i < H; ++i) x[i] = f2h(__fadd_rn(h2f(x[i]), h2f(dn[i])));
        }
        const auto hf = norm(x, final_norm.data(), H);
        std::vector<unsigned short> logits(vocab);
        hostgrid::launch_seq(vocab / 8, 256, [&] { tv_emb_q4_h(ew.data(), es.data(), eb.data(), hf.data(), logits.data(), H, egpr, 0u, 0u); });
        unsigned long long best = 0ull;
        u32 next = 0;
        for (u32 i = 0; i < vocab; ++i) {
            u32 b = logits[i];
            if (b == 0x8000u) b = 0u;
            const u32 ord = (b & 0x8000u) ? ((~b) & 0xFFFFu) : (b | 0x8000u);
            const unsigned long long key = ((unsigned long long)ord << 32) | ((unsigned long long)(1023u - (__brev(i & 1023u) >> 22)) << 22) | (0x3FFFFFu - (i >> 10));
            if (key > best) { best = key; next = i; }
        }
        ring_run[t] = next;
        tok = next;
        if (tok == eos) break;
    }
  }
#else
    const F1rTables tb = tables();
    std::vector<unsigned short> qs(NH * 128), ks(NKV * 128), vs(NKV * 128), x_b(H), q_out(NH * 128), ctx(NH * 128), act(I);
    std::vector<float> scratch((size_t)NKV * ((cap + 255) / 256) * GRP * 130);
    std::vector<u32> bar(1, 0u), ab(1, 0u), nan(1, 0u), ticket(NKV, 0u);
    std::vector<unsigned long long> slots(2, 0ull);
    const auto gu = [&] { std::vector<u32> v; for (u32 g = 0; g < grid; ++g) { v.push_back((u32)((uint64_t)I * g / grid)); v.push_back((u32)((uint64_t)I * (g + 1) / grid)); } return v; }();
    const auto dn = [&] { std::vector<u32> v; for (u32 g = 0; g < grid; ++g) { v.push_back((u32)((uint64_t)H * g / grid)); v.push_back((u32)((uint64_t)H * (g + 1) / grid)); } return v; }();
    auto segt = [](const TMat& m) { return SegTetra{m.words.data(), m.rscale.data(), m.tail.data(), m.gscale.data(), m.stride, m.nblocks, m.tail_w, m.d_out}; };
    std::vector<Pers4eLayer> PL(NL);
    for (u32 l = 0; l < NL; ++l) {
        Layer& Y = Ls[l];
        Pers4eLayer& P = PL[l];
        P = Pers4eLayer{};
        P.ln1 = Y.ln1.data(); P.signs_a = ra.signs.data(); P.small_a = ra.small.data(); P.m_a = ra.m; P.k_a = ra.k; P.inv_a = ra.inv;
        P.qkv = Pers4eQkv{Y.q.words.data(), Y.q.rscale.data(), Y.q.tail.data(), Y.q.gscale.data(), Y.q.stride, NH * 128,
                          Y.k.words.data(), Y.k.rscale.data(), Y.k.tail.data(), Y.k.gscale.data(), Y.k.stride, NKV * 128,
                          Y.vw.data(), Y.vs.data(), Y.vb.data(), NKV * 128, 20u, Y.q.nblocks, Y.q.tail_w, H,
                          qs.data(), ks.data(), vs.data()};
        P.q_norm = Y.qn.data(); P.k_norm = Y.kn.data(); P.kc = Y.kc.data(); P.vc = Y.vc.data();
        P.ow = Y.ow.data(); P.osc = Y.os.data(); P.obi = Y.ob.data(); P.o_gpr = NH * 128 / 128; P.tetra_down = Y.tetra_down ? 1u : 0u;
        SegArgs& M = P.mlp;
        M.alpha = Y.ln2.data(); M.signs_h = rm.signs.data(); M.small_h = rm.small.data(); M.silu = silu.data(); M.act = act.data();
        M.gate = segt(Y.gate); M.up = segt(Y.up); M.down = segt(Y.down);
        M.signs_i = rd_.signs.data(); M.small_i = rd_.small.data();
        M.q4w = Y.dqw.data(); M.q4s = Y.dqs.data(); M.q4b = Y.dqb.data();
        M.rows = tb.rows; M.pref = tb.prefixes; M.br = tb.branches; M.suf = tb.suffixes; M.invnorm = g_inv.data();
        M.slice_gu = gu.data(); M.slice_dn = dn.data(); M.bar = bar.data(); M.abort_flag = ab.data();
        M.hidden = H; M.inter = I; M.m_h = rm.m; M.k_h = rm.k; M.m_i = rd_.m; M.k_i = rd_.k;
        M.inv_h = rm.inv; M.inv_i = rd_.inv; M.eps = eps; M.gpr = I / 128;
    }
#if PERS_4E_ROWS >= 7
    // Form 7 reads its int4 rows (v, o, the int4 down) from words relaid by
    // pers_q4_tile_index, as persist4e_cuda.rs relays them on the card.
    std::vector<std::vector<u32>> tiled;
    auto tile = [&](const std::vector<u32>& w, u32 d_out, u32 d_in) -> const u32* {
        const u32 nw = d_in / 8;
        std::vector<u32> t(w.size());
        for (u32 row = 0; row < d_out; ++row)
            for (u32 wi = 0; wi < nw; ++wi) t[(size_t)row * nw + pers_q4_tile_index(wi, nw)] = w[(size_t)row * nw + wi];
        tiled.push_back(std::move(t));
        return tiled.back().data();
    };
    for (u32 l = 0; l < NL; ++l) {
        PL[l].qkv.vw = tile(Ls[l].vw, NKV * 128, H);
        PL[l].ow = tile(Ls[l].ow, H, NH * 128);
        if (!Ls[l].tetra_down) PL[l].mlp.q4w = tile(Ls[l].dqw, H, I);
    }
#endif
    Pers4eArgs A{};
    A.layers = PL.data(); A.emb_w = ew.data(); A.emb_s = es.data(); A.emb_b = eb.data(); A.final_norm = final_norm.data();
    A.cosr = cosr.data(); A.sinr = sinr.data(); A.x_a = x.data(); A.x_b = x_b.data(); A.q_out = q_out.data(); A.ctx = ctx.data();
    A.attn_scratch = scratch.data(); A.bar = bar.data(); A.abort_flag = ab.data(); A.slots = slots.data(); A.nan_flag = nan.data();
    A.rows = tb.rows; A.pref = tb.prefixes; A.br = tb.branches; A.suf = tb.suffixes; A.invnorm = g_inv.data();
    A.attn_ticket = ticket.data();
    A.n_layers = NL; A.vocab = vocab; A.hidden = H; A.emb_gpr = egpr; A.n_heads = NH; A.n_kv = NKV; A.groups = GRP; A.cap = cap;
    A.eps = eps; A.c = scale;
    typedef void (*Fn)(const Pers4eArgs*, const Pers4eLaunch);
    const Fn table[4][3] = {{pers_4e_tokens_f0, pers_4e_tokens_f1, pers_4e_tokens_f2},
                            {pers_4e_stamps_f0_t512, pers_4e_stamps_f1_t512, pers_4e_stamps_f2_t512},
                            {pers_4e_tokens_f0_t512, pers_4e_tokens_f1_t512, pers_4e_tokens_f2_t512},
                            {pers_4e_stamps_f0_t768, pers_4e_stamps_f1_t768, pers_4e_stamps_f2_t768}};
    if (variant > 3u) { std::fprintf(stderr, "host_pers: 4E variant %u\n", variant); std::exit(2); }
    const size_t n_ev = 3u + 15u * NL;
    std::vector<unsigned long long> stamps(stamp ? ring_len * n_ev * grid : 0u, 0ull);
    const size_t smem = PERS_SMEM_HEADER + 18816 + std::max(4 * (size_t)std::max(I, 2 * H), (size_t)PERS_ATTN_SMEM(GRP)) +
                        PERS_4E_SCRATCH;
    // Launch after launch through the same state, as Runtime4e::decode runs
    // them: the barrier base is the counter's value (no abort), a fresh ring.
    size_t ring_at = 0;
    for (size_t run_i = 0; run_i < runs.size(); run_i += 3) {
        const Pers4eLaunch L{bar[0], runs[run_i], runs[run_i + 1], runs[run_i + 2], eos, 0u, ring.data() + ring_at,
                             stamp ? stamps.data() + ring_at * n_ev * grid : nullptr};
        ring_at += runs[run_i + 1];
        hostgrid::launch_coop(grid, threads, smem, seed + run_i, [&] { table[variant][form](&A, L); });
    }
    tail[0] = bar[0];
    tail[1] = ab[0];
    for (u32 t : ticket) tail[1] += t;  // the split fold leaves every ticket at zero
#endif
    wr(ring);
    wr(x);
    for (auto& Y : Ls) { wr(Y.kc); wr(Y.vc); }
    wr(tail);
#if PERS_SIDE == 1
    wr(stamps);
#endif
}
