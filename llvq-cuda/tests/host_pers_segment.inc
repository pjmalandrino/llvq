// Command 'S' of host_pers.cpp: route P's step-A segment against the eager
// served chain.
//
// in : u32 grid, threads, normv, tetra_down, u64 seed, u32 hidden, inter,
//      m_h, k_h, m_i, k_i, f32 inv_h, inv_i, eps, u32 bar_base,
//      u32 signs_h[hidden/32], signs_i[inter/32], f32 small_h[1024], small_i[1024],
//      u32 slice_gu[2*grid], slice_dn[2*grid]
// out: u16 out[hidden], u16 act[inter], u32 bar, u32 abort
{
    const u32 grid = rd1<u32>(), threads = rd1<u32>(), normv = rd1<u32>(), tetra_down = rd1<u32>();
    const uint64_t seed = rd1<uint64_t>();
    const u32 H = rd1<u32>(), I = rd1<u32>(), m_h = rd1<u32>(), k_h = rd1<u32>(), m_i = rd1<u32>(), k_i = rd1<u32>();
    const float inv_h = rd1<float>(), inv_i = rd1<float>(), eps = rd1<float>();
    const u32 bar_base = rd1<u32>();
    const auto signs_h = rd<u32>(H / 32), signs_i = rd<u32>(I / 32);
    const auto small_h = rd<float>(1024), small_i = rd<float>(1024);
    const auto slice_gu = rd<u32>(2 * grid), slice_dn = rd<u32>(2 * grid);

    const auto x = act_h(H, seed + 1, 10, 16);
    const auto alpha = act_h(H, seed + 2, 13, 16);
    const TMat gate = tmat(I, H, seed + 3), up = tmat(I, H, seed + 4), down = tmat(H, I, seed + 5);
    // A stand-in for candle's SiLU table: any table, the same on both sides.
    // On the card the table is filled by candle's own usilu_f16.
    std::vector<unsigned short> silu(65536);
    {
        Rng r(seed + 6);
        for (auto& e : silu) e = r.h16(8, 15);
    }
    Rng rq(seed + 7);
    const u32 gpr = I / 128u;
    std::vector<u32> q4w((size_t)H * I / 8);
    for (auto& w : q4w) w = (u32)rq.next();
    std::vector<unsigned short> q4s((size_t)H * gpr), q4b((size_t)H * gpr);
    for (auto& e : q4s) e = rq.h16(6, 10);
    for (auto& e : q4b) e = rq.h16(8, 12);

    std::vector<unsigned short> out(H, 0xFFFFu), act(I, 0xFFFFu);
    std::vector<u32> tail_out(2, 0u);
#if PERS_SIDE == 0
    (void)grid; (void)threads; (void)bar_base; (void)slice_gu; (void)slice_dn;
    const float sc = candle_rms_scale(x.data(), H, eps, (int)normv);
    std::vector<unsigned short> h0(H);
    for (u32 i = 0; i < H; ++i) h0[i] = f2h(__fmul_rn(__fmul_rn(sc, h2f(x[i])), h2f(alpha[i])));
    std::vector<float> xr(H);
    hostgrid::launch_seq(1, 256, [&] { rot_apply(h0.data(), signs_h.data(), small_h.data(), xr.data(), H, m_h, k_h, inv_h, 0u); });
    const auto g = served_ref(gate, xr), u = served_ref(up, xr);
    for (u32 r = 0; r < I; ++r) act[r] = f2h(__fmul_rn(h2f(silu[g[r]]), h2f(u[r])));
    std::vector<unsigned short> dn(H);
    if (tetra_down) {
        std::vector<float> ar(I);
        hostgrid::launch_seq(1, 256, [&] { rot_apply(act.data(), signs_i.data(), small_i.data(), ar.data(), I, m_i, k_i, inv_i, 0u); });
        dn = served_ref(down, ar);
    } else {
        hostgrid::launch_seq(H / 8, 256, [&] { tv_q4_hh(q4w.data(), q4s.data(), q4b.data(), act.data(), dn.data(), I, gpr); });
    }
    for (u32 r = 0; r < H; ++r) out[r] = f2h(__fadd_rn(h2f(x[r]), h2f(dn[r])));
#else
    static std::vector<u32> bar(1, 0u), abort_flag(1, 0u);
    bar[0] = bar_base;  // the counter as the ledger says earlier launches left it
    const F1rTables t = tables();
    auto seg = [](const TMat& m) {
        return SegTetra{m.words.data(), m.rscale.data(), m.tail.data(), m.gscale.data(), m.stride, m.nblocks, m.tail_w, m.d_out};
    };
    SegArgs A{};
    A.x = x.data(); A.alpha = alpha.data();
    A.signs_h = signs_h.data(); A.small_h = small_h.data();
    A.silu = silu.data(); A.act = act.data(); A.out = out.data();
    A.gate = seg(gate); A.up = seg(up); A.down = seg(down);
    A.signs_i = signs_i.data(); A.small_i = small_i.data();
    A.q4w = q4w.data(); A.q4s = q4s.data(); A.q4b = q4b.data();
    A.rows = t.rows; A.pref = t.prefixes; A.br = t.branches; A.suf = t.suffixes; A.invnorm = g_inv.data();
    A.slice_gu = slice_gu.data(); A.slice_dn = slice_dn.data();
    A.bar = bar.data(); A.abort_flag = abort_flag.data(); A.bar_base = bar_base;
    A.hidden = H; A.inter = I; A.m_h = m_h; A.k_h = k_h; A.m_i = m_i; A.k_i = k_i;
    A.inv_h = inv_h; A.inv_i = inv_i; A.eps = eps; A.gpr = gpr;
    typedef void (*SegFn)(const SegArgs*);
    const SegFn fn = tetra_down ? (normv ? pers_seg_tetra_v1 : pers_seg_tetra_v0) : (normv ? pers_seg_q4_v1 : pers_seg_q4_v0);
    const size_t smem_bytes = PERS_SMEM_HEADER + 18816 + 4 * (size_t)std::max(H, I);
    hostgrid::launch_coop(grid, threads, smem_bytes, seed, [&] { fn(&A); });
    tail_out[0] = bar[0];
    tail_out[1] = abort_flag[0];
#endif
    wr(out);
    wr(act);
    wr(tail_out);
}
