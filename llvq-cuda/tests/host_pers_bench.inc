// Command 'B' of host_pers.cpp: the L2 bench arms over a site list.
//
// in : u32 arm (index into pers.rs::PERS_ARMS), grid, threads, n_launches,
//      u64 seed, u32 nsites, u32 rec[nsites][8] (pers.rs::site_record, words
//      address 0: filled here), u32 lo[nsites*grid], hi[nsites*grid],
//      u32 hidden, inter, m_h, k_h, m_i, k_i, f32 inv_h, inv_i, eps,
//      u32 signs_h[hidden/32], signs_i[inter/32], f32 small_h[1024], small_i[1024],
//      then per launch u32 bar_base, epoch, cnt_offset (pers.rs::Ledger)
// out: u16 y_all[total rows], f32 dbg[total rows], u16 norm[n_norm*hidden],
//      f32 rot[n_prol*inter], u32 bar, u32 abort
// The served side writes the same layout: tv_tetra48_h per site, candle's
// norm replay and rot_apply for the prologue vectors, its bar = 0, abort = 0.
{
    const u32 arm_i = rd1<u32>(), grid = rd1<u32>(), threads = rd1<u32>(), n_launches = rd1<u32>();
    const uint64_t seed = rd1<uint64_t>();
    const u32 nsites = rd1<u32>();
    auto rec = rd<u32>((size_t)nsites * 8);
    const auto lo = rd<u32>((size_t)nsites * grid);
    const auto hi = rd<u32>((size_t)nsites * grid);
    const u32 hidden = rd1<u32>(), inter = rd1<u32>(), m_h = rd1<u32>(), k_h = rd1<u32>(), m_i = rd1<u32>(),
              k_i = rd1<u32>();
    const float inv_h = rd1<float>(), inv_i = rd1<float>(), eps = rd1<float>();
    const auto signs_h = rd<u32>(hidden / 32), signs_i = rd<u32>(inter / 32);
    const auto small_h = rd<float>(1024), small_i = rd<float>(1024);
    const auto ledger = rd<u32>((size_t)n_launches * 3);

    std::vector<TMat> mats;
    std::vector<u32> row0(nsites);
    u32 total = 0, n_norm = 0, n_prol = 0;
    for (u32 si = 0; si < nsites; ++si) {
        const u32 d_out = rec[si * 8 + 2], nb = rec[si * 8 + 3] & 0xffffu, tw = rec[si * 8 + 3] >> 16;
        mats.push_back(tmat(d_out, nb * 24 + tw, seed + 100u + si));
        row0[si] = total;
        total += d_out;
        const u32 pk = (rec[si * 8 + 7] >> 1) & 3u;
        // a prologue runs where the kernel restages: the width or the layer changed
        const bool restage = si == 0 || (rec[si * 8 + 3] != rec[(si - 1) * 8 + 3]) ||
                             ((rec[si * 8 + 7] >> 8) != (rec[(si - 1) * 8 + 7] >> 8));
        if (restage && pk != 0u) {
            n_prol += 1;
            n_norm += pk == 1u ? 1u : 0u;
        }
    }
    const auto x = act(9728, seed + 1);
    const auto resid = act_h(9728, seed + 2, 10, 16);
    const auto alpha = act_h(hidden, seed + 3, 13, 16);
    std::vector<unsigned short> y_all(total, 0xFFFFu), norm_out((size_t)n_norm * hidden, 0);
    std::vector<float> dbg(total, 0.0f), rot_out((size_t)n_prol * inter, 0.0f);
    std::vector<u32> out_tail(2, 0u);
#if PERS_SIDE == 0
    (void)arm_i; (void)grid; (void)threads; (void)lo; (void)hi; (void)ledger; (void)m_h; (void)k_h;
    for (u32 si = 0; si < nsites; ++si) {
        const auto y = served_ref(mats[si], x, dbg.data() + row0[si]);
        for (u32 r = 0; r < mats[si].d_out; ++r) y_all[row0[si] + r] = y[r];
    }
    // The prologue vectors as the eager path makes them: candle's rmsnorm,
    // then rot_apply on its binary16 output; or rot_apply on the residual.
    u32 pn = 0, pr = 0;
    for (u32 si = 0; si < nsites; ++si) {
        const u32 pk = (rec[si * 8 + 7] >> 1) & 3u;
        const bool restage = si == 0 || (rec[si * 8 + 3] != rec[(si - 1) * 8 + 3]) ||
                             ((rec[si * 8 + 7] >> 8) != (rec[(si - 1) * 8 + 7] >> 8));
        if (!restage || pk == 0u) continue;
        std::vector<unsigned short> in;
        u32 n, m, k;
        float inv;
        const u32* sg;
        const float* sm;
        if (pk == 1u) {
            n = hidden; m = m_h; k = k_h; inv = inv_h; sg = signs_h.data(); sm = small_h.data();
            const float sc = candle_rms_scale(resid.data(), hidden, eps, 0);
            in.resize(hidden);
            for (u32 i = 0; i < hidden; ++i) in[i] = f2h(__fmul_rn(__fmul_rn(sc, h2f(resid[i])), h2f(alpha[i])));
            for (u32 i = 0; i < hidden; ++i) norm_out[(size_t)pn * hidden + i] = in[i];
            ++pn;
        } else {
            n = inter; m = m_i; k = k_i; inv = inv_i; sg = signs_i.data(); sm = small_i.data();
            in.assign(resid.begin(), resid.begin() + inter);
        }
        std::vector<float> r(n);
        hostgrid::launch_seq(1, 256, [&] { rot_apply(in.data(), sg, sm, r.data(), n, m, k, inv, 0u); });
        for (u32 i = 0; i < n; ++i) rot_out[(size_t)pr * inter + i] = r[i];
        ++pr;
    }
#else
    // Device state that lives across launches, as on the card.
    static std::vector<u32> bar(1, 0u), abort_flag(1, 0u);
    std::vector<u32> cnts(4096, 0u);
    std::vector<const float*> gs(nsites), rs(nsites);
    std::vector<const unsigned short*> tl(nsites);
    for (u32 si = 0; si < nsites; ++si) {
        const uint64_t a = (uint64_t)(uintptr_t)mats[si].words.data();
        rec[si * 8 + 0] = (u32)a;
        rec[si * 8 + 1] = (u32)(a >> 32);
        rec[si * 8 + 4] = row0[si];
        rec[si * 8 + 5] = mats[si].stride;
        gs[si] = mats[si].gscale.data();
        rs[si] = mats[si].rscale.data();
        tl[si] = mats[si].tail.data();
    }
    const F1rTables t = tables();
    PersArgs A{};
    A.sites = reinterpret_cast<const PersSite*>(rec.data());
    A.rows = t.rows; A.pref = t.prefixes; A.br = t.branches; A.suf = t.suffixes;
    A.invnorm = g_inv.data();
    A.gscale = gs.data(); A.rscale = rs.data(); A.tail = tl.data();
    A.x = x.data();
    A.y_all = y_all.data();
    PersLaunch PL{};
    A.slice_lo = lo.data(); A.slice_hi = hi.data();
    A.bar = bar.data(); A.cnts = cnts.data(); A.abort_flag = abort_flag.data();
    A.resid = resid.data(); A.alpha = alpha.data();
    A.signs_h = signs_h.data(); A.signs_i = signs_i.data();
    A.small_h = small_h.data(); A.small_i = small_i.data();
    A.nsites = nsites;
    A.hidden = hidden; A.inter = inter; A.m_h = m_h; A.k_h = k_h; A.m_i = m_i; A.k_i = k_i;
    A.inv_h = inv_h; A.inv_i = inv_i; A.eps = eps;
    typedef void (*ArmFn)(const PersArgs*, const PersLaunch);
    // pers.rs::PERS_ARMS order; pers_matches_served.rs checks the count.
    const ArmFn fns[15] = {tv_pers_null, tv_pers_word, tv_pers_dec_l1, tv_pers_dec_smem, tv_pers_l1,
                           tv_pers_l1_t28, tv_pers_smem, tv_pers_smem_s25, tv_pers_smem_t28,
                           tv_pers_smem_t28_u2, tv_pers_t28_512, tv_pers_t28_1024,
                           tv_pers_chain_w, tv_pers_chain_w2, tv_pers_chain_wp};
    const u32 strides[15] = {28, 28, 28, 28, 24, 28, 24, 25, 28, 28, 28, 28, 28, 28, 28};
    const u32 modes[15] = {0, 1, 4, 5, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3};
    if (arm_i >= 15) {
        std::fprintf(stderr, "host_pers: arm index %u out of range\n", arm_i);
        std::exit(2);
    }
    const bool shared_tables = modes[arm_i] == 3 || modes[arm_i] == 5;
    const size_t smem_bytes = PERS_SMEM_HEADER + (shared_tables ? 18816 : 0) +
                              std::max<size_t>((size_t)(405 * strides[arm_i] + 8) * 4, 9728 * 4);
    for (u32 L = 0; L < n_launches; ++L) {
        std::fill(y_all.begin(), y_all.end(), (unsigned short)0xFFFFu);
        const bool last = L + 1 == n_launches;
        PL.dbg = last ? dbg.data() : nullptr;
        PL.dbg_norm = last ? norm_out.data() : nullptr;
        PL.dbg_rot = last ? rot_out.data() : nullptr;
        PL.bar_base = ledger[L * 3 + 0];
        PL.epoch = ledger[L * 3 + 1];
        PL.cnt_offset = ledger[L * 3 + 2];
        hostgrid::launch_coop(grid, threads, smem_bytes, seed + L, [&] { fns[arm_i](&A, PL); });
    }
    out_tail[0] = bar[0];
    out_tail[1] = abort_flag[0];
#endif
    wr(y_all);
    wr(dbg);
    wr(norm_out);
    wr(rot_out);
    wr(out_tail);
}
