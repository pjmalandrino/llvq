// Command 'e' of host_pers.cpp: the research row bodies as 4E runs them (kernels/pers_4e_rc.cuh, step R6), on
// RECORDED words, against the served row. Included inside main()'s command chain; `tag` is in scope.
//
// in : u32 body (1 endpass, 2 flat; 3 ep2, 4 ep3, 5 fl2, 6 fl3: pers_4e_rc2.cuh), mode (0 a site's rows, the padded copy beside the natural staging, as 4E's q/k
//      and gate/up phases place it; 1 the same, restaged IN PLACE, as the down and the o; 2 gate/up pairs, sites s and
//      s + 1 of one shape as ONE list, the gate's slice then the up's, as 4E's gate/up phase), grid, threads, nsites,
//      then the sites and the activation exactly as command 'C' reads them.
// out: every site's rows in order, binary16, then their f32 values before f2h. The served side runs served_ref on
//      each site (command 'C''s reference) and ignores body, mode, grid and threads.
//
// Each CTA takes the rows [d_out c / grid, d_out (c + 1) / grid) of its site (of each site of the pair), stages the
// activation in 4E's row vector (pers_stage_f32, natural stride 24), sets the v7 offset of its header to the end of
// the region this command allocates, and calls pers_rc_rows: the function the rc arms' modules call.
{
    const u32 body = rd1<u32>(), mode = rd1<u32>(), grid = rd1<u32>(), threads = rd1<u32>();
    const u32 nsites = rd1<u32>();
    std::vector<TMat> mats;
    u32 maxd = 0u;
    for (u32 s = 0; s < nsites; ++s) {
        TMat m;
        m.d_out = rd1<u32>();
        const u32 d_in = rd1<u32>();
        m.nblocks = d_in / 24u;
        m.tail_w = d_in % 24u;
        m.stride = (6u * m.nblocks + 7u) / 8u * 2u;
        m.words = rd<u32>((size_t)m.d_out * m.stride);
        m.gscale = rd<float>(2);
        m.rscale = rd<float>(m.d_out);
        m.tail = rd<unsigned short>((size_t)m.d_out * m.tail_w);
        maxd = std::max(maxd, d_in);
        mats.push_back(std::move(m));
    }
    const u32 nx = rd1<u32>();
    const std::vector<float> x = rd<float>(nx);
    if (nx < maxd) {
        std::fprintf(stderr, "host_pers: activation of %u values, the widest site reads %u\n", nx, maxd);
        return 2;
    }
    std::vector<unsigned short> out;
    std::vector<float> outf;
#if PERS_SIDE == 0
    (void)body;
    (void)mode;
    (void)grid;
    (void)threads;
    for (const TMat& m : mats) {
        const std::vector<float> xi(x.begin(), x.begin() + m.nblocks * 24u + m.tail_w);
        std::vector<float> f(m.d_out);
        const auto y = served_ref(m, xi, f.data());
        out.insert(out.end(), y.begin(), y.end());
        outf.insert(outf.end(), f.begin(), f.end());
    }
#else
    if (body < 1u || body > 6u || mode > 2u || (mode == 2u && nsites % 2u != 0u)) {
        std::fprintf(stderr, "host_pers: rc body %u mode %u on %u sites\n", body, mode, nsites);
        return 2;
    }
    if (mode == 2u)
        for (u32 s = 0; s < nsites; s += 2u)
            if (mats[s].d_out != mats[s + 1u].d_out || mats[s].nblocks != mats[s + 1u].nblocks ||
                mats[s].tail_w != mats[s + 1u].tail_w) {
                std::fprintf(stderr, "host_pers: rc gate/up pair %u: the two sites differ in shape\n", s / 2u);
                return 2;
            }
    std::vector<std::vector<unsigned short>> ys;
    std::vector<std::vector<float>> fs;
    for (const TMat& m : mats) {
        ys.emplace_back(m.d_out, 0xFFFFu);
        fs.emplace_back(m.d_out, -1.0f);
    }
    // the row vector: the natural staging, then flat's padded copy, then the slots and the gate/up values
    const u32 nat = (maxd + 3u) & ~3u;
    const u32 region = nat + ((24u * (maxd / 24u) + (maxd / 24u) / 8u + maxd % 24u + 3u) & ~3u) + PERS_RC_SLOTS + 2048u;
    const size_t smem = PERS_SMEM_HEADER + 18816u + 4u * (size_t)region;
    const F1rTables tb = tables();
    const u32 step = mode == 2u ? 2u : 1u;
    for (u32 s = 0; s < nsites; s += step) {
        const TMat& a = mats[s];
        const TMat& b = mats[s + step - 1u];
        unsigned short* ya[2] = {ys[s].data(), ys[s + step - 1u].data()};
        float* fa[2] = {fs[s].data(), fs[s + step - 1u].data()};
        hostgrid::launch_coop(grid, threads, smem, 0x7e80ull + s + 16u * mode + 64u * body, [&] {
            if (threadIdx.x == 0u) pers_hdr()[PERS_HDR_V7] = region;
            float* v = pers_rc_v();
            pers_stage_f32<24u, false>(x.data(), v, a.nblocks, a.tail_w);
            const u32 lo = (u32)((uint64_t)a.d_out * blockIdx.x / gridDim.x);
            const u32 hi = (u32)((uint64_t)a.d_out * (blockIdx.x + 1u) / gridDim.x);
            const u32 np = hi - lo, n = step * np;
            const auto row_at = [&](u32 idx) {
                const u32 k = idx >= np ? 1u : 0u;
                const TMat& m = k ? b : a;
                const u32 r = lo + idx - k * np;
                return PersTRow{m.words.data() + (size_t)r * m.stride, m.gscale.data(), m.rscale.data(), m.tail.data(), r};
            };
            const auto epi = [&](u32 idx, float y) {
                const u32 k = idx >= np ? 1u : 0u;
                const u32 r = lo + idx - k * np;
                ya[k][r] = f2h(y);
                fa[k][r] = y;
            };
            const bool in_place = mode == 1u;
            float* work = in_place ? v + pers_rc_padded_floats(a.nblocks, a.tail_w) : v + nat;
            switch (body) {
            case 1u: pers_rc_rows<1u>(n, a.nblocks, a.tail_w, tb, g_inv.data(), v, row_at, epi, work, in_place); break;
            case 2u: pers_rc_rows<2u>(n, a.nblocks, a.tail_w, tb, g_inv.data(), v, row_at, epi, work, in_place); break;
            case 3u: pers_rc2_rows<3u>(n, a.nblocks, a.tail_w, tb, g_inv.data(), v, row_at, epi, work, in_place); break;
            case 4u: pers_rc2_rows<4u>(n, a.nblocks, a.tail_w, tb, g_inv.data(), v, row_at, epi, work, in_place); break;
            case 5u: pers_rc2_rows<5u>(n, a.nblocks, a.tail_w, tb, g_inv.data(), v, row_at, epi, work, in_place); break;
            default: pers_rc2_rows<6u>(n, a.nblocks, a.tail_w, tb, g_inv.data(), v, row_at, epi, work, in_place); break;
            }
        });
    }
    for (const auto& y : ys) out.insert(out.end(), y.begin(), y.end());
    for (const auto& f : fs) outf.insert(outf.end(), f.begin(), f.end());
#endif
    wr(out);
    wr(outf);
}
