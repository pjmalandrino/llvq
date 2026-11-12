// The decode-speed research phase's stage-1 gate (kernels/rowcand.cu, tests/rowcand_gate.rs), included inside
// main()'s command chain of host_pers.cpp ('C', 'd' and 'b'). `tag` is in scope.
if (tag == 'C') {
    // CANDIDATE ROWS on RECORDED words: arm (ROWC_ARMS order), mode (0 the rows body, 1 the gate/up body, whose
    // sites come in pairs of one shape), grid, threads, nsites; then per site d_out, d_in, the words (d_out x
    // stride u32, the served Tetra48 stream), gscale[2] f32, rscale[d_out] f32, tail[d_out x tail_w] binary16;
    // then n and the activation, n f32 (at least the widest d_in). Writes every site's rows in order, binary16,
    // then their f32 values before f2h. The served side runs the served reference (served_ref: tv_tetra48_h with
    // NVVM's three fusions, which the replay test pins to the served text) on each site and ignores arm and mode.
    const u32 arm = rd1<u32>(), mode = rd1<u32>(), grid = rd1<u32>(), threads = rd1<u32>();
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
    (void)arm;
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
    if (arm >= ROWC_HOST_COUNT || mode > 1u) {
        std::fprintf(stderr, "host_pers: candidate arm %u mode %u, %zu arms registered\n", arm, mode, ROWC_HOST_COUNT);
        return 2;
    }
    if (mode == 1u)
        for (u32 s = 0; s + 1u < nsites; s += 2u)
            if (mats[s].d_out != mats[s + 1u].d_out || mats[s].nblocks != mats[s + 1u].nblocks ||
                mats[s].tail_w != mats[s + 1u].tail_w) {
                std::fprintf(stderr, "host_pers: gate/up pair %u: the two sites differ in shape\n", s / 2u);
                return 2;
            }
    std::vector<std::vector<unsigned short>> ys;
    std::vector<std::vector<float>> fs;
    std::vector<std::vector<u32>> relaid(mats.size());
    std::vector<unsigned long long> words;
    ys.reserve(mats.size());
    fs.reserve(mats.size());
    for (size_t k = 0; k < mats.size(); ++k) {
        const TMat& m = mats[k];
        ys.emplace_back(m.d_out, 0xFFFFu);
        fs.emplace_back(m.d_out, -1.0f);
        const u32* wsrc = m.words.data();
        if (ROWC_HOST_RELAID[arm]) {
            // RELAY: the arm's own copy, rewritten once, in the served stride
            relaid[k].assign(m.words.size(), 0u);
            hostgrid::launch_seq(3, 64, [&] {
                ROWC_HOST_RELAYS[arm](m.words.data(), relaid[k].data(), m.stride, m.d_out, m.nblocks);
            });
            wsrc = relaid[k].data();
        }
        const unsigned long long w[TROWS_SITE_WORDS] = {
            (unsigned long long)(uintptr_t)wsrc, (unsigned long long)(uintptr_t)m.gscale.data(),
            (unsigned long long)(uintptr_t)m.rscale.data(), (unsigned long long)(uintptr_t)m.tail.data(),
            (unsigned long long)(uintptr_t)ys.back().data(),
            (unsigned long long)m.d_out | ((unsigned long long)m.nblocks << 32),
            (unsigned long long)m.tail_w | ((unsigned long long)m.stride << 32),
            (unsigned long long)(uintptr_t)fs.back().data(), 0ull, 0ull};
        words.insert(words.end(), w, w + TROWS_SITE_WORDS);
    }
    const size_t smem = PERS_SMEM_HEADER + 18816u + 4u * maxd + 16u;
    hostgrid::launch_coop(grid, threads, smem, 0x7e70ull + arm, [&] {
        (mode == 0u ? ROWC_HOST_BODIES : ROWC_HOST_GU_BODIES)[arm](words.data(), nsites, g_rows.data(), g_pref.data(),
                                                                   g_br.data(), g_suf.data(), g_inv.data(), x.data());
    });
    for (const auto& y : ys) out.insert(out.end(), y.begin(), y.end());
    for (const auto& f : fs) outf.insert(outf.end(), f.begin(), f.end());
#endif
    wr(out);
    wr(outf);
} else if (tag == 'd') {
    // READ DIAGNOSTICS (rowcand.cu, ROWD_ARMS): arm (ROWD_ARMS order), mode (0 rows, 1 gate/up pairs), grid,
    // threads, nsites, then the sites and the activation as 'C' reads them. Runs the arm's body with every site's
    // word [8] naming one buffer of 2 u64 a thread, and writes the wrapping sum of the threads' read checks
    // (out[2 t]), then the thread count. The served side writes zeros: the reference is the host's own sum of the
    // words (rowcand_gate.rs).
    const u32 arm = rd1<u32>(), mode = rd1<u32>(), grid = rd1<u32>(), threads = rd1<u32>();
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
    std::vector<unsigned long long> res(2, 0ull);
#if PERS_SIDE == 1
    if (arm >= ROWD_HOST_COUNT || mode > 1u) {
        std::fprintf(stderr, "host_pers: diagnostic arm %u mode %u, %zu arms registered\n", arm, mode, ROWD_HOST_COUNT);
        return 2;
    }
    std::vector<unsigned long long> slots(2ull * grid * threads, 0x5a5a5a5a5a5a5a5aull);
    std::vector<std::vector<unsigned short>> ys;
    std::vector<std::vector<float>> fs;
    std::vector<unsigned long long> words;
    ys.reserve(mats.size());
    fs.reserve(mats.size());
    for (const TMat& m : mats) {
        ys.emplace_back(m.d_out, 0xFFFFu);
        fs.emplace_back(m.d_out, -1.0f);
        const unsigned long long w[TROWS_SITE_WORDS] = {
            (unsigned long long)(uintptr_t)m.words.data(), (unsigned long long)(uintptr_t)m.gscale.data(),
            (unsigned long long)(uintptr_t)m.rscale.data(), (unsigned long long)(uintptr_t)m.tail.data(),
            (unsigned long long)(uintptr_t)ys.back().data(),
            (unsigned long long)m.d_out | ((unsigned long long)m.nblocks << 32),
            (unsigned long long)m.tail_w | ((unsigned long long)m.stride << 32),
            (unsigned long long)(uintptr_t)fs.back().data(), (unsigned long long)(uintptr_t)slots.data(), 0ull};
        words.insert(words.end(), w, w + TROWS_SITE_WORDS);
    }
    const size_t smem = PERS_SMEM_HEADER + 18816u + 4u * maxd + 16u;
    hostgrid::launch_coop(grid, threads, smem, 0x7e71ull + arm, [&] {
        (mode == 0u ? ROWD_HOST_BODIES : ROWD_HOST_GU_BODIES)[arm](words.data(), nsites, g_rows.data(), g_pref.data(),
                                                                   g_br.data(), g_suf.data(), g_inv.data(), x.data());
    });
    for (size_t t = 0; t < slots.size(); t += 2) res[0] += slots[t];
    res[1] = (unsigned long long)grid * threads;
#else
    (void)arm;
    (void)mode;
    (void)grid;
    (void)threads;
#endif
    wr(res);
} else {
    // BANDWIDTH PROBE (bwprobe.cu): kernel (rowcand.rs::BW_KERNELS order), n words, seed, grid, threads, prefetch
    // steps. Fills the buffer with tv_bw_fill, reads it with the kernel, writes the sum of the CTAs' partials
    // (u64) and the partial count. The served side writes zeros: the probe has no served twin, its reference is
    // the host's own sum (rowcand.rs::bw_expected_sum).
    const u32 k = rd1<u32>();
    const uint64_t n = rd1<uint64_t>();
    const u32 seed = rd1<u32>(), grid = rd1<u32>(), threads = rd1<u32>(), pf = rd1<u32>();
    std::vector<unsigned long long> res(2, 0ull);
#if PERS_SIDE == 1
    std::vector<u32> buf(n + 4u, 0u);
    std::vector<unsigned long long> part(grid, 0ull);
    hostgrid::launch_seq(grid, threads, [&] { tv_bw_fill(buf.data(), n, seed); });
    hostgrid::launch_coop(grid, threads, PERS_SMEM_HEADER + 8u * (threads / 32u), seed, [&] {
        switch (k) {
        case 0: tv_bw_read_v1u1(buf.data(), n, part.data(), pf); break;
        case 1: tv_bw_read_v4u1(buf.data(), n, part.data(), pf); break;
        case 2: tv_bw_read_v4u4(buf.data(), n, part.data(), pf); break;
        default: std::abort();
        }
    });
    for (unsigned long long p : part) res[0] += p;
    res[1] = part.size();
#else
    (void)k;
    (void)n;
    (void)seed;
    (void)grid;
    (void)threads;
    (void)pf;
#endif
    wr(res);
}
