// plan-300 step 1, included inside main()'s command chain of host_pers.cpp
// ('y' and 'c'). `tag` is in scope.
#ifndef CTR_GROUP
#define CTR_GROUP 8u  // pers_ctr.cu's, for the served side, which does not compile it
#define CTR_LINE 32u
#endif
if (tag == 'y') {
    // INT4 ROWS ALONE (pers_q4rows.cu): arm (0 Q4, 1 NC, 2 TILE, 3 READ), grid, threads, stamps, x seed,
    // nsites, then per site d_out, d_in, seed. Every site's rows, binary16, then their f32 values before
    // f2h. The served side runs tv_q4_hh on each site, as 'Q' generates it, and ignores the arm.
    const u32 arm = rd1<u32>(), grid = rd1<u32>(), threads = rd1<u32>(), stamp = rd1<u32>();
    const uint64_t xseed = rd1<uint64_t>();
    const u32 nsites = rd1<u32>();
    struct Q4M {
        u32 d_out, d_in;
        std::vector<u32> wq;
        std::vector<unsigned short> sc, bi;
    };
    std::vector<Q4M> mats;
    u32 maxd = 0u;
    for (u32 s = 0; s < nsites; ++s) {
        Q4M m;
        m.d_out = rd1<u32>();
        m.d_in = rd1<u32>();
        Rng r(rd1<uint64_t>());
        const u32 gpr = m.d_in / 128u;
        m.wq.resize((size_t)m.d_out * m.d_in / 8);
        for (auto& w : m.wq) w = (u32)r.next();
        m.sc.resize((size_t)m.d_out * gpr);
        m.bi.resize((size_t)m.d_out * gpr);
        for (auto& v : m.sc) v = r.h16(6, 10);
        for (auto& v : m.bi) v = r.h16(8, 12);
        maxd = std::max(maxd, m.d_in);
        mats.push_back(std::move(m));
    }
    const auto xh = act_h(maxd, xseed);
    std::vector<unsigned short> out;
    std::vector<float> outf;
#if PERS_SIDE == 0
    (void)arm;
    (void)grid;
    (void)threads;
    (void)stamp;
    for (const Q4M& m : mats) {
        std::vector<unsigned short> y(m.d_out, 0xFFFFu);
        std::vector<float> f(m.d_out, 0.0f);
        g_capture = f.data();
        hostgrid::launch_seq(m.d_out / 8, 256, [&] {
            tv_q4_hh(m.wq.data(), m.sc.data(), m.bi.data(), xh.data(), y.data(), m.d_in, m.d_in / 128u);
        });
        g_capture = nullptr;
        out.insert(out.end(), y.begin(), y.end());
        outf.insert(outf.end(), f.begin(), f.end());
    }
#else
    std::vector<std::vector<unsigned short>> ys;
    std::vector<std::vector<float>> fs;
    std::vector<std::vector<u32>> tiled(mats.size());
    std::vector<unsigned long long> words;
    ys.reserve(mats.size());
    fs.reserve(mats.size());
    for (size_t k = 0; k < mats.size(); ++k) {
        const Q4M& m = mats[k];
        ys.emplace_back(m.d_out, 0xFFFFu);
        fs.emplace_back(m.d_out, -1.0f);
        const u32* wsrc = m.wq.data();
        if (arm == Q4ROWS_TILE) {
            tiled[k].assign(m.wq.size(), 0u);
            hostgrid::launch_seq(3, 64, [&] { tv_q4rows_tile_relayout(m.wq.data(), tiled[k].data(), m.d_out, m.d_in / 8u); });
            wsrc = tiled[k].data();
        }
        const unsigned long long w[Q4ROWS_SITE_WORDS] = {
            (unsigned long long)(uintptr_t)wsrc, (unsigned long long)(uintptr_t)m.sc.data(),
            (unsigned long long)(uintptr_t)m.bi.data(), (unsigned long long)(uintptr_t)ys.back().data(),
            (unsigned long long)m.d_out | ((unsigned long long)m.d_in << 32), (unsigned long long)(uintptr_t)fs.back().data()};
        words.insert(words.end(), w, w + Q4ROWS_SITE_WORDS);
    }
    std::vector<u32> st((size_t)nsites * grid * (threads / 32u + 1u), 0u);
    std::vector<unsigned long long> ends(4u * grid, 0ull);
    u32* sp = stamp ? st.data() : nullptr;
    unsigned long long* ep = stamp ? ends.data() : nullptr;
    hostgrid::launch_coop(grid, threads, PERS_SMEM_HEADER + 4u * maxd + 16u, xseed, [&] {
        switch (arm) {
        case 0: q4rows_body<Q4ROWS_Q4>(words.data(), nsites, xh.data(), sp, ep); break;
        case 1: q4rows_body<Q4ROWS_NC>(words.data(), nsites, xh.data(), sp, ep); break;
        case 2: q4rows_body<Q4ROWS_TILE>(words.data(), nsites, xh.data(), sp, ep); break;
        case 3: q4rows_body<Q4ROWS_READ>(words.data(), nsites, xh.data(), sp, ep); break;
        default: std::abort();
        }
    });
    if (stamp)
        for (u32 c = 0; c < grid; ++c)
            if (ends[4u * c + 2u] <= ends[4u * c] || ends[4u * c + 3u] <= ends[4u * c + 1u]) {
                std::fprintf(stderr, "host_pers: CTA %u: launch ends out of order\n", c);
                return 2;
            }
    for (const auto& y : ys) out.insert(out.end(), y.begin(), y.end());
    for (const auto& f : fs) outf.insert(outf.end(), f.begin(), f.end());
#endif
    wr(out);
    wr(outf);
} else {
    // COUNTERS (pers_ctr.cu): arm (pers.rs::CTR_ARMS order), grid, threads, n hops, work cycles. Two launches
    // on the same counters, the second with base n. Writes the head of every counter line (one, or one a
    // group of 8 for ctr8), then the abort flag.
    const u32 arm = rd1<u32>(), grid = rd1<u32>(), threads = rd1<u32>(), n = rd1<u32>(), work = rd1<u32>();
    const u32 ng = (grid + CTR_GROUP - 1u) / CTR_GROUP;
    std::vector<u32> ctr((size_t)ng * CTR_LINE, 0u), abort_flag(1, 0u);
#if PERS_SIDE == 1
    for (u32 launch = 0; launch < 2u; ++launch) {
        const u32 base = launch * n;
        hostgrid::launch_coop(grid, threads, PERS_SMEM_HEADER, 0x5eedull + launch, [&] {
            switch (arm) {
            case 0: tv_ctr_bar(ctr.data(), base, n, work, abort_flag.data()); break;
            case 1: tv_ctr_tight(ctr.data(), base, n, work, abort_flag.data()); break;
            case 2: tv_ctr_rel(ctr.data(), base, n, work, abort_flag.data()); break;
            case 3: tv_ctr_8(ctr.data(), base, n, work, abort_flag.data()); break;
            case 4: tv_ctr_split(ctr.data(), base, n, work, abort_flag.data()); break;
            case 5: tv_ctr_work(ctr.data(), base, n, work, abort_flag.data()); break;
            default: std::abort();
            }
        });
    }
#else
    (void)arm;
    (void)threads;
    (void)n;
    (void)work;
#endif
    std::vector<u32> heads(arm == 3u ? ng : 1u);
    for (size_t p = 0; p < heads.size(); ++p) heads[p] = ctr[p * CTR_LINE];
    wr(heads);
    wr(abort_flag);
}
