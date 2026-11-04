// spin_ns: one thread holds its stream for `ns` nanoseconds of %globaltimer.
// Measurement only (LLVQ_GESTURE_AB), never on a served path: the host enqueues
// a segment of the decode step behind it while the device is busy, so the
// host's enqueue span and the device's work can be read apart.

extern "C" __global__ void spin_ns(unsigned long long ns)
{
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        __nanosleep(1000);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}
