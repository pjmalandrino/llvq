"""flock(1) for a Mac without util-linux: take an exclusive flock(2) on the
file, run the command, release. Same lock as `flock <file> <cmd>`."""
import fcntl, subprocess, sys, time
with open(sys.argv[1], "a") as f:
    t = time.time()
    fcntl.flock(f, fcntl.LOCK_EX)
    print(f"[lock {sys.argv[1]} held after {time.time() - t:.1f} s]", file=sys.stderr, flush=True)
    sys.exit(subprocess.call(sys.argv[2:]))
