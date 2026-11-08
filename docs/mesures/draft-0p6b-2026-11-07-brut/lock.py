import fcntl, subprocess, sys
with open("/tmp/llvq-mac-bench.lock", "a") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    sys.exit(subprocess.call(sys.argv[1:]))
