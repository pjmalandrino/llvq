The *.logits.f32 files (25.6 MB each), the cache states under work/ (1.41 GB each) and the stores under
serve13/home, serve13b/home, prefix/home and prefix/home-off (prefix files of 1.41 GB) stay out of git; logits.sha256 pins the
logits. Each store is the throwaway LLVQ_HOME of a test server.
