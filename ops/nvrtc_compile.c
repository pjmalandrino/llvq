/* Compile one NVRTC unit the way llvq-cuda/src/gpu.rs does, with no card:
 * the parts joined by "\n", only --gpu-architecture=<arch> passed. Writes the
 * PTX to argv[2] and the compile log to stderr; exit 1 on a refused unit.
 *
 *   nvrtc_compile <arch> <out.ptx> <part> [<part> ...]
 *
 * A part that starts with "#define" or "//" is taken literally, anything else
 * is a file path. Built and run inside the llvq-check image by
 * ops/nvrtc-check.sh. */
#include <nvrtc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char* slurp(const char* path, size_t* n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* b = malloc((size_t)len + 1);
    if (fread(b, 1, (size_t)len, f) != (size_t)len) { fprintf(stderr, "short read %s\n", path); exit(2); }
    fclose(f);
    b[len] = 0;
    *n = (size_t)len;
    return b;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: nvrtc_compile <arch> <out.ptx> <part>...\n"); return 2; }
    size_t cap = 1 << 20, len = 0;
    char* src = malloc(cap);
    for (int i = 3; i < argc; ++i) {
        size_t n;
        char* p;
        if (strncmp(argv[i], "#define", 7) == 0 || strncmp(argv[i], "//", 2) == 0) {
            n = strlen(argv[i]);
            p = strdup(argv[i]);
        } else {
            p = slurp(argv[i], &n);
        }
        while (len + n + 2 > cap) { cap *= 2; src = realloc(src, cap); }
        if (i > 3) src[len++] = '\n';
        memcpy(src + len, p, n);
        len += n;
        free(p);
    }
    src[len] = 0;
    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, src, "unit.cu", 0, NULL, NULL) != NVRTC_SUCCESS) return 1;
    char opt[64];
    snprintf(opt, sizeof opt, "--gpu-architecture=%s", argv[1]);
    const char* opts[] = {opt};
    nvrtcResult r = nvrtcCompileProgram(prog, 1, opts);
    size_t ln;
    nvrtcGetProgramLogSize(prog, &ln);
    char* log = malloc(ln + 1);
    nvrtcGetProgramLog(prog, log);
    if (ln > 1) fprintf(stderr, "%s\n", log);
    if (r != NVRTC_SUCCESS) { fprintf(stderr, "NVRTC refused the unit: %s\n", nvrtcGetErrorString(r)); return 1; }
    size_t pn;
    nvrtcGetPTXSize(prog, &pn);
    char* ptx = malloc(pn);
    nvrtcGetPTX(prog, ptx);
    FILE* o = fopen(argv[2], "wb");
    fwrite(ptx, 1, pn - 1, o);
    fclose(o);
    fprintf(stderr, "unit: %zu bytes, %d parts, %s: PTX %zu bytes\n", len, argc - 3, argv[1], pn - 1);
    return 0;
}
