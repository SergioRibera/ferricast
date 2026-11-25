// NVENC session churn against the bare driver API, laid out like
// ferricast-nvcodec's encode engine: a session owns a CUDA stream (also
// NVENC's IO stream), an event, one bitstream buffer and one pitched input
// buffer registered on first use and unregistered at close. Reproduces the
// driver crash recorded in design/NVCODEC.md ("NVENC and context
// hardening"): a SIGSEGV in a libnvcuvid worker thread while a session is
// destroyed.
//
// build: gcc -O1 -g -I<nv-codec-headers>/include nvenc_churn.c -o nvenc_churn -ldl -lpthread
// run:   LD_LIBRARY_PATH=/run/opengl-driver/lib ./nvenc_churn <threads> <seconds>
// env:   FRAMES=n     pictures per session (default 1; 10 crashes sooner)
//        PFRAMES=1    only the first picture is a forced IDR
//        LOCK=1       open..bitstream and destroy under a process mutex
//        RW=1         lifecycle exclusive, pictures shared (rwlock)
//        LIFE=1       every open and close on one dedicated thread
//        STICKY=1|2|3 keep the context current (all / lifecycle / worker threads)
//        KEEP=1       open the next session before closing the previous one
//        DRAIN=bits   before destroy: 1 EOS picture, 2 sync the IO stream, 4 sync the context
//        DELAY=us     sleep before destroy
//        ANCHOR=1|2   keep an idle initialised (1) or bare (2) session open
//        CAPS=0, EARLY=1 (unregister and free right after the picture)
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "ffnvcodec/dynlink_cuda.h"
#include "ffnvcodec/nvEncodeAPI.h"

typedef NVENCSTATUS (*tCreate)(NV_ENCODE_API_FUNCTION_LIST *);
static NV_ENCODE_API_FUNCTION_LIST fl;
static tcuInit *cuInit_;
static tcuDeviceGet *cuDeviceGet_;
static tcuDevicePrimaryCtxRetain *cuRetain_;
static tcuCtxPushCurrent_v2 *cuPush_;
static tcuCtxPopCurrent_v2 *cuPop_;
static tcuStreamCreate *cuStreamCreate_;
static tcuStreamDestroy_v2 *cuStreamDestroy_;
static tcuEventCreate *cuEventCreate_;
static tcuEventDestroy_v2 *cuEventDestroy_;
static tcuEventRecord *cuEventRecord_;
static tcuEventSynchronize *cuEventSync_;
static tcuMemAllocPitch_v2 *cuMemAllocPitch_;
static tcuMemFree_v2 *cuMemFree_;
static tcuMemcpy2DAsync_v2 *cuMemcpy2DAsync_;

static CUcontext primary;
static pthread_rwlock_t rwl = PTHREAD_RWLOCK_INITIALIZER;
static int rw, pframes;
static int lock_mode, keep_mode, life_mode, frames = 1, sticky, do_caps = 1, early, drain_mode, delay_us;
static tcuStreamSynchronize *cuStreamSynchronize_;
static tcuCtxSynchronize *cuCtxSynchronize_;
static __thread int t_life;
static int sticky_here(void) { return sticky == 1 || (sticky == 2 && t_life) || (sticky == 3 && !t_life); }
static unsigned W = 1280, H = 720;
static atomic_long sessions, pictures;
static volatile int stop;
static pthread_mutex_t big = PTHREAD_MUTEX_INITIALIZER;

#define CK(x) do { CUresult r_ = (x); if (r_) { fprintf(stderr, "%s:%d cuda %d\n", __FILE__, __LINE__, r_); abort(); } } while (0)
#define NK(x) do { NVENCSTATUS s_ = (x); if (s_) { fprintf(stderr, "%s:%d nvenc %d\n", __FILE__, __LINE__, s_); abort(); } } while (0)

struct sess {
    void *enc;
    CUstream stream, *io;
    CUevent ev;
    void *bb;
    CUdeviceptr dptr;
    size_t pitch;
    void *reg;
    int frame_no;
};

static void *sym(void *lib, const char *name) {
    void *p = dlsym(lib, name);
    if (!p) { fprintf(stderr, "missing %s\n", name); exit(1); }
    return p;
}

static void open_sess(struct sess *s) {
    memset(s, 0, sizeof *s);
    if (rw) pthread_rwlock_wrlock(&rwl);
    if (!sticky_here()) CK(cuPush_(primary));
    if (lock_mode) pthread_mutex_lock(&big);
    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS op = { .version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER,
        .deviceType = NV_ENC_DEVICE_TYPE_CUDA, .device = primary, .apiVersion = NVENCAPI_VERSION };
    NK(fl.nvEncOpenEncodeSessionEx(&op, &s->enc));
    uint32_t n = 0; GUID guids[16];
    NK(fl.nvEncGetEncodeGUIDCount(s->enc, &n));
    NK(fl.nvEncGetEncodeGUIDs(s->enc, guids, 16, &n));
    int v;
    for (int cap = NV_ENC_CAPS_NUM_MAX_BFRAMES; do_caps && cap <= NV_ENC_CAPS_SUPPORT_LOOKAHEAD; cap++) {
        NV_ENC_CAPS_PARAM cp = { .version = NV_ENC_CAPS_PARAM_VER, .capsToQuery = cap };
        NK(fl.nvEncGetEncodeCaps(s->enc, NV_ENC_CODEC_H264_GUID, &cp, &v));
    }
    NV_ENC_BUFFER_FORMAT fmts[32];
    NK(fl.nvEncGetInputFormatCount(s->enc, NV_ENC_CODEC_H264_GUID, &n));
    NK(fl.nvEncGetInputFormats(s->enc, NV_ENC_CODEC_H264_GUID, fmts, 32, &n));
    NV_ENC_PRESET_CONFIG pc = { .version = NV_ENC_PRESET_CONFIG_VER, .presetCfg = { .version = NV_ENC_CONFIG_VER } };
    NK(fl.nvEncGetEncodePresetConfigEx(s->enc, NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_P3_GUID, NV_ENC_TUNING_INFO_LOW_LATENCY, &pc));
    NV_ENC_CONFIG cfg = pc.presetCfg;
    cfg.version = NV_ENC_CONFIG_VER;
    cfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
    cfg.rcParams.averageBitRate = cfg.rcParams.maxBitRate = 2000000;
    NV_ENC_INITIALIZE_PARAMS ip = { .version = NV_ENC_INITIALIZE_PARAMS_VER, .encodeGUID = NV_ENC_CODEC_H264_GUID,
        .presetGUID = NV_ENC_PRESET_P3_GUID, .encodeWidth = W, .encodeHeight = H, .darWidth = W, .darHeight = H,
        .frameRateNum = 30, .frameRateDen = 1, .enablePTD = 1, .encodeConfig = &cfg, .maxEncodeWidth = W,
        .maxEncodeHeight = H, .tuningInfo = NV_ENC_TUNING_INFO_LOW_LATENCY, .bufferFormat = NV_ENC_BUFFER_FORMAT_NV12 };
    NK(fl.nvEncInitializeEncoder(s->enc, &ip));
    CK(cuStreamCreate_(&s->stream, 1));
    CK(cuEventCreate_(&s->ev, 2));
    s->io = malloc(sizeof *s->io);
    *s->io = s->stream;
    NK(fl.nvEncSetIOCudaStreams(s->enc, (NV_ENC_CUSTREAM_PTR)s->io, (NV_ENC_CUSTREAM_PTR)s->io));
    NV_ENC_CREATE_BITSTREAM_BUFFER bb = { .version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER };
    NK(fl.nvEncCreateBitstreamBuffer(s->enc, &bb));
    s->bb = bb.bitstreamBuffer;
    if (lock_mode) pthread_mutex_unlock(&big);
    if (!sticky_here()) { CUcontext p; CK(cuPop_(&p)); }
    if (rw) pthread_rwlock_unlock(&rwl);
}

static void encode_sess(struct sess *s, unsigned char *pic) {
    if (rw) pthread_rwlock_rdlock(&rwl);
    if (!sticky_here()) CK(cuPush_(primary));
    if (!s->reg) {
        CK(cuMemAllocPitch_(&s->dptr, &s->pitch, W, H * 3 / 2, 16));
        NV_ENC_REGISTER_RESOURCE rr = { .version = NV_ENC_REGISTER_RESOURCE_VER,
            .resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR, .width = W, .height = H, .pitch = s->pitch,
            .resourceToRegister = (void *)s->dptr, .bufferFormat = NV_ENC_BUFFER_FORMAT_NV12, .bufferUsage = NV_ENC_INPUT_IMAGE };
        NK(fl.nvEncRegisterResource(s->enc, &rr));
        s->reg = rr.registeredResource;
    }
    CUDA_MEMCPY2D c = { .srcMemoryType = CU_MEMORYTYPE_HOST, .srcHost = pic, .srcPitch = W,
        .dstMemoryType = CU_MEMORYTYPE_DEVICE, .dstDevice = s->dptr, .dstPitch = s->pitch, .WidthInBytes = W, .Height = H * 3 / 2 };
    CK(cuMemcpy2DAsync_(&c, s->stream));
    CK(cuEventRecord_(s->ev, s->stream));
    CK(cuEventSync_(s->ev));
    NV_ENC_MAP_INPUT_RESOURCE mr = { .version = NV_ENC_MAP_INPUT_RESOURCE_VER, .registeredResource = s->reg };
    NK(fl.nvEncMapInputResource(s->enc, &mr));
    NV_ENC_PIC_PARAMS pp = { .version = NV_ENC_PIC_PARAMS_VER, .inputWidth = W, .inputHeight = H, .inputPitch = s->pitch,
        .encodePicFlags = (pframes && s->frame_no++ > 0) ? 0 : (NV_ENC_PIC_FLAG_FORCEIDR | NV_ENC_PIC_FLAG_OUTPUT_SPSPPS), .inputBuffer = mr.mappedResource,
        .outputBitstream = s->bb, .bufferFmt = mr.mappedBufferFmt, .pictureStruct = NV_ENC_PIC_STRUCT_FRAME };
    NK(fl.nvEncEncodePicture(s->enc, &pp));
    NV_ENC_LOCK_BITSTREAM lb = { .version = NV_ENC_LOCK_BITSTREAM_VER, .outputBitstream = s->bb };
    NK(fl.nvEncLockBitstream(s->enc, &lb));
    NK(fl.nvEncUnlockBitstream(s->enc, s->bb));
    NK(fl.nvEncUnmapInputResource(s->enc, mr.mappedResource));
    if (early) { NK(fl.nvEncUnregisterResource(s->enc, s->reg)); s->reg = NULL; CK(cuMemFree_(s->dptr)); s->dptr = 0; }
    if (!sticky_here()) { CUcontext p; CK(cuPop_(&p)); }
    atomic_fetch_add(&pictures, 1);
    if (rw) pthread_rwlock_unlock(&rwl);
}

static void close_sess(struct sess *s) {
    if (rw) pthread_rwlock_wrlock(&rwl);
    if (!sticky_here()) CK(cuPush_(primary));
    if (s->reg) NK(fl.nvEncUnregisterResource(s->enc, s->reg));
    if (drain_mode & 1) { NV_ENC_PIC_PARAMS eos = { .version = NV_ENC_PIC_PARAMS_VER, .encodePicFlags = NV_ENC_PIC_FLAG_EOS }; fl.nvEncEncodePicture(s->enc, &eos); }
    if (drain_mode & 2) CK(cuStreamSynchronize_(s->stream));
    if (drain_mode & 4) CK(cuCtxSynchronize_());
    if (delay_us) usleep(delay_us);
    if (lock_mode) pthread_mutex_lock(&big);
    NK(fl.nvEncDestroyBitstreamBuffer(s->enc, s->bb));
    NK(fl.nvEncDestroyEncoder(s->enc));
    if (lock_mode) pthread_mutex_unlock(&big);
    CK(cuStreamDestroy_(s->stream));
    CK(cuEventDestroy_(s->ev));
    if (s->dptr) CK(cuMemFree_(s->dptr));
    free(s->io);
    if (!sticky_here()) { CUcontext p; CK(cuPop_(&p)); }
    atomic_fetch_add(&sessions, 1);
    if (rw) pthread_rwlock_unlock(&rwl);
}

// The dedicated lifecycle thread (LIFE=1): one request at a time.
static pthread_mutex_t req_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t req_cond = PTHREAD_COND_INITIALIZER, done_cond = PTHREAD_COND_INITIALIZER;
static struct sess *req_sess;
static int req_kind, req_done; // 1 open, 2 close

static void life_call(int kind, struct sess *s) {
    pthread_mutex_lock(&req_lock);
    while (req_kind) pthread_cond_wait(&done_cond, &req_lock);
    req_kind = kind; req_sess = s; req_done = 0;
    pthread_cond_broadcast(&req_cond);
    while (!req_done || req_sess != s) pthread_cond_wait(&done_cond, &req_lock);
    req_kind = 0; req_sess = NULL; req_done = 0;
    pthread_cond_broadcast(&done_cond);
    pthread_mutex_unlock(&req_lock);
}

static void *life_thread(void *arg) {
    (void)arg;
    t_life = 1; if (sticky == 1 || sticky == 2) CK(cuPush_(primary));
    pthread_mutex_lock(&req_lock);
    for (;;) {
        while (!req_kind || req_done) {
            if (stop == 2) { pthread_mutex_unlock(&req_lock); return NULL; }
            pthread_cond_wait(&req_cond, &req_lock);
        }
        int kind = req_kind; struct sess *s = req_sess;
        pthread_mutex_unlock(&req_lock);
        if (kind == 1) open_sess(s); else close_sess(s);
        pthread_mutex_lock(&req_lock);
        req_done = 1;
        pthread_cond_broadcast(&done_cond);
    }
}

static void do_open(struct sess *s) { if (life_mode) life_call(1, s); else open_sess(s); }
static void do_close(struct sess *s) { if (life_mode) life_call(2, s); else close_sess(s); }

static void *worker(void *arg) {
    (void)arg;
    if (sticky == 1 || sticky == 3) CK(cuPush_(primary));
    unsigned char *pic = malloc(W * H * 3 / 2);
    memset(pic, 77, W * H * 3 / 2);
    struct sess a, b, *cur = &a, *next = &b;
    if (keep_mode) do_open(cur);
    while (!stop) {
        if (keep_mode) {
            do_open(next);
            do_close(cur);
            struct sess *t = cur; cur = next; next = t;
        } else {
            do_open(cur);
        }
        for (int i = 0; i < frames; i++) encode_sess(cur, pic);
        if (!keep_mode) do_close(cur);
    }
    if (keep_mode) do_close(cur);
    free(pic);
    return NULL;
}

static int env_int(const char *name, int def) { const char *v = getenv(name); return v ? atoi(v) : def; }

int main(int argc, char **argv) {
    int threads = argc > 1 ? atoi(argv[1]) : 2, secs = argc > 2 ? atoi(argv[2]) : 60;
    lock_mode = env_int("LOCK", 0); keep_mode = env_int("KEEP", 0); life_mode = env_int("LIFE", 0);
    frames = env_int("FRAMES", 1); sticky = env_int("STICKY", 0); do_caps = env_int("CAPS", 1); early = env_int("EARLY", 0); drain_mode = env_int("DRAIN", 0); rw = env_int("RW", 0); pframes = env_int("PFRAMES", 0); delay_us = env_int("DELAY", 0);
    void *cu = dlopen("libcuda.so.1", RTLD_NOW), *ne = dlopen("libnvidia-encode.so.1", RTLD_NOW);
    if (!cu || !ne) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    cuInit_ = sym(cu, "cuInit"); cuDeviceGet_ = sym(cu, "cuDeviceGet");
    cuRetain_ = sym(cu, "cuDevicePrimaryCtxRetain");
    cuPush_ = sym(cu, "cuCtxPushCurrent_v2"); cuPop_ = sym(cu, "cuCtxPopCurrent_v2");
    cuStreamCreate_ = sym(cu, "cuStreamCreate"); cuStreamDestroy_ = sym(cu, "cuStreamDestroy_v2");
    cuEventCreate_ = sym(cu, "cuEventCreate"); cuEventDestroy_ = sym(cu, "cuEventDestroy_v2");
    cuEventRecord_ = sym(cu, "cuEventRecord"); cuEventSync_ = sym(cu, "cuEventSynchronize");
    cuMemAllocPitch_ = sym(cu, "cuMemAllocPitch_v2"); cuMemFree_ = sym(cu, "cuMemFree_v2");
    cuMemcpy2DAsync_ = sym(cu, "cuMemcpy2DAsync_v2");
    cuStreamSynchronize_ = sym(cu, "cuStreamSynchronize"); cuCtxSynchronize_ = sym(cu, "cuCtxSynchronize");
    tCreate create_instance = sym(ne, "NvEncodeAPICreateInstance");
    CUdevice dev;
    CK(cuInit_(0)); CK(cuDeviceGet_(&dev, 0)); CK(cuRetain_(&primary, dev));
    fl.version = NV_ENCODE_API_FUNCTION_LIST_VER;
    NK(create_instance(&fl));
    static struct sess anchor;
    if (getenv("ANCHOR") && atoi(getenv("ANCHOR")) == 1) { open_sess(&anchor); }
    if (getenv("ANCHOR") && atoi(getenv("ANCHOR")) == 2) { CK(cuPush_(primary)); NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS op = { .version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER, .deviceType = NV_ENC_DEVICE_TYPE_CUDA, .device = primary, .apiVersion = NVENCAPI_VERSION }; NK(fl.nvEncOpenEncodeSessionEx(&op, &anchor.enc)); CUcontext p; CK(cuPop_(&p)); }
    pthread_t t[64], lt;
    if (life_mode) pthread_create(&lt, NULL, life_thread, NULL);
    for (int i = 0; i < threads; i++) pthread_create(&t[i], NULL, worker, NULL);
    for (int s = 0; s < secs; s++) {
        sleep(1);
        if ((s + 1) % 30 == 0) { printf("t=%ds sessions=%ld pictures=%ld\n", s + 1, atomic_load(&sessions), atomic_load(&pictures)); fflush(stdout); }
    }
    stop = 1;
    for (int i = 0; i < threads; i++) pthread_join(t[i], NULL);
    if (life_mode) {
        pthread_mutex_lock(&req_lock); stop = 2; pthread_cond_broadcast(&req_cond); pthread_mutex_unlock(&req_lock);
        pthread_join(lt, NULL);
    }
    printf("done sessions=%ld pictures=%ld\n", atomic_load(&sessions), atomic_load(&pictures));
    return 0;
}
