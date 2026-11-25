// GPU image kernels of ferricast-nvcodec: scale, crop, rotate / flip,
// letterbox, RGB -> NV12 and 16 -> 8 bit narrowing, in one pass per plane.
//
// Compiled to PTX by `scripts/kernels.sh`; the PTX next to this file is
// committed and JIT-compiled by the driver at run time, so building the
// crate needs no CUDA toolkit. Every entry point takes one parameter struct
// by value. `src/kernels/params.rs` mirrors each struct with `#[repr(C)]`,
// and a test checks the sizes against the `.param` declarations in the PTX.
//
// Geometry, per output plane:
// - the canvas (`canvas_w` x `canvas_h`) is the whole output plane;
// - the picture lands in the placement rectangle; the rest is filled;
// - placement coordinates map to the oriented picture (the cropped source
//   after rotation / flip, `oriented_w` x `oriented_h`) by the resampling
//   filter; every tap of the oriented picture maps to a source pixel by the
//   integer affine map `x = xx * X + xy * Y + x0`, `y = yx * X + yy * Y + y0`
//   (crop origin folded into `x0` / `y0`).
//
// The filters are fast_image_resize's (what ferricast's CPU scaler uses)
// with its adaptive support: a triangle (bilinear) or box kernel stretched
// by the downscale ratio, so a downscale averages every source pixel it
// covers instead of aliasing. The host checks that every tap lands inside
// the source and every write inside the destination before a launch.

#include <stdint.h>

typedef unsigned long long u64;

struct Geometry {
    int32_t canvas_w, canvas_h;
    int32_t place_x, place_y, place_w, place_h;
    int32_t oriented_w, oriented_h;
    int32_t xx, xy, x0;
    int32_t yx, yy, y0;
    float ratio_x, ratio_y;
    // 0: triangle (bilinear), 1: box (area).
    int32_t filter;
    int32_t reserved;
};
static_assert(sizeof(Geometry) == 72, "Geometry layout");

// A texture object (CUDA arrays) or a device pointer, and its pitch.
struct Source {
    u64 handle;
    u64 pitch;
};

struct Plane {
    u64 ptr;
    u64 pitch;
};

struct RgbParams {
    Geometry g;
    Source src;
    Plane dst;
    // Swap the first and third byte (BGRA <-> RGBA).
    int32_t swap_rb;
    int32_t reserved;
};
static_assert(sizeof(RgbParams) == 112, "RgbParams layout");

struct Nv12Params {
    Geometry g;
    Source src;
    Plane luma;
    Plane chroma;
    // yr, yg, yb, cb_r, cb_g, cb_b, cr_r, cr_g, cr_b at 13 fractional bits.
    int32_t coeff[9];
    int32_t bias_y, bias_uv;
    // Source bytes are R, G, B, A (else B, G, R, A).
    int32_t src_rgba;
    int32_t reserved[2];
};
static_assert(sizeof(Nv12Params) == 176, "Nv12Params layout");

struct PlaneParams {
    Geometry g;
    Plane src;
    Plane dst;
    // out = min(floor(v * out_scale + 0.5), out_max) << out_shift
    float out_scale;
    int32_t out_max;
    int32_t out_shift;
    // Letterbox value of channel 0 and 1, in output units.
    uint32_t fill0, fill1;
    int32_t reserved;
};
static_assert(sizeof(PlaneParams) == 128, "PlaneParams layout");

// Fixed point of the RGB -> YCbCr coefficients (the `yuv` crate's
// "balanced" precision, which ferricast's CPU conversion uses).
#define CSC_PRECISION 13

__device__ __forceinline__ float2 operator+(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ float2 operator*(float s, float2 a) { return make_float2(s * a.x, s * a.y); }
__device__ __forceinline__ float4 operator+(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__device__ __forceinline__ float4 operator*(float s, float4 a) {
    return make_float4(s * a.x, s * a.y, s * a.z, s * a.w);
}

__device__ __forceinline__ float filter_weight(int filter, float t) {
    if (filter == 0) {
        t = fabsf(t);
        return t < 1.0f ? 1.0f - t : 0.0f;
    }
    return (t > -0.5f && t <= 0.5f) ? 1.0f : 0.0f;
}

// The taps of one output coordinate along one axis.
struct Axis {
    int lo, hi;
    float center, inv_scale, norm;
};

__device__ __forceinline__ Axis axis(int out, float ratio, int size, int filter) {
    Axis a;
    if (ratio == 1.0f) {
        // Same size: the one source sample, bit-exact.
        a.lo = out;
        a.hi = out + 1;
        a.center = (float)out;
        a.inv_scale = 1.0f;
        a.norm = 1.0f;
        return a;
    }
    float in_center = ((float)out + 0.5f) * ratio;
    float scale = fmaxf(ratio, 1.0f);
    float radius = (filter == 0 ? 1.0f : 0.5f) * scale;
    a.lo = max((int)floorf(in_center - radius), 0);
    a.hi = min((int)ceilf(in_center + radius), size);
    a.center = in_center - 0.5f;
    a.inv_scale = 1.0f / scale;
    float sum = 0.0f;
    #pragma unroll 1
    for (int i = a.lo; i < a.hi; ++i) {
        sum += filter_weight(filter, ((float)i - a.center) * a.inv_scale);
    }
    if (sum > 0.0f) {
        a.norm = 1.0f / sum;
    } else {
        // No tap with weight (cannot happen for in-range centres): nearest.
        a.lo = min(max((int)floorf(in_center), 0), size - 1);
        a.hi = a.lo + 1;
        a.center = (float)a.lo;
        a.inv_scale = 1.0f;
        a.norm = 1.0f;
    }
    return a;
}

// The resampled value of placement pixel (u, v).
template <class S>
__device__ __forceinline__ typename S::Value sample(const S& s, const Geometry& g, int u, int v) {
    Axis ax = axis(u, g.ratio_x, g.oriented_w, g.filter);
    Axis ay = axis(v, g.ratio_y, g.oriented_h, g.filter);
    typename S::Value acc = S::zero();
    #pragma unroll 1
    for (int Y = ay.lo; Y < ay.hi; ++Y) {
        float wy = filter_weight(g.filter, ((float)Y - ay.center) * ay.inv_scale);
        if (wy == 0.0f) {
            continue;
        }
        typename S::Value row = S::zero();
        #pragma unroll 1
        for (int X = ax.lo; X < ax.hi; ++X) {
            float wx = filter_weight(g.filter, ((float)X - ax.center) * ax.inv_scale);
            if (wx == 0.0f) {
                continue;
            }
            int sx = g.xx * X + g.xy * Y + g.x0;
            int sy = g.yx * X + g.yy * Y + g.y0;
            row = row + wx * s.at(sx, sy);
        }
        acc = acc + wy * row;
    }
    return (ax.norm * ay.norm) * acc;
}

// Round half up, like integer `(v + half) >> shift`.
__device__ __forceinline__ int round_half_up(float v) { return (int)floorf(v + 0.5f); }

__device__ __forceinline__ unsigned char to_u8(float v) {
    return (unsigned char)min(max(round_half_up(v), 0), 255);
}

// Packed 4 x 8-bit pixels in a CUDA array, read through a texture object
// (point filtering, element read mode, unnormalised coordinates).
struct ArrayRgba {
    typedef float4 Value;
    cudaTextureObject_t tex;
    __device__ static float4 zero() { return make_float4(0.0f, 0.0f, 0.0f, 0.0f); }
    __device__ float4 at(int x, int y) const {
        uchar4 p = tex2D<uchar4>(tex, (float)x + 0.5f, (float)y + 0.5f);
        return make_float4(p.x, p.y, p.z, p.w);
    }
};

// Packed 4 x 8-bit pixels in pitched device memory.
struct PitchedRgba {
    typedef float4 Value;
    const unsigned char* base;
    size_t pitch;
    __device__ static float4 zero() { return make_float4(0.0f, 0.0f, 0.0f, 0.0f); }
    __device__ float4 at(int x, int y) const {
        const uchar4* row = reinterpret_cast<const uchar4*>(base + (size_t)y * pitch);
        uchar4 p = __ldg(row + x);
        return make_float4(p.x, p.y, p.z, p.w);
    }
};

// One plane of `C` interleaved samples of type `T` (8 or 16 bit).
template <class T, int C>
struct PitchedSamples {
    typedef float2 Value;
    const unsigned char* base;
    size_t pitch;
    __device__ static float2 zero() { return make_float2(0.0f, 0.0f); }
    __device__ float2 at(int x, int y) const {
        const T* row = reinterpret_cast<const T*>(base + (size_t)y * pitch);
        if (C == 1) {
            return make_float2((float)__ldg(row + x), 0.0f);
        }
        return make_float2((float)__ldg(row + 2 * x), (float)__ldg(row + 2 * x + 1));
    }
};

__device__ __forceinline__ bool in_placement(const Geometry& g, int x, int y, int* u, int* v) {
    *u = x - g.place_x;
    *v = y - g.place_y;
    return *u >= 0 && *v >= 0 && *u < g.place_w && *v < g.place_h;
}

template <class S>
__device__ __forceinline__ void rgb_to_rgb(const RgbParams& p, const S& s) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.g.canvas_w || y >= p.g.canvas_h) {
        return;
    }
    uchar4 out;
    int u, v;
    if (in_placement(p.g, x, y, &u, &v)) {
        float4 c = sample(s, p.g, u, v);
        out = make_uchar4(to_u8(c.x), to_u8(c.y), to_u8(c.z), to_u8(c.w));
        if (p.swap_rb) {
            unsigned char t = out.x;
            out.x = out.z;
            out.z = t;
        }
    } else {
        // Opaque black in both byte orders.
        out = make_uchar4(0, 0, 0, 255);
    }
    uchar4* row = reinterpret_cast<uchar4*>(p.dst.ptr + (u64)y * p.dst.pitch);
    row[x] = out;
}

// One 8-bit R, G, B pixel of the output picture; black outside the
// placement.
template <class S>
__device__ __forceinline__ int3 rgb_pixel(const Nv12Params& p, const S& s, int x, int y) {
    int u, v;
    if (!in_placement(p.g, x, y, &u, &v)) {
        return make_int3(0, 0, 0);
    }
    float4 c = sample(s, p.g, u, v);
    int first = to_u8(c.x), second = to_u8(c.y), third = to_u8(c.z);
    return p.src_rgba ? make_int3(first, second, third) : make_int3(third, second, first);
}

// One thread per 2 x 2 block: four luma samples and the chroma pair from
// the rounded average of the block's RGB (the `yuv` crate's 4:2:0 rule,
// with the pixels that exist at an odd right or bottom edge).
template <class S>
__device__ __forceinline__ void rgb_to_nv12(const Nv12Params& p, const S& s) {
    int bx = blockIdx.x * blockDim.x + threadIdx.x;
    int by = blockIdx.y * blockDim.y + threadIdx.y;
    int x0 = 2 * bx, y0 = 2 * by;
    if (x0 >= p.g.canvas_w || y0 >= p.g.canvas_h) {
        return;
    }
    const int* k = p.coeff;
    int sr = 0, sg = 0, sb = 0, n = 0;
    for (int dy = 0; dy < 2; ++dy) {
        int y = y0 + dy;
        if (y >= p.g.canvas_h) {
            break;
        }
        unsigned char* luma = reinterpret_cast<unsigned char*>(p.luma.ptr + (u64)y * p.luma.pitch);
        for (int dx = 0; dx < 2; ++dx) {
            int x = x0 + dx;
            if (x >= p.g.canvas_w) {
                break;
            }
            int3 c = rgb_pixel(p, s, x, y);
            int luma_value = (c.x * k[0] + c.y * k[1] + c.z * k[2] + p.bias_y) >> CSC_PRECISION;
            luma[x] = (unsigned char)min(max(luma_value, 0), 255);
            sr += c.x;
            sg += c.y;
            sb += c.z;
            ++n;
        }
    }
    int r = (sr + n / 2) / n, g = (sg + n / 2) / n, b = (sb + n / 2) / n;
    int cb = (r * k[3] + g * k[4] + b * k[5] + p.bias_uv) >> CSC_PRECISION;
    int cr = (r * k[6] + g * k[7] + b * k[8] + p.bias_uv) >> CSC_PRECISION;
    unsigned char* chroma = reinterpret_cast<unsigned char*>(p.chroma.ptr + (u64)by * p.chroma.pitch);
    chroma[2 * bx] = (unsigned char)min(max(cb, 0), 255);
    chroma[2 * bx + 1] = (unsigned char)min(max(cr, 0), 255);
}

template <class TIn, class TOut, int C>
__device__ __forceinline__ void plane(const PlaneParams& p) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p.g.canvas_w || y >= p.g.canvas_h) {
        return;
    }
    unsigned int out0 = p.fill0, out1 = p.fill1;
    int u, v;
    if (in_placement(p.g, x, y, &u, &v)) {
        PitchedSamples<TIn, C> s;
        s.base = reinterpret_cast<const unsigned char*>(p.src.ptr);
        s.pitch = (size_t)p.src.pitch;
        float2 c = sample(s, p.g, u, v);
        out0 = (unsigned int)min(max(round_half_up(c.x * p.out_scale), 0), p.out_max) << p.out_shift;
        out1 = (unsigned int)min(max(round_half_up(c.y * p.out_scale), 0), p.out_max) << p.out_shift;
    }
    TOut* row = reinterpret_cast<TOut*>(p.dst.ptr + (u64)y * p.dst.pitch);
    if (C == 1) {
        row[x] = (TOut)out0;
    } else {
        row[2 * x] = (TOut)out0;
        row[2 * x + 1] = (TOut)out1;
    }
}

#define BLOCK 256

extern "C" __global__ void __launch_bounds__(BLOCK) ferricast_rgb_to_rgb_array(const RgbParams p) {
    ArrayRgba s;
    s.tex = (cudaTextureObject_t)p.src.handle;
    rgb_to_rgb(p, s);
}

extern "C" __global__ void __launch_bounds__(BLOCK) ferricast_rgb_to_rgb_pitched(const RgbParams p) {
    PitchedRgba s;
    s.base = reinterpret_cast<const unsigned char*>(p.src.handle);
    s.pitch = (size_t)p.src.pitch;
    rgb_to_rgb(p, s);
}

extern "C" __global__ void __launch_bounds__(BLOCK) ferricast_rgb_to_nv12_array(const Nv12Params p) {
    ArrayRgba s;
    s.tex = (cudaTextureObject_t)p.src.handle;
    rgb_to_nv12(p, s);
}

extern "C" __global__ void __launch_bounds__(BLOCK) ferricast_rgb_to_nv12_pitched(const Nv12Params p) {
    PitchedRgba s;
    s.base = reinterpret_cast<const unsigned char*>(p.src.handle);
    s.pitch = (size_t)p.src.pitch;
    rgb_to_nv12(p, s);
}

extern "C" __global__ void __launch_bounds__(BLOCK) ferricast_plane_u8_u8_c1(const PlaneParams p) {
    plane<unsigned char, unsigned char, 1>(p);
}

extern "C" __global__ void __launch_bounds__(BLOCK) ferricast_plane_u8_u8_c2(const PlaneParams p) {
    plane<unsigned char, unsigned char, 2>(p);
}

extern "C" __global__ void __launch_bounds__(BLOCK) ferricast_plane_u16_u16_c1(const PlaneParams p) {
    plane<unsigned short, unsigned short, 1>(p);
}

extern "C" __global__ void __launch_bounds__(BLOCK) ferricast_plane_u16_u16_c2(const PlaneParams p) {
    plane<unsigned short, unsigned short, 2>(p);
}

extern "C" __global__ void __launch_bounds__(BLOCK) ferricast_plane_u16_u8_c1(const PlaneParams p) {
    plane<unsigned short, unsigned char, 1>(p);
}

extern "C" __global__ void __launch_bounds__(BLOCK) ferricast_plane_u16_u8_c2(const PlaneParams p) {
    plane<unsigned short, unsigned char, 2>(p);
}
