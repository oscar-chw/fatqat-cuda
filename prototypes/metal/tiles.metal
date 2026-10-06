// IEEE 754 binary64 multiply and add in software, for GPUs without FP64.
// Round to nearest, ties to even; subnormal inputs and results; signed zeros;
// infinities. Every non-NaN result is bit-identical to the CPU's. A NaN
// operand comes back quieted and an invalid operation gives the ARM default
// NaN (0x7FF8...); which NaN payload wins when both operands are NaN is not
// matched. Values are ulong bit patterns.
#include <metal_stdlib>
using namespace metal;

constant ulong SIGN = 0x8000000000000000ul;
constant ulong EXP_MASK = 0x7FF0000000000000ul;
constant ulong FRAC_MASK = 0x000FFFFFFFFFFFFFul;
constant ulong QUIET_NAN = 0x7FF8000000000000ul;

static inline bool is_nan(ulong a) { return (a & ~SIGN) > EXP_MASK; }

// Round a 64-bit significand m (leading 1 at bit 63 for normal results) to 53
// bits; `sticky` says nonzero bits were dropped below m. `e` is the biased
// exponent of the value 1.m. Returns the magnitude's bit pattern.
static inline ulong round_pack(ulong m, bool sticky, int e) {
    int shift = 11;
    if (e <= 0) {           // subnormal result: shift further right
        const int extra = 1 - e;
        if (extra >= 64) return 0;  // far below halfway the least subnormal
        shift += extra;
        e = 0;
    }
    ulong q, rem, halfway;
    if (shift >= 64) {
        q = 0;
        rem = m;
        halfway = 1ul << 63;
        if (shift > 64) { sticky = sticky || rem != 0; rem = 0; }
    } else {
        q = m >> shift;
        rem = m & ((1ul << shift) - 1ul);
        halfway = 1ul << (shift - 1);
    }
    if (rem > halfway || (rem == halfway && (sticky || (q & 1ul)))) q += 1;
    // A normal q carries its leading 1 at bit 52, so (e - 1) << 52 + q is the
    // packed value; a rounding carry or a subnormal reaching 2^52 bumps the
    // exponent field on its own.
    const ulong packed = (e > 0 ? (ulong(e - 1) << 52) : 0ul) + q;
    return packed >= EXP_MASK ? EXP_MASK : packed;  // overflow: infinity
}

static inline ulong f64_mul(ulong a, ulong b) {
    const ulong sign = (a ^ b) & SIGN;
    int ea = int((a >> 52) & 0x7FF), eb = int((b >> 52) & 0x7FF);
    ulong ma = a & FRAC_MASK, mb = b & FRAC_MASK;
    if (is_nan(a)) return a | 0x0008000000000000ul;
    if (is_nan(b)) return b | 0x0008000000000000ul;
    const bool a_zero = ea == 0 && ma == 0, b_zero = eb == 0 && mb == 0;
    if (ea == 0x7FF || eb == 0x7FF) {
        if (a_zero || b_zero) return QUIET_NAN;  // inf * 0: the ARM default NaN
        return sign | EXP_MASK;
    }
    if (a_zero || b_zero) return sign;
    if (ea == 0) { const int lz = int(clz(ma)) - 11; ma <<= lz; ea = 1 - lz; } else { ma |= 1ul << 52; }
    if (eb == 0) { const int lz = int(clz(mb)) - 11; mb <<= lz; eb = 1 - lz; } else { mb |= 1ul << 52; }
    const ulong A = ma << 11, B = mb << 11;  // leading 1 at bit 63
    ulong hi = mulhi(A, B), lo = A * B;     // product in [2^126, 2^128)
    int e = ea + eb - 1023;
    if (hi & SIGN) { e += 1; } else { hi = (hi << 1) | (lo >> 63); lo <<= 1; }
    return sign | round_pack(hi, lo != 0, e);
}

static inline ulong f64_add(ulong a, ulong b) {
    if (is_nan(a)) return a | 0x0008000000000000ul;
    if (is_nan(b)) return b | 0x0008000000000000ul;
    int ea = int((a >> 52) & 0x7FF), eb = int((b >> 52) & 0x7FF);
    if (ea == 0x7FF || eb == 0x7FF) {
        if (ea == 0x7FF && eb == 0x7FF && ((a ^ b) & SIGN)) return QUIET_NAN;
        return ea == 0x7FF ? a : b;
    }
    const bool a_zero = (a & ~SIGN) == 0, b_zero = (b & ~SIGN) == 0;
    if (a_zero && b_zero) return a & b & SIGN;  // -0 only for -0 + -0
    if (a_zero) return b;
    if (b_zero) return a;
    // Order by magnitude: |a| >= |b|.
    if ((b & ~SIGN) > (a & ~SIGN)) { const ulong t = a; a = b; b = t; const int te = ea; ea = eb; eb = te; }
    const ulong sign = a & SIGN;
    ulong ma = a & FRAC_MASK, mb = b & FRAC_MASK;
    if (ea == 0) ea = 1; else ma |= 1ul << 52;
    if (eb == 0) eb = 1; else mb |= 1ul << 52;
    ma <<= 9; mb <<= 9;  // leading 1 at bit 61: room for a carry and 9 guard bits
    const int d = ea - eb;
    if (d > 0) {
        if (d >= 63) { mb = mb != 0 ? 1ul : 0ul; }
        else { const ulong dropped = mb & ((1ul << d) - 1ul); mb = (mb >> d) | (dropped != 0 ? 1ul : 0ul); }
    }
    int e = ea;
    ulong m;
    if (((a ^ b) & SIGN) == 0) {
        m = ma + mb;
        if (m & (1ul << 62)) { m = (m >> 1) | (m & 1ul); e += 1; }
    } else {
        m = ma - mb;
        if (m == 0) return 0;  // exact cancellation: +0 when rounding to nearest
        const int lz = int(clz(m)) - 2;  // bring the leading 1 to bit 61
        if (lz > 0) {
            if (e - lz >= 1) { m <<= lz; e -= lz; }
            else { m <<= (e - 1); e = 0; }  // subnormal: the low bits shifted in are zeros
        }
    }
    // Hand round_pack a significand with its leading 1 at bit 63.
    if (e == 0) {
        // Subnormal, exactly: the value is m * 2^-(1074 + 9); pack directly.
        const ulong q0 = m >> 9, rem = m & 511ul;
        ulong q = q0;
        if (rem > 256ul || (rem == 256ul && (q & 1ul))) q += 1;
        return sign | q;
    }
    return sign | round_pack(m << 2, false, e);
}

static inline ulong f64_neg(ulong a) { return a ^ SIGN; }

// Complex multiply in the CPU engines' order: (mr*ar - mi*ai, mr*ai + mi*ar).
static inline ulong2 c_mul(ulong2 m, ulong2 a) {
    return ulong2(f64_add(f64_mul(m.x, a.x), f64_neg(f64_mul(m.y, a.y))),
                  f64_add(f64_mul(m.x, a.y), f64_mul(m.y, a.x)));
}

constant ulong ONE = 0x3FF0000000000000ul;
// Equal to 1 as a value, like the CPU engines' test: either sign of zero.
static inline bool is_one(ulong2 m) { return m.x == ONE && (m.y << 1) == 0ul; }
// acc + m*a with the CPU's rounding: 0 + x first turns -0 into +0.
static inline ulong2 c_mul_add(ulong2 acc, ulong2 m, ulong2 a) {
    const ulong2 t = c_mul(m, a);
    return ulong2(f64_add(acc.x, t.x), f64_add(acc.y, t.y));
}
static inline int spread(int k, int count, constant int* places, constant int* values) {
    for (int i = 0; i < count; ++i) {
        const int p = places[i];
        k = (k & ((1 << p) - 1)) | ((k >> p) << (p + 1)) | (values[i] << p);
    }
    return k;
}
#define TILE_BITS 11
// One threadgroup per tile; the same descriptors as the CUDA tile kernel
// (_GateTiles._apply_tile_batch), with ulong2 bit patterns for complex128.
kernel void gate_tile(device ulong2* state [[buffer(0)]],
                      constant ulong2* matrices [[buffer(1)]],
                      constant int* gates [[buffer(2)]],
                      constant ulong* masks [[buffer(3)]],
                      constant int& n_gates [[buffer(4)]],
                      constant int* tile_bits [[buffer(5)]],
                      constant int* rest_bits [[buffer(6)]],
                      constant int& n_rest [[buffer(7)]],
                      constant uint& first_block [[buffer(8)]],
                      uint group [[threadgroup_position_in_grid]],
                      uint lane [[thread_position_in_threadgroup]],
                      uint threads [[threads_per_threadgroup]]) {
    threadgroup ulong2 tile[1 << TILE_BITS];
    const int size = 1 << TILE_BITS;
    const uint block = first_block + group;
    ulong base = 0;
    for (int r = 0; r < n_rest; ++r)
        if ((block >> r) & 1u) base |= 1ul << rest_bits[r];
    for (int j = lane; j < size; j += threads) {
        ulong index = base;
        for (int b = 0; b < TILE_BITS; ++b) if ((j >> b) & 1) index |= 1ul << tile_bits[b];
        tile[j] = state[index];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = 0; g < n_gates; ++g) {
        if ((base & masks[2 * g]) != masks[2 * g + 1]) continue;  // uniform per threadgroup
        constant int* d = gates + 15 * g;
        const int kind = d[0], width = d[1], permutation = d[6], fixed = d[7], count = d[8];
        constant int* places = d + 9;
        constant int* values = d + 12;
        constant ulong2* matrix = matrices + d[5];
        const int visits = size >> count;
        if (kind == 3) {
            int fixed_row = 0, inside = 0, b0 = 0, b1 = 0, b2 = 0, r0 = 0, r1 = 0, r2 = 0;
            for (int p = 0; p < width; ++p) {
                const int target = d[2 + p], row_bit = width - 1 - p;
                if (target < 0) {
                    if ((base >> (-1 - target)) & 1ul) fixed_row |= 1 << row_bit;
                } else {
                    if (inside == 0) { b0 = target; r0 = row_bit; }
                    else if (inside == 1) { b1 = target; r1 = row_bit; }
                    else { b2 = target; r2 = row_bit; }
                    ++inside;
                }
            }
            const int dim = 1 << inside;
            bool trivial = true;
            for (int r = 0; r < dim; ++r) {
                const int row = fixed_row | ((r & 1) << r0) | (((r >> 1) & 1) << r1) | (((r >> 2) & 1) << r2);
                trivial = trivial && is_one(matrix[row]);
            }
            if (trivial) continue;
            for (int k = lane; k < visits; k += threads) {
                const int start = spread(k, count, places, values);
                for (int r = 0; r < dim; ++r) {
                    const int row = fixed_row | ((r & 1) << r0) | (((r >> 1) & 1) << r1) | (((r >> 2) & 1) << r2);
                    const ulong2 m = matrix[row];
                    if (is_one(m)) continue;
                    const int i = start | ((r & 1) << b0) | (((r >> 1) & 1) << b1) | (((r >> 2) & 1) << b2);
                    tile[i] = c_mul(m, tile[i]);
                }
            }
        } else {
            const int first = d[2], second = d[3], dim = 1 << width;
            for (int k = lane; k < visits; k += threads) {
                const int local = spread(k, count, places, values);
                ulong2 amplitudes[4];
                int indices[4];
                for (int r = 0; r < dim; ++r) {
                    const int offset = width == 1 ? (r << first) : (((r >> 1) << first) | ((r & 1) << second));
                    indices[r] = local | offset;
                    amplitudes[r] = tile[indices[r]];
                }
                for (int r = 0; r < dim; ++r) {
                    if (kind == 0) { tile[indices[r]] = c_mul(matrix[r * dim + r], amplitudes[r]); continue; }
                    if (kind == 1) {
                        if (fixed & (1 << r)) continue;
                        const int c = (permutation >> (2 * r)) & 3;
                        const ulong2 m = matrix[r * dim + c];
                        tile[indices[r]] = is_one(m) ? amplitudes[c] : c_mul(m, amplitudes[c]);
                        continue;
                    }
                    ulong2 sum = ulong2(0ul, 0ul);
                    for (int c = 0; c < dim; ++c) sum = c_mul_add(sum, matrix[r * dim + c], amplitudes[c]);
                    tile[indices[r]] = sum;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int j = lane; j < size; j += threads) {
        ulong index = base;
        for (int b = 0; b < TILE_BITS; ++b) if ((j >> b) & 1) index |= 1ul << tile_bits[b];
        state[index] = tile[j];
    }
}
