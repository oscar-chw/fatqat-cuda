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

kernel void check_mul(device const ulong* a [[buffer(0)]], device const ulong* b [[buffer(1)]],
                      device ulong* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    out[i] = f64_mul(a[i], b[i]);
}
kernel void check_add(device const ulong* a [[buffer(0)]], device const ulong* b [[buffer(1)]],
                      device ulong* out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    out[i] = f64_add(a[i], b[i]);
}

// Complex multiply in the CPU engines' order: (mr*ar - mi*ai, mr*ai + mi*ar).
static inline ulong2 c_mul(ulong2 m, ulong2 a) {
    return ulong2(f64_add(f64_mul(m.x, a.x), f64_neg(f64_mul(m.y, a.y))),
                  f64_add(f64_mul(m.x, a.y), f64_mul(m.y, a.x)));
}
// acc = 0 + m0*a0 + m1*a1, as Numba's dense loop computes it.
static inline ulong2 c_dot2(ulong2 m0, ulong2 a0, ulong2 m1, ulong2 a1) {
    ulong2 acc = c_mul(m0, a0);
    // 0 + x is x, except that -0 becomes +0.
    acc = ulong2(acc.x == SIGN ? 0ul : acc.x, acc.y == SIGN ? 0ul : acc.y);
    const ulong2 t = c_mul(m1, a1);
    return ulong2(f64_add(acc.x, t.x), f64_add(acc.y, t.y));
}

kernel void dense_1q(device ulong2* state [[buffer(0)]], constant ulong2* matrix [[buffer(1)]],
                     constant uint& bit [[buffer(2)]], uint g [[thread_position_in_grid]]) {
    const uint low = g & ((1u << bit) - 1u);
    const uint i0 = ((g >> bit) << (bit + 1u)) | low, i1 = i0 | (1u << bit);
    const ulong2 a0 = state[i0], a1 = state[i1];
    state[i0] = c_dot2(matrix[0], a0, matrix[1], a1);
    state[i1] = c_dot2(matrix[2], a0, matrix[3], a1);
}
