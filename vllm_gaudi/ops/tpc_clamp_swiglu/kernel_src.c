// SPDX-License-Identifier: Apache-2.0
//
// clamp_swiglu_fwd_bf16 — fused clamped-SwiGLU MoE epilogue for Intel Gaudi 2 (HL-225).
//
// Input  h   : [T, 2I] bf16, dense row-major. Declared to GC with dim0 = 2I (FCD),
//              dim1 = T. The gate half of row t is h[t, 0:I], the up half h[t, I:2I].
// Output out : [T, I]  bf16 (dim0 = I, dim1 = T).
// Scalar     : limit (fp32 scalar kernel param; production value 10.0).
//
// Math per element (fp32 internal, single bf16 round at the very end):
//   g = min(h_g, limit);  u = clamp(h_u, -limit, limit);
//   out = silu(g) * u,  silu(g) = g * sigmoid(g).
//
// sigmoid comes from no_saturation_sigmoid_f32 (gaudi2/gaudi3 builtin): a short
// polynomial that keeps full precision for large-negative gates. It replaces an
// earlier hand-rolled 1/(1+exp(-|x|)) chain built from v_exp_f32 +
// v_reciprocal_f32; that pair costs ~45 VPU ops per 64-lane half against ~20
// here, and the two produce bit-identical bf16 output over the whole input
// range (simulator A/B, sim_accuracy.cpp). Do NOT swap in v_sigmoid_f32: that
// one is the LUT tanh path, which saturates to exact 0/1 at |x| >= 9 and so
// zeroes out every deep-negative gate. no_saturation_sigmoid_bf16 is likewise
// not a substitute — a bf16-domain sigmoid loses ~13% relative accuracy on
// those same gates.
//
// Note the multiply order: (sigmoid(g) * g) * u. Folding it to
// sigmoid(g) * (g * u) looks equivalent and schedules slightly better, but g*u
// overflows fp32 for a large-magnitude gate (the gate is clamped only from
// ABOVE), and 0 * inf is NaN. Keeping silu(g) as the first product bounds every
// intermediate by limit^2.
//
// Vectorization: dim0 (I direction) in 128-lane bf16 steps (one 256 B VPU
// vector). Two TOKENS are processed per iteration so that two independent
// sigmoid chains are in flight and fill the VLIW bubble slots — that is the
// axis with slack, because GC splits the index space along dim0 and typically
// hands each of the 24 TPCs only one or two 128-element chunks per row. A
// leftover odd row falls back to a 2x unroll along the FCD so a single-token
// call still gets instruction-level parallelism. The trailing partial vector
// (I % 128 != 0) is peeled out of the hot loop entirely: expressing it as a
// ternary inside the loop made the compiler issue BOTH the full and the partial
// load every iteration, doubling tensor traffic.

// One 128-lane step: load gate at CG and up at CU, clamp, silu*up in fp32.
#define BODY(CG, CU, G, U, S_)                                                   \
    float128 G = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(CG, h));           \
    float128 U = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_b(CU, h));           \
    G.v1 = v_f32_min_b(G.v1, lim);                                               \
    G.v2 = v_f32_min_b(G.v2, lim);                                               \
    U.v1 = v_f32_max_b(v_f32_min_b(U.v1, lim), nlim);                            \
    U.v2 = v_f32_max_b(v_f32_min_b(U.v2, lim), nlim);                            \
    float128 S_;                                                                 \
    S_.v1 = v_f32_mul_b(v_f32_mul_b(no_saturation_sigmoid_f32(G.v1), G.v1), U.v1); \
    S_.v2 = v_f32_mul_b(v_f32_mul_b(no_saturation_sigmoid_f32(G.v2), G.v2), U.v2);

// Same, for a trailing REM < 128 lane group.
#define BODYP(CG, CU, REM, G, U, S_)                                             \
    float128 G = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_partial_b(CG, h, REM, 0)); \
    float128 U = v_convert_bf16_to_f32_all_b(v_bf16_ld_tnsr_partial_b(CU, h, REM, 0)); \
    G.v1 = v_f32_min_b(G.v1, lim);                                               \
    G.v2 = v_f32_min_b(G.v2, lim);                                               \
    U.v1 = v_f32_max_b(v_f32_min_b(U.v1, lim), nlim);                            \
    U.v2 = v_f32_max_b(v_f32_min_b(U.v2, lim), nlim);                            \
    float128 S_;                                                                 \
    S_.v1 = v_f32_mul_b(v_f32_mul_b(no_saturation_sigmoid_f32(G.v1), G.v1), U.v1); \
    S_.v2 = v_f32_mul_b(v_f32_mul_b(no_saturation_sigmoid_f32(G.v2), G.v2), U.v2);

#define ST(CG, S_)        v_bf16_st_tnsr(CG, out, v_convert_f32_to_bf16_all_b(S_));
#define STP(CG, S_, REM)  v_bf16_st_tnsr_partial(CG, out, v_convert_f32_to_bf16_all_b(S_), REM, 0);

void main(tensor h, tensor out, float limit)
{
    const int5 start = get_index_space_offset();
    const int5 end   = start + get_index_space_size();

    // Elements per output half-row: h dim0 = 2I.
    const int I = get_dim_size(h, 0) >> 1;

    float64 lim  = limit;    // scalar broadcast into vector lanes
    float64 nlim = -limit;

    const int s0 = start[0], e0 = end[0], t1 = end[1];

    int5 ca = start, cb = start;   // gate (and output) coords, chains a and b
    int5 ua = start, ub = start;   // up coords: same row, +I along the FCD

    int t = start[1];
    for (; t + 1 < t1; t += 2)     // two tokens per iteration
    {
        ca[1] = t;     ua[1] = t;
        cb[1] = t + 1; ub[1] = t + 1;

        int c0 = s0;
        for (; c0 < e0 && (I - c0) >= 128; c0 += 128)
        {
            ca[0] = c0; ua[0] = c0 + I;
            cb[0] = c0; ub[0] = c0 + I;
            BODY(ca, ua, ga, ura, sa)
            BODY(cb, ub, gb, urb, sb)
            ST(ca, sa) ST(cb, sb)
        }
        if (c0 < e0)
        {
            const int rem = I - c0;
            ca[0] = c0; ua[0] = c0 + I;
            cb[0] = c0; ub[0] = c0 + I;
            BODYP(ca, ua, rem, ga, ura, sa)
            BODYP(cb, ub, rem, gb, urb, sb)
            STP(ca, sa, rem) STP(cb, sb, rem)
        }
    }
    if (t < t1)   // odd leftover row: take the parallelism from the FCD instead
    {
        ca[1] = t; ua[1] = t;
        cb[1] = t; ub[1] = t;

        int c0 = s0;
        // both chunks must lie inside THIS member's slice, not just inside the
        // row: GC guarantees 128-element granularity along dim0, not 256.
        for (; c0 + 256 <= e0 && (I - c0) >= 256; c0 += 256)
        {
            ca[0] = c0;       ua[0] = c0 + I;
            cb[0] = c0 + 128; ub[0] = c0 + 128 + I;
            BODY(ca, ua, ga, ura, sa)
            BODY(cb, ub, gb, urb, sb)
            ST(ca, sa) ST(cb, sb)
        }
        for (; c0 < e0 && (I - c0) >= 128; c0 += 128)
        {
            ca[0] = c0; ua[0] = c0 + I;
            BODY(ca, ua, g, ur, s)
            ST(ca, s)
        }
        if (c0 < e0)
        {
            const int rem = I - c0;
            ca[0] = c0; ua[0] = c0 + I;
            BODYP(ca, ua, rem, g, ur, s)
            STP(ca, s, rem)
        }
    }
}
