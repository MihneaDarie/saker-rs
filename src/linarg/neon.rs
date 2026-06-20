use rayon::prelude::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::linarg::utils::aprox_sigmoid_f32;
use crate::linarg::utils::aprox_silu_f32;
#[cfg(target_arch = "aarch64")]
use crate::linarg::utils::leaky_relu_f32;
use crate::linarg::utils::relu_f32;

#[cfg(target_arch = "aarch64")]
use crate::activations::Activation;

macro_rules! env_usize {
    ($key:literal) => {{
        const S: &str = env!($key);
        const V: usize = {
            let b = S.as_bytes();
            assert!(!b.is_empty(), "env var is empty");
            let mut v = 0usize;
            let mut i = 0;
            while i < b.len() {
                let digit = b[i];
                assert!(digit >= b'0' && digit <= b'9', "env var contains non-digit byte");
                v = v * 10 + (digit - b'0') as usize;
                i += 1;
            }
            v
        };
        V
    }};
}

pub const MC: usize = env_usize!("SAKER_MC");
pub const KC: usize = env_usize!("SAKER_KC");
pub const NC: usize = env_usize!("SAKER_NC");
pub const MR: usize = env_usize!("SAKER_MR");
pub const NR: usize = env_usize!("SAKER_NR");

#[inline(always)]
unsafe fn micro_kernel_8x8_scalar(
    mr: usize,
    nr: usize,
    k: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
    accumulate: bool,
) {
    let mut acc = [[0.0f32; NR]; MR];

    unsafe {
        for p in 0..k {
            for (i, rows) in acc.iter_mut().enumerate().take(mr) {
                let a_val = *a.add(i * lda + p);
                for (j, val) in rows.iter_mut().enumerate().take(nr) {
                    *val += a_val * *b.add(p * ldb + j);
                }
            }
        }

        if accumulate {
            for (i, rows) in acc.iter().enumerate().take(mr) {
                for (j, value) in rows.iter().enumerate().take(nr) {
                    *c.add(i * ldc + j) += *value;
                }
            }
        } else {
            for (i, rows) in acc.iter().enumerate().take(mr) {
                for (j, value) in rows.iter().enumerate().take(nr) {
                    *c.add(i * ldc + j) = *value;
                }
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
macro_rules! set_zero_neon {
    ( $( $name:ident ),+ $(,)? ) => {
        $(
            let mut $name = vdupq_n_f32(0.0);
        )+
    };
}

#[cfg(target_arch = "aarch64")]
macro_rules! fmadd_n_neon {
    ($b:expr, $a:expr, $lda:expr, $p:expr => $( $c:ident : $i:expr ),+ $(,)?) => {
        $(
            $c = vfmaq_n_f32($c, $b, *$a.add($i * $lda + $p));
        )+
    };
}

#[cfg(target_arch = "aarch64")]
macro_rules! accumulate_neon {
    ($c:expr, $ldc:expr => $( $name:ident : $i:expr ),+ $(,)?) => {
        $(
            let old = vld1q_f32($c.add($i * $ldc));
            vst1q_f32($c.add($i * $ldc), vaddq_f32(old, $name));
        )+
    };
}

#[cfg(target_arch = "aarch64")]
macro_rules! store_neon {
    ($c:expr, $ldc:expr => $( $name:ident : $i:expr ),+ $(,)?) => {
        $(
            vst1q_f32($c.add($i * $ldc), $name);
        )+
    };
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn micro_kernel_8x8_neon(
    k: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
    accumulate: bool,
) {
    set_zero_neon!(
        c0_0, c1_0, c2_0, c3_0, c4_0, c5_0, c6_0, c7_0, c0_1, c1_1, c2_1, c3_1, c4_1, c5_1, c6_1,
        c7_1,
    );

    unsafe {
        for p in 0..k {
            let b_lo = vld1q_f32(b.add(p * ldb));
            let b_hi = vld1q_f32(b.add(p * ldb + 4));

            fmadd_n_neon!(b_lo, a, lda, p =>
                c0_0: 0, c1_0: 1, c2_0: 2, c3_0: 3,
                c4_0: 4, c5_0: 5, c6_0: 6, c7_0: 7,);
            fmadd_n_neon!(b_hi, a, lda, p =>
                c0_1: 0, c1_1: 1, c2_1: 2, c3_1: 3,
                c4_1: 4, c5_1: 5, c6_1: 6, c7_1: 7,);
        }

        if accumulate {
            accumulate_neon!(c, ldc =>
                c0_0: 0, c1_0: 1, c2_0: 2, c3_0: 3,
                c4_0: 4, c5_0: 5, c6_0: 6, c7_0: 7,);
            accumulate_neon!(c.add(4), ldc =>
                c0_1: 0, c1_1: 1, c2_1: 2, c3_1: 3,
                c4_1: 4, c5_1: 5, c6_1: 6, c7_1: 7,);
        } else {
            store_neon!(c, ldc =>
                c0_0: 0, c1_0: 1, c2_0: 2, c3_0: 3,
                c4_0: 4, c5_0: 5, c6_0: 6, c7_0: 7,);
            store_neon!(c.add(4), ldc =>
                c0_1: 0, c1_1: 1, c2_1: 2, c3_1: 3,
                c4_1: 4, c5_1: 5, c6_1: 6, c7_1: 7,);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn apply_bias_neon(c: *mut f32, n: usize, bias: f32) {
    let bias_v = vdupq_n_f32(bias);
    let safe_n = n - (n % 4);
    unsafe {
        for i in (0..safe_n).step_by(4) {
            let val = vld1q_f32(c.add(i));
            vst1q_f32(c.add(i), vaddq_f32(val, bias_v));
        }
        for i in safe_n..n {
            *c.add(i) = *c.add(i) + bias;
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn gemm_bias_blocked_neon(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    c: &mut [f32],
    activation: Activation,
) {

    let lda = k;
    let ldb = n;
    let ldc = n;

    let a_base = a.as_ptr() as usize;
    let b_base = b.as_ptr() as usize;
    let c_base = c.as_mut_ptr() as usize;

    let mt = m.div_ceil(MC);
    let nt = n.div_ceil(NC);

    (0..mt * nt).into_par_iter().for_each(|t| {
        let a_ptr_base = a_base as *const f32;
        let b_ptr_base = b_base as *const f32;
        let c_ptr_base = c_base as *mut f32;

        let i0 = (t / nt) * MC;
        let j0 = (t % nt) * NC;

        let mc = (m - i0).min(MC);
        let nc = (n - j0).min(NC);

        for p0 in (0..k).step_by(KC) {
            let kc = (k - p0).min(KC);
            let accumulate = p0 != 0;

            for i in (0..mc).step_by(MR) {
                let mr = (mc - i).min(MR);

                for j in (0..nc).step_by(NR) {
                    let nr = (nc - j).min(NR);

                    unsafe {
                        let a_ptr = a_ptr_base.add((i0 + i) * lda + p0);
                        let b_ptr = b_ptr_base.add(p0 * ldb + (j0 + j));
                        let c_ptr = c_ptr_base.add((i0 + i) * ldc + (j0 + j));

                        if mr == MR && nr == NR {
                            micro_kernel_8x8_neon(
                                kc, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, accumulate,
                            );
                        } else {
                            micro_kernel_8x8_scalar(
                                mr, nr, kc, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, accumulate,
                            );
                        }
                    }
                }
            }
        }
    });

    match bias {
        Some(bias) => match activation {
            Activation::Sigmoid => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_sigmoid_and_bias_neon(row.as_mut_ptr(), n, bias[i]);
                });
            }
            Activation::Silu => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_silu_and_bias_neon(row.as_mut_ptr(), n, bias[i]);
                });
            }
            Activation::Relu => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_relu_and_bias_neon(row.as_mut_ptr(), n, bias[i]);
                });
            }
            Activation::LeakyRelu(alpha) => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_leaky_relu_and_bias_neon(row.as_mut_ptr(), alpha, n, bias[i]);
                });
            }
            Activation::None => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_bias_neon(row.as_mut_ptr(), n, bias[i]);
                });
            }
        },
        None => match activation {
            Activation::Sigmoid => {
                c.par_chunks_mut(n).for_each(|row| unsafe {
                    apply_sigmoid_neon(row.as_mut_ptr(), row.len());
                });
            }
            Activation::Silu => {
                c.par_chunks_mut(n).for_each(|row| unsafe {
                    apply_silu_neon(row.as_mut_ptr(), row.len());
                });
            }
            Activation::Relu => {
                c.par_chunks_mut(n).for_each(|row| unsafe {
                    apply_relu_neon(row.as_mut_ptr(), row.len());
                });
            }
            Activation::LeakyRelu(alpha) => {
                c.par_chunks_mut(n).for_each(|row| unsafe {
                    apply_leaky_relu_neon(row.as_mut_ptr(), alpha, n);
                });
            }
            Activation::None => {}
        },
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn relu_neon(x: float32x4_t) -> float32x4_t {
    let zeros = vdupq_n_f32(0.0);
    vmaxq_f32(x, zeros)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn leaky_relu_neon(x: float32x4_t, alpha: f32) -> float32x4_t {
    let alphaed = vmulq_n_f32(x, alpha);
    vmaxq_f32(x, alphaed)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
fn silu_neon(x: float32x4_t) -> float32x4_t {
    let left_margin = vdupq_n_f32(-4.0);
    let right_margin = vdupq_n_f32(4.0);
    let zeros = vdupq_n_f32(0.0);
    let quarter = vdupq_n_f32(0.25);
    let one_over_eight = vdupq_n_f32(0.125);
    let half = vdupq_n_f32(0.5);

    let abs_x = vabsq_f32(x);

    // 0.25 * |x| * x * 0.125
    let part1 = vmulq_f32(vmulq_f32(quarter, vmulq_f32(x, abs_x)), one_over_eight);

    // 0.5 + 0.25 * x - part1
    let part2 = vsubq_f32(vaddq_f32(half, vmulq_f32(quarter, x)), part1);

    let mut result = vmulq_f32(x, part2);

    // masks are uint32x4_t: all-ones lanes where the comparison holds
    let mask_low = vcltq_f32(x, left_margin);
    let mask_high = vcgtq_f32(x, right_margin);

    // vbslq_f32(mask, a, b): picks a where mask is set, b otherwise
    result = vbslq_f32(mask_low, zeros, result);
    result = vbslq_f32(mask_high, x, result);

    result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn sigmoid_neon(x: float32x4_t) -> float32x4_t {
    let x = vaddq_f32(x, vdupq_n_f32(f32::EPSILON));
    let left_margin = vdupq_n_f32(-4.0);
    let right_margin = vdupq_n_f32(4.0);
    let zeros = vdupq_n_f32(0.0);
    let quarter = vdupq_n_f32(0.25);
    let one_over_eight = vdupq_n_f32(0.125);
    let half = vdupq_n_f32(0.5);

    let abs_x = vabsq_f32(x);

    // 0.25 * |x| * x * 0.125
    let part1 = vmulq_f32(vmulq_f32(quarter, vmulq_f32(x, abs_x)), one_over_eight);

    // 0.5 + 0.25 * x - part1
    let part2 = vsubq_f32(vaddq_f32(half, vmulq_f32(quarter, x)), part1);

    let mut result = vmulq_f32(x, part2);

    let mask_low = vcltq_f32(x, left_margin);
    let mask_high = vcgtq_f32(x, right_margin);

    result = vbslq_f32(mask_low, zeros, result);
    result = vbslq_f32(mask_high, x, result);

    // silu = x * sig <=> sig = silu / x
    result = vdivq_f32(result, x);

    result
}

macro_rules! unsafe_apply_activation_and_bias_neon {
    ($func_name:ident, $neon_activation_func:ident, $scalar_activation_func:ident) => {
        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        unsafe fn $func_name(c: *mut f32, n: usize, bias: f32) {
            let bias_v = vdupq_n_f32(bias);
            let safe_n = n - (n % 4);
            unsafe {
                for i in (0..safe_n).step_by(4) {
                    let val = vld1q_f32(c.add(i));
                    let activated = $neon_activation_func(vaddq_f32(val, bias_v));
                    vst1q_f32(c.add(i), activated);
                }
                for i in safe_n..n {
                    let v = *c.add(i) + bias;
                    *c.add(i) = $scalar_activation_func(v);
                }
            }
        }
    };
}

unsafe_apply_activation_and_bias_neon!(
    apply_sigmoid_and_bias_neon,
    sigmoid_neon,
    aprox_sigmoid_f32
);
unsafe_apply_activation_and_bias_neon!(apply_silu_and_bias_neon, silu_neon, aprox_silu_f32);
unsafe_apply_activation_and_bias_neon!(apply_relu_and_bias_neon, relu_neon, relu_f32);

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn apply_leaky_relu_and_bias_neon(c: *mut f32, alpha: f32, n: usize, bias: f32) {
    let bias_v = vdupq_n_f32(bias);
    let safe_n = n - (n % 4);
    unsafe {
        for i in (0..safe_n).step_by(4) {
            let val = vld1q_f32(c.add(i));
            let activated = leaky_relu_neon(vaddq_f32(val, bias_v), alpha);
            vst1q_f32(c.add(i), activated);
        }
        for i in safe_n..n {
            let v = *c.add(i) + bias;
            *c.add(i) = leaky_relu_f32(v, alpha);
        }
    }
}

macro_rules! unsafe_apply_activation_neon {
    ($func_name:ident, $neon_activation_func:ident, $scalar_activation_func:ident) => {
        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        pub unsafe fn $func_name(dst: *mut f32, n: usize) {
            let safe_n = n - (n % 4);
            unsafe {
                for i in (0..safe_n).step_by(4) {
                    let val = vld1q_f32(dst.add(i));
                    let activated = $neon_activation_func(val);
                    vst1q_f32(dst.add(i), activated);
                }
                for i in safe_n..n {
                    let v = *dst.add(i);
                    *dst.add(i) = $scalar_activation_func(v);
                }
            }
        }
    };
}

unsafe_apply_activation_neon!(apply_sigmoid_neon, sigmoid_neon, aprox_sigmoid_f32);
unsafe_apply_activation_neon!(apply_silu_neon, silu_neon, aprox_silu_f32);
unsafe_apply_activation_neon!(apply_relu_neon, relu_neon, relu_f32);

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn apply_leaky_relu_neon(dst: *mut f32, alpha: f32, n: usize) {
    unsafe {
        let safe_n = n - (n % 4);
        for i in (0..safe_n).step_by(4) {
            let val = vld1q_f32(dst.add(i));
            let activated = leaky_relu_neon(val, alpha);
            vst1q_f32(dst.add(i), activated);
        }
        for i in safe_n..n {
            let v = *dst.add(i);
            *dst.add(i) = leaky_relu_f32(v, alpha);
        }
    }
}

macro_rules! unsafe_apply_activation_from_src_neon {
    ($func_name:ident, $neon_activation_func:ident, $scalar_activation_func:ident) => {
        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        pub unsafe fn $func_name(dst: *mut f32, src: *const f32, n: usize) {
            let safe_n = n - (n % 4);
            unsafe {
                for i in (0..safe_n).step_by(4) {
                    let val = vld1q_f32(src.add(i));
                    let activated = $neon_activation_func(val);
                    vst1q_f32(dst.add(i), activated);
                }
                for i in safe_n..n {
                    let v = *src.add(i);
                    *dst.add(i) = $scalar_activation_func(v);
                }
            }
        }
    };
}

unsafe_apply_activation_from_src_neon!(
    apply_sigmoid_neon_from_src,
    sigmoid_neon,
    aprox_sigmoid_f32
);
unsafe_apply_activation_from_src_neon!(apply_silu_neon_from_src, silu_neon, aprox_silu_f32);
unsafe_apply_activation_from_src_neon!(apply_relu_neon_from_src, relu_neon, relu_f32);

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn apply_leaky_relu_neon_from_src(dst: *mut f32, alpha: f32, src: *const f32, n: usize) {
    let safe_n = n - (n % 4);

    unsafe {
        for i in (0..safe_n).step_by(4) {
            let val = vld1q_f32(src.add(i));
            let activated = leaky_relu_neon(val, alpha);
            vst1q_f32(dst.add(i), activated);
        }
    }
    for i in safe_n..n {
        let v = *src.add(i);
        *dst.add(i) = leaky_relu_f32(v, alpha);
    }
}

macro_rules! binop_neon {
    ($func_name:ident, $fast_neon_func:ident, $op:tt, $chunk_size:expr) => {
        #[cfg(target_arch = "aarch64")]
        #[target_feature(enable = "neon")]
        pub unsafe fn $func_name(a: *const f32, b: *const f32, dst: *mut f32, n: usize) {
            unsafe {
                let chunks = n / $chunk_size * $chunk_size;
                for i in (0..chunks).step_by($chunk_size) {
                    let a_chunck = vld1q_f32(a.add(i));
                    let b_chunck = vld1q_f32(b.add(i));
                    vst1q_f32(dst.add(i), $fast_neon_func(a_chunck, b_chunck));
                }
                for i in chunks..n {
                    *dst.add(i) = *a.add(i) $op *b.add(i);
                }
            }
        }
    };
}

binop_neon!(add_neon, vaddq_f32, +, 4);
binop_neon!(sub_neon, vsubq_f32, -, 4);
binop_neon!(mul_neon, vmulq_f32, *, 4);
binop_neon!(div_neon, vdivq_f32, /, 4);
