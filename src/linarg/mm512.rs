use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::activations::Activation;
#[cfg(target_arch = "x86_64")]
use crate::{
    accumulate_simd, fmadd_ps_simd, linarg::utils::aprox_silu_f32, set_zero_simd, set1_ps_simd,
    storeu_ps_simd,
};

const MC: usize = 64;
const KC: usize = 256;
const NC: usize = 256;
const MR: usize = 16;
const NR: usize = 16;

#[inline(always)]
unsafe fn micro_kernel_16x16_scalar(
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
    let mut acc = [0.0f32; 256];

    unsafe {
        for p in 0..k {
            for i in 0..mr {
                let val = *a.add(i * lda + p);
                for j in 0..nr {
                    acc[i * 16 + j] += val * *b.add(p * ldb + j);
                }
            }
        }
        for i in 0..mr {
            for j in 0..nr {
                if accumulate {
                    *c.add(i * ldc + j) += acc[i * 16 + j];
                } else {
                    *c.add(i * ldc + j) = acc[i * 16 + j]
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn micro_kernel_16x16_avx_512(
    k: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
    accumulate: bool,
) {
    set_zero_simd!(
        _mm512_setzero_ps,
        c0,
        c1,
        c2,
        c3,
        c4,
        c5,
        c6,
        c7,
        c8,
        c9,
        c10,
        c11,
        c12,
        c13,
        c14,
        c15
    );

    unsafe {
        for p in 0..k {
            let b = _mm512_loadu_ps(b.add(p * ldb));

            set1_ps_simd!(_mm512_set1_ps, a, lda, p =>
                a0: 0, a1: 1, a2: 2, a3: 3,
                a4: 4, a5: 5, a6: 6, a7: 7,
                a8: 8, a9: 9, a10:10, a11:11,
                a12:12, a13:13, a14:14, a15:15,
            );

            fmadd_ps_simd!(_mm512_fmadd_ps, b => a0:c0, a1:c1, a2:c2, a3:c3,
                                a4:c4, a5:c5, a6:c6, a7:c7,
                                a8:c8, a9:c9, a10:c10, a11:c11,
                                a12:c12, a13:c13, a14:c14, a15:c15);
        }

        if accumulate {

            accumulate_simd!(_mm512_loadu_ps,_mm512_storeu_ps,_mm512_add_ps, c, ldc =>  c0: 0, c1: 1, c2: 2, c3: 3,
                                                                                        c4: 4, c5: 5, c6: 6, c7: 7,
                                                                                        c8: 8, c9: 9, c10:10, c11:11,
                                                                   c12:12, c13:13, c14:14, c15:15,);
        } else {
            storeu_ps_simd!(_mm512_storeu_ps,c,ldc => c0: 0, c1: 1, c2: 2, c3: 3,
                                                    c4: 4, c5: 5, c6: 6, c7: 7,
                                                    c8: 8, c9: 9, c10:10, c11:11,
                                                    c12:12, c13:13, c14:14, c15:15,);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn apply_silu_and_bias_avx512(c: *mut f32, n: usize, bias: f32) {
    let bias_v = _mm512_set1_ps(bias);
    unsafe {
        for i in (0..n).step_by(16) {
            let val = _mm512_loadu_ps(c.add(i));
            let activated = silu_avx512(_mm512_add_ps(val, bias_v));
            _mm512_storeu_ps(c.add(i), activated);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn apply_sigmoid_and_bias_avx512(c: *mut f32, n: usize, bias: f32) {
    let bias_v = _mm512_set1_ps(bias);
    unsafe {
        for i in (0..n).step_by(16) {
            let val = _mm512_loadu_ps(c.add(i));
            let activated = sigmoid_avx512(_mm512_add_ps(val, bias_v));
            _mm512_storeu_ps(c.add(i), activated);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn apply_bias_avx512(c: *mut f32, n: usize, bias: f32) {
    let bias_v = _mm512_set1_ps(bias);
    unsafe {
        for i in (0..n).step_by(16) {
            let val = _mm512_loadu_ps(c.add(i));
            _mm512_storeu_ps(c.add(i), _mm512_add_ps(val, bias_v));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
unsafe fn silu_avx512(x: __m512) -> __m512 {
    let left_margin = _mm512_set1_ps(-4.0);
    let right_margin = _mm512_set1_ps(4.0);
    let zeros = _mm512_setzero_ps();
    let quarter = _mm512_set1_ps(0.25);
    let one_over_eight = _mm512_set1_ps(0.125);
    let half = _mm512_set1_ps(0.5);

    let abs_x = unsafe { _mm512_andnot_ps(_mm512_set1_ps(-0.0), x) };

    // 0.25 * |x| * x * 0.125
    let part1 = _mm512_mul_ps(
        _mm512_mul_ps(quarter, _mm512_mul_ps(x, abs_x)),
        one_over_eight,
    );

    //0.5 + 0.25 * x - part1
    let part2 = _mm512_sub_ps(_mm512_add_ps(half, _mm512_mul_ps(quarter, x)), part1);

    let mut result = _mm512_mul_ps(x, part2);

    let mask_low = _mm512_cmp_ps_mask(x, left_margin, _CMP_LT_OQ);
    let mask_high = _mm512_cmp_ps_mask(x, right_margin, _CMP_GT_OQ);

    result = _mm512_mask_mov_ps(result, mask_low, zeros);
    result = _mm512_mask_mov_ps(result, mask_high, x);

    result
}

pub fn gemm_bias_blocked_scalar(
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
    let mt = m.div_ceil(MC);
    let nt = n.div_ceil(NC);

    let a_base = a.as_ptr() as usize;
    let b_base = b.as_ptr() as usize;
    let c_base = c.as_mut_ptr() as usize;

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

                        micro_kernel_16x16_scalar(
                            mr, nr, kc, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, accumulate,
                        );
                    }
                }
            }
        }
    });
    match bias {
        Some(bias) => match activation {
            Activation::Sigmoid => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_sigmoid_and_bias_avx512(row.as_mut_ptr(), n, bias[i]);
                });
            }
            Activation::Silu => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_silu_and_bias_avx512(row.as_mut_ptr(), n, bias[i]);
                });
            }
            Activation::None => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_bias_avx512(row.as_mut_ptr(), n, bias[i]);
                });
            }
        },
        None => match activation {
            Activation::Sigmoid => {
                c.par_chunks_mut(n).for_each(|row| unsafe {
                    apply_sigmoid_avx512(row.as_mut_ptr(), row.len());
                });
            }
            Activation::Silu => {
                c.par_chunks_mut(n).for_each(|row| unsafe {
                    apply_silu_avx512(row.as_mut_ptr(), row.len());
                });
            }
            Activation::None => {}
        },
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn gemm_bias_blocked_avx512(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    c: &mut [f32],
    activation: Activation,
) {
    if let Some(bb) = bias {
        debug_assert_eq!(bb.len(), m);
    }

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
                            micro_kernel_16x16_avx_512(
                                kc, a_ptr, lda, b_ptr, ldb, c_ptr, ldc, accumulate,
                            );
                        } else {
                            micro_kernel_16x16_scalar(
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
                    apply_sigmoid_and_bias_avx512(row.as_mut_ptr(), n, bias[i]);
                });
            }
            Activation::Silu => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_silu_and_bias_avx512(row.as_mut_ptr(), n, bias[i]);
                });
            }
            Activation::None => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_bias_avx512(row.as_mut_ptr(), n, bias[i]);
                });
            }
        },
        None => match activation {
            Activation::Sigmoid => {
                c.par_chunks_mut(n).for_each(|row| unsafe {
                    apply_sigmoid_avx512(row.as_mut_ptr(), row.len());
                });
            }
            Activation::Silu => {
                c.par_chunks_mut(n).for_each(|row| unsafe {
                    apply_silu_avx512(row.as_mut_ptr(), row.len());
                });
            }
            Activation::None => {}
        },
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn sigmoid_avx512(x: __m512) -> __m512 {
    let left_margin = _mm512_set1_ps(-4.0);
    let right_margin = _mm512_set1_ps(4.0);
    let zeros = _mm512_setzero_ps();
    let quarter = _mm512_set1_ps(0.25);
    let one_over_eight = _mm512_set1_ps(0.125);
    let half = _mm512_set1_ps(0.5);

    let abs_x = unsafe { _mm512_andnot_ps(_mm512_set1_ps(-0.0), x) };

    // 0.25 * |x| * x * 0.125
    let part1 = _mm512_mul_ps(
        _mm512_mul_ps(quarter, _mm512_mul_ps(x, abs_x)),
        one_over_eight,
    );

    //0.5 + 0.25 * x - part1
    let part2 = _mm512_sub_ps(_mm512_add_ps(half, _mm512_mul_ps(quarter, x)), part1);

    let mut result = _mm512_mul_ps(x, part2);

    let mask_low = _mm512_cmp_ps_mask(x, left_margin, _CMP_LT_OQ);
    let mask_high = _mm512_cmp_ps_mask(x, right_margin, _CMP_GT_OQ);

    result = _mm512_mask_mov_ps(result, mask_low, zeros);
    result = _mm512_mask_mov_ps(result, mask_high, x);

    result = _mm512_div_ps(result, x);

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn apply_silu_avx512(dst: *mut f32, n: usize) {
    unsafe {
        for i in (0..n).step_by(16) {
            let val = _mm512_loadu_ps(dst.add(i));
            let activated = silu_avx512(val);
            _mm512_storeu_ps(dst.add(i), activated);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn apply_sigmoid_avx512(dst: *mut f32, n: usize) {
    unsafe {
        for i in (0..n).step_by(16) {
            let val = _mm512_loadu_ps(dst.add(i));
            let activated = sigmoid_avx512(val);
            _mm512_storeu_ps(dst.add(i), activated);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn apply_silu_avx512_from_src(dst: *mut f32, src: *const f32, n: usize) {
    unsafe {
        let chunks = n / 16 * 16;
        for i in (0..chunks).step_by(16) {
            let val = _mm512_loadu_ps(src.add(i));
            let activated = silu_avx512(val);
            _mm512_storeu_ps(dst.add(i), activated);
        }
        for i in chunks..n {
            let x = *src.add(i);
            *dst.add(i) = aprox_silu_f32(x);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn apply_sigmoid_avx512_from_src(dst: *mut f32, src: *const f32, n: usize) {
    unsafe {
        for i in (0..n).step_by(16) {
            let val = _mm512_loadu_ps(src.add(i));
            let activated = sigmoid_avx512(val);
            _mm512_storeu_ps(dst.add(i), activated);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn add_avx512(a: *const f32, b: *const f32, dst: *mut f32, n: usize) {
    unsafe {
        let chunks = n / 16 * 16;
        for i in (0..chunks).step_by(16) {
            let a_chunck = _mm512_loadu_ps(a.add(i));
            let b_chunck = _mm512_loadu_ps(b.add(i));
            _mm512_storeu_ps(dst.add(i), _mm512_add_ps(a_chunck, b_chunck));
        }
        for i in chunks..n {
            *dst.add(i) = *a.add(i) + *b.add(i);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn sub_avx512(a: *const f32, b: *const f32, dst: *mut f32, n: usize) {
    unsafe {
        let chunks = n / 16 * 16;
        for i in (0..chunks).step_by(16) {
            let a_chunck = _mm512_loadu_ps(a.add(i));
            let b_chunck = _mm512_loadu_ps(b.add(i));
            _mm512_storeu_ps(dst.add(i), _mm512_sub_ps(a_chunck, b_chunck));
        }
        for i in chunks..n {
            *dst.add(i) = *a.add(i) - *b.add(i);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn mul_avx512(a: *const f32, b: *const f32, dst: *mut f32, n: usize) {
    unsafe {
        let chunks = n / 16 * 16;
        for i in (0..chunks).step_by(16) {
            let a_chunck = _mm512_loadu_ps(a.add(i));
            let b_chunck = _mm512_loadu_ps(b.add(i));
            _mm512_storeu_ps(dst.add(i), _mm512_mul_ps(a_chunck, b_chunck));
        }
        for i in chunks..n {
            *dst.add(i) = *a.add(i) * *b.add(i);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,fma")]
pub unsafe fn div_avx512(a: *const f32, b: *const f32, dst: *mut f32, n: usize) {
    unsafe {
        let chunks = n / 16 * 16;
        for i in (0..chunks).step_by(16) {
            let a_chunck = _mm512_loadu_ps(a.add(i));
            let b_chunck = _mm512_loadu_ps(b.add(i));
            _mm512_storeu_ps(dst.add(i), _mm512_div_ps(a_chunck, b_chunck));
        }
        for i in chunks..n {
            *dst.add(i) = *a.add(i) / *b.add(i);
        }
    }
}
