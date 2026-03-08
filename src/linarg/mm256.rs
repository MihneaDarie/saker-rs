use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
use crate::{
    accumulate_simd, activations::Activation, fmadd_ps_simd, set_zero_simd, set1_ps_simd,
    storeu_ps_simd,
};

const MC: usize = 64;
const KC: usize = 256;
const NC: usize = 256;
const MR: usize = 8;
const NR: usize = 8;

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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn micro_kernel_8x8_avx2(
    k: usize,
    a: *const f32,
    lda: usize,
    b: *const f32,
    ldb: usize,
    c: *mut f32,
    ldc: usize,
    accumulate: bool,
) {
    set_zero_simd!(_mm256_setzero_ps, c0, c1, c2, c3, c4, c5, c6, c7,);

    unsafe {
        for p in 0..k {
            let b_row = _mm256_loadu_ps(b.add(p * ldb));

            set1_ps_simd!(_mm256_set1_ps, a, lda, p =>
                a0: 0, a1: 1, a2: 2, a3: 3,
                a4: 4, a5: 5, a6: 6, a7: 7,
            );

            fmadd_ps_simd!(_mm256_fmadd_ps, b_row => a0:c0, a1:c1, a2:c2, a3:c3,
                                a4:c4, a5:c5, a6:c6, a7:c7,);
        }

        if accumulate {
            accumulate_simd!(_mm256_loadu_ps,_mm256_storeu_ps,_mm256_add_ps, c, ldc =>  c0: 0, c1: 1, c2: 2, c3: 3,
            c4: 4, c5: 5, c6: 6, c7: 7,);
        } else {
            storeu_ps_simd!(_mm256_storeu_ps,c,ldc => c0: 0, c1: 1, c2: 2, c3: 3,
            c4: 4, c5: 5, c6: 6, c7: 7,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_silu_and_bias_avx2(c: *mut f32, n: usize, bias: f32) {
    let bias_v = _mm256_set1_ps(bias);
    unsafe {
        for i in (0..n).step_by(8) {
            let val = _mm256_loadu_ps(c.add(i));
            let activated = silu_avx2(_mm256_add_ps(val, bias_v));
            _mm256_storeu_ps(c.add(i), activated);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_sigmoid_and_bias_avx2(c: *mut f32, n: usize, bias: f32) {
    let bias_v = _mm256_set1_ps(bias);
    unsafe {
        for i in (0..n).step_by(8) {
            let val = _mm256_loadu_ps(c.add(i));
            let activated = sigmoid_avx2(_mm256_add_ps(val, bias_v));
            _mm256_storeu_ps(c.add(i), activated);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn silu_avx2(x: __m256) -> __m256 {
    let left_margin = _mm256_set1_ps(-4.0);
    let right_margin = _mm256_set1_ps(4.0);
    let zeros = _mm256_setzero_ps();
    let quarter = _mm256_set1_ps(0.25);
    let one_over_eight = _mm256_set1_ps(0.125);
    let half = _mm256_set1_ps(0.5);

    let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);

    // 0.25 * |x| * x * 0.125
    let part1 = _mm256_mul_ps(
        _mm256_mul_ps(quarter, _mm256_mul_ps(x, abs_x)),
        one_over_eight,
    );

    //0.5 + 0.25 * x - part1
    let part2 = _mm256_sub_ps(_mm256_add_ps(half, _mm256_mul_ps(quarter, x)), part1);

    let mut result = _mm256_mul_ps(x, part2);

    let mask_low = _mm256_cmp_ps(x, left_margin, _CMP_LT_OQ);
    let mask_high = _mm256_cmp_ps(x, right_margin, _CMP_GT_OQ);

    result = _mm256_blendv_ps(result, zeros, mask_low);

    result = _mm256_blendv_ps(result, x, mask_high);

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn apply_bias_avx2(c: *mut f32, n: usize, bias: f32) {
    let bias_v = _mm256_set1_ps(bias);

    unsafe {
        for i in (0..n).step_by(8) {
            let val = _mm256_loadu_ps(c.add(i));
            _mm256_storeu_ps(c.add(i), _mm256_add_ps(val, bias_v));
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn gemm_bias_blocked_avx2(
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
                            micro_kernel_8x8_avx2(
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
                    apply_sigmoid_and_bias_avx2(row.as_mut_ptr(), n, bias[i]);
                });
            }
            Activation::Silu => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_silu_and_bias_avx2(row.as_mut_ptr(), n, bias[i]);
                });
            }
            Activation::None => {
                c.par_chunks_mut(n).enumerate().for_each(|(i, row)| unsafe {
                    apply_bias_avx2(row.as_mut_ptr(), n, bias[i]);
                });
            }
        },
        None => match activation {
            Activation::Sigmoid => {
                c.par_chunks_mut(n).for_each(|row| unsafe {
                    apply_sigmoid_avx2(row.as_mut_ptr(), row.len());
                });
            }
            Activation::Silu => {
                c.par_chunks_mut(n).for_each(|row| unsafe {
                    apply_silu_avx2(row.as_mut_ptr(), row.len());
                });
            }
            Activation::None => {}
        },
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn sigmoid_avx2(x: __m256) -> __m256 {
    let left_margin = _mm256_set1_ps(-4.0);
    let right_margin = _mm256_set1_ps(4.0);
    let zeros = _mm256_setzero_ps();
    let quarter = _mm256_set1_ps(0.25);
    let one_over_eight = _mm256_set1_ps(0.125);
    let half = _mm256_set1_ps(0.5);

    let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);

    // 0.25 * |x| * x * 0.125
    let part1 = _mm256_mul_ps(
        _mm256_mul_ps(quarter, _mm256_mul_ps(x, abs_x)),
        one_over_eight,
    );

    //0.5 + 0.25 * x - part1
    let part2 = _mm256_sub_ps(_mm256_add_ps(half, _mm256_mul_ps(quarter, x)), part1);

    let mut result = _mm256_mul_ps(x, part2);

    let mask_low = _mm256_cmp_ps(x, left_margin, _CMP_LT_OQ);
    let mask_high = _mm256_cmp_ps(x, right_margin, _CMP_GT_OQ);

    result = _mm256_blendv_ps(result, zeros, mask_low);

    result = _mm256_blendv_ps(result, x, mask_high);

    //silu = x * sig <=> sig = silu / x
    result = _mm256_div_ps(result, x);

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn apply_silu_avx2(dst: *mut f32, n: usize) {
    unsafe {
        for i in (0..n).step_by(8) {
            let val = _mm256_loadu_ps(dst.add(i));
            let activated = silu_avx2(val);
            _mm256_storeu_ps(dst.add(i), activated);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn apply_sigmoid_avx2(dst: *mut f32, n: usize) {
    unsafe {
        for i in (0..n).step_by(8) {
            let val = _mm256_loadu_ps(dst.add(i));
            let activated = sigmoid_avx2(val);
            _mm256_storeu_ps(dst.add(i), activated);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn apply_silu_avx2_from_src(dst: *mut f32, src: *const f32, n: usize) {
    unsafe {
        for i in (0..n).step_by(8) {
            let val = _mm256_loadu_ps(src.add(i));
            let activated = silu_avx2(val);
            _mm256_storeu_ps(dst.add(i), activated);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn apply_sigmoid_avx2_from_src(dst: *mut f32, src: *const f32, n: usize) {
    unsafe {
        for i in (0..n).step_by(8) {
            let val = _mm256_loadu_ps(src.add(i));
            let activated = sigmoid_avx2(val);
            _mm256_storeu_ps(dst.add(i), activated);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn add_avx2(a: *const f32, b: *const f32, dst: *mut f32, n: usize) {
    unsafe {
        let chunks = n / 8 * 8;
        for i in (0..chunks).step_by(8) {
            let a_chunck = _mm256_loadu_ps(a.add(i));
            let b_chunck = _mm256_loadu_ps(b.add(i));
            _mm256_storeu_ps(dst.add(i), _mm256_add_ps(a_chunck, b_chunck));
        }
        for i in chunks..n {
            *dst.add(i) = *a
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn sub_avx2(a: *const f32, b: *const f32, dst: *mut f32, n: usize) {
    unsafe {
        let chunks = n / 8 * 8;
        for i in (0..chunks).step_by(8) {
            let a_chunck = _mm256_loadu_ps(a.add(i));
            let b_chunck = _mm256_loadu_ps(b.add(i));
            _mm256_storeu_ps(dst.add(i), _mm256_sub_ps(a_chunck, b_chunck));
        }
        for i in chunks..n {
            *dst.add(i) = *a
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn mul_avx2(a: *const f32, b: *const f32, dst: *mut f32, n: usize) {
    unsafe {
        let chunks = n / 8 * 8;
        for i in (0..chunks).step_by(8) {
            let a_chunck = _mm256_loadu_ps(a.add(i));
            let b_chunck = _mm256_loadu_ps(b.add(i));
            _mm256_storeu_ps(dst.add(i), _mm256_mul_ps(a_chunck, b_chunck));
        }
        for i in chunks..n {
            *dst.add(i) = *a
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn div_avx2(a: *const f32, b: *const f32, dst: *mut f32, n: usize) {
    unsafe {
        let chunks = n / 8 * 8;
        for i in (0..chunks).step_by(8) {
            let a_chunck = _mm256_loadu_ps(a.add(i));
            let b_chunck = _mm256_loadu_ps(b.add(i));
            _mm256_storeu_ps(dst.add(i), _mm256_div_ps(a_chunck, b_chunck));
        }
        for i in chunks..n {
            *dst.add(i) = *a.add(i) / *b.add(i);
        }
    }
}
