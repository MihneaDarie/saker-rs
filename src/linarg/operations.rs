use rayon::{
    iter::{
        IndexedParallelIterator as _, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator as _,
    },
    slice::ParallelSliceMut,
};

#[cfg(target_arch = "x86_64")]
use crate::linarg::mm512::gemm_bias_blocked_avx512;
use crate::{
    activations::Activation,
    appcontext::{get_global_context, Device, GemmType},
    linarg::{
        mm256::{
            add_avx2, apply_leaky_relu_avx2_from_src, apply_relu_avx2_from_src,
            apply_sigmoid_avx2_from_src, apply_silu_avx2_from_src, div_avx2,
            gemm_bias_blocked_avx2, mul_avx2, sub_avx2,
        },
        mm512::{
            add_avx512, apply_leaky_relu_avx512_from_src, apply_relu_avx512_from_src,
            apply_sigmoid_avx512_from_src, apply_silu_avx512_from_src, div_avx512,
            gemm_bias_blocked_scalar, mul_avx512, sub_avx512,
        },
        utils::{aprox_sigmoid_f32, aprox_silu_f32, leaky_relu_f32, relu_f32},
    },
};

pub fn sgemm_bias_parallel(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    c: &mut [f32],
    activation: Activation,
) {
    if m == 0 || n == 0 || k == 0 {
        return;
    }

    let context = get_global_context();

    if context.get_device() == Device::Cpu {
        match context.get_gemm_type() {
            GemmType::Avx2 => {
                unsafe { gemm_bias_blocked_avx2(m, n, k, a, b, bias, c, activation) };
            }
            GemmType::Avx512 => {
                unsafe { gemm_bias_blocked_avx512(m, n, k, a, b, bias, c, activation) };
            }
            _ => {
                gemm_bias_blocked_scalar(m, n, k, a, b, bias, c, activation);
            }
        }
    }
}

const CHUNK_SIZE: usize = 32_768;

macro_rules! apply_activation_from_src {
    ($func_name:ident, $avx2_func:ident, $avx512_func:ident, $single_activation_func:ident) => {
        pub fn $func_name(dst: &mut [f32], src: &[f32]) {
            let context = get_global_context();
            let len = dst.len();
            let dst_ptr = dst.as_mut_ptr();
            let src_ptr = src.as_ptr();

            match context.get_gemm_type() {
                GemmType::Avx2 => {
                    unsafe { $avx2_func(dst_ptr, src_ptr, len) };
                }
                GemmType::Avx512 => unsafe {
                    $avx512_func(dst_ptr, src_ptr, len);
                },
                _ => {
                    dst.par_iter_mut()
                        .zip(src.par_iter())
                        .for_each(|(d, s)| *d = $single_activation_func(*s));
                }
            }
        }
    };
}

apply_activation_from_src!(
    apply_silu,
    apply_silu_avx2_from_src,
    apply_silu_avx512_from_src,
    aprox_silu_f32
);

apply_activation_from_src!(
    apply_relu,
    apply_relu_avx2_from_src,
    apply_relu_avx512_from_src,
    relu_f32
);

apply_activation_from_src!(
    apply_sigmoid,
    apply_sigmoid_avx2_from_src,
    apply_sigmoid_avx512_from_src,
    aprox_sigmoid_f32
);

pub fn apply_leaky_relu(dst: &mut [f32], alpha: f32, src: &[f32]) {
    let context = get_global_context();
    let len = dst.len();
    let dst_ptr = dst.as_mut_ptr();
    let src_ptr = src.as_ptr();
    match context.get_gemm_type() {
        GemmType::Avx2 => {
            unsafe { apply_leaky_relu_avx2_from_src(dst_ptr, alpha, src_ptr, len) };
        }
        GemmType::Avx512 => unsafe {
            apply_leaky_relu_avx512_from_src(dst_ptr, alpha, src_ptr, len);
        },
        _ => {
            dst.par_iter_mut()
                .zip(src.par_iter())
                .for_each(|(d, s)| *d = leaky_relu_f32(*s, alpha));
        }
    }
}

macro_rules! something_maybe_simd {
    ($func_name:ident, $axv2_func:ident, $avx512_func:ident, $op:tt) => {
        pub fn $func_name(a: &[f32], b: &[f32], dst: &mut [f32]) {
            match get_global_context().get_gemm_type() {
                GemmType::Avx2 => {
                    dst.par_chunks_mut(CHUNK_SIZE)
                        .enumerate()
                        .for_each(|(i, dst_chunk)| {
                            let offset = CHUNK_SIZE * i;
                            let len = dst_chunk.len();
                            unsafe {
                                $axv2_func(
                                    a.as_ptr().add(offset),
                                    b.as_ptr().add(offset),
                                    dst_chunk.as_mut_ptr(),
                                    len,
                                )
                            };
                        });
                }
                GemmType::Avx512 => {
                    dst.par_chunks_mut(CHUNK_SIZE)
                        .enumerate()
                        .for_each(|(i, dst_chunk)| {
                            let offset = CHUNK_SIZE * i;
                            let len = dst_chunk.len();
                            unsafe {
                                $avx512_func(
                                    a.as_ptr().add(offset),
                                    b.as_ptr().add(offset),
                                    dst_chunk.as_mut_ptr(),
                                    len,
                                )
                            };
                        });
                }
                _ => {
                    dst.par_iter_mut()
                        .zip(a.par_iter().zip(b.par_iter()))
                        .for_each(|(d, (a, b))| *d = *a $op *b);
                }
            }
        }
    };
}

something_maybe_simd!(add_maybe_simd, add_avx2, add_avx512, +);
something_maybe_simd!(sub_maybe_simd, sub_avx2, sub_avx512, -);
something_maybe_simd!(mul_maybe_simd, mul_avx2, mul_avx512, *);
something_maybe_simd!(div_maybe_simd, div_avx2, div_avx512, /);
