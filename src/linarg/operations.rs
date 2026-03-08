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
    activations::Activation, appcontext::{Device, GemmType, get_global_context}, linarg::{
        mm256::{
            add_avx2, apply_sigmoid_avx2_from_src, apply_silu_avx2_from_src, div_avx2,
            gemm_bias_blocked_avx2, mul_avx2, sub_avx2,
        },
        mm512::{
            add_avx512, apply_sigmoid_avx512_from_src, apply_silu_avx512_from_src, div_avx512,
            gemm_bias_blocked_scalar, mul_avx512, sub_avx512,
        },
        utils::{aprox_sigmoid_f32, aprox_silu_f32},
    }
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

pub fn apply_silu(dst: &mut [f32], src: &[f32], n: usize) {
    let context = get_global_context();
    let dst_ptr = dst.as_mut_ptr();
    let src_ptr = src.as_ptr();

    match context.get_gemm_type() {
        GemmType::Avx2 => {
            unsafe { apply_silu_avx2_from_src(dst_ptr, src_ptr, n) };
        }
        GemmType::Avx512 => unsafe {
            apply_silu_avx512_from_src(dst_ptr, src_ptr, n);
        },
        _ => {
            dst.par_iter_mut()
                .zip(src.par_iter())
                .for_each(|(d, s)| *d = aprox_silu_f32(*s));
        }
    }
}

pub fn apply_sigmoid(dst: &mut [f32], src: &[f32]) {
    match get_global_context().get_gemm_type() {
        GemmType::Avx2 => {
            dst.par_chunks_mut(CHUNK_SIZE)
                .enumerate()
                .for_each(|(i, dst_chunk)| {
                    let offset = CHUNK_SIZE * i;
                    let len = dst_chunk.len();
                    unsafe {
                        apply_sigmoid_avx2_from_src(
                            dst_chunk.as_mut_ptr(),
                            src.as_ptr().add(offset),
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
                        apply_sigmoid_avx512_from_src(
                            dst_chunk.as_mut_ptr(),
                            src.as_ptr().add(offset),
                            len,
                        )
                    };
                });
        }
        _ => {
            dst.par_iter_mut()
                .zip(src.par_iter())
                .for_each(|(d, s)| *d = aprox_sigmoid_f32(*s));
        }
    }
}

pub fn add_maybe_simd(a: &[f32], b: &[f32], dst: &mut [f32]) {
    match get_global_context().get_gemm_type() {
        GemmType::Avx2 => {
            dst.par_chunks_mut(CHUNK_SIZE)
                .enumerate()
                .for_each(|(i, dst_chunk)| {
                    let offset = CHUNK_SIZE * i;
                    let len = dst_chunk.len();
                    unsafe {
                        add_avx2(
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
                        add_avx512(
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
                .for_each(|(d, (a, b))| *d = *a + *b);
        }
    }
}

pub fn sub_maybe_simd(a: &[f32], b: &[f32], dst: &mut [f32]) {
    match get_global_context().get_gemm_type() {
        GemmType::Avx2 => {
            dst.par_chunks_mut(CHUNK_SIZE)
                .enumerate()
                .for_each(|(i, dst_chunk)| {
                    let offset = CHUNK_SIZE * i;
                    let len = dst_chunk.len();
                    unsafe {
                        sub_avx2(
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
                        sub_avx512(
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
                .for_each(|(d, (a, b))| *d = *a - *b);
        }
    }
}

pub fn mul_maybe_simd(a: &[f32], b: &[f32], dst: &mut [f32]) {
    match get_global_context().get_gemm_type() {
        GemmType::Avx2 => {
            dst.par_chunks_mut(CHUNK_SIZE)
                .enumerate()
                .for_each(|(i, dst_chunk)| {
                    let offset = i * CHUNK_SIZE;
                    let len = dst_chunk.len();
                    unsafe {
                        mul_avx2(
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
                        mul_avx512(
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
                .for_each(|(d, (a, b))| *d = *a * *b);
        }
    }
}

pub fn div_maybe_simd(a: &[f32], b: &[f32], dst: &mut [f32], n: usize) {
    let dst_ptr_mut = dst.as_mut_ptr();
    let sa_ptr = a.as_ptr();
    let sb_ptr = b.as_ptr();

    match get_global_context().get_gemm_type() {
        GemmType::Avx2 => {
            unsafe { div_avx2(sa_ptr, sb_ptr, dst_ptr_mut, n) };
        }
        GemmType::Avx512 => {
            unsafe { div_avx512(sa_ptr, sb_ptr, dst_ptr_mut, n) };
        }
        _ => {
            dst.par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(d, (a, b))| *d = *a / *b);
        }
    }
}
