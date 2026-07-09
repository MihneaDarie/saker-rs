use std::{collections::HashSet, sync::OnceLock};

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmType {
    Avx2,
    Avx512,
    Neon,
    Scalar,
}

impl Default for GemmType {
    fn default() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx512f") {
                Self::Avx512
            } else if std::is_x86_feature_detected!("avx2") {
                Self::Avx2
            } else {
                Self::Scalar
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self::Neon
        }
    }
}

#[derive(Default, Debug)]
pub struct AppContext {
    gemm_type: GemmType,
}

impl AppContext {

    pub fn get_gemm_type(&self) -> GemmType {
        self.gemm_type
    }

    pub fn parse_command_line_arguments(args: &[String]) -> Result<Self, String> {
        let mut context = AppContext::default();

        if args.is_empty() {
            return Ok(context);
        }

        for flag in args.iter() {

            context.gemm_type = match flag.as_str() {
                "--AVX2" => GemmType::Avx2,
                "--AVX512" => GemmType::Avx512,
                "--Neon" => GemmType::Neon,
                "--Scalar" => GemmType::Scalar,
                _ => {context.gemm_type}
            };
        }
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))] {
            if context.gemm_type == GemmType::Avx512 && !std::is_x86_feature_detected!("avx512f") {
                context.gemm_type = if std::is_x86_feature_detected!("avx2") {
                    GemmType::Avx2
                } else {
                    GemmType::Scalar
                }
            } else if !std::is_x86_feature_detected!("avx2") && context.gemm_type == GemmType::Avx2 {
                context.gemm_type = GemmType::Scalar;
            } else if context.gemm_type == GemmType::Neon {
                context.gemm_type = if std::is_x86_feature_detected!("avx512f") {
                    GemmType::Avx512
                } else if std::is_x86_feature_detected!("avx2") {
                    GemmType::Avx2
                } else {
                    GemmType::Scalar
                }
            } else {
                context.gemm_type = GemmType::Scalar;
            }
        }

        #[cfg(target_arch = "aarch64")]{
            if context.gemm_type != GemmType::Neon || context.gemm_type != GemmType::Scalar {
                context.gemm_type = GemmType::Neon;
            } 
        }

        Ok(context)
    }
}

static GLOBAL_CONTEXT: OnceLock<AppContext> = OnceLock::new();

pub fn get_global_context() -> &'static AppContext {
    GLOBAL_CONTEXT.get_or_init(|| {
        let args: Vec<String> = std::env::args().skip(1).collect();
        match AppContext::parse_command_line_arguments(&args) {
            std::result::Result::Ok(context) => context,
            Err(e) => {
                panic!("{e}")
            }
        }
    })
}
