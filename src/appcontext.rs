use std::{collections::HashSet, sync::OnceLock};

#[repr(u8)]
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    #[default]
    Cpu,
    Gpu,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmType {
    Avx2,
    Avx512,
    Scalar,
    None,
}

impl Default for GemmType {
    fn default() -> Self {
        if std::is_x86_feature_detected!("avx512f") {
            Self::Avx512
        } else if std::is_x86_feature_detected!("avx2") {
            Self::Avx2
        } else {
            Self::Scalar
        }
    }
}

#[derive(Default, Debug)]
pub struct AppContext {
    device: Device,
    gemm_type: GemmType,
}

impl AppContext {
    fn new(device: Device, gemm_type: GemmType) -> Self {
        Self { device, gemm_type }
    }

    pub fn get_device(&self) -> Device {
        self.device
    }

    pub fn get_gemm_type(&self) -> GemmType {
        self.gemm_type
    }

    pub fn parse_command_line_arguments(args: &[String]) -> Result<Self, String> {
        let mut context = AppContext::default();

        if args.is_empty() {
            return Ok(context);
        }

        if args.len().is_multiple_of(2){
            return Err("Odd number of arguments !".to_string());
        }

        let mut seen_flags = HashSet::new();
        for i in (0..args.len()).step_by(2) {
            if !seen_flags.insert(&args[i]) {
                return Err(format!("Duplicate flag: {}", args[i]));
            }
        }

        let valid_args: HashSet<&str> = ["--camera", "-c", "--device", "-d", "--type", "-t"]
            .into_iter()
            .collect();

        for i in (0..args.len()).step_by(2) {
            let flag = &args[i];
            let value = &args[i + 1];

            if !valid_args.contains(flag.as_str()) {
                return Err(format!("Invalid argument {}
                \n Valid arguments are: [--camera , -c, --device, -d, --type, -t]
                \n|--camera, -c| -> select the camera from you computer you would like to use
                \n|--device, -d| Select the device ypu wuold like the model to run on (CPU(default)/GPU)
                \n|--type, -t| If the used device is the CPU you can choose what type of gemm you model is using",
                flag));
            }

            if args.contains(&String::from("gpu"))
                && (args.contains(&String::from("--type")) || args.contains(&String::from("-t")))
            {
                return Err("Can't set cpu specific features for gpu usage !".to_string());
            }

            if !flag.starts_with('-') {
                return Err(format!("Expected a flag, got: {}", flag));
            }

            if value.starts_with('-') {
                return Err(format!(
                    "Expected a value after {}, got another flag: {}",
                    flag, value,
                ));
            }

            match flag.as_str() {
                "--device" | "-d" => {
                    context.device = match value.to_lowercase().as_str() {
                        "cpu" => {
                            context.gemm_type = GemmType::Avx2;
                            Device::Cpu
                        }
                        "gpu" => {
                            context.gemm_type = GemmType::None;
                            Device::Gpu
                        }
                        _ => {
                            return Err(format!(
                                "Unknown device type: {}. Use 'cpu' or 'gpu'.",
                                value
                            ));
                        }
                    };
                }
                "--type" | "-t" => {
                    context.gemm_type = match value.to_lowercase().as_str() {
                        "avx2" => GemmType::Avx2,
                        "avx512" => GemmType::Avx512,
                        "scalar" => GemmType::Scalar,
                        _ => {
                            return Err(format!(
                                "Unknown GEMM type: {}. Use 'scalar', 'avx2', or 'avx512'.",
                                value
                            ));
                        }
                    };
                }
                _ => {}
            }
        }

        match context.check_context_compatibility() {
            Ok(()) => Ok(context),
            Err(e) => Err(e),
        }
    }

    pub fn check_context_compatibility(&mut self) -> Result<(), String> {
        if self.device == Device::Gpu && self.gemm_type != GemmType::None {
            return Err("Can't use cpu features on gpu !".to_string());
        }

        if self.device == Device::Cpu {
            match self.gemm_type {
                GemmType::Avx2 => {
                    if !std::is_x86_feature_detected!("avx2")
                        || !std::is_x86_feature_detected!("fma")
                    {
                        println!("AVX2 selected but it is not avaible, switching to scalar !");
                        self.gemm_type = GemmType::Scalar;
                    }
                }
                GemmType::Avx512 => {
                    if !std::is_x86_feature_detected!("fma")
                        || !std::is_x86_feature_detected!("avx512f")
                    {
                        let mut message =
                            "AVX512f selected but it is not avaible, switching to avx2 !"
                                .to_string();
                        self.gemm_type = GemmType::Avx2;
                        if !std::is_x86_feature_detected!("avx2")
                            || !std::is_x86_feature_detected!("fma")
                        {
                            message = "AVX512f selected but it is not avaible, neither avx2, switching to scalar !".to_string();
                            self.gemm_type = GemmType::Scalar;
                        }
                        println!("{message}");
                    }
                }
                _ => {}
            }
        }

        Ok(())
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
