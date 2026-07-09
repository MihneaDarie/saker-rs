use std::env;

#[cfg(target_arch = "aarch64")]
use std::fs;

struct CacheParams {
    l1_bytes: u64,
    l2_bytes: u64,
    l3_bytes: u64,
}

struct GemmBlocking {
    mc: usize,
    kc: usize,
    nc: usize,
    mr: usize,
    nr: usize,
}

impl GemmBlocking {
    fn from_cache(p: &CacheParams) -> Self {
        const F32: u64 = 4;
        
        let (mut mc_div,mut mc_max): (u64, u64) = (2, 256); 

        let (mut nc_div, mut nc_max) = (2, 1024); 
        
        let mut l1_div = 2;
        
        let (mut mr, mut nr) = (4,4);

        #[cfg(target_arch = "x86_64")]
        {   l1_div = 4;
            (mc_div, mc_max) = (2, 64);
            (nc_div, nc_max) = (2, 256);

            (mr, nr) = if std::is_x86_feature_detected!("avx512f") {
                (16, 16)
            } else if std::is_x86_feature_detected!("avx2") {
                (8, 8)
            } else {
                (4, 4)
            };
            

        }

        #[cfg(target_arch = "aarch64")] {
            (mc_div, mc_max) = (8, 64);
            (nc_div, nc_max) = (4, 256);
            (mr, nr) = (8,8);
            l1_div = 4;
        }

        let kc_max: u64 = 256;
        let kc = ((p.l1_bytes / l1_div) / (mr as u64 * F32))
            .next_power_of_two()
            .clamp(64, kc_max) as usize;

        let mc = ((p.l2_bytes / mc_div) / (kc as u64 * F32))
            .next_power_of_two()
            .clamp(32, mc_max) as usize;

        let l3_eff = if p.l3_bytes > 0 { p.l3_bytes } else { p.l2_bytes };
        
        let nc = ((l3_eff / nc_div) / (kc as u64 * F32))
            .next_power_of_two()
            .clamp(64, nc_max) as usize;

        Self { mc, kc, nc, mr, nr }
    }
}

#[cfg(target_arch = "x86_64")]
fn query_caches() -> CacheParams {
    let (mut l1, mut l2, mut l3) = (0u64, 0u64, 0u64);

    for subleaf in 0u32..32 {
        let (eax, ebx, ecx, _edx): (u32, u32, u32, u32);
        unsafe {
            std::arch::asm!(
                "mov {tmp}, rbx",
                "cpuid",
                "mov {ebx_out:e}, ebx",
                "mov rbx, {tmp}",
                tmp = out(reg) _,
                ebx_out = out(reg) ebx,
                inout("eax") 4u32 => eax,
                inout("ecx") subleaf => ecx,
                out("edx") _edx,
                options(nostack, preserves_flags),
            );
        }

        let cache_type = eax & 0x1f;
        if cache_type == 0 { break; }
        if cache_type == 2 { continue; }

        let level = (eax >> 5) & 0x7;
        let line_size = ((ebx & 0xfff) + 1) as u64;
        let partitions = (((ebx >> 12) & 0x3ff) + 1) as u64;
        let ways= (((ebx >> 22) & 0x3ff) + 1) as u64;
        let sets = (ecx as u64) + 1;
        let size = ways * partitions * line_size * sets;

        match level {
            1 => l1 = l1.max(size),
            2 => l2 = l2.max(size),
            3 => l3 = l3.max(size),
            _ => {}
        }
    }

    if l1 == 0 { l1 = 32  * 1024; }
    if l2 == 0 { l2 = 256 * 1024; }
    if l3 == 0 { l3 = 8   * 1024 * 1024; }
 
    CacheParams { l1_bytes: l1, l2_bytes: l2, l3_bytes: l3 }
}


#[cfg(target_arch = "aarch64")]
fn query_caches() -> CacheParams {
    #[cfg(target_os = "macos")]
    {
        fn sysctl_u64(name: &std::ffi::CStr) -> Option<u64> {
            let mut val: u64 = 0;
            let mut size = std::mem::size_of::<u64>();
            let ret = unsafe {
                libc::sysctlbyname(
                    name.as_ptr(),
                    &mut val as *mut u64 as *mut libc::c_void,
                    &mut size,
                    std::ptr::null_mut(),
                    0,
                )
            };
            if ret == 0 { Some(val) } else { None }
        }

        let l1 = sysctl_u64(c"hw.l1dcachesize").unwrap_or(64 * 1024);
        let l2 = sysctl_u64(c"hw.l2cachesize" ).unwrap_or(4  * 1024 * 1024);
        let l3 = sysctl_u64(c"hw.l3cachesize" ).unwrap_or(0);
        return CacheParams { l1_bytes: l1, l2_bytes: l2, l3_bytes: l3 };
    }

    #[cfg(not(target_os = "macos"))]
    {
        fn read_sysfs_cache(index: u32) -> Option<(u32, u64)> {
            let base = format!("/sys/devices/system/cpu/cpu0/cache/index{index}");
            let level: u32 = fs::read_to_string(format!("{base}/level"))
                .ok()?.trim().parse().ok()?;
            let size_str = fs::read_to_string(format!("{base}/size")).ok()?;
            let size_str = size_str.trim();
            let size: u64 = if let Some(k) = size_str.strip_suffix('K') {
                k.parse::<u64>().ok()? * 1024
            } else if let Some(m) = size_str.strip_suffix('M') {
                m.parse::<u64>().ok()? * 1024 * 1024
            } else {
                size_str.parse().ok()?
            };
            Some((level, size))
        }

        let (mut l1, mut l2, mut l3) = (0u64, 0u64, 0u64);
        for i in 0u32..8 {
            if let Some((lvl, sz)) = read_sysfs_cache(i) {
                match lvl {
                    1 => l1 = l1.max(sz),
                    2 => l2 = l2.max(sz),
                    3 => l3 = l3.max(sz),
                    _ => {}
                }
            }
        }
        if l1 == 0 { l1 = 64 * 1024; }
        if l2 == 0 { l2 = 4  * 1024 * 1024; }
        CacheParams { l1_bytes: l1, l2_bytes: l2, l3_bytes: l3 }
    }
}


#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn query_caches() -> CacheParams {
    CacheParams {
        l1_bytes: 32 * 1024,
        l2_bytes: 256 * 1024,
        l3_bytes: 4 * 1024 * 1024,
        has_avx512f: false,
    }
}

fn main() {
    let params   = query_caches();
    let blocking = GemmBlocking::from_cache(&params);

    println!("cargo:warning= === saker build.rs ===");
    println!("cargo:warning=  MC={} KC={} NC={}", blocking.mc, blocking.kc, blocking.nc);
    println!("cargo:warning=  MR={} NR={}", blocking.mr, blocking.nr);


    println!("cargo:rustc-env=SAKER_MC={}", blocking.mc);
    println!("cargo:rustc-env=SAKER_KC={}", blocking.kc);
    println!("cargo:rustc-env=SAKER_NC={}", blocking.nc);
    println!("cargo:rustc-env=SAKER_MR={}", blocking.mr);
    println!("cargo:rustc-env=SAKER_NR={}", blocking.nr);

    println!("cargo:rerun-if-changed=build.rs");
}