#[inline(always)]
pub fn aprox_silu_f32(x: f32) -> f32 {
    if x < -4.0 {
        0.0
    } else if x > 4.0 {
        x
    } else {
        let a = 0.25;
        x * (0.5 + a * x - a * x.abs() * x / 8.0)
    }
}

#[inline(always)]
pub fn aprox_silu_f64(x: f64) -> f64 {
    if x < -4.0 {
        0.0
    } else if x > 4.0 {
        x
    } else {
        let a = 0.25;
        x * (0.5 + a * x - a * x.abs() * x / 8.0)
    }
}

#[inline(always)]
pub fn aprox_sigmoid_f32(x: f32) -> f32 {
    let x = x + f32::EPSILON;
    aprox_silu_f32(x) / x
}

#[inline(always)]
pub fn aprox_sigmoid_f64(x: f64) -> f64 {
    let x = x + f64::EPSILON;
    aprox_silu_f64(x) / x
}

#[inline(always)]
pub fn relu_f32(x: f32) -> f32 {
    x.max(0.0)
}

#[inline(always)]
pub fn relu_f64(x: f64) -> f64 {
    x.max(0.0)
}
