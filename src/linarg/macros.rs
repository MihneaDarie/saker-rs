#[macro_export]
macro_rules! set1_ps_simd {
    ($func:path, $a:expr, $lda:expr, $p:expr => $( $name:ident : $i:expr ),+ $(,)?) => {
        $(
            let $name = $func(*$a.add($i * $lda + $p));
        )+
    };
}

#[macro_export]
macro_rules! accumulate_simd {
    ($load_func:path, $store_func:path, $add_func:path, $c:expr, $ldc:expr=> $( $name:ident : $i:expr ),+ $(,)?) => {
        $(
            let old = $load_func($c.add($i * $ldc));
            $store_func($c.add($i * $ldc), $add_func(old,$name));
        )+
    };
}

#[macro_export]
macro_rules! storeu_ps_simd {
    ($store_func:path, $c:expr, $ldc:expr=> $( $name:ident : $i:expr ),+ $(,)?) => {
        $(
            $store_func($c.add($i * $ldc), $name);
        )+
    };
}

#[macro_export]
macro_rules! loadu_ps_simd {
    ($func:path, $a:expr, $lda:expr, $p:expr => $( $name:ident : $i:expr ),+ $(,)?) => {
        $(
            let $name = $func(*$a.add($i * $lda + $p));
        )+
    };
}

#[macro_export]
macro_rules! set_zero_simd {
    ($func:path, $( $name:ident ),+ $(,)? ) => {
       $(
            let mut $name = $func();
       )+
    };
}

#[macro_export]
macro_rules! fmadd_ps_simd {
    ($func:path, $b:expr => $( $name_a:ident : $name_c:ident ),+ $(,)?) => {
        $(
          $name_c = $func($name_a, $b, $name_c);
        )+
    };
}



#[macro_export]
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