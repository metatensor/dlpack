fn main() {
    let config = pyo3_build_config::get();

    if let Some(ref lib_dir) = config.lib_dir {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }
}
