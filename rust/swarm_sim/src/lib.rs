use pyo3::prelude::*;

pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn version() -> &'static str {
    crate_version()
}

#[pymodule]
fn swarm_sim(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(version, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::crate_version;

    #[test]
    fn crate_version_matches_package_version() {
        assert_eq!(crate_version(), "0.1.0");
    }
}
