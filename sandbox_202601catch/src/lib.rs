use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};

#[pyfunction]
fn sum_f64(x: PyReadonlyArray1<f64>) -> PyResult<f64> {
    // 連続メモリならコピーせず参照して合計
    let slice = x.as_slice()?;
    Ok(slice.iter().sum())
}

#[pymodule]
fn catch(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_f64, m)?)?;
    Ok(())
}