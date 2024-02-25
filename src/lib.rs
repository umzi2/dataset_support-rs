use ndarray::{Array1};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

fn sin_tile(shape_sin: usize, alpha: f32) -> Array1<f32> {
    let mut shape_sin_zero: Array1<f32> = ndarray::Array::zeros(shape_sin);
    for x in 0..shape_sin {
        shape_sin_zero[x] = (3.14 * x as f32 / (shape_sin as f32 / 2.0)).sin() * alpha;
    }
    shape_sin_zero
}

#[pyfunction]
fn sin_patern(
    input: PyReadonlyArray2<f32>,
    shape_sin: usize,
    alpha: f32,
    vertical: bool,
    bias: f32,
    py: Python,
) -> PyResult<Py<PyArray2<f32>>> {
    let mut  array = input.as_array().to_owned();
    let tile = sin_tile(shape_sin, alpha);
    let (height, width) = (array.shape()[0], array.shape()[1]);
    let shape_sin = shape_sin as i16;
    let mut i :i16;

    if vertical {
        for lx in 0..height {
            let xx = (lx as f32 * bias) as i16;
            for ly in 0..width {
                i = (ly as i16 - xx) % shape_sin;
                if i < 0 {
                    i += shape_sin
                }

                array[[lx, ly]] = (array[[lx, ly]] + tile[i as usize]).max(0.0).min(1.0);
            }
        }
    } else {
        for ly in 0..width {
            let yy = (ly as f32 * bias) as i16;
            for lx in 0..height {
                i = (yy - lx as i16) % shape_sin;
                if i < 0 {
                    i = i + shape_sin
                }

                array[[lx, ly]] = (array[[lx, ly]] + tile[i as usize]).max(0.0).min(1.0);
            }
        }
    }

    Ok(array.into_pyarray(py).to_owned())
}

#[pymodule]
fn dataset_support(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sin_patern, m)?)?;
    Ok(())
}
