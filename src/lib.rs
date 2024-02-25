use ndarray::{Array1, ArrayD};
use numpy::{ PyArrayDyn,  PyReadonlyArrayDyn, ToPyArray};
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
    input: PyReadonlyArrayDyn<f32>,
    shape_sin: usize,
    alpha: f32,
    vertical: bool,
    bias: f32,
    py: Python,
) -> PyResult<Py<PyArrayDyn<f32>>> {
    let array = input.as_array();
    let tile = sin_tile(shape_sin, alpha);
    let (height, width, depth) = (
        array.shape()[0],
        array.shape()[1],
        array.shape().get(2).cloned().unwrap_or(1),
    );
    let shape_sin = shape_sin as i16;
    let mut result_array = ArrayD::<f32>::zeros(array.raw_dim());
    let mut i: i16;


    if vertical {
        if depth>1{
            for lx in 0..height {
                let xx = (lx as f32 * bias) as i16;
                for ly in 0..width {
                    for lz in 0..depth {
                        i = (ly as i16 - xx) % shape_sin;
                        if i < 0 {
                            i += shape_sin
                        }

                        result_array[[lx, ly, lz]] =
                            (array[[lx, ly, lz]] + tile[i as usize]).max(0.0).min(1.0);
                    }
                }
            }
        }else{
            for lx in 0..height {
                let xx = (lx as f32 * bias) as i16;
                for ly in 0..width {
                    i = (ly as i16 - xx) % shape_sin;
                    if i < 0 {
                        i += shape_sin
                    }

                    result_array[[lx, ly]] =
                        (array[[lx, ly]] + tile[i as usize]).max(0.0).min(1.0);}
                }

        }} else {
        if depth > 1 {
            for ly in 0..width {
                let yy = (ly as f32 * bias) as i16;
                for lx in 0..height {
                    for lz in 0..depth {
                        i = (yy - lx as i16) % shape_sin;
                        if i < 0 {
                            i = i + shape_sin
                        }

                        result_array[[lx, ly, lz]] =
                            (array[[lx, ly, lz]] + tile[i as usize]).max(0.0).min(1.0);
                    }
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

                    result_array[[lx, ly]] =
                        (array[[lx, ly]] + tile[i as usize]).max(0.0).min(1.0);
                }
            }
        }
    }



    Ok(result_array.to_pyarray(py).to_owned())
}
#[pyfunction]
fn color_levels(
    input: PyReadonlyArrayDyn<f32>,
    in_low: f32,
    in_high: f32,
    out_low: f32,
    out_high: f32,
    gamma: f32,
    py: Python,
) -> PyResult<Py<PyArrayDyn<f32>>> {
    let array = input.as_array();
    let in_range = in_high - in_low;
    let out_range = out_high - out_low;
    let result_array = array.mapv(|x| ((x - in_low) / (in_range) * (out_range) + out_low).max(0.0).min(1.0).powf(gamma));

    // Возвращаем результат в виде PyArrayDyn<f32>
    Ok(result_array.to_pyarray(py).to_owned())
}

#[pymodule]
fn dataset_support(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sin_patern, m)?)?;
    m.add_function(wrap_pyfunction!(color_levels,m)?)?;
    Ok(())
}
