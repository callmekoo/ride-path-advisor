//! Python bindings for the path search kernel.
//!
//! The kernel knows nothing about tags, preferences or penalties: the caller
//! supplies one weight per arc and gets paths back.

mod search;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use search::Csr;

type SsspArrays<'py> = (
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<u32>>,
);

#[pyclass(frozen)]
struct Graph {
    csr: Csr,
}

impl Graph {
    fn check_node(&self, name: &str, node: u32) -> PyResult<()> {
        if (node as usize) < self.csr.n_nodes() {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "{name}={node} is out of range: the graph has {} nodes",
                self.csr.n_nodes()
            )))
        }
    }

    fn check_len(&self, name: &str, actual: usize, expected: usize) -> PyResult<()> {
        if actual == expected {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "{name} has {actual} entries, expected {expected}"
            )))
        }
    }
}

#[pymethods]
impl Graph {
    #[new]
    fn new(
        edge_u: PyReadonlyArray1<'_, u32>,
        edge_v: PyReadonlyArray1<'_, u32>,
        node_lonlat: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let lonlat = node_lonlat.as_array();
        if lonlat.ncols() != 2 {
            return Err(PyValueError::new_err("node_lonlat must have shape [N, 2]"));
        }
        let points: Vec<[f64; 2]> = (0..lonlat.nrows())
            .map(|i| [lonlat[[i, 0]], lonlat[[i, 1]]])
            .collect();
        let csr = Csr::new(edge_u.as_slice()?, edge_v.as_slice()?, &points)
            .map_err(PyValueError::new_err)?;
        Ok(Graph { csr })
    }

    #[getter]
    fn n_nodes(&self) -> usize {
        self.csr.n_nodes()
    }

    #[getter]
    fn n_edges(&self) -> usize {
        self.csr.n_edges()
    }

    /// Cheapest path from `src` to `dst`: `(arc_ids, cost)` or `None`.
    #[pyo3(signature = (src, dst, w, h_scale = 0.999))]
    fn route<'py>(
        &self,
        py: Python<'py>,
        src: u32,
        dst: u32,
        w: PyReadonlyArray1<'py, f32>,
        h_scale: f64,
    ) -> PyResult<Option<(Bound<'py, PyArray1<u32>>, f64)>> {
        let w = w.as_slice()?;
        self.check_len("w", w.len(), 2 * self.csr.n_edges())?;
        self.check_node("src", src)?;
        self.check_node("dst", dst)?;
        let found = py.detach(|| self.csr.route(src, dst, w, h_scale));
        Ok(found.map(|(arcs, cost)| (arcs.into_pyarray(py), cost)))
    }

    /// Cheapest costs from `src` to all nodes (or to `src` with `reverse`):
    /// `(cost, summed_aux, pred_arc)`.
    #[pyo3(signature = (src, w, aux, max_cost = f64::INFINITY, reverse = false))]
    fn sssp<'py>(
        &self,
        py: Python<'py>,
        src: u32,
        w: PyReadonlyArray1<'py, f32>,
        aux: PyReadonlyArray1<'py, f32>,
        max_cost: f64,
        reverse: bool,
    ) -> PyResult<SsspArrays<'py>> {
        let w = w.as_slice()?;
        let aux = aux.as_slice()?;
        self.check_len("w", w.len(), 2 * self.csr.n_edges())?;
        self.check_len("aux", aux.len(), self.csr.n_edges())?;
        self.check_node("src", src)?;
        let (cost, aux_sum, pred) = py.detach(|| self.csr.sssp(src, w, aux, max_cost, reverse));
        Ok((
            cost.into_pyarray(py),
            aux_sum.into_pyarray(py),
            pred.into_pyarray(py),
        ))
    }
}

#[pymodule]
mod rpa_kernel {
    #[pymodule_export]
    use super::Graph;
}
