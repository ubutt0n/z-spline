#include "pybind11/pybind11.h"
#include "Z-spline.h"
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(zspline, m)
{
	py::class_<Z_spline<double>>(m, "Z_spline")
        .def(py::init<int, std::vector<double>&, std::vector<double>&>())
        .def("__call__", [](const Z_spline<double>& zsp, double x) {return zsp(x);})
        .def("__call__", [](const Z_spline<double>& zsp, std::vector<double> x) {return zsp(x);});
}