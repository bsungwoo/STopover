#include "make_original_dendrogram.h"
#include "make_smoothed_dendrogram.h"
#include "make_dendrogram_bar.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

// Expose to Python via Pybind11
PYBIND11_MODULE(connected_components, m) {  // Module name within the STopover package
    m.def("make_original_dendrogram", &make_original_dendrogram_cc, "make_original_dendrogram",
          py::arg("U"), py::arg("A"), py::arg("threshold"));

    m.def("make_smoothed_dendrogram", &make_smoothed_dendrogram, "make_smoothed_dendrogram",
          py::arg("cCC"), py::arg("cE"), py::arg("cduration"), py::arg("chistory"), 
          py::arg("lim_size"));

    m.def("make_dendrogram_bar", &make_dendrogram_bar, "make_dendrogram_bar",
          py::arg("history"), py::arg("duration"), py::arg("cvertical_x"),
          py::arg("cvertical_y"), py::arg("chorizontal_x"), py::arg("chorizontal_y"),
          py::arg("cdots"));
}