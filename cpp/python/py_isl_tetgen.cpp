#include <isl/geometry/tetgen/tetrahedralize.h>

#include <pybind11/pybind11.h>
//#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

namespace py = pybind11;

// Shows prints in jupyter
py::scoped_ostream_redirect stream_cout(
  std::cout,                               // std::ostream&
  py::module::import("sys").attr("stdout") // Python output
);

// Shows error messages in jupyter
py::scoped_ostream_redirect stream_cerr(
  std::cerr,                               // std::ostream&
  py::module::import("sys").attr("stderr") // Python output
);

PYBIND11_MODULE(pyisl, m)
{
    
    m.doc() = "This is our python wrapper for C++ code.";
    
    /** TETGEN DEFINITIONS */
    // Define the struct for TetGen
    py::class_<isl::geometry::tetgen::ReturnVectors1>(m, "Return Vectors")
        .def_readwrite("succ", &isl::geometry::tetgen::ReturnVectors1::succ)
        .def_readwrite("V", &isl::geometry::tetgen::ReturnVectors1::V)
        .def_readwrite("T", &isl::geometry::tetgen::ReturnVectors1::T)
        .def_readwrite("F", &isl::geometry::tetgen::ReturnVectors1::F);
    
    auto mtet = m.def_submodule("tetgen");
    mtet.doc() = "Python version of TetGen's tetrahedralize functions";
    
    // Define the function
    mtet.def("tetrahedralize", [](const std::vector<std::vector<REAL>> & V,
                                  const std::vector<std::vector<int>> & F,
                                  const std::string & switches,
                                  bool & save,
                                  const std::string & path
                                 ){
                 return isl::geometry::tetgen::tetrahedralize(V, F, switches, save, path);
             },
            "Tetrahedralize with the parameters given.");
    
    mtet.def("tetrahedralize", [](const std::vector<std::vector<REAL>> &V, 
                                  const std::vector<std::vector<int>> &F,
                                  const std::string &switches,
                                  const std::vector<std::vector<REAL>> &VB, 
                                  const std::vector<std::vector<int>> &TB,
                                  const std::vector<std::vector<REAL>> &AB,
                                  bool &save,
                                  const std::string &path
                                 ){
        return isl::geometry::tetgen::tetrahedralize(V, F, switches, VB, TB, AB, save, path);
        }, 
        "Tetrahedralize with background mesh.");
}
