#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <tight_inclusion/ccd.hpp>


namespace py = pybind11;

using namespace ticcd;

//TODO: Problably expose some of the parameters.
std::tuple<bool,float> runEdgeEdge(Vector3 a0s,
                                   Vector3 a1s,
                                   Vector3 b0s,
                                   Vector3 b1s,
                                   Vector3 a0e,
                                   Vector3 a1e,
                                   Vector3 b0e,
                                   Vector3 b1e
                                   )
                                   {
                                       Eigen::Array<Scalar, 3, 1> err = {-1,-1,-1};
                                       Scalar ms = 1e-3;
                                       Scalar toi;
                                       const Scalar tolerance = 1e-6;
                                       Scalar t_max = 1;
                                       Scalar max_itr = 10000;
                                       Scalar out_tol;
                                       bool result = ticcd::edgeEdgeCCD(a0s,a1s,b0s,b1s,a0e,a1e,b0e,b1e,err,ms,toi,tolerance,t_max,max_itr,out_tol);
                                       return std::tuple<bool,float>{result,(float)toi};
                                       

}

std::tuple<bool,float> runVertexFace(Vector3 vs,
                                   Vector3 f0s,
                                   Vector3 f1s,
                                   Vector3 f2s,
                                   Vector3 ve,
                                   Vector3 f0e,
                                   Vector3 f1e,
                                   Vector3 f2e
                                   )
                                   {
                                       Eigen::Array<Scalar, 3, 1> err = {-1,-1,-1};
                                       Scalar ms = 1e-3;
                                       Scalar toi;
                                       const Scalar tolerance = 1e-6;
                                       Scalar t_max = 1;
                                       Scalar max_itr = 10000;
                                       Scalar out_tol;
                                       bool result = ticcd::vertexFaceCCD(vs,f0s,f1s,f2s,ve,f0e,f1e,f2e,err,ms,toi,tolerance,t_max,max_itr,out_tol);
                                       return std::tuple<bool,float>{result,(float)toi};
}

PYBIND11_MODULE(TightInclusionWrap, m) {
    m.doc() = "A Wrapper allowing the use of the \"Tight Inclusion\" library";
    m.def("edgeEdgeCCD", &runEdgeEdge, R"pbdoc(
        Compares 2 edges up against eachother
    )pbdoc");

    m.def("vertexFaceCCD", &runVertexFace, R"pbdoc(
        Compares a vertex against a face triangle
    )pbdoc");
}