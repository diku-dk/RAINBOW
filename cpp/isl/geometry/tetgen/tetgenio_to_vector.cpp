#include "tetgenio_to_vector.h"

#include <iostream>
#include <unordered_map>


isl::geometry::tetgen::ReturnVectors isl::geometry::tetgen::error_res()
{
    return {false, std::vector<std::vector<REAL> >(), std::vector<std::vector<int> >(), std::vector<std::vector<int> >()};
}

isl::geometry::tetgen::ReturnVectors isl::geometry::tetgen::tetgenio_to_vector(tetgenio const & out)
{
    using namespace std;
    
    bool succ;
    
    // process points
    if(out.pointlist == nullptr)
    {
        cerr << "tetgenio_to_vector Error: point list is NULL" << endl;
        error_res();
    }
    
    std::vector<std::vector<REAL> > V;
    V.resize(out.numberofpoints,vector<REAL>(3));
    
    // loop over points
    for(int i = 0; i < out.numberofpoints; ++i)
    {
        V[i][0] = out.pointlist[i*3+0];
        V[i][1] = out.pointlist[i*3+1];
        V[i][2] = out.pointlist[i*3+2];
    }
    
    // process tets
    if(out.tetrahedronlist == nullptr)
    {
        cerr << "tetgenio_to_vector Error: tet list is NULL" << endl;
        error_res();
    }
    
    // Each tetrahedron occupies numberofcorners (4 or 10) ints. 
    // When initializerd the variable ’numberofcorners’, which is 4 (a tetrahedron has 4 nodes).
    assert(out.numberofcorners == 4);
    
    std::vector<std::vector<int> > T;
    T.resize(out.numberoftetrahedra,vector<int>(out.numberofcorners));
    int min_index = 1e7;
    int max_index = -1e7;
    // loop over tetrahedra
    for(int i = 0; i < out.numberoftetrahedra; ++i)
    {
        for(int j = 0; j<out.numberofcorners; ++j)
        {
            int index = out.tetrahedronlist[i * out.numberofcorners + j];
            T[i][j] = index;
            min_index = (min_index > index ? index : min_index);
            max_index = (max_index < index ? index : max_index);
        }
    }

    std::vector<std::vector<int> > F;
    F.clear();
    // loop over tetrahedra
    for(int k = 0; k < out.numberoftrifaces; ++k)
    {
      vector<int> face(3);
      for(int j = 0; j<3; j++)
      {
        face[j] = out.trifacelist[k * 3 + j];
      }
      F.push_back(face);
    }
    
    isl::geometry::tetgen::ReturnVectors collect = {true, V, T, F};
    return collect;
}
