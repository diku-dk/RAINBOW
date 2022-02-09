#include "vector_to_tetgenio.h"

#include <cassert>
#include <iostream>

bool isl::geometry::tetgen::vector_to_tetgenio( std::vector<std::vector<REAL > > const & V
                                               , std::vector<std::vector<int > > const & F
                                               , tetgenio & in
                                               )
{
    using namespace std;
  
    tetgenio::facet *f;
    tetgenio::polygon *p;

    in.firstnumber = 0;

    in.numberofpoints = V.size();
    in.pointlist = new REAL[in.numberofpoints * 3];
    for(int i = 0; i < (int)V.size(); i++)
    {
        assert(V[i].size() == 3);
        in.pointlist[i*3+0] = V[i][0];
        in.pointlist[i*3+1] = V[i][1];
        in.pointlist[i*3+2] = V[i][2];
    }

    in.numberoffacets = F.size();
    in.facetlist = new tetgenio::facet[in.numberoffacets];
    in.facetmarkerlist = new int[in.numberoffacets];

    for(int i = 0; i < (int)F.size(); i++)
    {
        in.facetmarkerlist[i] = i;
        f = &in.facetlist[i];
        f->numberofpolygons = 1;
        f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
        f->numberofholes = 0;
        f->holelist = NULL;
        p = &f->polygonlist[0];
        p->numberofvertices = F[i].size();
        p->vertexlist = new int[p->numberofvertices];
        for(int j = 0; j < (int)F[i].size(); j++)
        {
          p->vertexlist[j] = F[i][j];
        }
    }
    
    return true;
}

bool isl::geometry::tetgen::vector_to_tetgenio( std::vector<std::vector<REAL > > const & V
                                               , std::vector<std::vector<int > > const & T
                                               , std::vector<std::vector<REAL > > const & A
                                               , tetgenio & in
                                               )
{
    using namespace std;
    in.firstnumber = 0;
    
    in.numberofpoints = V.size();
    in.pointlist = new REAL[in.numberofpoints * 3];
    for(int i = 0; i < (int)V.size(); i++)
    {
        assert(V[i].size() == 3);
        in.pointlist[i*3+0] = V[i][0];
        in.pointlist[i*3+1] = V[i][1];
        in.pointlist[i*3+2] = V[i][2];
    }
    
    in.numberoftetrahedra = T.size();
    in.numberofcorners = 4;
    in.tetrahedronlist = new int[in.numberoftetrahedra * in.numberofcorners];
    int min_index = 1e7;
    int max_index = -1e7;
    for(int i = 0; i < in.numberoftetrahedra; i++)
    {
        for(int j = 0; j< in.numberofcorners; j++)
        {
            int index = T[i][j];
            in.tetrahedronlist[i * in.numberofcorners + j] = index;
            min_index = (min_index > index ? index : min_index);
            max_index = (max_index < index ? index : max_index);
        }
    }
    
    in.numberofpointmtrs = 1;
    in.pointmtrlist = new REAL[in.numberofpoints * in.numberofpointmtrs];
    for(int i = 0; i < (int)A.size(); i++)
    {
        assert(A[i].size() == 1);
        in.pointmtrlist[i] = A[i][0];
    }
    
    return true;
}
