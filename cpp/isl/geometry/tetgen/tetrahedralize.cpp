#include "tetrahedralize.h"
#include "vector_to_tetgenio.h"
#include "tetgenio_to_vector.h"

// STL includes
#include <cassert>
#include <iostream>


isl::geometry::tetgen::ReturnVectors1 isl::geometry::tetgen::error_res1()
{
    return {false, std::vector<std::vector<REAL> >(), std::vector<std::vector<int> >(), std::vector<std::vector<int> >()};
}

isl::geometry::tetgen::ReturnVectors1 isl::geometry::tetgen::tetrahedralize(
                                          std::vector<std::vector<REAL > > const & V
                                          , std::vector<std::vector<int > > const & F
                                          , const std::string switches
                                          , bool save
                                          , const std::string path
                                          )
{
  using namespace std;
  
  int succ;
  tetgenio in;
  tetgenio out;

  bool success = vector_to_tetgenio(V, F, in);
  if(!success)
  {
    error_res1();
  }

  try
  {
    char * cswitches = new char[switches.size() + 1];
    std::strcpy(cswitches,switches.c_str());
    ::tetrahedralize(cswitches,&in, &out);
    delete[] cswitches;
  }
  catch(int e)
  {
    cerr << "Error in "
         << __FUNCTION__
         << ": Tetgen crashed"
         << endl;
    error_res1();
  }

  if(out.numberoftetrahedra == 0)
  {
    cerr << "Error in "
         << __FUNCTION__
         << ": Tetgen failed to create tetrahedra"
         << endl;
    error_res1();
  }
  
  isl::geometry::tetgen::ReturnVectors collect = tetgenio_to_vector(out);
  if(!collect.succ)
  {
    error_res1();
  }
  
  isl::geometry::tetgen::ReturnVectors1 collectDone = {0, collect.V, collect.T, collect.F};
  
  if(save)
  {
      char * cpath = new char[path.size() + 1];
      std::strcpy(cpath,path.c_str());
      out.save_nodes(cpath);
      out.save_elements(cpath);
      out.save_faces(cpath);
      out.save_edges(cpath);
      out.save_faces2smesh(cpath);
  }
  
  return collectDone;
}

// Given a background mesh
isl::geometry::tetgen::ReturnVectors1 isl::geometry::tetgen::tetrahedralize(
                                          std::vector<std::vector<REAL > > const & V
                                          , std::vector<std::vector<int > > const & F
                                          , const std::string switches
                                          , std::vector<std::vector<REAL > > const & VB
                                          , std::vector<std::vector<int > > const & TB
                                          , std::vector<std::vector<REAL > > const & AB
                                          , bool save
                                          , const std::string path
                                          )
{
  using namespace std;
  
  int succ;
  tetgenio in;
  tetgenio out;
  tetgenio addin;
  tetgenio bgmin;

  bool success = vector_to_tetgenio(V, F, in);
  if(!success)
  {
    error_res1();
  }
  
  success = vector_to_tetgenio(VB, TB, AB, bgmin);
  if(!success)
  {
    cerr << "Error when making bgmin" << endl;
    return error_res1();
  }
  
  try
  {
    char * cswitches = new char[switches.size() + 1];
    std::strcpy(cswitches,switches.c_str());
    ::tetrahedralize(cswitches, &in, &out, NULL, &bgmin);
    delete[] cswitches;
  }
  catch(int e)
  {
    cerr << "Error in "
         << __FUNCTION__
         << ": Tetgen crashed with bgmesh"
         << endl;
    return error_res1();
  }
  
  if(out.numberoftetrahedra == 0)
  {
    cerr << "Error in "
         << __FUNCTION__
         << ": Tetgen failed to create tetrahedra"
         << endl;
    return error_res1();
  }
  
  isl::geometry::tetgen::ReturnVectors collect = tetgenio_to_vector(out);
  if(!collect.succ)
  {
    return error_res1();
  }
  
  isl::geometry::tetgen::ReturnVectors1 collectDone = {0, collect.V, collect.T, collect.F};
  
  if(save)
  {
      char * cpath = new char[path.size() + 1];
      std::strcpy(cpath,path.c_str());
      out.save_nodes(cpath);
      out.save_elements(cpath);
      out.save_faces(cpath);
      out.save_edges(cpath);
      out.save_faces2smesh(cpath);
  }
  
  return collectDone;
}
