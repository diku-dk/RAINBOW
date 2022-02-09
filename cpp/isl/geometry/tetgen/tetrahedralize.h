#ifndef ISL_GEOMETRY_TETGEN_TETRAHEEDRALIZE_H
#define ISL_GEOMETRY_TETGEN_TETRAHEEDRALIZE_H

#include <tetgen.h> // Needed for REAL

#include <vector>
#include <string>

namespace isl
{
  namespace geometry
  {
    namespace tetgen
    {
    
    /**
     * Struct to hold the information needed to be returned
     *
     * @param succ                 Output integer
     * @param V                    Vertex position list
     * @param T                    List of tetrahedra indices into V
     * @param F                    List of marked facets
     */
    /**typedef struct TetraDone {
       TetraDone(int succ, 
                 std::vector<std::vector<REAL>> V, 
                 std::vector<std::vector<int>> T,
                 std::vector<std::vector<int>> F) : succ(succ), V(V), T(T), F(F) { }
        
       int succ;
       std::vector<std::vector<REAL > > V;
       std::vector<std::vector<int > > T;
       std::vector<std::vector<int > > F;
     } TetraDone;*/
     typedef struct ReturnVectors1 {
        bool succ;
        std::vector<std::vector<REAL > > V;
        std::vector<std::vector<int > > T;
        std::vector<std::vector<int > > F;
      } ReturnVectors1;
     
     /**
       * Create the struct with empty vectores.
       * It is intended to help limit the implementation needed
       */
     ReturnVectors1 error_res1();
    
     /**
     * Mesh the interior of a surface mesh (V,F) using Tetgen
     *
     * @param V                    Inputs a #V by 3 vertex position list
     * @param F                    Inputs a #F list of polygon face indices into V (0-indexed)
     * @param switches    This is a string of tetgen options (See tetgen documentation) e.g.
     *                  "pq1.414a0.01" tries to mesh the interior of a given surface with quality
     *                  and area constraints. "" will mesh the convex hull constrained to pass
     *                  through V (ignores F).
     * @return          TetraDone type
     */
    ReturnVectors1 tetrahedralize(
                       std::vector<std::vector<REAL > > const & V
                       , std::vector<std::vector<int > > const & F
                       , const std::string switches
                       , bool save
                       , const std::string path
                       );
        
    ReturnVectors1 tetrahedralize(
                       std::vector<std::vector<REAL > > const & V
                       , std::vector<std::vector<int > > const & F
                       , const std::string switches
                       , std::vector<std::vector<REAL > > const & VB
                       , std::vector<std::vector<int > > const & TB
                       , std::vector<std::vector<REAL > > const & AB
                       , bool save
                       , const std::string path
                       );
   }
  }
}

//ISL_GEOMETRY_TETGEN_TETRAHEEDRALIZE_H
#endif

