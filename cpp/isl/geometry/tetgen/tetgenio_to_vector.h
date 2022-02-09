#ifndef ISL_GEOMETRY_TETGEN_TETGENIO_TO_VECTOR_H
#define ISL_GEOMETRY_TETGEN_TETGENIO_TO_VECTOR_H

#include <tetgen.h> // Needed for tetgenio, REAL

#include <vector>


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
      typedef struct ReturnVectors {
        bool succ;
        std::vector<std::vector<REAL > > V;
        std::vector<std::vector<int > > T;
        std::vector<std::vector<int > > F;
      } ReturnVectors;
      
      /**
       * Create the struct with empty vectores.
       * It is intended to help limit the implementation needed
       */
      ReturnVectors error_res();
    
      /**
       * Extract a tetrahedral mesh from a tetgenio object
       *
       * @param out   tetgenio output object
       * @return      Tetra type
       */
      ReturnVectors tetgenio_to_vector(tetgenio const & out);
    }
  }
}


//ISL_GEOMETRY_TETGEN_TETGENIO_TO_VECTOR_H
#endif
