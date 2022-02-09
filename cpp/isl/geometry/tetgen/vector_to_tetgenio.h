#ifndef ISL_GEOMETRY_TETGEN_VECTOR_TO_TETGENIO_H
#define ISL_GEOMETRY_TETGEN_VECTOR_TO_TETGENIO_H

#include <tetgen.h> // Defined tetgenio, REAL

#include <vector>

namespace isl
{
  namespace geometry
  {
    namespace tetgen
    {
      /**
       * Load a vertex list and face list into a tetgenio object.
       *
       * @param V    #V by 3 vertex position list
       * @param F    #F list of polygon face indices into V (0-indexed)
       * @param in  Upon return this tetgenio object contains the V and F data.
       * @return   Returns true on success, false on error.
       */
       bool vector_to_tetgenio( std::vector<std::vector<REAL > > const & V
                                   , std::vector<std::vector<int > > const & F
                                   , tetgenio & in
                                   );
        
       bool vector_to_tetgenio( std::vector<std::vector<REAL > > const & V
                                  , std::vector<std::vector<int > > const & T
                                  , std::vector<std::vector<REAL > > const & A
                                  , tetgenio & in
                                  );
    }
  }
}

//ISL_GEOMETRY_TETGEN_VECTOR_TO_TETGENIO_H
#endif

