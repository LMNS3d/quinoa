// *****************************************************************************
/*!
  \file      src/Mesh/DerivedData.h
  \copyright 2012-2015, J. Bakosi, 2016-2017, Los Alamos National Security, LLC.
  \brief     Generate data structures derived from unstructured mesh
  \details   Generate data structures derived from the connectivity information
     of an unstructured mesh.
*/
// *****************************************************************************
#ifndef DerivedData_h
#define DerivedData_h

#include <vector>
#include <map>
#include <utility>
#include <cstddef>

namespace tk {

//! Generate derived data structure, elements surrounding points
std::pair< std::vector< std::size_t >, std::vector< std::size_t > >
genEsup( const std::vector< std::size_t >& inpoel, std::size_t nnpe );

//! Generate derived data structure, points surrounding points
std::pair< std::vector< std::size_t >, std::vector< std::size_t > >
genPsup( const std::vector< std::size_t >& inpoel,
         std::size_t nnpe,
         const std::pair< std::vector< std::size_t >,
                          std::vector< std::size_t > >& esup );

//! Generate derived data structure, edges surrounding points
std::pair< std::vector< std::size_t >, std::vector< std::size_t > >
genEdsup( const std::vector< std::size_t >& inpoel,
          std::size_t nnpe,
          const std::pair< std::vector< std::size_t >,
                           std::vector< std::size_t > >& esup );

//! Generate derived data structure, edge connectivity
std::vector< std::size_t >
genInpoed( const std::vector< std::size_t >& inpoel,
           std::size_t nnpe,
           const std::pair< std::vector< std::size_t >,
                            std::vector< std::size_t > >& esup );

//! Generate derived data structure, elements surrounding points of elements
std::pair< std::vector< std::size_t >, std::vector< std::size_t > >
genEsupel( const std::vector< std::size_t >& inpoel,
           std::size_t nnpe,
           const std::pair< std::vector< std::size_t >,
                            std::vector< std::size_t > >& esup );

//! Generate derived data structure, elements surrounding elements
std::pair< std::vector< std::size_t >, std::vector< std::size_t > >
genEsuel( const std::vector< std::size_t >& inpoel,
          std::size_t nnpe,
          const std::pair< std::vector< std::size_t >,
                           std::vector< std::size_t > >& esup );

//! \brief Generate derived data structure, elements surrounding elements
//!   as a fixed length data structure as a full vector, including boundary
//!   elements as -1.
std::vector< int >
genEsuelTet( const std::vector< std::size_t >& inpoel,
             const std::pair< std::vector< std::size_t >,
                              std::vector< std::size_t > >& esup );

//! Generate derived data structure, edges of elements
std::vector< std::size_t >
genInedel( const std::vector< std::size_t >& inpoel,
           std::size_t nnpe,
           const std::vector< std::size_t >& inpoed );

//! Generate derived data structure, elements surrounding edges
std::pair< std::vector< std::size_t >, std::vector< std::size_t > >
genEsued( const std::vector< std::size_t >& inpoel,
          std::size_t nnpe,
          const std::pair< std::vector< std::size_t >,
                           std::vector< std::size_t > >& esup );

//! Generate total number of boundary faces in this chunk
std::size_t
genNbfacTet( std::size_t tnbfac,
             const std::vector< std::size_t >& inpoel,
             const std::vector< std::size_t >& t_triinpoel,
             std::vector< std::size_t >& triinpoel );

//! Generate derived data structure, total number of faces
std::size_t
genNtfac( std::size_t nfpe,
          std::size_t nbfac,
          const std::vector< int >& esuelTet );

//! Generate derived data structure, elements surrounding faces
std::vector< int >
genEsuf( std::size_t nfpe,
         std::size_t ntfac,
         std::size_t nbfac,
         const std::vector< std::size_t >& belem,
         const std::vector< int >& esuelTet );

//! Generate derived data structure, node-face connectivity
std::vector< std::size_t >
genInpofaTet( std::size_t ntfac,
              std::size_t nbfac,
              const std::vector< std::size_t >& inpoel,
              const std::vector< std::size_t >& triinpoel,
              const std::vector< int >& esuelTet );

//! Generate derived data structure, host/boundary element
std::vector< std::size_t >
genBelemTet( std::size_t nbfac,
              const std::vector< std::size_t >& inpofa,
              const std::pair< std::vector< std::size_t >,
                               std::vector< std::size_t > >& esup );
} // tk::

#endif // DerivedData_h
