// *****************************************************************************
/*!
  \file      src/PDE/Indicator.hpp
  \copyright 2012-2015 J. Bakosi,
             2016-2018 Los Alamos National Security, LLC.,
             2019 Triad National Security, LLC.
             All rights reserved. See the LICENSE file for details.
  \brief     Adaptive indicators for p-adaptive discontiunous Galerkin methods
  \details   This file contains functions that provide adaptive indicator
    function calculations for marking the number of degree of freedom of each 
    element.
*/
// *****************************************************************************
#ifndef Indicator_h
#define Indicator_h

#include <array>
#include <vector>
#include <algorithm>

#include "Types.hpp"
#include "Fields.hpp"
#include "DerivedData.hpp"
#include "UnsMesh.hpp"
#include "Integrate/Quadrature.hpp"
#include "Integrate/Basis.hpp"
#include "Inciter/InputDeck/InputDeck.hpp"

namespace inciter {

using ncomp_t = kw::ncomp::info::expect::type;

//! Evaluate the adaptive indicator and mark the ndof for each element
void eval_ndof( const std::size_t nunk,
                const tk::UnsMesh::Coords& coord,
                const std::vector< std::size_t >& inpoel,
                const inciter::FaceData& fd,
                const tk::Fields& unk,
                std::vector< std::size_t >& ndofel );

//! Evaluate the spectral-decay indicator and mark the ndof for each element
void spectral_decay( const std::size_t nunk,
                     const std::vector< int >& esuel,
                     const tk::Fields& unk,
                     std::vector< std::size_t >& ndofel );

//! Evaluate the non-conformity indicator and mark the ndof for each element
void non_conformity( const std::size_t nunk,
                     const std::size_t Nbfac,
                     const std::vector< std::size_t >& inpoel,
                     const tk::UnsMesh::Coords& coord,
                     const std::vector< int >& esuel,
                     const std::vector< int > esuf,
                     const std::vector< std::size_t >& inpofa,
                     const tk::Fields& unk,
                     std::vector< std::size_t >& ndofel );

} // inciter::

#endif // Indicator_h
