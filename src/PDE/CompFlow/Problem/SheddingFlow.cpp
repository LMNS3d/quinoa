// *****************************************************************************
/*!
  \file      src/PDE/CompFlow/Problem/SheddingFlow.cpp
  \copyright 2012-2015 J. Bakosi,
             2016-2018 Los Alamos National Security, LLC.,
             2019 Triad National Security, LLC.
             All rights reserved. See the LICENSE file for details.
  \brief     Problem configuration for the compressible flow equations
  \details   This file defines a Problem policy class for the compressible flow
    equations, defined in PDE/CompFlow/CompFlow.h. See PDE/CompFlow/Problem.h
    for general requirements on Problem policy classes for CompFlow.
*/
// *****************************************************************************

#include "SheddingFlow.hpp"
#include "Inciter/InputDeck/InputDeck.hpp"
#include "FieldOutput.hpp"

namespace inciter {

extern ctr::InputDeck g_inputdeck;

} // ::inciter

using inciter::CompFlowProblemSheddingFlow;

tk::SolutionFn::result_type
CompFlowProblemSheddingFlow::solution( ncomp_t system,
                                       [[maybe_unused]] ncomp_t ncomp,
                                       tk::real,
                                       tk::real,
                                       tk::real,
                                       tk::real )
// *****************************************************************************
//! Evaluate initial solution at (x,y,z,t) for all components
//! \param[in] system Equation system index, i.e., which compressible
//!   flow equation system we operate on among the systems of PDEs
//! \param[in] ncomp Number of scalar components in this PDE system
//! \param[in] x X coordinate where to evaluate the solution
//! \return Values of all components evaluated at (x)
//! \note The function signature must follow tk::SolutionFn
// *****************************************************************************
{
  Assert( ncomp == ncomp, "Number of scalar components must be " +
                          std::to_string(ncomp) );
  using tag::param;

  // Assign uniform initial condition according to the farfield state
  auto r = g_inputdeck.get< tag::param, eq,
                            tag::farfield_density >()[ system ];
  auto p = g_inputdeck.get< tag::param, eq,
                            tag::farfield_pressure >()[ system ];
  auto u = g_inputdeck.get< tag::param, eq,
                            tag::farfield_velocity >()[ system ];
  auto rE = eos_totalenergy< eq >( system, r, u[0], u[1], u[2], p );

  return {{ r, r*u[0], r*u[1], r*u[2], rE }};
}

std::vector< tk::real >
CompFlowProblemSheddingFlow::solinc( ncomp_t system, ncomp_t ncomp, tk::real x,
  tk::real y, tk::real z, tk::real t, tk::real dt ) const
// *****************************************************************************
// Evaluate the increment from t to t+dt of the analytical solution at (x,y,z)
// for all components
//! \param[in] system Equation system index, i.e., which compressible
//!   flow equation system we operate on among the systems of PDEs
//! \param[in] ncomp Number of scalar components in this PDE system
//! \param[in] x X coordinate where to evaluate the solution
//! \param[in] y Y coordinate where to evaluate the solution
//! \param[in] z Z coordinate where to evaluate the solution
//! \param[in] t Time where to evaluate the solution increment starting from
//! \param[in] dt Time increment at which evaluate the solution increment to
//! \return Increment in values of all components evaluated at (x,y,z,t+dt)
// *****************************************************************************
{
  auto st1 = solution( system, ncomp, x, y, z, t );
  auto st2 = solution( system, ncomp, x, y, z, t+dt );

  std::transform( begin(st1), end(st1), begin(st2), begin(st2),
                  []( tk::real s, tk::real& d ){ return d -= s; } );

  return st2;
}

tk::SrcFn::result_type
CompFlowProblemSheddingFlow::src( ncomp_t, ncomp_t, tk::real,
                                  tk::real, tk::real, tk::real )
// *****************************************************************************
//  Compute and return source term for manufactured solution
//! \return Array of reals containing the source for all components
//! \note The function signature must follow tk::SrcFn
// *****************************************************************************
{
  return {{ 0.0, 0.0, 0.0, 0.0, 0.0 }};
}

std::vector< std::string >
CompFlowProblemSheddingFlow::fieldNames( ncomp_t ) const
// *****************************************************************************
// Return field names to be output to file
//! \return Vector of strings labelling fields output in file
// *****************************************************************************
{
  const auto pref = inciter::g_inputdeck.get< tag::pref, tag::pref >();

  auto n = CompFlowFieldNames();

  if(pref)
    n.push_back( "number of degree of freedom" );

  return n;
}

std::vector< std::vector< tk::real > >
CompFlowProblemSheddingFlow::fieldOutput(
  ncomp_t system,
  ncomp_t,
  ncomp_t offset,
  tk::real,
  tk::real,
  const std::vector< tk::real >&,
  const std::array< std::vector< tk::real >, 3 >&,
  tk::Fields& U ) const
// *****************************************************************************
//  Return field output going to file
//! \param[in] system Equation system index, i.e., which compressible
//!   flow equation system we operate on among the systems of PDEs
//! \param[in] offset System offset specifying the position of the system of
//!   PDEs among other systems
//! \param[in] U Solution vector at recent time step
//! \return Vector of vectors to be output to file
// *****************************************************************************
{
  return CompFlowFieldOutput(system, offset, U);
}

std::vector< std::string >
CompFlowProblemSheddingFlow::names( ncomp_t ) const
// *****************************************************************************
//  Return names of integral variables to be output to diagnostics file
//! \return Vector of strings labelling integral variables output
// *****************************************************************************
{
  return { "r", "ru", "rv", "rw", "re" };
}