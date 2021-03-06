// *****************************************************************************
/*!
  \file      src/PDE/CompFlow/Problem/RayleighTaylor.hpp
  \copyright 2012-2015 J. Bakosi,
             2016-2018 Los Alamos National Security, LLC.,
             2019-2020 Triad National Security, LLC.
             All rights reserved. See the LICENSE file for details.
  \brief     Problem configuration for the compressible flow equations
  \details   This file defines a policy class for the compressible flow
    equations, defined in PDE/CompFlow/CompFlow.h. See PDE/CompFlow/Problem.h
    for general requirements on Problem policy classes for CompFlow.
*/
// *****************************************************************************
#ifndef CompFlowProblemRayleighTaylor_h
#define CompFlowProblemRayleighTaylor_h

#include <string>
#include <unordered_set>

#include "Types.hpp"
#include "Fields.hpp"
#include "FunctionPrototypes.hpp"
#include "SystemComponents.hpp"
#include "Inciter/Options/Problem.hpp"
#include "Inciter/InputDeck/InputDeck.hpp"

namespace inciter {

extern ctr::InputDeck g_inputdeck;

//! CompFlow system of PDEs problem: Rayleigh-Taylor
//! \see Waltz, et. al, "Manufactured solutions for the three-dimensional Euler
//!   equations with relevance to Inertial Confinement Fusion", Journal of
//!   Computational Physics 267 (2014) 196-209.
class CompFlowProblemRayleighTaylor {

  private:
    using ncomp_t = tk::ctr::ncomp_t;
    using eq = tag::compflow;

  public:
    //! Evaluate analytical solution at (x,y,z,t) for all components
    static tk::SolutionFn::result_type
    solution( ncomp_t system, ncomp_t ncomp, tk::real x, tk::real y, tk::real z,
              tk::real t, int& );

    //! Compute and return source term for Rayleigh-Taylor manufactured solution
    //! \param[in] system Equation system index, i.e., which compressible
    //!   flow equation system we operate on among the systems of PDEs
    //! \param[in] x X coordinate where to evaluate the solution
    //! \param[in] y Y coordinate where to evaluate the solution
    //! \param[in] z Z coordinate where to evaluate the solution
    //! \param[in] t Physical time at which to evaluate the source
    //! \param[in,out] r Density source
    //! \param[in,out] ru X momentum source
    //! \param[in,out] rv Y momentum source
    //! \param[in,out] rw Z momentum source
    //! \param[in,out] re Specific total energy source
    //! \note The function signature must follow tk::SrcFn
    static tk::CompFlowSrcFn::result_type
    src( ncomp_t system, tk::real x, tk::real y, tk::real z, tk::real t,
         tk::real& r, tk::real& ru, tk::real& rv, tk::real& rw, tk::real& re )
    {
      using tag::param; using std::sin; using std::cos;

      // manufactured solution parameters
      auto a = g_inputdeck.get< param, eq, tag::alpha >()[system];
      auto bx = g_inputdeck.get< param, eq, tag::betax >()[system];
      auto by = g_inputdeck.get< param, eq, tag::betay >()[system];
      auto bz = g_inputdeck.get< param, eq, tag::betaz >()[system];
      auto k = g_inputdeck.get< param, eq, tag::kappa >()[system];
      auto p0 = g_inputdeck.get< param, eq, tag::p0 >()[system];
      // ratio of specific heats
      tk::real g = g_inputdeck.get< param, eq, tag::gamma >()[system][0];

      // evaluate solution at x,y,z,t
      int inbox = 0;
      auto s = solution( system, 5, x, y, z, t, inbox );

      // density, velocity, energy, pressure
      auto rho = s[0];
      auto u = s[1]/s[0];
      auto v = s[2]/s[0];
      auto w = s[3]/s[0];
      auto E = s[4]/s[0];
      auto p = p0 + a*(bx*x*x + by*y*y + bz*z*z);

      // spatial gradients
      std::array< tk::real, 3 > drdx{{ -2.0*bx*x, -2.0*by*y, -2.0*bz*z }};
      std::array< tk::real, 3 > dpdx{{ 2.0*a*bx*x, 2.0*a*by*y, 2.0*a*bz*z }};
      tk::real ft = cos(k*M_PI*t);
      std::array< tk::real, 3 > dudx{{ ft*M_PI*z*cos(M_PI*x),
                                       0.0,
                                       ft*sin(M_PI*x) }};
      std::array< tk::real, 3 > dvdx{{ 0.0,
                                       -ft*M_PI*z*sin(M_PI*y),
                                       ft*cos(M_PI*y) }};
      std::array< tk::real, 3 > dwdx{{ ft*M_PI*0.5*M_PI*z*z*sin(M_PI*x),
                                       ft*M_PI*0.5*M_PI*z*z*cos(M_PI*y),
                                      -ft*M_PI*z*(cos(M_PI*x) - sin(M_PI*y)) }};
      std::array< tk::real, 3 > dedx{{
        dpdx[0]/rho/(g-1.0) - p/(g-1.0)/rho/rho*drdx[0]
        + u*dudx[0] + v*dvdx[0] + w*dwdx[0],
        dpdx[1]/rho/(g-1.0) - p/(g-1.0)/rho/rho*drdx[1]
        + u*dudx[1] + v*dvdx[1] + w*dwdx[1],
        dpdx[2]/rho/(g-1.0) - p/(g-1.0)/rho/rho*drdx[2]
        + u*dudx[2] + v*dvdx[2] + w*dwdx[2] }};

      // time derivatives
      auto dudt = -k*M_PI*sin(k*M_PI*t)*z*sin(M_PI*x);
      auto dvdt = -k*M_PI*sin(k*M_PI*t)*z*cos(M_PI*y);
      auto dwdt =  k*M_PI*sin(k*M_PI*t)/2*M_PI*z*z*(cos(M_PI*x) - sin(M_PI*y));
      auto dedt = u*dudt + v*dvdt + w*dwdt;

      // density source
      r = u*drdx[0] + v*drdx[1] + w*drdx[2];
      // momentum source
      ru = rho*dudt+u*r+dpdx[0] + s[1]*dudx[0]+s[2]*dudx[1]+s[3]*dudx[2];
      rv = rho*dvdt+v*r+dpdx[1] + s[1]*dvdx[0]+s[2]*dvdx[1]+s[3]*dvdx[2];
      rw = rho*dwdt+w*r+dpdx[2] + s[1]*dwdx[0]+s[2]*dwdx[1]+s[3]*dwdx[2];
      // energy source
      re = rho*dedt + E*r + s[1]*dedx[0]+s[2]*dedx[1]+s[3]*dedx[2]
           + u*dpdx[0]+v*dpdx[1]+w*dpdx[2];
    }

    //! Return field names to be output to file
    std::vector< std::string > fieldNames( ncomp_t ) const;

    //! Return field output going to file
    std::vector< std::vector< tk::real > >
    fieldOutput( ncomp_t system,
                 ncomp_t ncomp,
                 ncomp_t offset,
                 std::size_t nunk,
                 tk::real t,
                 tk::real V,
                 const std::vector< tk::real >& vol,
                 const std::array< std::vector< tk::real >, 3 >& coord,
                 tk::Fields& U ) const;

    //! Return names of integral variables to be output to diagnostics file
    std::vector< std::string > names( ncomp_t /*ncomp*/ ) const;

    //! Return problem type
    static ctr::ProblemType type() noexcept
    { return ctr::ProblemType::RAYLEIGH_TAYLOR; }
};

} // inciter::

#endif // CompFlowProblemRayleighTaylor_h
