// *****************************************************************************
/*!
  \file      src/DiffEq/Dirichlet/GeneralizedDirichletCoeffPolicy.hpp
  \copyright 2012-2015 J. Bakosi,
             2016-2018 Los Alamos National Security, LLC.,
             2019-2020 Triad National Security, LLC.
             All rights reserved. See the LICENSE file for details.
  \brief     Lochner's generalized Dirichlet coefficients policies
  \details   This file defines coefficients policy classes for the generalized
    Dirichlet SDE, defined in DiffEq/GeneralizedDirichlet.h.

    General requirements on generalized Dirichlet SDE coefficients policy
    classes:

    - Must define a _constructor_, which is used to initialize the SDE
      coefficients, b, S, kappa, and c. Required signature:
      \code{.cpp}
        CoeffPolicyName(
          tk::ctr::ncomp_t ncomp,
          const std::vector< kw::sde_b::info::expect::type >& b_,
          const std::vector< kw::sde_S::info::expect::type >& S_,
          const std::vector< kw::sde_kappa::info::expect::type >& k_,
          const std::vector< kw::sde_c::info::expect::type >& c_,
          std::vector< kw::sde_b::info::expect::type  >& b,
          std::vector< kw::sde_S::info::expect::type >& S,
          std::vector< kw::sde_kappa::info::expect::type >& k,
          std::vector< kw::sde_c::info::expect::type >& c )
      \endcode
      where
      - ncomp denotes the number of scalar components of the system of beta
        SDEs.
      - Constant references to b_, S_, k_, and c_, which denote four vectors of
        real values used to initialize the parameter vectors of the generalized
        Dirichlet SDEs. The length of the vectors b_, S_, and kappa_, must be
        equal to the number of components given by ncomp, while the length of
        vector c_ must be ncomp*(ncomp-1)/2.
      - References to b, S, k, and c, which denote the parameter vectors to be
        initialized based on b_, S_, k_, and c_.

    - Must define the static function _type()_, returning the enum value of the
      policy option. Example:
      \code{.cpp}
        static ctr::CoeffPolicyType type() noexcept {
          return ctr::CoeffPolicyType::CONST_COEFF;
        }
      \endcode
      which returns the enum value of the option from the underlying option
      class, collecting all possible options for coefficients policies.
*/
// *****************************************************************************
#ifndef GeneralizedDirichletCoeffPolicy_h
#define GeneralizedDirichletCoeffPolicy_h

#include <brigand/sequences/list.hpp>

#include "Types.hpp"
#include "Walker/Options/CoeffPolicy.hpp"
#include "SystemComponents.hpp"

namespace walker {

//! Generalized Dirichlet constant coefficients policity: constants in time
class GeneralizedDirichletCoeffConst {

  public:
    //! Constructor: initialize coefficients
    GeneralizedDirichletCoeffConst(
      tk::ctr::ncomp_t ncomp,
      const std::vector< kw::sde_b::info::expect::type >& b_,
      const std::vector< kw::sde_S::info::expect::type >& S_,
      const std::vector< kw::sde_kappa::info::expect::type >& k_,
      const std::vector< kw::sde_c::info::expect::type >& c_,
      std::vector< kw::sde_b::info::expect::type >& b,
      std::vector< kw::sde_S::info::expect::type >& S,
      std::vector< kw::sde_kappa::info::expect::type >& k,
      std::vector< kw::sde_c::info::expect::type >& c );

    static ctr::CoeffPolicyType type() noexcept
    { return ctr::CoeffPolicyType::CONST_COEFF; }
};

//! List of all generalized Dirichlet's coefficients policies
using GeneralizedDirichletCoeffPolicies =
  brigand::list< GeneralizedDirichletCoeffConst >;

} // walker::

#endif // GeneralizedDirichletCoeffPolicy_h
