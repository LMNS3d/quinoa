// *****************************************************************************
/*!
  \file      src/PDE/Integrate/MultiMatTerms.cpp
  \copyright 2012-2015 J. Bakosi,
             2016-2018 Los Alamos National Security, LLC.,
             2019-2020 Triad National Security, LLC.
             All rights reserved. See the LICENSE file for details.
  \brief     Functions for computing volume integrals of multi-material terms
     using DG methods
  \details   This file contains functionality for computing volume integrals of
     non-conservative and pressure relaxation terms that appear in the
     multi-material hydrodynamic equations, using the discontinuous Galerkin
     method for various orders of numerical representation.
*/
// *****************************************************************************

#include "MultiMatTerms.hpp"
#include "Vector.hpp"
#include "Quadrature.hpp"
#include "EoS/EoS.hpp"
#include "MultiMat/MultiMatIndexing.hpp"

namespace tk {

void
nonConservativeInt( [[maybe_unused]] ncomp_t system,
                    std::size_t nmat,
                    ncomp_t offset,
                    const std::size_t ndof,
                    const std::size_t rdof,
                    const std::vector< std::size_t >& inpoel,
                    const UnsMesh::Coords& coord,
                    const Fields& geoElem,
                    const Fields& U,
                    const Fields& P,
                    const std::vector< std::vector< tk::real > >&
                      riemannDeriv,
                    const std::vector< std::size_t >& ndofel,
                    Fields& R )
// *****************************************************************************
//  Compute volume integrals for multi-material DG
//! \details This is called for multi-material DG, computing volume integrals of
//!   terms in the volume fraction and energy equations, which do not exist in
//!   the single-material flow formulation (for `CompFlow` DG). For further
//!   details see Pelanti, M., & Shyue, K. M. (2019). A numerical model for
//!   multiphase liquid–vapor–gas flows with interfaces and cavitation.
//!   International Journal of Multiphase Flow, 113, 208-230.
//! \param[in] system Equation system index
//! \param[in] nmat Number of materials in this PDE system
//! \param[in] offset Offset this PDE system operates from
//! \param[in] ndof Maximum number of degrees of freedom
//! \param[in] rdof Maximum number of reconstructed degrees of freedom
//! \param[in] inpoel Element-node connectivity
//! \param[in] coord Array of nodal coordinates
//! \param[in] geoElem Element geometry array
//! \param[in] U Solution vector at recent time step
//! \param[in] P Vector of primitive quantities at recent time step
//! \param[in] riemannDeriv Derivatives of partial-pressures and velocities
//!   computed from the Riemann solver for use in the non-conservative terms
//! \param[in] ndofel Vector of local number of degrees of freedome
//! \param[in,out] R Right-hand side vector added to
// *****************************************************************************
{
  using inciter::volfracIdx;
  using inciter::densityIdx;
  using inciter::momentumIdx;
  using inciter::energyIdx;
  using inciter::velocityIdx;

  const auto& cx = coord[0];
  const auto& cy = coord[1];
  const auto& cz = coord[2];

  auto ncomp = U.nprop()/rdof;
  auto nprim = P.nprop()/rdof;

  // compute volume integrals
  for (std::size_t e=0; e<U.nunk(); ++e)
  {
    auto ng = tk::NGvol(ndofel[e]);

    // arrays for quadrature points
    std::array< std::vector< real >, 3 > coordgp;
    std::vector< real > wgp;

    coordgp[0].resize( ng );
    coordgp[1].resize( ng );
    coordgp[2].resize( ng );
    wgp.resize( ng );

    GaussQuadratureTet( ng, coordgp, wgp );

    // Extract the element coordinates
    std::array< std::array< real, 3>, 4 > coordel {{
      {{ cx[ inpoel[4*e  ] ], cy[ inpoel[4*e  ] ], cz[ inpoel[4*e  ] ] }},
      {{ cx[ inpoel[4*e+1] ], cy[ inpoel[4*e+1] ], cz[ inpoel[4*e+1] ] }},
      {{ cx[ inpoel[4*e+2] ], cy[ inpoel[4*e+2] ], cz[ inpoel[4*e+2] ] }},
      {{ cx[ inpoel[4*e+3] ], cy[ inpoel[4*e+3] ], cz[ inpoel[4*e+3] ] }}
    }};

    auto jacInv =
            inverseJacobian( coordel[0], coordel[1], coordel[2], coordel[3] );

    // Compute the derivatives of basis function for DG(P1)
    std::array< std::vector<tk::real>, 3 > dBdx;
    if (ndofel[e] > 1)
      dBdx = eval_dBdx_p1( ndofel[e], jacInv );

    // Gaussian quadrature
    for (std::size_t igp=0; igp<ng; ++igp)
    {
      if (ndofel[e] > 4)
        eval_dBdx_p2( igp, coordgp, jacInv, dBdx );

      // If an rDG method is set up (P0P1), then, currently we compute the P1
      // basis functions and solutions by default. This implies that P0P1 is
      // unsupported in the p-adaptive DG (PDG).
      std::size_t dof_el;
      if (rdof > ndof)
      {
        dof_el = rdof;
      }
      else
      {
        dof_el = ndofel[e];
      }

      // Compute the basis function
      auto B =
        eval_basis( dof_el, coordgp[0][igp], coordgp[1][igp], coordgp[2][igp] );

      auto wt = wgp[igp] * geoElem(e, 0, 0);

      auto ugp = eval_state( ncomp, offset, rdof, dof_el, e, U, B );
      auto pgp = eval_state( nprim, offset, rdof, dof_el, e, P, B );

      // get bulk properties
      tk::real rhob(0.0);
      for (std::size_t k=0; k<nmat; ++k)
          rhob += ugp[densityIdx(nmat, k)];

      std::array< tk::real, 3 > vel{{ pgp[velocityIdx(nmat, 0)],
                                      pgp[velocityIdx(nmat, 1)],
                                      pgp[velocityIdx(nmat, 2)] }};

      std::vector< tk::real > ymat(nmat, 0.0);
      std::array< tk::real, 3 > dap{{0.0, 0.0, 0.0}};
      for (std::size_t k=0; k<nmat; ++k)
      {
        ymat[k] = ugp[densityIdx(nmat, k)]/rhob;

        for (std::size_t idir=0; idir<3; ++idir)
          dap[idir] += riemannDeriv[3*k+idir][e];
      }

      // compute non-conservative terms
      std::vector< tk::real > ncf(ncomp, 0.0);

      for (std::size_t idir=0; idir<3; ++idir)
        ncf[momentumIdx(nmat, idir)] = 0.0;

      for (std::size_t k=0; k<nmat; ++k)
      {
        ncf[densityIdx(nmat, k)] = 0.0;
        ncf[volfracIdx(nmat, k)] = ugp[volfracIdx(nmat, k)]
                                   * riemannDeriv[3*nmat][e];
        for (std::size_t idir=0; idir<3; ++idir)
          ncf[energyIdx(nmat, k)] -= vel[idir] * ( ymat[k]*dap[idir]
                                                - riemannDeriv[3*k+idir][e] );
      }

      update_rhs_ncn( ncomp, offset, ndof, ndofel[e], wt, e, dBdx, ncf, R );
    }
  }
}

void
update_rhs_ncn(
  ncomp_t ncomp,
  ncomp_t offset,
  const std::size_t ndof,
  [[maybe_unused]] const std::size_t ndof_el,
  const tk::real wt,
  const std::size_t e,
  [[maybe_unused]] const std::array< std::vector<tk::real>, 3 >& dBdx,
  const std::vector< tk::real >& ncf,
  Fields& R )
// *****************************************************************************
//  Update the rhs by adding the non-conservative term integrals
//! \param[in] ncomp Number of scalar components in this PDE system
//! \param[in] offset Offset this PDE system operates from
//! \param[in] ndof Maximum number of degrees of freedom
//! \param[in] ndof_el Number of degrees of freedom for local element
//! \param[in] wt Weight of gauss quadrature point
//! \param[in] e Element index
//! \param[in] dBdx Vector of basis function derivatives
//! \param[in] ncf Vector of non-conservative terms
//! \param[in,out] R Right-hand side vector computed
// *****************************************************************************
{
  //Assert( dBdx[0].size() == ndof_el,
  //        "Size mismatch for basis function derivatives" );
  //Assert( dBdx[1].size() == ndof_el,
  //        "Size mismatch for basis function derivatives" );
  //Assert( dBdx[2].size() == ndof_el,
  //        "Size mismatch for basis function derivatives" );
  //Assert( ncf.size() == ncomp,
  //        "Size mismatch for non-conservative term" );
  Assert( ncf.size() == ncomp, "Size mismatch for non-conservative term" );

  for (ncomp_t c=0; c<ncomp; ++c)
  {
    auto mark = c*ndof;
    R(e, mark, offset) += wt * ncf[c];
  }
}

void
pressureRelaxationInt( ncomp_t system,
                       std::size_t nmat,
                       std::size_t e,
                       ncomp_t offset,
                       const std::size_t ndof,
                       const std::size_t rdof,
                       real volElem,
                       const Fields& U,
                       const Fields& P,
                       std::size_t ndofel,
                       const tk::real ct,
                       Fields& R )
// *****************************************************************************
//  Compute volume integrals of pressure relaxation terms in multi-material DG
//! \details This is called for multi-material DG to compute volume integrals of
//!   finite pressure relaxation terms in the volume fraction and energy
//!   equations, which do not exist in the single-material flow formulation (for
//!   `CompFlow` DG). For further details see Dobrev, V. A., Kolev, T. V.,
//!   Rieben, R. N., & Tomov, V. Z. (2016). Multi‐material closure model for
//!   high‐order finite element Lagrangian hydrodynamics. International Journal
//!   for Numerical Methods in Fluids, 82(10), 689-706.
//! \param[in] system Equation system index
//! \param[in] nmat Number of materials in this PDE system
//! \param[in] e Element for which pressure relaxation is to be computed
//! \param[in] offset Offset this PDE system operates from
//! \param[in] ndof Maximum number of degrees of freedom
//! \param[in] rdof Maximum number of reconstructed degrees of freedom
//! \param[in] volElem Element volume
//! \param[in] U Solution vector at recent time step
//! \param[in] P Vector of primitive quantities at recent time step
//! \param[in] ndofel Local number of degrees of freedome
//! \param[in] ct Pressure relaxation time-scale for this system
//! \param[in,out] R Right-hand side vector added to
// *****************************************************************************
{
  using inciter::volfracIdx;
  using inciter::densityIdx;
  using inciter::momentumIdx;
  using inciter::energyIdx;
  using inciter::pressureIdx;
  using inciter::velocityIdx;

  auto ncomp = U.nprop()/rdof;
  auto nprim = P.nprop()/rdof;

  // compute volume integrals
  auto dx = std::cbrt(volElem);
  auto ng = NGvol(ndofel);

  // arrays for quadrature points
  std::array< std::vector< real >, 3 > coordgp;
  std::vector< real > wgp;

  coordgp[0].resize( ng );
  coordgp[1].resize( ng );
  coordgp[2].resize( ng );
  wgp.resize( ng );

  GaussQuadratureTet( ng, coordgp, wgp );

  // Compute the derivatives of basis function for DG(P1)
  std::array< std::vector<real>, 3 > dBdx;

  // Gaussian quadrature
  for (std::size_t igp=0; igp<ng; ++igp)
  {
    // If an rDG method is set up (P0P1), then, currently we compute the P1
    // basis functions and solutions by default. This implies that P0P1 is
    // unsupported in the p-adaptive DG (PDG).
    std::size_t dof_el;
    if (rdof > ndof)
    {
      dof_el = rdof;
    }
    else
    {
      dof_el = ndofel;
    }

    // Compute the basis function
    auto B =
      eval_basis( dof_el, coordgp[0][igp], coordgp[1][igp], coordgp[2][igp] );

    auto wt = wgp[igp] * volElem;

    auto ugp = eval_state( ncomp, offset, rdof, dof_el, e, U, B );
    auto pgp = eval_state( nprim, offset, rdof, dof_el, e, P, B );

    // calculate equilibrium pressure and builk-modulii
    std::vector< real > kmat(nmat, 0.0);
    real apow(1.0), almax(0.0), pb(0.0), trelax(0.0), p_relax(0.0);
    equilibriumPressure(system, nmat, ct, dx, apow, ugp, pgp, kmat, almax, pb,
      trelax, p_relax);

    // compute pressure relaxation terms
    std::vector< real > s_prelax(ncomp, 0.0);
    for (std::size_t k=0; k<nmat; ++k)
    {
      auto s_alpha = (pgp[pressureIdx(nmat, k)]
        -p_relax*ugp[volfracIdx(nmat, k)])
        * std::pow(ugp[volfracIdx(nmat, k)], apow)
        * std::pow(almax, 1.0-apow) / (trelax*kmat[k]);
      s_prelax[volfracIdx(nmat, k)] = s_alpha;
      s_prelax[energyIdx(nmat, k)] = - pb*s_alpha;
    }

    update_rhs_ncn( ncomp, offset, ndof, ndofel, wt, e, dBdx, s_prelax, R );
  }
}

void
pressureRelaxationJac( ncomp_t system,
                       std::size_t nmat,
                       std::size_t e,
                       ncomp_t offset,
                       const std::size_t ndof,
                       const std::size_t rdof,
                       const Fields& geoElem,
                       const Fields& U,
                       const Fields& P,
                       std::size_t dof_el,
                       const tk::real ct,
                       std::vector< tk::real >& J,
                       tk::Fields& rhs_si )
// *****************************************************************************
//  Compute element Jacobian of pressure relaxation terms in multi-material DG
//! \details This is called for multi-material DG to compute Jacobian of
//!   finite pressure relaxation terms in the volume fraction and energy
//!   equations, which do not exist in the single-material flow formulation (for
//!   `CompFlow` DG).
//! \param[in] system Equation system index
//! \param[in] nmat Number of materials in this PDE system
//! \param[in] e Element whose Jacobian is to be computed
//! \param[in] offset Offset this PDE system operates from
//! \param[in] ndof Maximum number of degrees of freedom
//! \param[in] rdof Maximum number of reconstructed degrees of freedom
//! \param[in] geoElem Element geometry array
//! \param[in] U Solution vector at recent time step
//! \param[in] P Vector of primitive quantities at recent time step
//! \param[in] dof_el Local number of degrees of freedom
//! \param[in] ct Pressure relaxation time-scale for this system
//! \param[in,out] J Jacobian vector computed
//! \param[in,out] rhs_si Semi-implicit RHS vector computed
// *****************************************************************************
{
  using inciter::volfracIdx;
  using inciter::volfracDofIdx;
  using inciter::energyDofIdx;
  using inciter::pressureIdx;

  auto ncomp = U.nprop()/rdof;
  auto nprim = P.nprop()/rdof;

  Assert( J.size() == ncomp*ndof, "Size mismatch for Jacobian vector" );

  auto dx = std::cbrt(geoElem(e, 0, 0));
  auto ng = NGvol(dof_el);

  // arrays for quadrature points
  std::array< std::vector< real >, 3 > coordgp;
  std::vector< real > wgp;

  coordgp[0].resize( ng );
  coordgp[1].resize( ng );
  coordgp[2].resize( ng );
  wgp.resize( ng );

  GaussQuadratureTet( ng, coordgp, wgp );

  // Gaussian quadrature
  for (std::size_t igp=0; igp<ng; ++igp)
  {
    // Compute the basis function
    auto B =
      eval_basis( dof_el, coordgp[0][igp], coordgp[1][igp], coordgp[2][igp] );

    auto wt = wgp[igp] * geoElem(e, 0, 0);

    auto ugp = eval_state( ncomp, offset, rdof, dof_el, e, U, B );
    auto pgp = eval_state( nprim, offset, rdof, dof_el, e, P, B );

    // calculate equilibrium pressure and builk-modulii
    std::vector< real > kmat(nmat, 0.0);
    real apow(1.0), almax(0.0), pb(0.0), trelax(0.0), p_relax(0.0);
    equilibriumPressure(system, nmat, ct, dx, apow, ugp, pgp, kmat, almax, pb,
      trelax, p_relax);

    real deno = 0.0;
    for (std::size_t k=0; k<nmat; ++k)
      deno += ugp[volfracIdx(nmat, k)]/kmat[k];

    // compute pressure relaxation Jacobian terms
    for (std::size_t k=0; k<nmat; ++k)
    {
      auto gk =
        inciter::g_inputdeck.get< tag::param, tag::multimat, tag::gamma >()[ system ][k];
      auto dsda = (pgp[pressureIdx(nmat,k)] - p_relax*ugp[volfracIdx(nmat, k)])
        / (trelax*kmat[k]);
      auto dsdE = -((gk - 1.0) - ((gk-1.0)*ugp[volfracIdx(nmat, k)]/kmat[k])/deno)
        * pb * ugp[volfracIdx(nmat, k)]/(trelax*kmat[k]);

      J[volfracDofIdx(nmat, k, ndof, 0)] += wt * dsda;
      J[energyDofIdx(nmat, k, ndof, 0)] += wt * dsdE;
      rhs_si(0, volfracDofIdx(nmat, k, ndof, 0), offset) += wt * dsda;
      rhs_si(0, energyDofIdx(nmat, k, ndof, 0), offset) -= wt * dsda * pb;
    }
  }
}

void equilibriumPressure(ncomp_t system,
  std::size_t nmat,
  real ct,
  real dx,
  real apow,
  std::vector< real >& u_gp,
  std::vector< real >& p_gp,
  std::vector< real >& kmat,
  real& almax,
  real& pb,
  real& trelax,
  real& p_relax)
// *****************************************************************************
//  Calculate equilibrium pressure and associated quantities at quadrature point
//! \details This function calculates equilibrium pressure between materials in
//!   the cell, and associated quantities required for the pressure relaxation
//!   term.
//! \param[in] system Equation system index
//! \param[in] nmat Number of materials in this PDE system
//! \param[in] ct Pressure relaxation time-scale for this system
//! \param[in] dx Characteristic length scale for pressure relaxation
//! \param[in] apow Power for volume-fraction in relaxation term
//! \param[in,out] u_gp Reconstructed solution at quadrature point
//! \param[in,out] p_gp Reconstructed primitive quantities at quadrature point
//! \param[in,out] kmat Vector of material bulk-modulii
//! \param[in,out] almax Maximum volume-fraction at quadrature point
//! \param[in,out] pb Bulk pressure
//! \param[in,out] trelax Relaxation time
//! \param[in,out] p_relax Expected equilibrium pressure
// *****************************************************************************
{
  using inciter::volfracIdx;
  using inciter::densityIdx;
  using inciter::momentumIdx;
  using inciter::energyIdx;
  using inciter::pressureIdx;
  using inciter::velocityIdx;

  auto al_eps = 1.0e-12;

  // get majority material, and correct trace materials
  std::size_t kmax(0);
  real al_corr(0.0),
    rmin(std::numeric_limits< real >::max()),
    rmax(std::numeric_limits< real >::min());
  almax = 0.0;
  for (std::size_t k=0; k<nmat; ++k)
  {
    const auto al = u_gp[volfracIdx(nmat, k)];
    rmin = std::min(rmin, u_gp[densityIdx(nmat, k)]/al);
    rmax = std::max(rmax, u_gp[densityIdx(nmat, k)]/al);

    if (al > almax)
    {
      almax = al;
      kmax = k;
    }

    if (al < al_eps)
    {
      al_corr -= al_eps - al;
      u_gp[volfracIdx(nmat, k)] = al_eps;
      u_gp[densityIdx(nmat, k)] *= al_eps/al;
      p_gp[pressureIdx(nmat, k)] *= al_eps/al;
    }
  }

  u_gp[volfracIdx(nmat, kmax)] += al_corr;
  u_gp[densityIdx(nmat, kmax)] *= u_gp[volfracIdx(nmat, kmax)]/almax;
  p_gp[pressureIdx(nmat, kmax)] *= u_gp[volfracIdx(nmat, kmax)]/almax;
  almax = u_gp[volfracIdx(nmat, kmax)];

  auto rhoratio = 1.0; //rmin/rmax;

  // get pressures and bulk modulii
  real nume(0.0), deno(0.0);
  std::vector< real > apmat(nmat, 0.0);
  pb = 0.0;
  for (std::size_t k=0; k<nmat; ++k)
  {
    real arhomat = u_gp[densityIdx(nmat, k)];
    real alphamat = u_gp[volfracIdx(nmat, k)];
    apmat[k] = p_gp[pressureIdx(nmat, k)];
    real amat = inciter::eos_soundspeed< tag::multimat >( system, arhomat,
      apmat[k], alphamat, k );
    kmat[k] = arhomat * amat * amat;
    pb += apmat[k];

    // relaxation parameters
    trelax = std::max(trelax, ct*rhoratio*dx/amat);
    nume += std::pow(alphamat, apow) * apmat[k] / kmat[k];
    deno += std::pow(alphamat, apow+1.0) / kmat[k];
  }
  p_relax = nume/deno;
}

}// tk::
