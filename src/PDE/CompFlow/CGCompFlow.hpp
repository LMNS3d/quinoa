// *****************************************************************************
/*!
  \file      src/PDE/CompFlow/CGCompFlow.hpp
  \copyright 2012-2015 J. Bakosi,
             2016-2018 Los Alamos National Security, LLC.,
             2019 Triad National Security, LLC.
             All rights reserved. See the LICENSE file for details.
  \brief     Compressible single-material flow using continuous Galerkin
  \details   This file implements the physics operators governing compressible
    single-material flow using continuous Galerkin discretization.
*/
// *****************************************************************************
#ifndef CGCompFlow_h
#define CGCompFlow_h

#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>

#include "DerivedData.hpp"
#include "Exception.hpp"
#include "Vector.hpp"
#include "EoS/EoS.hpp"
#include "Mesh/Around.hpp"
#include "Integrate/Riemann/HLLC.hpp"

namespace inciter {

extern ctr::InputDeck g_inputdeck;

static constexpr tk::real muscl_eps = 1.0e-9;
static constexpr tk::real muscl_const = 1.0/3.0;
static constexpr tk::real muscl_m1 = 1.0 - muscl_const;
static constexpr tk::real muscl_p1 = 1.0 + muscl_const;

namespace cg {

//! \brief CompFlow used polymorphically with tk::CGPDE
//! \details The template arguments specify policies and are used to configure
//!   the behavior of the class. The policies are:
//!   - Physics - physics configuration, see PDE/CompFlow/Physics.h
//!   - Problem - problem configuration, see PDE/CompFlow/Problems.h
//! \note The default physics is Euler, set in inciter::deck::check_compflow()
template< class Physics, class Problem >
class CompFlow {

  private:
    using ncomp_t = kw::ncomp::info::expect::type;

  public:
    //! \brief Constructor
    //! \param[in] c Equation system index (among multiple systems configured)
    explicit CompFlow( ncomp_t c ) :
      m_physics(),
      m_problem(),
      m_system( c ),
      m_ncomp(
        g_inputdeck.get< tag::component >().get< tag::compflow >().at(c) ),
      m_offset(
        g_inputdeck.get< tag::component >().offset< tag::compflow >(c) )
    {
       Assert( m_ncomp == 5, "Number of CompFlow PDE components must be 5" );
    }

    //! Initalize the compressible flow equations, prepare for time integration
    //! \param[in] coord Mesh node coordinates
    //! \param[in,out] unk Array of unknowns
    //! \param[in] t Physical time
    void initialize( const std::array< std::vector< tk::real >, 3 >& coord,
                     tk::Fields& unk,
                     tk::real t ) const
    {
      Assert( coord[0].size() == unk.nunk(), "Size mismatch" );
      const auto& x = coord[0];
      const auto& y = coord[1];
      const auto& z = coord[2];
      // set initial and boundary conditions using problem policy
      for (ncomp_t i=0; i<coord[0].size(); ++i) {
        const auto s =
          Problem::solution( m_system, m_ncomp, x[i], y[i], z[i], t );
        unk(i,0,m_offset) = s[0]; // rho
        unk(i,1,m_offset) = s[1]; // rho * u
        unk(i,2,m_offset) = s[2]; // rho * v
        unk(i,3,m_offset) = s[3]; // rho * w
        unk(i,4,m_offset) = s[4]; // rho * e, e: total = kinetic + internal
      }
    }

    //! Return analytic solution (if defined by Problem) at xi, yi, zi, t
    //! \param[in] xi X-coordinate
    //! \param[in] yi Y-coordinate
    //! \param[in] zi Z-coordinate
    //! \param[in] t Physical time
    //! \return Vector of analytic solution at given location and time
    std::vector< tk::real >
    analyticSolution( tk::real xi, tk::real yi, tk::real zi, tk::real t ) const
    {
      auto s = Problem::solution( m_system, m_ncomp, xi, yi, zi, t );
      return std::vector< tk::real >( begin(s), end(s) );
    }

    //! Compute nodal gradients of primitive variables for ALECG
    //! \param[in] coord Mesh node coordinates
    //! \param[in] inpoel Mesh element connectivity
//    //! \param[in] bndel List of elements contributing to chare-boundary nodes
//    //! \param[in] gid Local->global node id map
//    //! \param[in] bid Local chare-boundary node ids (value) associated to
//    //!    global node ids (key)
    //! \param[in] U Solution vector at recent time step
    //! \param[in,out] G Nodal gradients of primitive variables
    void grad( const std::array< std::vector< tk::real >, 3 >& coord,
               const std::vector< std::size_t >& inpoel,
               const std::vector< std::size_t >& /* bndel */,
               const std::vector< std::size_t >& /* gid */,
               const std::unordered_map< std::size_t, std::size_t >& /* bid */,
               const tk::Fields& U,
               tk::Fields& G ) const
    {
      Assert( U.nunk() == coord[0].size(), "Number of unknowns in solution "
              "vector at recent time step incorrect" );
      Assert( G.nunk() == coord[0].size(),
              "Number of unknowns in gradient vector incorrect" );
      Assert( G.nprop() == m_ncomp*3,
              "Number of components in gradient vector incorrect" );

      const auto& x = coord[0];
      const auto& y = coord[1];
      const auto& z = coord[2];

      // compute gradients of primitive variables in points
      G.fill( 0.0 );

      for (std::size_t e=0; e<inpoel.size()/4; ++e) {
        // access node IDs
        const std::array< std::size_t, 4 >
          N{{ inpoel[e*4+0], inpoel[e*4+1], inpoel[e*4+2], inpoel[e*4+3] }};
        // compute element Jacobi determinant
        const std::array< tk::real, 3 >
          ba{{ x[N[1]]-x[N[0]], y[N[1]]-y[N[0]], z[N[1]]-z[N[0]] }},
          ca{{ x[N[2]]-x[N[0]], y[N[2]]-y[N[0]], z[N[2]]-z[N[0]] }},
          da{{ x[N[3]]-x[N[0]], y[N[3]]-y[N[0]], z[N[3]]-z[N[0]] }};
        const auto J = tk::triple( ba, ca, da );        // J = 6V
        Assert( J > 0, "Element Jacobian non-positive" );
        // shape function derivatives, nnode*ndim [4][3]
        std::array< std::array< tk::real, 3 >, 4 > grad;
        grad[1] = tk::crossdiv( ca, da, J );
        grad[2] = tk::crossdiv( da, ba, J );
        grad[3] = tk::crossdiv( ba, ca, J );
        for (std::size_t i=0; i<3; ++i)
          grad[0][i] = -grad[1][i]-grad[2][i]-grad[3][i];
        // access solution at element nodes
        std::vector< std::array< tk::real, 4 > > u( m_ncomp );
        for (ncomp_t c=0; c<m_ncomp; ++c) u[c] = U.extract( c, m_offset, N );
        // scatter-add gradient contributions to points
        auto J24 = J/24.0;
        for (std::size_t a=0; a<4; ++a)
          for (std::size_t b=0; b<4; ++b)
            for (std::size_t j=0; j<3; ++j)
              for (std::size_t c=0; c<m_ncomp; ++c)
                G(N[a],c*3+j,0) += J24 * grad[b][j] * u[c][b];
      }
    }

    //! Compute right hand side for ALECG
    //! \param[in] coord Mesh node coordinates
    //! \param[in] inpoel Mesh element connectivity
    //! \param[in] esued Elements surrounding edges
    //! \param[in] triinpoel Boundary triangle face connecitivity
//    //! \param[in] gid Local->global node id map
//    //! \param[in] bid Local chare-boundary node ids (value) associated to
//    //!    global node ids (key)
//    //! \param[in] lid Global->local node ids
//    //! \param[in] vol Nodal volumes
    //! \param[in] G Nodal gradients
    //! \param[in] U Solution vector at recent time step
    //! \param[in,out] R Right-hand side vector computed
    void rhs( tk::real /* t */,
              const std::array< std::vector< tk::real >, 3 >& coord,
              const std::vector< std::size_t >& inpoel,
              const std::unordered_map< tk::UnsMesh::Edge,
                      std::vector< std::size_t >, tk::UnsMesh::Hash<2>,
                      tk::UnsMesh::Eq<2> >& esued,
              const std::pair< std::vector< std::size_t >,
                               std::vector< std::size_t > >& /* psup */,
              const std::vector< std::size_t >& triinpoel,
              const std::vector< std::size_t >& /* gid */,
              const std::unordered_map< std::size_t, std::size_t >& /* bid */,
              const std::unordered_map< std::size_t, std::size_t >& /* lid */,
              const std::unordered_map< tk::UnsMesh::Edge,
                      std::array< tk::real, 3 >,
                      tk::UnsMesh::Hash<2>, tk::UnsMesh::Eq<2> >& /* norm */,
              const std::vector< tk::real >& /* vol */,
              const tk::Fields& G,
              const tk::Fields& U,
              tk::Fields& R ) const
    {
      Assert( G.nunk() == coord[0].size(),
              "Number of unknowns in gradient vector incorrect" );
      Assert( G.nprop() == m_ncomp*3,
              "Number of components in gradient vector incorrect" );
      Assert( U.nunk() == coord[0].size(), "Number of unknowns in solution "
              "vector at recent time step incorrect" );
      Assert( R.nunk() == coord[0].size(),
              "Number of unknowns and/or number of components in right-hand "
              "side vector incorrect" );

      const auto& x = coord[0];
      const auto& y = coord[1];
      const auto& z = coord[2];
      
      // zero right hand side for all components
      for (ncomp_t c=0; c<5; ++c) R.fill( c, m_offset, 0.0 );

      // access pointer to right hand side at component and offset
      std::array< const tk::real*, 5 > r;
      for (ncomp_t c=0; c<5; ++c) r[c] = R.cptr( c, m_offset );

      tk::Fields V( U.nunk(), 3 );
      V.fill( 0.0 );

      // Domain edge integral
      for (const auto& [edge,surr_elements] : esued) {
        // access edge-end point node IDs
        auto p1 = edge[0];
        auto p2 = edge[1];
        // edge vector = outward face normal of the dual mesh face
        std::array< tk::real, 3 > fn{ x[p2]-x[p1], y[p2]-y[p1], z[p2]-z[p1] };
        // Access primitive variables at edge-end points
        std::array< std::vector< tk::real >, 2 >
          ru{ std::vector<tk::real>(5,0.0), std::vector<tk::real>(5,0.0) };
        for (std::size_t p=0; p<2; ++p) {
          ru[p][0] = U(edge[p], 0, m_offset);
          for (std::size_t c=1; c<5; ++c)
            ru[p][c] = U(edge[p], c, m_offset) / ru[p][0];
        }

        // MUSCL reconstruction of edge-end-point primitive variables
        for (std::size_t c=0; c<5; ++c) {
          auto delta_2 = ru[1][c] - ru[0][c];
          std::array< tk::real, 3 >
             g1{ G(p1,c*3+0,0), G(p1,c*3+1,0), G(p1,c*3+2,0) },
             g2{ G(p2,c*3+0,0), G(p2,c*3+1,0), G(p2,c*3+2,0) };
          auto delta_1 = 2.0 * tk::dot(g1,fn) - delta_2;
          auto delta_3 = 2.0 * tk::dot(g2,fn) - delta_2;
          // form limiters
          auto rL = (delta_2 + muscl_eps) / (delta_1 + muscl_eps);
          auto rR = (delta_2 + muscl_eps) / (delta_3 + muscl_eps);
          auto rLinv = (delta_1 + muscl_eps) / (delta_2 + muscl_eps);
          auto rRinv = (delta_3 + muscl_eps) / (delta_2 + muscl_eps);
          auto phiL = (std::abs(rL) + rL) / (std::abs(rL) + 1.0);
          auto phiR = (std::abs(rR) + rR) / (std::abs(rR) + 1.0);
          auto phi_L_inv = (std::abs(rLinv) + rLinv) / (std::abs(rLinv) + 1.0);
          auto phi_R_inv = (std::abs(rRinv) + rRinv) / (std::abs(rRinv) + 1.0);
          // final form of higher-order unknown
          ru[0][c] += 0.25*(delta_1*muscl_m1*phiL + delta_2*muscl_p1*phi_L_inv);
          ru[1][c] -= 0.25*(delta_3*muscl_m1*phiR + delta_2*muscl_p1*phi_R_inv);
        }

        // Compute conserved variables from primitive reconstructed ones
        for (std::size_t p=0; p<2; ++p)
          for (std::size_t c=1; c<5; ++c)
            ru[p][c] *= ru[p][0];

        // Compute Riemann flux using edge-end point states
        auto f = HLLC::flux( fn, ru, {{0.0,0.0,0.0}} );

        // scatter-add flux contributions to interior-edge-end points
        for (auto e : surr_elements) {
          // access node IDs
          const std::array< std::size_t, 4 > N{ inpoel[e*4+0], inpoel[e*4+1],
                                                inpoel[e*4+2], inpoel[e*4+3] };
          // compute element Jacobi determinant
          const std::array< tk::real, 3 >
            ba{{ x[N[1]]-x[N[0]], y[N[1]]-y[N[0]], z[N[1]]-z[N[0]] }},
            ca{{ x[N[2]]-x[N[0]], y[N[2]]-y[N[0]], z[N[2]]-z[N[0]] }},
            da{{ x[N[3]]-x[N[0]], y[N[3]]-y[N[0]], z[N[3]]-z[N[0]] }};
          const auto J = tk::triple( ba, ca, da );        // J = 6V
          Assert( J > 0, "Element Jacobian non-positive" );

          // shape function derivatives, nnode*ndim [4][3]
          std::array< std::array< tk::real, 3 >, 4 > grad;
          grad[1] = tk::crossdiv( ca, da, J );
          grad[2] = tk::crossdiv( da, ba, J );
          grad[3] = tk::crossdiv( ba, ca, J );
          for (std::size_t i=0; i<3; ++i)
            grad[0][i] = -grad[1][i]-grad[2][i]-grad[3][i];

          // sum flux contributions to nodes
          auto J48 = J/48.0;
          for (const auto& l : tk::lpoed) {
            auto a = l[0];
            auto b = l[1];
            if ((N[a]==p1 && N[b]==p2) || (N[a]==p2 && N[b]==p1)) {
              for (std::size_t j=0; j<3; ++j) {
                for (std::size_t c=0; c<5; ++c) {
                  auto d = J48 * (grad[a][j] - grad[b][j]) * f[c];
                  R.var(r[c],N[a]) -= d;
                  R.var(r[c],N[b]) += d;
                }
                auto d = 2.0*J48 * (grad[a][j] - grad[b][j]);
                V(N[a],j,0) -= d;
                V(N[b],j,0) += d;
              }
            }
          }
        }
      }

      // Test 2*sum_{vw in v} D_i^{vw} = 0 for interior points.
      std::unordered_set< std::size_t > bp( triinpoel.cbegin(), triinpoel.cend() );
      //std::cout << "tr: " << triinpoel.size() << ": ";
      //for (auto b : bp) std::cout << b << ' ';
      //std::cout << '\n';
      for (std::size_t p=0; p<coord[0].size(); ++p) {
        if (bp.find(p) == end(bp))
          for (std::size_t j=0; j<3; ++j)
            if (std::abs(V(p,j,m_offset)) > 1.0e-15)
              std::cout << '!';
      }

      //// Optional source
      //for (std::size_t p=0; p<U.nunk(); ++p) {
      //  auto s = Problem::src( m_system, m_ncomp, x[p], y[p], z[p], t );
      //  for (std::size_t c=0; c<5; ++c) R.var(r[c],p) += vol[p] * s[c];
      //}

      // Boundary integrals
      for (std::size_t e=0; e<triinpoel.size()/3; ++e) {
        // access node IDs
        const std::array< std::size_t, 3 >
          N{ triinpoel[e*3+0], triinpoel[e*3+1], triinpoel[e*3+2] };
        // compute face area
        auto A = tk::area( { x[N[0]], x[N[1]], x[N[2]] },
                           { y[N[0]], y[N[1]], y[N[2]] },
                           { z[N[0]], z[N[1]], z[N[2]] } );
        auto A24 = A/24.0;
        auto A6 = A/6.0;
        // compute face normal
        auto n = tk::normal( { x[N[0]], x[N[1]], x[N[2]] },
                             { y[N[0]], y[N[1]], y[N[2]] },
                             { z[N[0]], z[N[1]], z[N[2]] } );
        // access solution at element nodes
        std::array< std::array< tk::real, 3 >, 5 > u;
        for (ncomp_t c=0; c<5; ++c) u[c] = U.extract( c, m_offset, N );
        // sum boundary integrals to boundary nodes
        for (std::size_t j=0; j<3; ++j) {
          for (const auto& l : tk::lpoet) {
            auto a = l[0];
            auto b = l[1];
            for (std::size_t c=0; c<5; ++c) {
              auto d = A24 * n[j] * (u[c][a] + u[c][b]);
              R.var(r[c],N[a]) += d;
              R.var(r[c],N[b]) += d;
              R.var(r[c],N[a]) += A6 * n[j] * u[c][a];
            }
            auto d = 2.0*A24 * n[j];
            V(N[a],j,0) += d;
            V(N[b],j,0) += d;
            d = A6 * n[j];
            V(N[a],j,0) += d;
          }
        }
      }

      // Test 2*sum_{vw in v} D_i^{vw} + 2*sum_{vw in v} B_i^{vw} + B_i^v = 0
      // for boundary points.
      for (std::size_t p=0; p<coord[0].size(); ++p) {
        if (bp.find(p) != end(bp))
          for (std::size_t j=0; j<3; ++j)
            if (std::abs(V(p,j,m_offset)) > 1.0e-15)
              std::cout << '$';
      }
      std::cout << "max(abs(V)): " << tk::maxabs(V) << '\n';
      //std::cout << "max(abs(R)): " << tk::maxabs(R) << '\n';
    }

    //! Compute right hand side for DiagCG (CG-FCT)
    //! \param[in] t Physical time
    //! \param[in] deltat Size of time step
    //! \param[in] coord Mesh node coordinates
    //! \param[in] inpoel Mesh element connectivity
    //! \param[in] U Solution vector at recent time step
    //! \param[in,out] Ue Element-centered solution vector at intermediate step
    //!    (used here internally as a scratch array)
    //! \param[in,out] R Right-hand side vector computed
    void rhs( tk::real t,
              tk::real deltat,
              const std::array< std::vector< tk::real >, 3 >& coord,
              const std::vector< std::size_t >& inpoel,
              const tk::Fields& U,
              tk::Fields& Ue,
              tk::Fields& R ) const
    {
      Assert( U.nunk() == coord[0].size(), "Number of unknowns in solution "
              "vector at recent time step incorrect" );
      Assert( R.nunk() == coord[0].size(),
              "Number of unknowns and/or number of components in right-hand "
              "side vector incorrect" );

      const auto& x = coord[0];
      const auto& y = coord[1];
      const auto& z = coord[2];

      // 1st stage: update element values from node values (gather-add)
      for (std::size_t e=0; e<inpoel.size()/4; ++e) {

        // access node IDs
        const std::array< std::size_t, 4 > N{{ inpoel[e*4+0], inpoel[e*4+1],
                                               inpoel[e*4+2], inpoel[e*4+3] }};
        // compute element Jacobi determinant
        const std::array< tk::real, 3 >
          ba{{ x[N[1]]-x[N[0]], y[N[1]]-y[N[0]], z[N[1]]-z[N[0]] }},
          ca{{ x[N[2]]-x[N[0]], y[N[2]]-y[N[0]], z[N[2]]-z[N[0]] }},
          da{{ x[N[3]]-x[N[0]], y[N[3]]-y[N[0]], z[N[3]]-z[N[0]] }};
        const auto J = tk::triple( ba, ca, da );        // J = 6V
        Assert( J > 0, "Element Jacobian non-positive" );

        // shape function derivatives, nnode*ndim [4][3]
        std::array< std::array< tk::real, 3 >, 4 > grad;
        grad[1] = tk::crossdiv( ca, da, J );
        grad[2] = tk::crossdiv( da, ba, J );
        grad[3] = tk::crossdiv( ba, ca, J );
        for (std::size_t i=0; i<3; ++i)
          grad[0][i] = -grad[1][i]-grad[2][i]-grad[3][i];

        // access solution at element nodes
        std::array< std::array< tk::real, 4 >, 5 > u;
        for (ncomp_t c=0; c<5; ++c) u[c] = U.extract( c, m_offset, N );
        // access solution at elements
        std::array< const tk::real*, 5 > ue;
        for (ncomp_t c=0; c<5; ++c) ue[c] = Ue.cptr( c, m_offset );

        // pressure
        std::array< tk::real, 4 > p;
        for (std::size_t a=0; a<4; ++a)
          p[a] = eos_pressure< tag::compflow >
                   ( m_system, u[0][a], u[1][a]/u[0][a], u[2][a]/u[0][a],
                     u[3][a]/u[0][a], u[4][a] );

        // sum nodal averages to element
        for (ncomp_t c=0; c<5; ++c) {
          Ue.var(ue[c],e) = 0.0;
          for (std::size_t a=0; a<4; ++a)
            Ue.var(ue[c],e) += u[c][a]/4.0;
        }

        // sum flux contributions to element
        tk::real d = deltat/2.0;
        for (std::size_t j=0; j<3; ++j)
          for (std::size_t a=0; a<4; ++a) {
            // mass: advection
            Ue.var(ue[0],e) -= d * grad[a][j] * u[j+1][a];
            // momentum: advection
            for (std::size_t i=0; i<3; ++i)
              Ue.var(ue[i+1],e) -= d * grad[a][j] * u[j+1][a]*u[i+1][a]/u[0][a];
            // momentum: pressure
            Ue.var(ue[j+1],e) -= d * grad[a][j] * p[a];
            // energy: advection and pressure
            Ue.var(ue[4],e) -= d * grad[a][j] *
                              (u[4][a] + p[a]) * u[j+1][a]/u[0][a];
          }

        // add (optional) source to all equations
        std::array< std::vector< tk::real >, 4 > s{{
          Problem::src( m_system, m_ncomp, x[N[0]], y[N[0]], z[N[0]], t ),
          Problem::src( m_system, m_ncomp, x[N[1]], y[N[1]], z[N[1]], t ),
          Problem::src( m_system, m_ncomp, x[N[2]], y[N[2]], z[N[2]], t ),
          Problem::src( m_system, m_ncomp, x[N[3]], y[N[3]], z[N[3]], t ) }};
        for (std::size_t c=0; c<5; ++c)
          for (std::size_t a=0; a<4; ++a)
            Ue.var(ue[c],e) += d/4.0 * s[a][c];

      }


      // zero right hand side for all components
      for (ncomp_t c=0; c<5; ++c) R.fill( c, m_offset, 0.0 );

      // 2nd stage: form rhs from element values (scatter-add)
      for (std::size_t e=0; e<inpoel.size()/4; ++e) {

        // access node IDs
        const std::array< std::size_t, 4 > N{{ inpoel[e*4+0], inpoel[e*4+1],
                                               inpoel[e*4+2], inpoel[e*4+3] }};
        // compute element Jacobi determinant
        const std::array< tk::real, 3 >
          ba{{ x[N[1]]-x[N[0]], y[N[1]]-y[N[0]], z[N[1]]-z[N[0]] }},
          ca{{ x[N[2]]-x[N[0]], y[N[2]]-y[N[0]], z[N[2]]-z[N[0]] }},
          da{{ x[N[3]]-x[N[0]], y[N[3]]-y[N[0]], z[N[3]]-z[N[0]] }};
        const auto J = tk::triple( ba, ca, da );        // J = 6V
        Assert( J > 0, "Element Jacobian non-positive" );

        // shape function derivatives, nnode*ndim [4][3]
        std::array< std::array< tk::real, 3 >, 4 > grad;
        grad[1] = tk::crossdiv( ca, da, J );
        grad[2] = tk::crossdiv( da, ba, J );
        grad[3] = tk::crossdiv( ba, ca, J );
        for (std::size_t i=0; i<3; ++i)
          grad[0][i] = -grad[1][i]-grad[2][i]-grad[3][i];

        // access solution at elements
        std::array< tk::real, 5 > ue;
        for (ncomp_t c=0; c<5; ++c) ue[c] = Ue( e, c, m_offset );
        // access pointer to right hand side at component and offset
        std::array< const tk::real*, 5 > r;
        for (ncomp_t c=0; c<5; ++c) r[c] = R.cptr( c, m_offset );

        // pressure
        auto p = eos_pressure< tag::compflow >
                   ( m_system, ue[0], ue[1]/ue[0], ue[2]/ue[0], ue[3]/ue[0],
                     ue[4] );

        // scatter-add flux contributions to rhs at nodes
        tk::real d = deltat * J/6.0;
        for (std::size_t j=0; j<3; ++j)
          for (std::size_t a=0; a<4; ++a) {
            // mass: advection
            R.var(r[0],N[a]) += d * grad[a][j] * ue[j+1];
            // momentum: advection
            for (std::size_t i=0; i<3; ++i)
              R.var(r[i+1],N[a]) += d * grad[a][j] * ue[j+1]*ue[i+1]/ue[0];
            // momentum: pressure
            R.var(r[j+1],N[a]) += d * grad[a][j] * p;
            // energy: advection and pressure
            R.var(r[4],N[a]) += d * grad[a][j] * (ue[4] + p) * ue[j+1]/ue[0];
          }

        // add (optional) source to all equations
        auto xc = (x[N[0]] + x[N[1]] + x[N[2]] + x[N[3]]) / 4.0;
        auto yc = (y[N[0]] + y[N[1]] + y[N[2]] + y[N[3]]) / 4.0;
        auto zc = (z[N[0]] + z[N[1]] + z[N[2]] + z[N[3]]) / 4.0;
        auto s = Problem::src( m_system, m_ncomp, xc, yc, zc, t+deltat/2 );
        for (std::size_t c=0; c<5; ++c)
          for (std::size_t a=0; a<4; ++a)
            R.var(r[c],N[a]) += d/4.0 * s[c];

      }
//         // add viscous stress contribution to momentum and energy rhs
//         m_physics.viscousRhs( deltat, J, N, grad, u, r, R );
//         // add heat conduction contribution to energy rhs
//         m_physics.conductRhs( deltat, J, N, grad, u, r, R );
    }

    //! Compute the minimum time step size
    //! \param[in] U Solution vector at recent time step
    //! \param[in] coord Mesh node coordinates
    //! \param[in] inpoel Mesh element connectivity
    //! \return Minimum time step size
    tk::real dt( const std::array< std::vector< tk::real >, 3 >& coord,
                 const std::vector< std::size_t >& inpoel,
                 const tk::Fields& U ) const
    {
      Assert( U.nunk() == coord[0].size(), "Number of unknowns in solution "
              "vector at recent time step incorrect" );
      const auto& x = coord[0];
      const auto& y = coord[1];
      const auto& z = coord[2];
      // ratio of specific heats
      auto g = g_inputdeck.get< tag::param, tag::compflow, tag::gamma >()[0][0];
      // compute the minimum dt across all elements we own
      tk::real mindt = std::numeric_limits< tk::real >::max();
      for (std::size_t e=0; e<inpoel.size()/4; ++e) {
        const std::array< std::size_t, 4 > N{{ inpoel[e*4+0], inpoel[e*4+1],
                                               inpoel[e*4+2], inpoel[e*4+3] }};
        // compute cubic root of element volume as the characteristic length
        const std::array< tk::real, 3 >
          ba{{ x[N[1]]-x[N[0]], y[N[1]]-y[N[0]], z[N[1]]-z[N[0]] }},
          ca{{ x[N[2]]-x[N[0]], y[N[2]]-y[N[0]], z[N[2]]-z[N[0]] }},
          da{{ x[N[3]]-x[N[0]], y[N[3]]-y[N[0]], z[N[3]]-z[N[0]] }};
        const auto L = std::cbrt( tk::triple( ba, ca, da ) / 6.0 );
        // access solution at element nodes at recent time step
        std::array< std::array< tk::real, 4 >, 5 > u;
        for (ncomp_t c=0; c<5; ++c) u[c] = U.extract( c, m_offset, N );
        // compute the maximum length of the characteristic velocity (fluid
        // velocity + sound velocity) across the four element nodes
        tk::real maxvel = 0.0;
        for (std::size_t j=0; j<4; ++j) {
          auto& r  = u[0][j];    // rho
          auto& ru = u[1][j];    // rho * u
          auto& rv = u[2][j];    // rho * v
          auto& rw = u[3][j];    // rho * w
          auto& re = u[4][j];    // rho * e
          auto p = eos_pressure< tag::compflow >
                     ( m_system, r, ru/r, rv/r, rw/r, re );
          if (p < 0) p = 0.0;
          auto c = eos_soundspeed< tag::compflow >( m_system, r, p );
          auto v = std::sqrt((ru*ru + rv*rv + rw*rw)/r/r) + c; // char. velocity
          if (v > maxvel) maxvel = v;
        }
        // compute element dt for the Euler equations
        auto euler_dt = L / maxvel;
        // compute element dt based on the viscous force
        auto viscous_dt = m_physics.viscous_dt( L, u );
        // compute element dt based on thermal diffusion
        auto conduct_dt = m_physics.conduct_dt( L, g, u );
        // compute minimum element dt
        auto elemdt = std::min( euler_dt, std::min( viscous_dt, conduct_dt ) );
        // find minimum dt across all elements
        if (elemdt < mindt) mindt = elemdt;
      }
      return mindt;
    }

    //! Extract the velocity field at cell nodes. Currently unused.
    //! \param[in] U Solution vector at recent time step
    //! \param[in] N Element node indices    
    //! \return Array of the four values of the velocity field
    std::array< std::array< tk::real, 4 >, 3 >
    velocity( const tk::Fields& U,
              const std::array< std::vector< tk::real >, 3 >&,
              const std::array< std::size_t, 4 >& N ) const
    {
      std::array< std::array< tk::real, 4 >, 3 > v;
      v[0] = U.extract( 1, m_offset, N );
      v[1] = U.extract( 2, m_offset, N );
      v[2] = U.extract( 3, m_offset, N );
      auto r = U.extract( 0, m_offset, N );
      std::transform( r.begin(), r.end(), v[0].begin(), v[0].begin(),
                      []( tk::real s, tk::real& d ){ return d /= s; } );
      std::transform( r.begin(), r.end(), v[1].begin(), v[1].begin(),
                      []( tk::real s, tk::real& d ){ return d /= s; } );
      std::transform( r.begin(), r.end(), v[2].begin(), v[2].begin(),
                      []( tk::real s, tk::real& d ){ return d /= s; } );
      return v;
    }

    //! \brief Query all side set IDs the user has configured for all components
    //!   in this PDE system
    //! \param[in,out] conf Set of unique side set IDs to add to
    void side( std::unordered_set< int >& conf ) const
    { m_problem.side( conf ); }

    //! \brief Query Dirichlet boundary condition value on a given side set for
    //!    all components in this PDE system
    //! \param[in] t Physical time
    //! \param[in] deltat Time step size
    //! \param[in] ss Pair of side set ID and (local) node IDs on the side set
    //! \param[in] coord Mesh node coordinates
    //! \return Vector of pairs of bool and boundary condition value associated
    //!   to mesh node IDs at which Dirichlet boundary conditions are set. Note
    //!   that instead of the actual boundary condition value, we return the
    //!   increment between t+dt and t, since that is what the solution requires
    //!   as we solve for the soution increments and not the solution itself.
    std::map< std::size_t, std::vector< std::pair<bool,tk::real> > >
    dirbc( tk::real t,
           tk::real deltat,
           const std::pair< const int, std::vector< std::size_t > >& ss,
           const std::array< std::vector< tk::real >, 3 >& coord ) const
    {
      using tag::param; using tag::compflow; using tag::bcdir;
      using NodeBC = std::vector< std::pair< bool, tk::real > >;
      std::map< std::size_t, NodeBC > bc;
      const auto& ubc = g_inputdeck.get< param, compflow, bcdir >();
      if (!ubc.empty()) {
        Assert( ubc.size() > 0, "Indexing out of Dirichlet BC eq-vector" );
        const auto& x = coord[0];
        const auto& y = coord[1];
        const auto& z = coord[2];
        for (const auto& b : ubc[0])
          if (std::stoi(b) == ss.first)
            for (auto n : ss.second) {
              Assert( x.size() > n, "Indexing out of coordinate array" );
              auto s = m_problem.solinc( m_system, m_ncomp, x[n], y[n], z[n],
                                         t, deltat );
              bc[n] = {{ {true,s[0]}, {true,s[1]}, {true,s[2]}, {true,s[3]},
                         {true,s[4]} }};
            }
      }
      return bc;
    }

    //! Set symmetry boundary conditions at nodes
    //! \param[in] U Solution vector at recent time step
    //! \param[in] bnorm Face normals in boundary points: key local node id,
    //!    value: unit normal
    void
    symbc( tk::Fields& U,
           const std::unordered_map<std::size_t,std::array<tk::real,4>>& bnorm )
    const {
      for (const auto& [ i, nr ] : bnorm ) {
        std::array< tk::real, 3 >
          n{ nr[0], nr[1], nr[2] },
          v{ U(i,1,m_offset), U(i,2,m_offset), U(i,3,m_offset) };
        auto v_dot_n = tk::dot( v, n );
        U(i,1,m_offset) -= v_dot_n * n[0];
        U(i,2,m_offset) -= v_dot_n * n[1];
        U(i,3,m_offset) -= v_dot_n * n[2];
      }
    }

    //! Query nodes at which symmetry boundary conditions are set
    //! \param[in] bface Boundary-faces mapped to side set ids
    //! \param[in] triinpoel Boundary-face connectivity
    //! \param[in,out] nodes Node ids at which symmetry BCs are set
    void
    symbcnodes( const std::map< int, std::vector< std::size_t > >& bface,
                const std::vector< std::size_t >& triinpoel,
                std::unordered_set< std::size_t >& nodes ) const
    {
      using tag::param; using tag::compflow; using tag::bcsym;
      const auto& bc = g_inputdeck.get< param, compflow, bcsym >();
      if (!bc.empty() && bc.size() > m_system) {
        const auto& ss = bc[ m_system ];// side sets with sym bcs specified
        for (const auto& s : ss) {
          auto k = bface.find( std::stoi(s) );
          if (k != end(bface)) {
            for (auto f : k->second) {  // face ids on symbc side set
              nodes.insert( triinpoel[f*3+0] );
              nodes.insert( triinpoel[f*3+1] );
              nodes.insert( triinpoel[f*3+2] );
            }
          }
        }
      }
    }

    //! Return field names to be output to file
    //! \return Vector of strings labelling fields output in file
    std::vector< std::string > fieldNames() const
    { return m_problem.fieldNames( m_ncomp ); }

    //! Return field output going to file
    //! \param[in] t Physical time
    //! \param[in] V Total mesh volume
    //! \param[in] coord Mesh node coordinates
    //! \param[in] v Nodal mesh volumes
    //! \param[in,out] U Solution vector at recent time step
    //! \return Vector of vectors to be output to file
    std::vector< std::vector< tk::real > >
    fieldOutput( tk::real t,
                 tk::real V,
                 const std::array< std::vector< tk::real >, 3 >& coord,
                 const std::vector< tk::real >& v,
                 tk::Fields& U ) const
    {
      return
        m_problem.fieldOutput( m_system, m_ncomp, m_offset, t, V, v, coord, U );
    }

    //! Return names of integral variables to be output to diagnostics file
    //! \return Vector of strings labelling integral variables output
    std::vector< std::string > names() const
    { return m_problem.names( m_ncomp ); }

  private:
    const Physics m_physics;            //!< Physics policy
    const Problem m_problem;            //!< Problem policy
    const ncomp_t m_system;             //!< Equation system index
    const ncomp_t m_ncomp;              //!< Number of components in this PDE
    const ncomp_t m_offset;             //!< Offset PDE operates from
};

} // cg::

} // inciter::

#endif // CGCompFlow_h
