// *****************************************************************************
/*!
  \file      src/PDE/CompFlow/DGCompFlow.hpp
  \copyright 2012-2015 J. Bakosi,
             2016-2018 Los Alamos National Security, LLC.,
             2019-2020 Triad National Security, LLC.
             All rights reserved. See the LICENSE file for details.
  \brief     Compressible single-material flow using discontinuous Galerkin
     finite elements
  \details   This file implements calls to the physics operators governing
    compressible single-material flow using discontinuous Galerkin
    discretizations.
*/
// *****************************************************************************
#ifndef DGCompFlow_h
#define DGCompFlow_h

#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <map>

#include <brigand/algorithms/for_each.hpp>

#include "Macro.hpp"
#include "Exception.hpp"
#include "Vector.hpp"
#include "ContainerUtil.hpp"
#include "UnsMesh.hpp"
#include "Inciter/InputDeck/InputDeck.hpp"
#include "Integrate/Basis.hpp"
#include "Integrate/Quadrature.hpp"
#include "Integrate/Initialize.hpp"
#include "Integrate/Mass.hpp"
#include "Integrate/Surface.hpp"
#include "Integrate/Boundary.hpp"
#include "Integrate/Volume.hpp"
#include "Integrate/Source.hpp"
#include "RiemannFactory.hpp"
#include "EoS/EoS.hpp"
#include "Reconstruction.hpp"
#include "Limiter.hpp"

namespace inciter {

extern ctr::InputDeck g_inputdeck;

namespace dg {

//! \brief CompFlow used polymorphically with tk::DGPDE
//! \details The template arguments specify policies and are used to configure
//!   the behavior of the class. The policies are:
//!   - Physics - physics configuration, see PDE/CompFlow/Physics.h
//!   - Problem - problem configuration, see PDE/CompFlow/Problem.h
//! \note The default physics is Euler, set in inciter::deck::check_compflow()
template< class Physics, class Problem >
class CompFlow {

  private:
    using eq = tag::compflow;

  public:
    //! Constructor
    //! \param[in] c Equation system index (among multiple systems configured)
    explicit CompFlow( ncomp_t c ) :
      m_physics(),
      m_problem(),
      m_system( c ),
      m_ncomp( g_inputdeck.get< tag::component, eq >().at(c) ),
      m_offset( g_inputdeck.get< tag::component >().offset< eq >(c) ),
      m_riemann(tk::cref_find(compflowRiemannSolvers(),
        g_inputdeck.get< tag::param, tag::compflow, tag::flux >().at(m_system)))
    {
      // associate boundary condition configurations with state functions, the
      // order in which the state functions listed matters, see ctr::bc::Keys
      brigand::for_each< ctr::bc::Keys >( ConfigBC< eq >( m_system, m_bc,
        { dirichlet
        , symmetry
        , invalidBC         // Inlet BC not implemented
        , invalidBC         // Outlet BC not implemented
        , characteristic
        , extrapolate } ) );
    }

    //! Find the number of primitive quantities required for this PDE system
    //! \return The number of primitive quantities required to be stored for
    //!   this PDE system
    std::size_t nprim() const
    {
      // compflow does not need/store any primitive quantities currently
      return 0;
    }

    //! Initalize the compressible flow equations, prepare for time integration
    //! \param[in] L Block diagonal mass matrix
    //! \param[in] inpoel Element-node connectivity
    //! \param[in] coord Array of nodal coordinates
    //! \param[in,out] unk Array of unknowns
    //! \param[in] t Physical time
    //! \param[in] nielem Number of internal elements
    void initialize( const tk::Fields& L,
                     const std::vector< std::size_t >& inpoel,
                     const tk::UnsMesh::Coords& coord,
                     tk::Fields& unk,
                     tk::real t,
                     const std::size_t nielem ) const
    {
      tk::initialize( m_system, m_ncomp, m_offset, L, inpoel, coord,
                      Problem::solution, unk, t, nielem );
    }

    //! Compute the left hand side block-diagonal mass matrix
    //! \param[in] geoElem Element geometry array
    //! \param[in,out] l Block diagonal mass matrix
    void lhs( const tk::Fields& geoElem, tk::Fields& l ) const {
      const auto ndof = g_inputdeck.get< tag::discr, tag::ndof >();
      tk::mass( m_ncomp, m_offset, ndof, geoElem, l );
    }

    //! Update the primitives for this PDE system
    //! \details This function computes and stores the dofs for primitive
    //!   quantities, which is currently unused for compflow. But if a limiter
    //!   requires primitive variables for example, this would be the place to
    //!   add the computation of the primitive variables.
    void updatePrimitives( const tk::Fields&,
                           tk::Fields&,
                           std::size_t ) const {}

    //! Clean up the state of trace materials for this PDE system
    //! \details This function cleans up the state of materials present in trace
    //!   quantities in each cell. This is unused for compflow.
    void cleanTraceMaterial( const tk::Fields&,
                             tk::Fields&,
                             tk::Fields&,
                             std::size_t ) const {}

    //! Reconstruct second-order solution from first-order using least-squares
    //! \param[in] t Physical time
    //! \param[in] geoFace Face geometry array
    //! \param[in] geoElem Element geometry array
    //! \param[in] fd Face connectivity and boundary conditions object
    //! \param[in] inpoel Element-node connectivity
    //! \param[in] coord Array of nodal coordinates
    //! \param[in,out] U Solution vector at recent time step
    //! \param[in,out] P Primitive vector at recent time step
    void reconstruct( tk::real t,
                      const tk::Fields& geoFace,
                      const tk::Fields& geoElem,
                      const inciter::FaceData& fd,
                      const std::map< std::size_t, std::vector< std::size_t > >&,
                      const std::vector< std::size_t >& inpoel,
                      const tk::UnsMesh::Coords& coord,
                      tk::Fields& U,
                      tk::Fields& P ) const
    {
      const auto rdof = g_inputdeck.get< tag::discr, tag::rdof >();
      const auto nelem = fd.Esuel().size()/4;

      Assert( U.nprop() == rdof*5, "Number of components in solution "
              "vector must equal "+ std::to_string(rdof*5) );
      Assert( fd.Inpofa().size()/3 == fd.Esuf().size()/2,
              "Mismatch in inpofa size" );

      // allocate and initialize matrix and vector for reconstruction
      std::vector< std::array< std::array< tk::real, 3 >, 3 > >
        lhs_ls( nelem, {{ {{0.0, 0.0, 0.0}},
                          {{0.0, 0.0, 0.0}},
                          {{0.0, 0.0, 0.0}} }} );
      std::vector< std::vector< std::array< tk::real, 3 > > >
        rhs_ls( nelem, std::vector< std::array< tk::real, 3 > >
          ( m_ncomp,
            {{ 0.0, 0.0, 0.0 }} ) );

      // reconstruct x,y,z-derivatives of unknowns
      // 0. get lhs matrix, which is only geometry dependent
      tk::lhsLeastSq_P0P1(fd, geoElem, geoFace, lhs_ls);

      // 1. internal face contributions
      tk::intLeastSq_P0P1( m_ncomp, m_offset, rdof, fd, geoElem, U, rhs_ls );

      // 2. boundary face contributions
      for (const auto& b : m_bc)
        tk::bndLeastSqConservedVar_P0P1( m_system, m_ncomp, m_offset, rdof,
          b.first, fd, geoFace, geoElem, t, b.second, P, U, rhs_ls );

      // 3. solve 3x3 least-squares system
      tk::solveLeastSq_P0P1( m_ncomp, m_offset, rdof, lhs_ls, rhs_ls, U );

      // 4. transform reconstructed derivatives to Dubiner dofs
      tk::transform_P0P1( m_ncomp, m_offset, rdof, nelem, inpoel, coord, U );
    }

    //! Limit second-order solution
    //! \param[in] t Physical time
    //! \param[in] geoFace Face geometry array
    //! \param[in] geoElem Element geometry array
    //! \param[in] fd Face connectivity and boundary conditions object
    //! \param[in] inpoel Element-node connectivity
    //! \param[in] coord Array of nodal coordinates
    //! \param[in] ndofel Vector of local number of degrees of freedome
    //! \param[in,out] U Solution vector at recent time step
    void limit( [[maybe_unused]] tk::real t,
                [[maybe_unused]] const tk::Fields& geoFace,
                [[maybe_unused]] const tk::Fields& geoElem,
                const inciter::FaceData& fd,
                const std::map< std::size_t, std::vector< std::size_t > >&,
                const std::vector< std::size_t >& inpoel,
                const tk::UnsMesh::Coords& coord,
                const std::vector< std::size_t >& ndofel,
                tk::Fields& U,
                tk::Fields& ) const
    {
      const auto limiter = g_inputdeck.get< tag::discr, tag::limiter >();

      if (limiter == ctr::LimiterType::WENOP1)
        WENO_P1( fd.Esuel(), m_offset, U );
      else if (limiter == ctr::LimiterType::SUPERBEEP1)
        Superbee_P1( fd.Esuel(), inpoel, ndofel, m_offset, coord, U );
    }

    //! Compute right hand side
    //! \param[in] t Physical time
    //! \param[in] geoFace Face geometry array
    //! \param[in] geoElem Element geometry array
    //! \param[in] fd Face connectivity and boundary conditions object
    //! \param[in] inpoel Element-node connectivity
    //! \param[in] coord Array of nodal coordinates
    //! \param[in] U Solution vector at recent time step
    //! \param[in] P Primitive vector at recent time step
    //! \param[in] ndofel Vector of local number of degrees of freedom
    //! \param[in,out] R Right-hand side vector computed
    void rhs( tk::real t,
              const tk::Fields& geoFace,
              const tk::Fields& geoElem,
              const inciter::FaceData& fd,
              const std::vector< std::size_t >& inpoel,
              const tk::UnsMesh::Coords& coord,
              const tk::Fields& U,
              const tk::Fields& P,
              const std::vector< std::size_t >& ndofel,
              tk::Fields& R ) const
    {
      const auto ndof = g_inputdeck.get< tag::discr, tag::ndof >();
      const auto rdof = g_inputdeck.get< tag::discr, tag::rdof >();

      Assert( U.nunk() == P.nunk(), "Number of unknowns in solution "
              "vector and primitive vector at recent time step incorrect" );
      Assert( U.nunk() == R.nunk(), "Number of unknowns in solution "
              "vector and right-hand side at recent time step incorrect" );
      Assert( U.nprop() == rdof*5, "Number of components in solution "
              "vector must equal "+ std::to_string(rdof*5) );
      Assert( P.nprop() == 0, "Number of components in primitive "
              "vector must equal "+ std::to_string(0) );
      Assert( R.nprop() == ndof*5, "Number of components in right-hand "
              "side vector must equal "+ std::to_string(ndof*5) );
      Assert( fd.Inpofa().size()/3 == fd.Esuf().size()/2,
              "Mismatch in inpofa size" );

      // set rhs to zero
      R.fill(0.0);

      // empty vector for non-conservative terms. This vector is unused for
      // single-material hydrodynamics since, there are no non-conservative
      // terms in the system of PDEs.
      std::vector< std::vector < tk::real > > riemannDeriv;

      // configure Riemann flux function
      auto rieflxfn =
        [this]( const std::array< tk::real, 3 >& fn,
                const std::array< std::vector< tk::real >, 2 >& u,
                const std::vector< std::array< tk::real, 3 > >& v )
              { return m_riemann.flux( fn, u, v ); };
      // configure a no-op lambda for prescribed velocity
      auto velfn = [this]( ncomp_t, ncomp_t, tk::real, tk::real, tk::real ){
        return std::vector< std::array< tk::real, 3 > >( m_ncomp ); };

      // compute internal surface flux integrals
      tk::surfInt( m_system, 1, m_offset, ndof, rdof, inpoel, coord,
                   fd, geoFace, rieflxfn, velfn, U, P, ndofel, R, riemannDeriv );

      // compute ptional source term
      tk::srcInt( m_system, m_offset, t, ndof, fd.Esuel().size()/4,
                  inpoel, coord, geoElem, Problem::src, ndofel, R );

      if(ndof > 1)
        // compute volume integrals
        tk::volInt( m_system, m_ncomp, m_offset, ndof, fd.Esuel().size()/4,
                    inpoel, coord, geoElem, flux, velfn, U, ndofel, R );

      // compute boundary surface flux integrals
      for (const auto& b : m_bc)
        tk::bndSurfInt( m_system, 1, m_offset, ndof, rdof, b.first, fd,
                        geoFace, inpoel, coord, t, rieflxfn, velfn, b.second, U,
                        P, ndofel, R, riemannDeriv );
    }

    //! Compute the minimum time step size
    //! \param[in] coord Mesh node coordinates
    //! \param[in] inpoel Mesh element connectivity
    //! \param[in] fd Face connectivity and boundary conditions object
    //! \param[in] geoFace Face geometry array
    //! \param[in] geoElem Element geometry array
    //! \param[in] ndofel Vector of local number of degrees of freedom
    //! \param[in] U Solution vector at recent time step
    //! \return Minimum time step size
    tk::real dt( const std::array< std::vector< tk::real >, 3 >& coord,
                 const std::vector< std::size_t >& inpoel,
                 const inciter::FaceData& fd,
                 const tk::Fields& geoFace,
                 const tk::Fields& geoElem,
                 const std::vector< std::size_t >& ndofel,
                 const tk::Fields& U,
                 const tk::Fields&,
                 const std::size_t /*nielem*/ ) const
    {
      const auto rdof = g_inputdeck.get< tag::discr, tag::rdof >();

      const auto& esuf = fd.Esuf();
      const auto& inpofa = fd.Inpofa();

      tk::real rho, u, v, w, rhoE, p, a, vn, dSV_l, dSV_r;
      std::vector< tk::real > delt( U.nunk(), 0.0 );

      const auto& cx = coord[0];
      const auto& cy = coord[1];
      const auto& cz = coord[2];

      // compute internal surface maximum characteristic speed
      for (std::size_t f=0; f<esuf.size()/2; ++f)
      {

        std::size_t el = static_cast< std::size_t >(esuf[2*f]);
        auto er = esuf[2*f+1];

        // Number of quadrature points for  face integration
        std::size_t ng;

        if(er > -1)
        {
          auto eR = static_cast< std::size_t >( er );

          auto ng_l = tk::NGfa(ndofel[el]);
          auto ng_r = tk::NGfa(ndofel[eR]);

          // When the number of gauss points for the left and right element are
          // different, choose the larger ng
          ng = std::max( ng_l, ng_r );
        }
        else
        {
          ng = tk::NGfa(ndofel[el]);
        }

        // arrays for quadrature points
        std::array< std::vector< tk::real >, 2 > coordgp;
        std::vector< tk::real > wgp;

        coordgp[0].resize( ng );
        coordgp[1].resize( ng );
        wgp.resize( ng );

        // get quadrature point weights and coordinates for triangle
        tk::GaussQuadratureTri( ng, coordgp, wgp );

        // Extract the left element coordinates
        std::array< std::array< tk::real, 3>, 4 > coordel_l {{
          {{ cx[inpoel[4*el  ]], cy[inpoel[4*el  ]], cz[inpoel[4*el  ]] }},
          {{ cx[inpoel[4*el+1]], cy[inpoel[4*el+1]], cz[inpoel[4*el+1]] }},
          {{ cx[inpoel[4*el+2]], cy[inpoel[4*el+2]], cz[inpoel[4*el+2]] }},
          {{ cx[inpoel[4*el+3]], cy[inpoel[4*el+3]], cz[inpoel[4*el+3]] }} }};

        // Compute the determinant of Jacobian matrix
        auto detT_l = 
           tk::Jacobian(coordel_l[0], coordel_l[1], coordel_l[2], coordel_l[3]);

        // Extract the face coordinates
        std::array< std::array< tk::real, 3>, 3 > coordfa {{
          {{ cx[ inpofa[3*f  ] ], cy[ inpofa[3*f  ] ], cz[ inpofa[3*f  ] ] }},
          {{ cx[ inpofa[3*f+1] ], cy[ inpofa[3*f+1] ], cz[ inpofa[3*f+1] ] }},
          {{ cx[ inpofa[3*f+2] ], cy[ inpofa[3*f+2] ], cz[ inpofa[3*f+2] ] }}
        }};

        dSV_l = 0.0;
        dSV_r = 0.0;

        // Gaussian quadrature
        for (std::size_t igp=0; igp<ng; ++igp)
        {
          // Compute the coordinates of quadrature point at physical domain
          auto gp = tk::eval_gp( igp, coordfa, coordgp );

          // Compute the basis function for the left element
          auto B_l = tk::eval_basis( ndofel[el],
            tk::Jacobian(coordel_l[0], gp, coordel_l[2], coordel_l[3])/detT_l,
            tk::Jacobian(coordel_l[0], coordel_l[1], gp, coordel_l[3])/detT_l,
            tk::Jacobian(coordel_l[0], coordel_l[1], coordel_l[2], gp)/detT_l );

          auto wt = wgp[igp] * geoFace(f,0,0);

          std::array< std::vector< tk::real >, 2 > ugp;

          // left element
          for (ncomp_t c=0; c<5; ++c)
          {
            auto mark = c*rdof;
            ugp[0].push_back( U(el, mark, m_offset) );

            if(ndofel[el] > 1)          //DG(P1)
              ugp[0][c] +=  U(el, mark+1, m_offset) * B_l[1]
                          + U(el, mark+2, m_offset) * B_l[2]
                          + U(el, mark+3, m_offset) * B_l[3];

            if(ndofel[el] > 4)          //DG(P2)
              ugp[0][c] +=  U(el, mark+4, m_offset) * B_l[4]
                          + U(el, mark+5, m_offset) * B_l[5]
                          + U(el, mark+6, m_offset) * B_l[6]
                          + U(el, mark+7, m_offset) * B_l[7]
                          + U(el, mark+8, m_offset) * B_l[8]
                          + U(el, mark+9, m_offset) * B_l[9];
          }

          rho = ugp[0][0];
          u = ugp[0][1]/rho;
          v = ugp[0][2]/rho;
          w = ugp[0][3]/rho;
          rhoE = ugp[0][4];
          p = eos_pressure< tag::compflow >( m_system, rho, u, v, w, rhoE );

          a = eos_soundspeed< tag::compflow >( m_system, rho, p );

          vn = u*geoFace(f,1,0) + v*geoFace(f,2,0) + w*geoFace(f,3,0);

          dSV_l = wt * (std::fabs(vn) + a);

          // right element
          if (er > -1) {

            // nodal coordinates of the right element
            std::size_t eR = static_cast< std::size_t >( er );

            // Extract the left element coordinates
            std::array< std::array< tk::real, 3>, 4 > coordel_r {{
              {{ cx[inpoel[4*eR  ]], cy[inpoel[4*eR  ]], cz[inpoel[4*eR  ]] }},
              {{ cx[inpoel[4*eR+1]], cy[inpoel[4*eR+1]], cz[inpoel[4*eR+1]] }},
              {{ cx[inpoel[4*eR+2]], cy[inpoel[4*eR+2]], cz[inpoel[4*eR+2]] }},
              {{ cx[inpoel[4*eR+3]], cy[inpoel[4*eR+3]], cz[inpoel[4*eR+3]] }}
            }};

            // Compute the determinant of Jacobian matrix
            auto detT_r =
              tk::Jacobian(coordel_r[0],coordel_r[1],coordel_r[2],coordel_r[3]);

            // Compute the coordinates of quadrature point at physical domain
            gp = tk::eval_gp( igp, coordfa, coordgp );

            // Compute the basis function for the right element
            auto B_r = tk::eval_basis( ndofel[eR],
              tk::Jacobian(coordel_r[0],gp,coordel_r[2],coordel_r[3])/detT_r,
              tk::Jacobian(coordel_r[0],coordel_r[1],gp,coordel_r[3])/detT_r,
              tk::Jacobian(coordel_r[0],coordel_r[1],coordel_r[2],gp)/detT_r );
 
            for (ncomp_t c=0; c<5; ++c)
            {
              auto mark = c*rdof;
              ugp[1].push_back( U(eR, mark, m_offset) );

              if(ndofel[eR] > 1)          //DG(P1)
                ugp[1][c] +=  U(eR, mark+1, m_offset) * B_r[1]
                            + U(eR, mark+2, m_offset) * B_r[2]
                            + U(eR, mark+3, m_offset) * B_r[3];

              if(ndofel[eR] > 4)         //DG(P2)
                ugp[1][c] +=  U(eR, mark+4, m_offset) * B_r[4]
                            + U(eR, mark+5, m_offset) * B_r[5]
                            + U(eR, mark+6, m_offset) * B_r[6]
                            + U(eR, mark+7, m_offset) * B_r[7]
                            + U(eR, mark+8, m_offset) * B_r[8]
                            + U(eR, mark+9, m_offset) * B_r[9];
            }

            rho = ugp[1][0];
            u = ugp[1][1]/rho;
            v = ugp[1][2]/rho;
            w = ugp[1][3]/rho;
            rhoE = ugp[1][4];
            p = eos_pressure< tag::compflow >( m_system, rho, u, v, w, rhoE );
            a = eos_soundspeed< tag::compflow >( m_system, rho, p );

            vn = u*geoFace(f,1,0) + v*geoFace(f,2,0) + w*geoFace(f,3,0);

            dSV_r = wt * (std::fabs(vn) + a);
            delt[eR] += std::max( dSV_l, dSV_r );
          }

          delt[el] += std::max( dSV_l, dSV_r );
        }
      }

      tk::real mindt = std::numeric_limits< tk::real >::max();
      tk::real dgp = 0.0;

      // compute allowable dt
      for (std::size_t e=0; e<fd.Esuel().size()/4; ++e)
      {
        dgp = 0.0;
        if (ndofel[e] == 4)
        {
          dgp = 1.0;
        }
        else if (ndofel[e] == 10)
        {
          dgp = 2.0;
        }

        // Scale smallest dt with CFL coefficient and the CFL is scaled by (2*p+1)
        // where p is the order of the DG polynomial by linear stability theory.
        mindt = std::min( mindt, geoElem(e,0,0)/ (delt[e] * (2.0*dgp + 1.0)) );
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

    //! Return field names to be output to file
    //! \return Vector of strings labelling fields output in file
    std::vector< std::string > fieldNames() const
    { return m_problem.fieldNames( m_ncomp ); }

    //! Return field output going to file
    //! \param[in] t Physical time
    //! \param[in] V Total mesh volume
    //! \param[in] nunk Number of unknowns to extract
    //! \param[in] geoElem Element geometry array
    //! \param[in,out] U Solution vector at recent time step
    //! \return Vector of vectors to be output to file
    std::vector< std::vector< tk::real > >
    fieldOutput( tk::real t,
                 tk::real V,
                 std::size_t nunk,
                 const tk::Fields& geoElem,
                 tk::Fields& U,
                 const tk::Fields& ) const
    {
      std::array< std::vector< tk::real >, 3 > coord{
        geoElem.extract(1,0), geoElem.extract(2,0), geoElem.extract(3,0) };

      return m_problem.fieldOutput( m_system, m_ncomp, m_offset, nunk, t, V,
                                    geoElem.extract(0,0), coord, U );
    }

    //! Return surface field output going to file
    std::vector< std::vector< tk::real > >
    surfOutput( const std::map< int, std::vector< std::size_t > >&,
                tk::Fields& ) const
    {
      std::vector< std::vector< tk::real > > s; // punt for now
      return s;
    }

    //! Return nodal field output going to file
    std::vector< std::vector< tk::real > >
    avgElemToNode( const std::vector< std::size_t >& inpoel,
                   const tk::UnsMesh::Coords& coord,
                   const tk::Fields& /*geoElem*/,
                   const tk::Fields& U ) const
    {
      const auto rdof = g_inputdeck.get< tag::discr, tag::rdof >();

      const auto& cx = coord[0];
      const auto& cy = coord[1];
      const auto& cz = coord[2];

      std::vector< tk::real > count(cx.size(), 0);
      std::vector< std::vector< tk::real > >
        out( 6, std::vector< tk::real >( cx.size(), 0.0 ) );

      for (std::size_t e=0; e<inpoel.size()/4 ; ++e)
      {
        // nodal coordinates of the left element
        std::array< std::array< tk::real, 3 >, 4 >
          pi{{ {{ cx[ inpoel[4*e] ],
                 cy[ inpoel[4*e] ],
                 cz[ inpoel[4*e] ] }},
              {{ cx[ inpoel[4*e+1] ],
                 cy[ inpoel[4*e+1] ],
                 cz[ inpoel[4*e+1] ] }},
              {{ cx[ inpoel[4*e+2] ],
                 cy[ inpoel[4*e+2] ],
                 cz[ inpoel[4*e+2] ] }},
              {{ cx[ inpoel[4*e+3] ],
                 cy[ inpoel[4*e+3] ],
                 cz[ inpoel[4*e+3] ] }} }};
        auto detT = tk::Jacobian( pi[0], pi[1], pi[2], pi[3] );

        for (std::size_t i=0; i<4; ++i)
        {
          tk::real detT_gp;
          // transformation of the physical coordinates of the quadrature point
          // to reference space for the left element to be able to compute
          // basis functions on the left element.
          detT_gp = tk::Jacobian( pi[0], pi[i], pi[2], pi[3] );
          auto xi = detT_gp / detT;
          detT_gp = tk::Jacobian( pi[0], pi[1], pi[i], pi[3] );
          auto eta = detT_gp / detT;
          detT_gp = tk::Jacobian( pi[0], pi[1], pi[2], pi[i] );
          auto zeta = detT_gp / detT;

          auto B2 = 2.0 * xi + eta + zeta - 1.0;
          auto B3 = 3.0 * eta + zeta - 1.0;
          auto B4 = 4.0 * zeta - 1.0;

          std::vector< tk::real > ugp(5,0);

          for (ncomp_t c=0; c<5; ++c)
          {
            if (rdof == 1) {
              ugp[c] =  U(e, c, m_offset);
            } else {
              auto mark = c*rdof;
              ugp[c] =  U(e, mark,   m_offset)
                      + U(e, mark+1, m_offset) * B2
                      + U(e, mark+2, m_offset) * B3
                      + U(e, mark+3, m_offset) * B4;
            }
          }

          auto u = ugp[1] / ugp[0];
          auto v = ugp[2] / ugp[0];
          auto w = ugp[3] / ugp[0];
          auto p =
            eos_pressure< tag::compflow >( m_system, ugp[0], u, v, w, ugp[4] );

          out[0][ inpoel[4*e+i] ] += ugp[0];
          out[1][ inpoel[4*e+i] ] += u;
          out[2][ inpoel[4*e+i] ] += v;
          out[3][ inpoel[4*e+i] ] += w;
          out[4][ inpoel[4*e+i] ] += ugp[4]/ugp[0];
          out[5][ inpoel[4*e+i] ] += p;
          count[ inpoel[4*e+i] ] += 1.0;
        }
      }

      // average
      for (std::size_t i=0; i<cx.size(); ++i)
        for (std::size_t c=0; c<6; ++c)
          out[c][i] /= count[i];

      return out;
    }

    //! Return names of integral variables to be output to diagnostics file
    //! \return Vector of strings labelling integral variables output
    std::vector< std::string > names() const
    { return m_problem.names( m_ncomp ); }

    //! Return analytic solution (if defined by Problem) at xi, yi, zi, t
    //! \param[in] xi X-coordinate at which to evaluate the analytic solution
    //! \param[in] yi Y-coordinate at which to evaluate the analytic solution
    //! \param[in] zi Z-coordinate at which to evaluate the analytic solution
    //! \param[in] t Physical time at which to evaluate the analytic solution
    //! \return Vector of analytic solution at given location and time
    std::vector< tk::real >
    analyticSolution( tk::real xi, tk::real yi, tk::real zi, tk::real t ) const
    {
      int inbox = 0;
      auto s = Problem::solution( m_system, m_ncomp, xi, yi, zi, t, inbox );
      return std::vector< tk::real >( std::begin(s), std::end(s) );
    }

  private:
    //! Physics policy
    const Physics m_physics;
    //! Problem policy
    const Problem m_problem;
    //! Equation system index
    const ncomp_t m_system;
    //! Number of components in this PDE system
    const ncomp_t m_ncomp;
    //! Offset PDE system operates from
    const ncomp_t m_offset;
    //! Riemann solver
    RiemannSolver m_riemann;
    //! BC configuration
    BCStateFn m_bc;

    //! Evaluate physical flux function for this PDE system
    //! \param[in] system Equation system index
    //! \param[in] ncomp Number of scalar components in this PDE system
    //! \param[in] ugp Numerical solution at the Gauss point at which to
    //!   evaluate the flux
    //! \return Flux vectors for all components in this PDE system
    //! \note The function signature must follow tk::FluxFn
    static tk::FluxFn::result_type
    flux( ncomp_t system,
          [[maybe_unused]] ncomp_t ncomp,
          const std::vector< tk::real >& ugp,
          const std::vector< std::array< tk::real, 3 > >& )
    {
      Assert( ugp.size() == ncomp, "Size mismatch" );

      auto u = ugp[1] / ugp[0];
      auto v = ugp[2] / ugp[0];
      auto w = ugp[3] / ugp[0];
      auto p =
        eos_pressure< tag::compflow >( system, ugp[0], u, v, w, ugp[4] );

      std::vector< std::array< tk::real, 3 > > fl( ugp.size() );

      fl[0][0] = ugp[1];
      fl[1][0] = ugp[1] * u + p;
      fl[2][0] = ugp[1] * v;
      fl[3][0] = ugp[1] * w;
      fl[4][0] = u * (ugp[4] + p);

      fl[0][1] = ugp[2];
      fl[1][1] = ugp[2] * u;
      fl[2][1] = ugp[2] * v + p;
      fl[3][1] = ugp[2] * w;
      fl[4][1] = v * (ugp[4] + p);

      fl[0][2] = ugp[3];
      fl[1][2] = ugp[3] * u;
      fl[2][2] = ugp[3] * v;
      fl[3][2] = ugp[3] * w + p;
      fl[4][2] = w * (ugp[4] + p);

      return fl;
    }

    //! \brief Boundary state function providing the left and right state of a
    //!   face at Dirichlet boundaries
    //! \param[in] system Equation system index
    //! \param[in] ncomp Number of scalar components in this PDE system
    //! \param[in] ul Left (domain-internal) state
    //! \param[in] x X-coordinate at which to compute the states
    //! \param[in] y Y-coordinate at which to compute the states
    //! \param[in] z Z-coordinate at which to compute the states
    //! \param[in] t Physical time
    //! \return Left and right states for all scalar components in this PDE
    //!   system
    //! \note The function signature must follow tk::StateFn
    static tk::StateFn::result_type
    dirichlet( ncomp_t system, ncomp_t ncomp, const std::vector< tk::real >& ul,
               tk::real x, tk::real y, tk::real z, tk::real t,
               const std::array< tk::real, 3 >& )
    {
      int inbox = 0;
      return {{ ul, Problem::solution( system, ncomp, x, y, z, t, inbox ) }};
    }

    //! \brief Boundary state function providing the left and right state of a
    //!   face at symmetry boundaries
    //! \param[in] ul Left (domain-internal) state
    //! \param[in] fn Unit face normal
    //! \return Left and right states for all scalar components in this PDE
    //!   system
    //! \note The function signature must follow tk::StateFn
    static tk::StateFn::result_type
    symmetry( ncomp_t, ncomp_t, const std::vector< tk::real >& ul,
              tk::real, tk::real, tk::real, tk::real,
              const std::array< tk::real, 3 >& fn )
    {
      std::vector< tk::real > ur(5);
      // Internal cell velocity components
      auto v1l = ul[1]/ul[0];
      auto v2l = ul[2]/ul[0];
      auto v3l = ul[3]/ul[0];
      // Normal component of velocity
      auto vnl = v1l*fn[0] + v2l*fn[1] + v3l*fn[2];
      // Ghost state velocity components
      auto v1r = v1l - 2.0*vnl*fn[0];
      auto v2r = v2l - 2.0*vnl*fn[1];
      auto v3r = v3l - 2.0*vnl*fn[2];
      // Boundary condition
      ur[0] = ul[0];
      ur[1] = ur[0] * v1r;
      ur[2] = ur[0] * v2r;
      ur[3] = ur[0] * v3r;
      ur[4] = ul[4];
      return {{ std::move(ul), std::move(ur) }};
    }

    //! \brief Boundary state function providing the left and right state of a
    //!   face at characteristic boundaries
    //! \param[in] ul Left (domain-internal) state
    //! \param[in] fn Unit face normal
    //! \return Left and right states for all scalar components in this PDE
    //!   system
    //! \details The characteristic boudary calculation, implemented here, is
    //!   based on the characteristic theory of hyperbolic systems.
    //! \note The function signature must follow tk::StateFn
    static tk::StateFn::result_type
    characteristic( ncomp_t system, ncomp_t, const std::vector< tk::real >& ul,
                    tk::real, tk::real, tk::real, tk::real,
                    const std::array< tk::real, 3 >& fn )
    {
      using tag::param; using tag::bc;

      // Primitive variables from farfield
      auto frho = g_inputdeck.get< param, eq,
                                   tag::farfield_density >()[ system ];
      auto fp   = g_inputdeck.get< param, eq,
                                   tag::farfield_pressure >()[ system ];
      auto fu   = g_inputdeck.get< param, eq,
                                   tag::farfield_velocity >()[ system ];

      // Speed of sound from farfield
      auto fa = eos_soundspeed< eq >( system, frho, fp );

      // Normal component from farfield
      auto fvn = fu[0]*fn[0] + fu[1]*fn[1] + fu[2]*fn[2];

      // Mach number from farfield
      auto fM = fvn / fa;

      // Specific total energy from farfield
      auto frhoE =
        eos_totalenergy< eq >( system, frho, fu[0], fu[1], fu[2], fp );

      // Pressure from internal cell
      auto p = eos_pressure< eq >( system, ul[0], ul[1]/ul[0], ul[2]/ul[0],
                                   ul[3]/ul[0], ul[4] );

      auto ur = ul;

      if(fM <= -1)                         // Supersonic inflow
      {
        // For supersonic inflow, all the characteristics are from outside.
        // Therefore, we calculate the ghost cell state using the primitive
        // variables from outside.
        ur[0] = frho;
        ur[1] = frho * fu[0];
        ur[2] = frho * fu[1];
        ur[3] = frho * fu[2];
        ur[4] = frhoE;
      } else if(fM > -1 && fM < 0)       // Subsonic inflow
      {
        // For subsonic inflow, there are 1 outgoing characteristcs and 4
        // incoming characteristic. Therefore, we calculate the ghost cell state
        // by taking pressure from the internal cell and other quantities from
        // the outside.
        ur[0] = frho;
        ur[1] = frho * fu[0];
        ur[2] = frho * fu[1];
        ur[3] = frho * fu[2];
        ur[4] =
          eos_totalenergy< eq >( system, frho, fu[0], fu[1], fu[2], p );
      } else if(fM >= 0 && fM < 1)       // Subsonic outflow
      {
        // For subsonic outflow, there are 1 incoming characteristcs and 4
        // outgoing characteristic. Therefore, we calculate the ghost cell state
        // by taking pressure from the outside and other quantities from the
        // internal cell.
        ur[4] = eos_totalenergy< eq >( system, ul[0], ul[1]/ul[0], ul[2]/ul[0],
                                       ul[3]/ul[0], fp );
      }
      // Otherwise, for supersonic outflow, all the characteristics are from
      // internal cell. Therefore, we calculate the ghost cell state using the
      // conservative variables from outside.

      return {{ ul, ur }};
    }

    //! \brief Boundary state function providing the left and right state of a
    //!   face at extrapolation boundaries
    //! \param[in] ul Left (domain-internal) state
    //! \return Left and right states for all scalar components in this PDE
    //!   system
    //! \note The function signature must follow tk::StateFn
    static tk::StateFn::result_type
    extrapolate( ncomp_t, ncomp_t, const std::vector< tk::real >& ul,
                 tk::real, tk::real, tk::real, tk::real,
                 const std::array< tk::real, 3 >& )
    {
      return {{ ul, ul }};
    }
};

} // dg::

} // inciter::

#endif // DGCompFlow_h
