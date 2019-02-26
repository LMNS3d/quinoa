// *****************************************************************************
/*!
  \file      src/Main/InciterPrint.h
  \copyright 2012-2015, J. Bakosi, 2016-2019, Los Alamos National Security, LLC.
  \brief     Inciter-specific pretty printer functionality
  \details   Inciter-specific pretty printer functionality.
*/
// *****************************************************************************
#ifndef InciterPrint_h
#define InciterPrint_h

#include <iostream>
#include <string>

#include <brigand/algorithms/for_each.hpp>

#include "NoWarning/format.h"

#include "Print.h"
#include "ContainerUtil.h"
#include "Inciter/InputDeck/InputDeck.h"
#include "Inciter/Options/Physics.h"
#include "Inciter/Options/Problem.h"

namespace inciter {

extern ctr::InputDeck g_inputdeck_defaults;
extern ctr::InputDeck g_inputdeck;

//! InciterPrint : tk::Print
class InciterPrint : public tk::Print {

  public:
    //! Constructor
    //! \param[in,out] str Verbose stream
    //! \param[in,out] qstr Quiet stream
    //! \see tk::RNGPrint::RNGPrint and tk::Print::Print
    explicit InciterPrint( std::ostream& str = std::clog,
                           std::ostream& qstr = std::cout ) :
      Print( str, qstr ) {}

    //! Print control option: 'group : option'
    template< typename Option, typename... tags >
    void Item() const {
      Option opt;
      m_stream << m_item_name_value_fmt
                  % m_item_indent % opt.group()
                  % opt.name( g_inputdeck.get< tags... >() );
    }

    //! Print list of codes of vector-valued option
    //! \param[in] v Vector of option types (enums) whose code vector to print
    template< typename Option, typename T >
    void ItemVec( const std::vector< T >& v ) const {
      Option opt;
      std::string codes;
      for (auto e : v) codes += opt.code(e);
      item( opt.group(), codes );
    }

    // Helper class for compact output of PDE policies
    class Policies {
      public:
        // Default constructor
        explicit Policies() : phys(), prob() {}
        // Initializer constructor
        explicit Policies( const std::string& p, const std::string& t ) :
          phys(p), prob(t) {}
        // Operator += for adding up two Policies structs
        Policies& operator+= ( const Policies& p ) {
          phys += p.phys;
          prob += p.prob;
          return *this;
        }
        // Output unique policies to output stream
        friend std::ostream& operator<< ( std::ostream& os, const Policies& p )
        {
          Policies copy( p );     // copy policies
          copy.unique();          // get rid of duplicate policies
          os << static_cast< char >( kw::physics::info::code::value ) << ':'
             << copy.phys << ", "
             << static_cast< char >( kw::problem::info::code::value ) << ':'
             << copy.prob;
          return os;
        }

      private:
        // Make all policies unique
        void unique() { tk::unique( phys ); tk::unique( prob ); }

        std::string phys;
        std::string prob;
    };

    //! Print PDE factory legend
    void eqlegend();

    //! Print equation list with policies
    //! \param[in] t Section title
    //! \param[in] factory Factory to get equation data from
    //! \param[in] ntypes Unique equation types
    template< class Factory >
    void eqlist( const std::string& t,
                 const Factory& factory,
                 std::size_t ntypes ) const
    {
      if (!factory.empty()) {
        section( t );
        item( "Unique equation types", ntypes );
        item( "With all policy combinations", factory.size() );
         // extract eqname and supported policies for output
        const auto p = ctr::Physics();
        const auto r = ctr::Problem();
        std::map< std::string, Policies > eqs;      // eqname : policies
        for (const auto& f : factory)
          eqs[ PDEName( f.first ) ] +=
            Policies( p.code( f.first.template get< tag::physics >() ),
                      r.code( f.first.template get< tag::problem >() ) );
        // output eqname and supported policies
        for (const auto& e : eqs)
          m_stream << m_item_name_value_fmt % m_item_indent
                                            % e.first % e.second;
      }
    }

    //! Print configuration of a stack of partial differential equations
    void pdes( const std::string& t,
      const std::vector< std::vector< std::pair< std::string, std::string > > >&
        info ) const;

    //! Print time integration header
    void inthead( const std::string& t, const std::string& name,
                  const std::string& legend, const std::string& head ) const;

    //! Print mesh refinement variables and their indices in the unknown vector
    void refvar( const std::vector< std::string >& rvar,
                 const std::vector< std::size_t >& refidx );

    //! Print initial mesh refinement edge-node pairs
    void edgeref( const std::vector< std::size_t >& edgenodes );

  private:
    //! Return partial differential equation name
    //! \param[in] key Equation key
    //! \return Partial differential equation name based on key
    template< class Key >
    std::string PDEName ( const Key& key ) const
    { return ctr::PDE().name( key.template get< tag::pde >() ); }
};

} // inciter::

#endif // InciterPrint_h
