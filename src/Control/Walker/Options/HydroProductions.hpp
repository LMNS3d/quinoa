// *****************************************************************************
/*!
  \file      src/Control/Walker/Options/HydroProductions.hpp
  \copyright 2012-2015 J. Bakosi,
             2016-2018 Los Alamos National Security, LLC.,
             2019-2020 Triad National Security, LLC.
             All rights reserved. See the LICENSE file for details.
  \brief     Hydrodynamics production divided by dissipation rate options
  \details   Hydrodynamics production divided by dissipation rate options
*/
// *****************************************************************************
#ifndef HydroProductionsOptions_h
#define HydroProductionsOptions_h

#include <brigand/sequences/list.hpp>

#include "Toggle.hpp"
#include "Keywords.hpp"
#include "PUPUtil.hpp"
#include "DiffEq/HydroProductions.hpp"

namespace walker {
namespace ctr {

//! Hydrodynamics production divided by dissipation rate types
enum class HydroProductionsType : uint8_t { PROD_A005H=0
                                          , PROD_A005S
                                          , PROD_A005L
                                          , PROD_A05H
                                          , PROD_A05S
                                          , PROD_A05L
                                          , PROD_A075H
                                          , PROD_A075S
                                          , PROD_A075L
                                          };

//! \brief Pack/Unpack HydroProductionsType: forward overload to generic enum
//!   class packer
inline void operator|( PUP::er& p, HydroProductionsType& e )
{ PUP::pup( p, e ); }

//! HydroProductions options: outsource searches to base templated on enum type
class HydroProductions : public tk::Toggle< HydroProductionsType > {

  public:
    //! Valid expected choices to make them also available at compile-time
    using keywords = brigand::list< kw::prod_A005H
                                  , kw::prod_A005S
                                  , kw::prod_A005L
                                  , kw::prod_A05H
                                  , kw::prod_A05S
                                  , kw::prod_A05L
                                  , kw::prod_A075H
                                  , kw::prod_A075S
                                  , kw::prod_A075L
                                  >;

    //! \brief Options constructor
    //! \details Simply initialize in-line and pass associations to base, which
    //!    will handle client interactions
    explicit HydroProductions() :
      tk::Toggle< HydroProductionsType >(
        //! Group, i.e., options, name
        "Hydrodynamics production divided by dissipation rate",
        //! Enums -> names
        { { HydroProductionsType::PROD_A005H, kw::prod_A005H::name() },
          { HydroProductionsType::PROD_A005S, kw::prod_A005S::name() },
          { HydroProductionsType::PROD_A005L, kw::prod_A005L::name() },
          { HydroProductionsType::PROD_A05H, kw::prod_A05H::name() },
          { HydroProductionsType::PROD_A05S, kw::prod_A05S::name() },
          { HydroProductionsType::PROD_A05L, kw::prod_A05L::name() },
          { HydroProductionsType::PROD_A075H, kw::prod_A075H::name() },
          { HydroProductionsType::PROD_A075S, kw::prod_A075S::name() },
          { HydroProductionsType::PROD_A075L, kw::prod_A075L::name() } },
        //! keywords -> Enums
        {  { kw::prod_A005H::string(), HydroProductionsType::PROD_A005H },
           { kw::prod_A005S::string(), HydroProductionsType::PROD_A005S },
           { kw::prod_A005L::string(), HydroProductionsType::PROD_A005L },
           { kw::prod_A05H::string(), HydroProductionsType::PROD_A05H },
           { kw::prod_A05S::string(), HydroProductionsType::PROD_A05S },
           { kw::prod_A05L::string(), HydroProductionsType::PROD_A05L },
           { kw::prod_A075H::string(), HydroProductionsType::PROD_A075H },
           { kw::prod_A075S::string(), HydroProductionsType::PROD_A075S },
           { kw::prod_A075L::string(), HydroProductionsType::PROD_A075L } } ) {}

    //! \brief Return table based on Enum
    //! \param[in] t Enum value of the option requested
    //! \return tk::Table associated to the option
    tk::Table table( HydroProductionsType t ) const {
      if (t == HydroProductionsType::PROD_A005H)
        return prod_A005H;
      else if (t == HydroProductionsType::PROD_A005S)
        return prod_A005S;
      else if (t == HydroProductionsType::PROD_A005L)
        return prod_A005L;
      else if (t == HydroProductionsType::PROD_A05H)
        return prod_A05H;
      else if (t == HydroProductionsType::PROD_A05S)
        return prod_A05S;
      else if (t == HydroProductionsType::PROD_A05L)
        return prod_A05L;
      else if (t == HydroProductionsType::PROD_A075H)
        return prod_A075H;
      else if (t == HydroProductionsType::PROD_A075S)
        return prod_A075S;
      else if (t == HydroProductionsType::PROD_A075L)
        return prod_A075L;
      else Throw( "Hydrodynamics P/e associated to " +
                  std::to_string( static_cast<uint8_t>(t) ) + " not found" );
    }

};

} // ctr::
} // walker::

#endif // HydroProductionsOptions_h
