//******************************************************************************
/*!
  \file      src/Control/Quinoa/Options/DiffEq.h
  \author    J. Bakosi
  \date      Tue 16 Sep 2014 08:15:06 AM MDT
  \copyright 2005-2014, Jozsef Bakosi.
  \brief     Differential equation options and associations
  \details   Differential equation options and associations
*/
//******************************************************************************
#ifndef QuinoaDiffEqOptions_h
#define QuinoaDiffEqOptions_h

#include <map>

#include <Toggle.h>
#include <Quinoa/InputDeck/Keywords.h>
#include <Quinoa/Options/InitPolicy.h>
#include <Quinoa/Options/CoeffPolicy.h>

namespace quinoa {
namespace ctr {

//! Differential equation types
enum class DiffEqType : uint8_t { NO_DIFFEQ=0,
                                  ORNSTEIN_UHLENBECK,
                                  LOGNORMAL,
                                  SKEWNORMAL,
                                  DIRICHLET,
                                  GENDIR };

//! Pack/Unpack: forward overload to generic enum class packer
inline void operator|( PUP::er& p, DiffEqType& e ) { PUP::pup( p, e ); }

//! Differential equation key used access a differential equation in a factory
using DiffEqKey =
  tk::tuple::tagged_tuple< tag::diffeq,      ctr::DiffEqType,
                           tag::initpolicy,  ctr::InitPolicyType,
                           tag::coeffpolicy, ctr::CoeffPolicyType >;

//! Class with base templated on the above enum class with associations
class DiffEq : public tk::Toggle< DiffEqType > {

  public:
    //! Constructor: pass associations references to base, which will handle
    //! class-user interactions
    explicit DiffEq() :
      Toggle< DiffEqType >( "Differential equation",
        //! Enums -> names
        { { DiffEqType::NO_DIFFEQ, "n/a" },
          { DiffEqType::ORNSTEIN_UHLENBECK, kw::ornstein_uhlenbeck().name() },
          { DiffEqType::LOGNORMAL, kw::lognormal().name() },
          { DiffEqType::SKEWNORMAL, kw::skewnormal().name() },
          { DiffEqType::DIRICHLET, kw::dirichlet().name() },
          { DiffEqType::GENDIR, kw::gendir().name() } },
        //! keywords -> Enums
        { { "no_diffeq", DiffEqType::NO_DIFFEQ },
          { kw::ornstein_uhlenbeck().string(), DiffEqType::ORNSTEIN_UHLENBECK },
          { kw::lognormal().string(), DiffEqType::LOGNORMAL },
          { kw::skewnormal().string(), DiffEqType::SKEWNORMAL },
          { kw::dirichlet().string(), DiffEqType::DIRICHLET },
          { kw::gendir().string(), DiffEqType::GENDIR } } ) {}
};

} // ctr::
} // quinoa::

#endif // QuinoaDiffEqOptions_h