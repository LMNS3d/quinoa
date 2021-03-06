// *****************************************************************************
/*!
  \file      src/Control/Random123Grammar.hpp
  \copyright 2012-2015 J. Bakosi,
             2016-2018 Los Alamos National Security, LLC.,
             2019-2020 Triad National Security, LLC.
             All rights reserved. See the LICENSE file for details.
  \brief     Random123-related grammar
  \details   This file defines Random1232 library related grammar, (re-)used by
     several executables.
*/
// *****************************************************************************
#ifndef Random123Grammar_h
#define Random123Grammar_h

#include "CommonGrammar.hpp"

namespace tk {
//! Toolkit, grammar definition for the Random123 library
namespace random123 {

  using namespace tao;

  // Random123 PEGTL grammar

  //! \brief rng: match any one of the Random123 random number generators
  template< template< class > class use >
  struct rng :
         pegtl::sor< typename use< kw::r123_threefry >::pegtl_string,
                     typename use< kw::r123_philox >::pegtl_string > {};

  //! \brief Match and set Random123 RNG seed
  template< template< class > class use, typename sel,
            typename vec, typename... tags >
  struct seed :
         tk::grm::process< use< kw::seed >,
                           tk::grm::insert_seed< sel, vec, tags... > > {};

  //! \brief Match Random123 RNGs in an rngs ... end block
  //! \see walker::deck::rngs
  template< template< class > class use, typename sel,
            typename vec, typename... tags >
  struct rngs :
         pegtl::if_must<
           tk::grm::scan< rng< use >,
                          tk::grm::store_back_option< use,
                                                      ctr::RNG,
                                                      sel, vec > >,
           tk::grm::block< use< kw::end >,
                           seed< use, sel, vec, tags... > > > {};

} // random123::
} // tk::

#endif // Random123Grammar_h
