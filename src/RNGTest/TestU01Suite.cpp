// *****************************************************************************
/*!
  \file      src/RNGTest/TestU01Suite.cpp
  \copyright 2012-2015 J. Bakosi,
             2016-2018 Los Alamos National Security, LLC.,
             2019-2020 Triad National Security, LLC.
             All rights reserved. See the LICENSE file for details.
  \brief     TestU01 random number generator test suite
  \details   This file declares the TestU01 random number generator test suite,
    which facilitates subjecting any supported random number generator to a
    battery of statistical tests interfaced to the TestU01 library.
*/
// *****************************************************************************

#include <string>
#include <iostream>
#include <cstddef>

#include "NoWarning/format.hpp"

#include "TestU01Suite.hpp"
#include "TestStack.hpp"
#include "SmallCrush.hpp"
#include "Crush.hpp"
#include "BigCrush.hpp"
#include "Options/RNG.hpp"
#include "RNGTest/Options/Battery.hpp"
#include "NoWarning/rngtest.decl.h"
#include "QuinoaConfig.hpp"

extern CProxy_Main mainProxy;

namespace rngtest {

extern TestStack g_testStack;

} // rngtest::

using rngtest::TestU01Suite;

TestU01Suite::TestU01Suite( ctr::BatteryType suite ) :
  m_ctrs(),
  m_tests(),
  m_name(),
  m_npval(0),
  m_ncomplete(0),
  m_ntest(0),
  m_nfail(),
  m_time(),
  m_failed()
// *****************************************************************************
// Constructor
//! \param[in] suite Enum id selecting TestU01 battery type
// *****************************************************************************
{
  // Add statistical tests to suite
  if ( suite == ctr::BatteryType::SMALLCRUSH )
    m_name = addTests< SmallCrush >();
  else if ( suite == ctr::BatteryType::CRUSH )
    m_name = addTests< Crush >();
  else if ( suite == ctr::BatteryType::BIGCRUSH )
    m_name = addTests< BigCrush >();
  else Throw( "Non-TestU01 RNG test suite passed to TestU01Suite" );

  // Construct all tests and store handles
  for (const auto& t : m_ctrs) m_tests.emplace_back( t() );

  // Collect number of results from all tests (one per RNG)
  for (std::size_t i=0; i<ntest(); ++i) m_tests[i].npval();
}

void
TestU01Suite::npval( std::size_t n )
// *****************************************************************************
// Collect number of p-values from a statistical test
//! \param[in] n Number of p-values the test contributes to the total
// *****************************************************************************
{
  m_npval += n;

  if ( ++m_ntest == ntest() ) {
    printer().battery( ntest(), m_npval );
    // Collect test names from all tests (one per RNG)
    m_ntest = 0;
    for (std::size_t i=0; i<ntest(); ++i) m_tests[i].names();
  }
}

void
TestU01Suite::names( std::vector< std::string > n )
// *****************************************************************************
// Collect test names from a statistical test
//! \param[in] n Vector of test names (there can be more than one from one test)
// *****************************************************************************
{
  auto print = printer();

  print.names( n );

  if ( ++m_ntest == ntest() ) {
    const auto& rngs = g_inputdeck.get< tag::selected, tag::rng >();
    std::stringstream ss;
    ss << "RNGs tested (" << rngs.size() << ")";
    print.section( ss.str() );
    #ifdef HAS_MKL
    print.MKLParams( rngs, g_inputdeck.get< tag::param, tag::rngmkl >() );
    #endif
    print.RNGSSEParams( rngs, g_inputdeck.get< tag::param, tag::rngsse >() );
    print.Random123Params( rngs,
                           g_inputdeck.get< tag::param, tag::rng123 >() );
    print.endpart();
    print.part( m_name );
    print.statshead( "Statistics computed",
                     m_npval*rngs.size(),
                     m_ctrs.size() );

    // Run battery of RNG tests
    for (const auto& t : m_tests) t.run();

    // Initialize space for counting the number failed tests per RNG. Note
    // that we could use tk::ctr::RNGType as the map-key here instead of the
    // RNG name (RNGType would be an enum vs the name which is a string),
    // but then the RNGType would have to be part of the status (instead of
    // the RNG name) returning from a completed test by TestU01Props::run().
    // That would require a return type that is less generic than the
    // current vector of vector of strings. Since the RNG name is already
    // part of the status, we just match the RNG name for counting failed
    // tests per RNG and keep the status more generic. See also the
    // discussion on the return type in TestU01Props::run().
    tk::ctr::RNG rng;
    for (const auto& r : rngs )
      m_nfail[ rng.name(r) ] = 0;
  }
}

void
TestU01Suite::evaluate( std::vector< std::vector< std::string > > status )
// *****************************************************************************
// Evaluate statistical test
//! \param[in] status Status vectors of strings for a test
// *****************************************************************************
{
  printer().test( ++m_ncomplete, m_ctrs.size(), m_nfail, status );

  // Store information on failed test for final assessment
  for (std::size_t p=0; p<status[1].size(); ++p)
    if (status[1][p].size() > 4)
      m_failed.emplace_back( status[0][p], status[2][0], status[1][p] );

  if ( m_ncomplete == m_ctrs.size() ) {
    // Collect measured test run times
    m_ncomplete = 0;
    for (const auto& t : m_tests) t.time();
  }
}

void
TestU01Suite::time( std::pair< std::string, tk::real > t )
// *****************************************************************************
// Collect test times measured in seconds from a statistical test
//! \param[in] t Measured time to do the test for an RNG
// *****************************************************************************
{
  m_time[ t.first ] += t.second;

  if ( ++m_ncomplete == m_ctrs.size() ) assess();
}

void
TestU01Suite::assess()
// *****************************************************************************
// Output final assessment
// *****************************************************************************
{
  auto print = printer();

  // Output summary of failed tests for all RNGs tested
  if ( !m_failed.empty() ) {
    const auto& rngs = g_inputdeck.get< tag::selected, tag::rng >();
    print.failed( "Failed statistics", m_npval*rngs.size(), m_failed );
  } else print.note< tk::QUIET >( "All tests passed" );

  // Cost and quality assessment only for more than one RNG
  if (m_time.size() > 1) {
    // Output measured times per RNG in order of computational cost
    print.cost( "Generator cost",
                "Measured times in seconds in increasing order (low is good)",
                m_time );
    // Output number of failed tests per RNG in order of decreasing quality
    print.rank( "Generator quality",
                "Number of failed tests in increasing order (low is good)",
                m_nfail );
  }

  // Quit
  mainProxy.finalize();
}

std::size_t
TestU01Suite::ntest() const
// *****************************************************************************
// Return the number of statistical tests per each RNG tested
//! \return The number of statistical tests for each RNG tested
// *****************************************************************************
{
  const auto& rngs = g_inputdeck.get< tag::selected, tag::rng >();
  return m_ctrs.size() / rngs.size();
}

#include "NoWarning/testu01suite.def.h"
