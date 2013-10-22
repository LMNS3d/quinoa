//******************************************************************************
/*!
  \file      src/Control/RNGTest/InputDeck/Keywords.h
  \author    J. Bakosi
  \date      Sun 06 Oct 2013 04:03:02 PM MDT
  \copyright Copyright 2005-2012, Jozsef Bakosi, All rights reserved.
  \details   All keywords recognized by Quinoa's random number generator (RNG)
  test suite input deck parser. The keywords are defined by specializing struct
  'keyword', defined in Control/Keyword.h. Introducing a new keyword requires a
  more human readable (bust still short) name as well as a short, few-line,
  help-like description.
*/
//******************************************************************************
#ifndef RNGTestInputDeckKeywords_h
#define RNGTestInputDeckKeywords_h

//! Signal to compiler that we are building a list of keywords. This is used by
//! the inline includes, such as *Keywords.h, below (if any) to make sure they
//! get included in the correct namespace and not polluting the global one.
#define Keywords

#include <Keyword.h>

namespace rngtest {
//! List of keywords the parser understands
namespace kw {

using namespace pegtl::ascii;
using quinoa::kw::keyword;

// Include base keywords recognized by all input deck parsers
#include <BaseKeywords.h>

// Include Intel's MKL's RNG keywords
#include <MKLRNGKeywords.h>

// Keyword 'smallcrush'
struct smallcrush_info {
  static const char* name() { return "SmallCrush"; }
  static const char* help() { return
    "This keyword is used to introduce the description of the random number "
    "generator test suite, i.e., battery, 'SmallCrush'. SmallCrush is a "
    "battery of relatively small number, O(10), of tests, defined in TestU01, "
    "a library for the empirical testing of random number generators. For more "
    "info, see http://www.iro.umontreal.ca/~simardr/testu01/tu01.html.";
  }
};
using smallcrush = keyword< smallcrush_info, s,m,a,l,l,c,r,u,s,h >;

// Keyword 'crush'
struct crush_info {
  static const char* crush_name() { return "Crush"; }
  static const char* crush_help() { return
    "This keyword is used to introduce the description of the random number "
    "generator test suite, i.e., battery, 'Crush'. Crush is a suite of "
    "stringent statistical tests, O(100), defined in TestU01, a library for "
    "the empirical testing of random number generators. For more info, see "
    "http://www.iro.umontreal.ca/~simardr/testu01/tu01.html.";
  }
};
using crush = keyword< crush_info, c,r,u,s,h >;

// Keyword 'bigcrush'
struct bigcrush_info {
  static const char* bigcrush_name() { return "BigCrush"; }
  static const char* bigcrush_help() { return
    "This keyword is used to introduce the description of the random number "
    "generator test suite, i.e., battery, 'BigCrush'. BigCrush is a "
    "suite of very stringent statistical tests, O(100), defined in TestU01, a "
    "library for the empirical testing of random number generators. For more "
    "info, see http://www.iro.umontreal.ca/~simardr/testu01/tu01.html.";
  }
};
using bigcrush = keyword< bigcrush_info, b,i,g,c,r,u,s,h >;

// Keyword 'rngs'
struct rngs_info {
  static const char* rngs_name() { return "RNGs start block"; }
  static const char* rngs_help() { return
    "This keyword is used to introduce a block that lists the names of the "
    "random number generators to test. Example:\n"
    "\trngs\n"
    "\t  mkl_r250\n"
    "\t  mkl_mcg31\n"
    "\t  mkl_mrg32k3a\n"
    "\tend";
  }
};
using rngs = keyword< rngs_info, r,n,g,s >;

} // kw::
} // rngtest::

#undef Keywords

#endif // RNGTestInputDeckKeywords_h