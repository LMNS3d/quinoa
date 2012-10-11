//******************************************************************************
/*!
  \file      src/Base/Exception.h
  \author    J. Bakosi
  \date      Wed 10 Oct 2012 04:11:31 PM EDT
  \copyright Copyright 2005-2012, Jozsef Bakosi, All rights reserved.
  \brief     Exception base class declaration
  \details   Exception base class declaration
*/
//******************************************************************************
#ifndef Exception_h
#define Exception_h

#include <QuinoaTypes.h>

namespace Quinoa {

//! Exception types
// ICC: no strongly typed enums yet
enum ExceptType { CUMULATIVE=0,  //!< Only several will produce a warning
                  WARNING,       //!< Warning: output message
                  ERROR,         //!< Error: output but will not interrupt
                  UNCAUGHT,      //!< Uncaught: will interrupt
                  FATAL,         //!< Fatal error: will interrupt
                  NUM_EXCEPT
};

//! Error codes for the OS (or whatever calls Quinoa)
enum ErrCode { NO_ERROR=0,       //!< Everything went fine
               NONFATAL,         //!< Exception occurred but continue
               FATAL_ERROR,      //!< Fatal error occurred
               NUM_ERR_CODE
};

class Driver;

//! Exception base class
class Exception {

  public:
    //! Constructor
    Exception(ExceptType except) : m_except(except) {}

    //! Destructor
    ~Exception() = default;

    //! Handle Exception passing pointer to driver
    ErrCode handleException(Driver* driver);

  protected:
    //! Move constructor, necessary for throws, default compiler generated,
    //! can only be thrown from within derived Exception classes
    Exception(Exception&&) = default;

    //! Don't permit copy constructor
    // ICC: should be deleted and private
    Exception(const Exception&);

  private:
    //! Don't permit copy assignment
    Exception& operator=(const Exception&) = delete;
    //! Don't permit move assignment
    Exception& operator=(Exception&&) = delete;

    //! Exception type (CUMULATIVE, WARNING, ERROR, etc.)
    ExceptType m_except;
};

} // namespace Quinoa

#endif // Exception_h
