// *****************************************************************************
/*!
  \file      src/Main/InciterDriver.h
  \copyright 2012-2015, J. Bakosi, 2016-2019, Los Alamos National Security, LLC.
  \brief     Inciter driver
  \details   Inciter driver.
*/
// *****************************************************************************
#ifndef InciterDriver_h
#define InciterDriver_h

#include "Inciter/CmdLine/CmdLine.h"

namespace inciter {

class InciterPrint;

//! Inciter driver used polymorphically with tk::Driver
class InciterDriver {

  public:
    //! Constructor
    explicit InciterDriver( const InciterPrint& print,
                            const ctr::CmdLine& cmdline );

    //! Execute driver
    void execute() const;

  private:
    const InciterPrint& m_print;        //!< Pretty printer
};

} // inciter::

#endif // InciterDriver_h
