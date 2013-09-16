//******************************************************************************
/*!
  \file      src/Model/Hydro/SLM/SLM.h
  \author    J. Bakosi
  \date      Sun 15 Sep 2013 05:21:33 PM MDT
  \copyright Copyright 2005-2012, Jozsef Bakosi, All rights reserved.
  \brief     Simplified Langevin hydrodynamics model
  \details   Simplified Langevin hydrodynamics model
*/
//******************************************************************************
#ifndef SLM_h
#define SLM_h

#include <Memory.h>
#include <Hydro.h>

namespace quinoa {

class Memory;
class Paradigm;
class JPDF;

//! SimplifiedLangevin : Hydro<SimplifiedLangevin> child for CRTP
//! See: http://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
class SimplifiedLangevin : public Hydro<SimplifiedLangevin> {

  public:
    //! Constructor
    explicit SimplifiedLangevin(const Base& base, real* const particles) :
      Hydro<SimplifiedLangevin>(base, particles),
      m_C0(base.control.get<control::param, control::slm, control::c0>()) {
      // Error out if hydro model selected at compile time does not match that
      // whose options are given in control file
      //control->matchModels<select::Hydro, select::HydroType, control::HYDRO>(
      //  select::HydroType::SLM);
      //ErrChk on m_C0
    }

    //! Destructor
    ~SimplifiedLangevin() noexcept override = default;

    //! Initialize particles
    void init();

    //! Advance particles
    void advance(const real& dt);

  private:
    //! Don't permit copy constructor
    SimplifiedLangevin(const SimplifiedLangevin&) = delete;
    //! Don't permit copy assigment
    SimplifiedLangevin& operator=(const SimplifiedLangevin&) = delete;
    //! Don't permit move constructor
    SimplifiedLangevin(SimplifiedLangevin&&) = delete;
    //! Don't permit move assigment
    SimplifiedLangevin& operator=(SimplifiedLangevin&&) = delete;

    const real m_C0;                //!< Parameter C0 in SLM
};

} // namespace quinoa

#endif // SLM_h
