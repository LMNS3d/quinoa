//******************************************************************************
/*!
  \file      src/Physics/HomRT/HomRT.h
  \author    J. Bakosi
  \date      Sun 15 Sep 2013 05:47:42 PM MDT
  \copyright Copyright 2005-2012, Jozsef Bakosi, All rights reserved.
  \brief     Homogeneous Rayleigh-Taylor
  \details   Homogeneous Rayleigh-Taylor
*/
//******************************************************************************
#ifndef HomRT_h
#define HomRT_h

#include <map>

#include <Physics.h>
#include <Mix.h>
#include <Timer.h>

namespace quinoa {

class Memory;
class Paradigm;
class QuinoaControl;

//! HomRT : Physics
class HomRT : public Physics {

  public:
    //! Constructor
    explicit HomRT(const Base& base) :
      Physics(base),
      m_totalTime(base.timer.create("Total solution")) {}

    //! Destructor
    ~HomRT() noexcept override = default;

    //! Initialize model
    void init() override;

    //! Solve model
    void solve() override;

  private:
    //! Don't permit copy constructor
    HomRT(const HomRT&) = delete;
    //! Don't permit copy assigment
    HomRT& operator=(const HomRT&) = delete;
    //! Don't permit move constructor
    HomRT(HomRT&&) = delete;
    //! Don't permit move assigment
    HomRT& operator=(HomRT&&) = delete;

    //! Echo information on homogeneous Rayleigh-Taylor physics
    void echo();

    //! One-liner report
    void reportHeader() const;
    void report(const uint64_t it,
                const uint64_t nstep,
                const real t,
                const real dt,
                const bool wroteJpdf,
                const bool wroteGlob,
                const bool wrotePlot);

    //! Output joint scalar PDF
    void outJpdf(const real t);

    const TimerIdx m_totalTime;           //!< Timer measuring the total run    
};

} // namespace quinoa

#endif // HomRT_h
