//******************************************************************************
/*!
  \file      src/Statistics/BiPDF.h
  \author    J. Bakosi
  \date      Tue 16 Sep 2014 10:56:49 PM MDT
  \copyright 2005-2014, Jozsef Bakosi.
  \brief     Joint bivariate PDF estimator
  \details   Joint bivariate PDF estimator
*/
//******************************************************************************
#ifndef BiPDF_h
#define BiPDF_h

#include <array>
#include <unordered_map>
#include <algorithm>

#include <Types.h>
#include <PUPUtil.h>

namespace quinoa {

//! Joint bivariate PDF estimator
class BiPDF {

    //! Key type
    using key_type = std::array< long, 2 >;

    // Hash function for key_type
    struct key_hash {
      long operator()( const key_type& key ) const {
        return std::hash< long >()( key[0] ) ^ std::hash< long >()( key[1] );
      }
    };

    //! Pair type
    using pair_type = std::pair< const key_type, tk::real >;

    //! Joint bivariate PDF is an unordered_map: key: two bin ids corresponding
    //! to the two sample space dimensions, mapped value: sample counter,
    //! hasher: XORed hash of the two bin ids
    using map_type = std::unordered_map< key_type, tk::real, key_hash >;

  public:
    //! Empty constructor for Charm++
    explicit BiPDF() : m_binsize( {{ 0, 0 }} ), m_nsample( 0 ) {}

    //! Constructor: Initialize joint bivariate PDF container
    //! \param[in]   binsize    Sample space bin size
    explicit BiPDF( const std::vector< tk::real >& bs ) :
      m_binsize( {{ bs[0], bs[1] }} ),
      m_nsample( 0 ) {}

    //! Accessor to number of samples
    //! \return Number of samples collected
    std::size_t nsample() const noexcept { return m_nsample; }

    //! Add sample to bivariate PDF
    //! \param[in]  sample  Sample to add
    void add( std::array< tk::real, 2 > sample ) {
      ++m_nsample;
      ++m_pdf[ {{ std::lround( sample[0] / m_binsize[0] ),
                  std::lround( sample[1] / m_binsize[1] ) }} ];
    }

    //! Add multiple samples from a PDF
    //! \param[in]  p  PDF whose samples to add
    void addPDF( const BiPDF& p ) {
      m_binsize = p.binsize();
      m_nsample += p.nsample();
      for (const auto& e : p.map()) m_pdf[ e.first ] += e.second;
    }

    //! Zero bins
    void zero() noexcept { m_nsample = 0; m_pdf.clear(); }

    //! Constant accessor to underlying PDF map
    //! \return Reference to underlying map
    const map_type& map() const noexcept { return m_pdf; }

    //! Constant accessor to bin sizes
    //! \return Sample space bin sizes
    const std::array< tk::real, 2 >& binsize() const noexcept
    { return m_binsize; }

    //! Return minimum and maximum bin ids of sample space in both dimensions
    //! \return  {minx,miny}{maxx,maxy}  Minima and maxima of the bin ids
    std::pair< key_type, key_type > extents() const {
      auto x = std::minmax_element( begin(m_pdf), end(m_pdf),
                 []( const pair_type& a, const pair_type& b )
                 { return a.first[0] < b.first[0]; } );
      auto y = std::minmax_element( begin(m_pdf), end(m_pdf),
                 []( const pair_type& a, const pair_type& b )
                 { return a.first[1] < b.first[1]; } );
      return { {{ x.first->first[0], y.first->first[1] }},
               {{ x.second->first[0], y.second->first[1] }} };
    }

    //! Pack/Unpack
    void pup( PUP::er& p ) {
      p | m_binsize;
      p | m_nsample;
      p | m_pdf;
    }

  private:
    std::array< tk::real, 2 > m_binsize;//!< Sample space bin sizes
    std::size_t m_nsample;              //!< Number of samples collected
    map_type m_pdf;                     //!< Probability density function
};

} // quinoa::

#endif // BiPDF_h