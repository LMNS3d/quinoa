title "Example problem"

walker

  term  10.0    # Max time
  dt    0.001   # Time step size
  npar  10000   # Number of particles
  ttyi  1000    # TTY output interval

  rngs
    mkl_mcg59 seed 0 end
  end

  diag_ou
    depvar o
    init raw
    coeff const_coeff
    ncomp 2
    sigmasq 0.25 1.0 end
    theta 1.0 1.0 end
    mu 0.0 1.5 end
    rng mkl_mcg59
  end

  pdfs
    interval          10000
    filetype          txt
    policy            overwrite
    centering         elem
    format            scientific
    precision         4
    f2( o1 o2 : 0.2 0.2 ; -2 2 -4 4 )
  end
end