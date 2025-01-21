FastXSF
-------

Fast X-ray spectral fitting.

Currently, there are the following issues in modern X-ray spectral fitting:

1. Response matrices have become huge (e.g. XRISM), making models slow to evaluate.
2. XSPEC is not developed openly and its quirks make it difficult to build upon.
3. Models are maintained by the community in XSPEC.
4. Maintaining additional software packages requires institutional efforts (CXC: sherpa, SRON: spex).
5. Not all models are differentiable. Reimplementing them in a differentiable language one by one is a significant effort, but has been tried but such projects tend to die out (see also 3ML astromodels).
6. Inference parameter spaces are complicated in X-ray astronomy, with multiple modes and 
7. Bayesian Model Comparison is nice to have.

Therefore, we want:

1) Performant code
2) Community packages from XSPEC
3) Nested sampling

FastXSF does that.

Approach
--------

1) Vectorization.
   Folding the spectrum through the RMF is vectorized.
   Handling many proposed spectra at once keeps memory low and efficiency high.
   Modern Bayesian sampling algorithms (e.g. UltraNest) can handle this.

2) Building upon the CXC (Doug Burke's) wrapper for Xspec models. https://github.com/cxcsds/xspec-models-cxc/
   All XSPEC models are available for use!

3) Some further niceties (all optional) include handling of backgrounds, redshifts and galactic NH:

   * Use BXA's autobackground folder to create a background spectral model from your background region.
   * Use BXA's galnh.py to fetch the galactic NH for the position of your observation and store it in my.pha.nh as a string (e.g. 1.2e20).
   * Store the redshift in my.pha.z as a string.

4) We treat X-ray spectral fitting as a normal inference problem like any other!

   Define a likelihood, prior and call a sampler. No need to carry around
   chi-square, C-stat, background-subtraction legacy awkwardness!

Getting started
---------------

To start, have a look at simple.py, which demonstrates:

* loading a spectrum
* loading a ATable
* setting up a XSPEC model
* passing the model through the ARF and RMF
* adding a background model
* computing the likelihood and print it
* plotting source and background fit and spectrum

Next, the vectorization is in simplev.py, which demonstrates the same as above plus:

* vectorized handling of proposals
* launching UltraNest

Credits
--------

This builds upon work by the Chandra X-ray Center (in particular Doug Burke's wrapper),
and Daniela Huppenkothen's RMF/ARF reader (based in turn on sherpa code, IIRC).
