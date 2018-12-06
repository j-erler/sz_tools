.. sz_tools:

.. currentmodule:: sz_tools


:mod:`sz_tools` -- What's available
===================================

Planck-related tools
--------------------
.. autosummary::
   :toctree: generated/

   planck_uc
   planck_beams

SED models
----------
.. autosummary::
   :toctree: generated/

   tsz_spec
   tsz_spec_planck
   ksz_spec
   mbb_spec

clusters: general
-----------------
.. autosummary::
   :toctree: generated/

   m500_2_r500
   r500_2_m500
   M_500_planck
   Y_500_planck
   Y_500_sph
   Y_500_cyl
   r200r500
   m200m500

clusters: pressure and temperature profiles
-------------------------------------------
.. autosummary::
   :toctree: generated/

   beta_model
   beta_projected
   simulate_cluster_beta
   gnfw
   gnfw_projected
   gnfw_projected_fast
   gnfw_abel
   T_e_profile
   T_sz
   T_sz_fast
   tau_fast
   simulate_cluster
   simulate_rel_cluster
   deproject

ILC
---
.. autosummary::
   :toctree: generated/

   create_header
   project_maps
   ilc_windows
   remove_offset
   run_ilc
   ilc
   ilc_scales
   ilc_allsky
	
statistics
----------
.. autosummary::
   :toctree: generated/

   cov2corr
   corr2cov
   compact_error
   quantile
   find_level

wrappers
--------
.. autosummary::
   :toctree: generated/

   make_table
   write_file
   read_file
   writefits
   readfits
   hubble
   angular_dist
   luminosity_dist
   dist
   gaussian
   create_histogram
   convert_units
