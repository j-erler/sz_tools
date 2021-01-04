
# -*- coding: utf-8 -*-
# 
#  This file is part of sz_tools.
# 
#  sz_tools is free software; you can redistribute it and/or modify
#  it under the terms of the MIT License.
# 
#  sz_tools is distributed in the hope that it will be useful,but 
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See 
#  the provided copy of the MIT License for more details.

"""sz_tools is a python implementation of matched filtering techniques for 
   applications in astronomy.
"""

__version__ = "1.1"

__bibtex__ = """
"""

from .sz_tools import (planck_uc, planck_beams, tsz_spec, tsz_spec_planck, ksz_spec, 
                       m500_2_r500, r500_2_m500, beta_model, beta_projected, gnfw, 
                       gnfw_projected, gnfw_projected_fast, gnfw_abel, simulate_cluster, 
                       simulate_cluster_beta, deproject, Y_500_sph, Y_500_cyl, T_e_profile, 
                       T_sz, T_sz_fast, tau_fast, simulate_rel_cluster, Y_500_planck, 
                       M_500_planck, r200r500, m200m500, rebin)
from .shortcuts import (make_table, write_file, read_file, writefits, readfits, hubble, 
                        angular_dist, luminosity_dist, dist, gaussian, create_histogram, 
                        convert_units, mbb_spec, cov2corr, corr2cov, compact_error, quantile, 
                        find_level, angle, radial_profile, sample_sphere_uniform)
from .ilc import (create_header, project_maps, ilc_windows, remove_offset, run_ilc, ilc_scales, 
                  ilc, ilc_allsky)
