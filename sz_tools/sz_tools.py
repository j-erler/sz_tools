import numpy as np
from astropy.io import ascii
from astropy.io import fits
from scipy import interpolate
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import M_sun, pc, e, m_e, c, h, k_B
from scipy.integrate import quad, simps
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import derivative
from scipy.interpolate import make_interp_spline
from scipy.optimize import bisect
import os.path

datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.7255)

h = h.si.value
k_B = k_B.si.value
c = c.si.value
thomson = 6.6524587e-29
T_CMB = cosmo.Tcmb0.si.value
m_e = m_e.si.value
e = e.si.value
pc = pc.si.value
M_sun = M_sun.si.value
H0 = cosmo.H0.value
I_0 = 2*(k_B*T_CMB)**3/(h*c)**2*1e20


fname1 = os.path.join(datapath, "tsz_tabulated.fits")
hdul = fits.open(fname1)
tsz_grid = np.transpose(hdul[0].data)
hdul.close()

T_e = np.linspace(0,75,760)
f = np.geomspace(1e10,2.8e12,1000)
tsz_interpol = interpolate.RectBivariateSpline(f, T_e, tsz_grid, kx=1, ky=1)

fname2 = os.path.join(datapath, "planck_tsz_tabulated.csv")
data = ascii.read(fname2)
tsz_table = {'T_e': np.array(data[:]['T_e']),
            30: np.array(data[:]['LFI_1']),
            44: np.array(data[:]['LFI_2']),
            70: np.array(data[:]['LFI_3']),
            100: np.array(data[:]['HFI_1']),
            143: np.array(data[:]['HFI_2']),
            217: np.array(data[:]['HFI_3']),
            353: np.array(data[:]['HFI_4']),
            545: np.array(data[:]['HFI_5']),
            857: np.array(data[:]['HFI_6'])}


def planck_uc(nu):
	'''Returns the K_CMB --> MJy unit conversion factors form the
	2015 PlancK explanatory supplement. 

	Parameters
	----------
	nu: int
		Planck central band frequency in GHz

	Returns
	-------
	uc: float
		Unit conversion factor at given frequency
	'''

	conversion = {30: 23.5099, 44: 55.7349, 70: 129.1869, 
                  100: 244.0960, 143: 371.7327, 217: 483.6874, 
                  353: 287.4517, 545: 58.0356, 857: 2.2681}

	uc = conversion[nu]

	return(uc)


def planck_beams(nu):
	'''Returns the FWHM of the Planck LFI and HFI form the
	2015 PlancK explanatory supplement. 

	Parameters
	----------
	nu: int
		Planck central band frequency in GHz

	Returns
	-------
	uc: float
		Planck beam FWHM at given frequency
	'''

	beams = {30: 32.29, 44: 27.00, 70: 13.21, 
             100: 9.68, 143: 7.30, 217: 5.02, 
             353: 4.94, 545: 4.83, 857: 4.64}

	fwhm = beams[nu]

	return(fwhm)


def tsz_spec(nu, y, T_e=0, MJy=False):
	'''Returns the tSZ spectrum with optional relativistic corrections. 

	Parameters
	----------
	nu: float or array
		Frequency in Hz
	y: float
		Value of the compton-y parameter
	T_e: float, optional
		Electron temperature; values 0 keV < T_e < 75 keV are valid.
		The applied relativistic corrections for T_e > 0 were computed 
		with SZ-pack. Default: 0
	MJy: bool, optional
		Determines the units of the output; The default MJy=False returns 
		the tSZ spectrum in units of K_CMB, if set to True the values will be
		MJy/sr. Default: False

	Returns
	-------
	tsz: array
		tsz spectrum
	'''

	x=h*nu/k_B/T_CMB
	exp_x=np.exp(x)
	h_x=x**4*exp_x/(exp_x - 1)**2

	if T_e == 0:
		f_x = x*(exp_x + 1)/(exp_x - 1) - 4
		tsz = y*f_x*h_x*I_0

	else:
		tsz = y*tsz_interpol(nu, T_e)[:,0]

	if MJy is False:
		tsz *= T_CMB/I_0/h_x 
			
	return(tsz)


def tsz_spec_planck(nu, y, T_e=0, MJy=False):
	'''Returns the tSZ spectrum with corrections for the Planck LFI and HFI bandpasses. 

	Parameters
	----------
	nu: float or array
		Frequency in GHz; has to be a valid Planck LFI or HFI band
	y: float
		Value of the compton-y parameter
	T_e: float, optional
		Electron temperature; values 0 keV < T_e < 75 keV are valid.
		The applied relativistic corrections for T_e > 0 were computed 
		with SZ-pack. Default: 0
	MJy: bool, optional
		Determines the units of the output; The default MJy=False returns 
		the tSZ spectrum in units of K_CMB, if set to True the values will be
		MJy/sr. Default: False

	Returns
	-------
	tsz: array
		Bandpass-corrected tsz spectrum
	'''

	tsz = y*np.array([np.interp(T_e, tsz_table['T_e'], tsz_table[f]) for f in nu])
	
	if MJy is False:
		tsz /= np.array([planck_uc(f) for f in nu])

	return(tsz)


def ksz_spec(nu, v_pec, tau=None, y=None, T_e=None, MJy=False):
	'''Returns the (non-relativistic) kSZ spectrum. 

	Parameters
	----------
	nu: float or array
		Frequency in Hz
	v_pec: float
		Peculiar velocity of the cluster in km/s
	tau: float
		Optical depth of the cluster. Can be computed from the optional values
		for y and T_e assuming an isothermal cluster 
	y: float
		Compton-y parameter; will be used together with T_e to compute tau
		assuming and isothermal cluster. Default: None
	T_e: float, optional
		Electron temperature; will be used together with y to compute tau
		assuming and isothermal cluster. Default: None
	MJy: bool, optional
		Determines the units of the output; The default MJy=False returns 
		the kSZ spectrum in units of K_CMB, if set to True the values will be
		MJy/sr. Default: False

	Returns
	-------
	ksz: array
		bandpass-corrected ksz spectrum
	'''

	x = h*nu/k_B/T_CMB
	exp_x = np.exp(x)
	h_x = x**4*exp_x/(exp_x - 1)**2

	if y is not None and T_e is not None:
		tau = y*m_e*c**2 / (T_e * 1e3 * e) 

	ksz = (-1)*tau*v_pec*1000/c * T_CMB
	
	if MJy is True:
		ksz *= I_0 * h_x / T_CMB

	return(ksz)


def m500_2_r500(m, z, factor = 500):
	'''Computes the radius r_500 of a galaxy cluster from its mass M_500.
		r_500 is defined as the radius of a sphere within which the average
		density is 500 times the critical density at the cluster's redshift. 

	Parameters
	----------
	m: float
		Mass of the cluster enclosed within r_500
	z: float
		Cluster redshift
	factor: float, optional
		Overdensity factor. Default: 500

	Returns
	-------
	r_500: float
		Cluster radius r_500
	'''

	rho_c = cosmo.critical_density(z).si.value * (1e6 * pc)**3 / M_sun
	r_500 = (3/4. * m/(np.pi * factor*rho_c))**(1/3.)
	return(r_500)


def r500_2_m500(r, z, factor = 500):
	'''Computes the Mass M_500 of a galaxy cluster from its radius r_500.
	M_500 is defined as the mass of a sphere with an average density 
	of 500 times the critical density and radius r_500. 

	Parameters
	----------
	r: float
		Cluster radius r_500
	z: float
		Cluster redshift
	factor: float, optional
		Overdensity factor. Default: 500

	Returns
	-------
	M_500: float
		Cluster Mass M_500
	'''

	rho_c = cosmo.critical_density(z).si.value * (1e6 * pc)**3 / M_sun
	M_500 = factor * rho_c * 4/3. * np.pi * r**3.
	return(M_500)


def beta_model(r, n_0, r_c, beta = 1.0):
	'''Computes the radial 3D electron number density 
	profile of a galaxy cluster using a beta model. 

	Parameters
	----------
	r: float or float array
		Radial distance from the cluster center
	n_0: float
		Central electron number density
	r_c: float
		Core radius of galaxy cluster
	beta: float, optional
		Value of exponent beta. Default: 1.0

	Returns
	-------
	n_e: float or float array
		Radial 3D electron number density profile
	'''

	n_e = n_0 * (1+(r/r_c)**2)**(-3/2 * beta)
	return(n_e)


def beta_projected(r, A_0, core_radius, beta = 1, xray = False):
	'''Computes the projected radial electron number density 
	profile of a galaxy cluster using an untruncated 
	analytically projected beta model. 

	Parameters
	----------
	r: float or float array
		Projected radial distance from the cluster center
	n_0: float
		Projected central electron number density
	r_c: float
		Projected core radius of galaxy cluster
	beta: float, optional
		Value of exponent beta. Default: 1.0
	xray: bool, optional
		If set to True, projected squared electron number 
		density profile will be returned. Default: False

	Returns
	-------
	n_e: float or float array
		Projected radial 2D electron number density profile
	'''
	
	if xray is False:
		n_e = n_0 * (1+(r/r_c)**2)**((1-3*beta)/2)
	elif xray is True:
		n_e = n_0 * (1+(r/r_c)**2)**((1/2-3*beta))

	return(n_e)


def gnfw(r, z, M_500, p = 'Arnaud', alpha_p_prime = False, xx = None, yy = None):
	'''Computes the radial 3D electron pressure profile of 
	a galaxy cluster using a GNFW model (Nagai et al. 2007)

	Parameters
	----------
	r: float or float array
		Radial distance from the cluster center in meters
	z: float
		Cluster redshift
	M_500: float
		Mass of the cluster enclosed within r_500
	p: float array or string, optional
		GNFW parameter values. If the values are provided 
		as an array, the order is P_0, c_500, gamma, alpha, 
		and beta. Alternatively, p can be set to 'Arnaud' 
		or 'Planck', in which case the best-fist parameters
		from Arnaud et al. (2010) and Planck intermediate 
		results V (2013) are used. Default: 'Arnaud'
	alpha_p_prime: bool, optional
		If set to True, variation of the slope of the power law
		in the normalization of the pressure profile as defined 
		in Eq. (9) of Arnaud et al. (2010) is applied. 
		Default: False
	xx: float or float array
		x-coordinate. Used for projection only. Default: None
	yy: float or float array
		y-coordinate. Used for projection only. Default: None

	Returns
	-------
	pressure: float or float array
		Radial 3D electron pressure at radius r in units of J/m^3
	'''
	
	if r is None:
		r = np.sqrt(xx**2 + yy**2)	

	r_500 = m500_2_r500(M_500, z, factor = 500) * 1e6 * pc
	x = r/r_500
	h = H0/100
	h_z = cosmo.H(z).value/100

	if p == 'Arnaud':
		P_0, c_500, gamma, alpha, beta = 8.403 * (H0/70.)**(-3/2.), 1.177,0.3081, 1.0510, 5.4905
	elif p == 'Planck':
		P_0, c_500, gamma, alpha, beta = 6.41 * (H0/70.)**(-3/2.), 1.81,0.31, 1.33, 4.13
	else:
		P_0, c_500, gamma, alpha, beta = p[0],p[1],p[2],p[3],p[4]

	px = P_0/((c_500*x)**gamma*(1+(c_500*x)**alpha)**((beta-gamma)/alpha))

	if alpha_p_prime is True:
		alpha_p_prime = 0.1-(0.12+0.10)*((x/0.5)**3./(1+(x/0.5)**3.))
	else:
		alpha_p_prime = 0
	
	Pnorm = 1.65e-3 * (h_z/h)**(8./3.) * ((M_500) /(3e14*(h/0.7)**(-1.0)))**(2/3.+0.12+alpha_p_prime) * (h/0.7)**2.0

	pressure = Pnorm * px * 1e9 * e

	return(pressure)


def gnfw_projected(r, z, M_500, p = "Arnaud", alpha_p_prime = False, r_max = 5.0, 
		   norm_planck = False):
	'''Computes the radial Compton-y profile of a galaxy cluster 
	by numerically projecting a GNFW electron pressure model.
	(Nagai et al. 2007)

	Parameters
	----------
	r: float array
		projected radial distance from the cluster center in meters
	z: float
		Cluster redshift
	M_500: float
		Mass of the cluster enclosed within r_500
	p: float array or string, optional
		GNFW parameter values. If the values are provided 
		as an array, the order is P_0, c_500, gamma, alpha, 
		and beta. Alternatively, p can be set to 'Arnaud' 
		or 'Planck', in which case the best-fist parameters
		from Arnaud et al. (2010) and Planck intermediate 
		results V (2013) are used. Default: 'Arnaud'
	alpha_p_prime: bool, optional
		If set to True, variation of the slope of the power law
		in the normalization of the pressure profile as defined 
		in Eq. (9) of Arnaud et al. (2010) is applied. 
		Default: False		
	r_max: float, optional
		Cluster radius in units of r_500. Determines the line 
		of sight extent used during the projection. Default: 5.0
	norm_planck: bool, optional
		If set to True, the Planck Y_500 M_500 relation is
		used to re-normalize the cluster pressure profile.
		Default: False 

	Returns
	-------
	compton: float array
		Unitless radial Compton-y profile.
	'''

	r_500 = m500_2_r500(M_500, z, factor = 500) * 1e6 * pc
	compton = []
	for x in r:
		if x <= r_max*r_500:
			integral = quad(lambda y: gnfw(None, z, M_500, p, alpha_p_prime = alpha_p_prime, x, y), 0, np.sqrt((r_max*r_500)**2-x**2))
			compton.append(thomson/m_e/c**2 * 2*integral[0])
		else:
			compton.append(0)

	compton = np.array(compton)

	if norm_planck is True:
		compton *= Y_500_planck(M_500, z) / Y_500_sph(M_500, z, p=p)

	return(compton)
	
	
def gnfw_projected_fast(r, z, M_500, p = "Arnaud", alpha_p_prime = False, r_max = 5.0, 
			r_min = 1e-3, bins = 1000, norm_planck = False):
	'''Computes the radial Compton-y profile of a galaxy cluster 
	by numerically projecting a GNFW electron pressure model
	(Nagai et al. 2007). This function uses faster tabulated 
	integration than the otherwise identical fuction 
	gnfw_projected().

	Parameters
	----------
	r: float array
		projected radial distance from the cluster center in meters
	z: float
		Cluster redshift
	M_500: float
		Mass of the cluster enclosed within r_500
	p: float array or string, optional
		GNFW parameter values. If the values are provided 
		as an array, the order is P_0, c_500, gamma, alpha, 
		and beta. Alternatively, p can be set to 'Arnaud' 
		or 'Planck', in which case the best-fist parameters
		from Arnaud et al. (2010) and Planck intermediate 
		results V (2013) are used. Default: 'Arnaud'
	alpha_p_prime: bool, optional
		If set to True, variation of the slope of the power law
		in the normalization of the pressure profile as defined 
		in Eq. (9) of Arnaud et al. (2010) is applied. 
		Default: False		
	r_max: float, optional
		Cluster radius in units of r_500. Determines the line 
		of sight extent used during the projection. Default: 5.0
	r_min: float, optional
		Lower bound of the cluster radius in units of r_500. 
		A lower bound for the projection is necessary due to 
		the central cusp of the GNFW model. Default: 1e-3
	bins: float, optional
		Number of bins used for the projection along the line 
		of sight. Default: 1000
	norm_planck: bool, optional
		If set to True, the Planck Y_500 M_500 relation is
		used to re-normalize the cluster pressure profile.
		Default: False 

	Returns
	-------
	compton: float array
		Unitless radial compton-y profile.
	'''

	r_500 = m500_2_r500(M_500, z, factor = 500) * 1e6 * pc
	y = r / r_500

	compton = []

	for yy in y:
		if yy <= r_max:
			if yy < r_min: 
				x_min = r_min 
			else: 
				x_min = 0 
			x = np.linspace(x_min,np.sqrt(r_max**2-yy**2),bins)
			r = np.sqrt(yy**2. + x**2.) * r_500
			integrant = gnfw(r, z, M_500, p, alpha_p_prime = alpha_p_prime)
			result = 2*thomson/m_e/c**2 * simps(integrant, x*r_500)
		else:
			result = 0

		compton.append(result)

	compton = np.array(compton)

	if norm_planck is True:
		compton *= Y_500_planck(M_500, z) / Y_500_sph(M_500, z, p=p)

	return(compton)


def gnfw_abel(r, z, M_500, p = "Arnaud", alpha_p_prime = False, r_max = 5.0, 
	      norm_planck = False):
	'''Computes the radial Compton-y profile of a galaxy 
	cluster by numerically projecting a GNFW electron 
	pressure model (Nagai et al. 2007).

	Parameters
	----------
	r: float array
		Projected radial distance from the cluster center in meters
	z: float
		Cluster redshift
	M_500: float
		Mass of the cluster enclosed within r_500
	p: float array or string, optional
		GNFW parameter values. If the values are provided 
		as an array, the order is P_0, c_500, gamma, alpha, 
		and beta. Alternatively, p can be set to 'Arnaud' 
		or 'Planck', in which case the best-fist parameters
		from Arnaud et al. (2010) and Planck intermediate 
		results V (2013) are used. Default: 'Arnaud'
	alpha_p_prime: bool, optional
		If set to True, variation of the slope of the power law
		in the normalization of the pressure profile as defined 
		in Eq. (9) of Arnaud et al. (2010) is applied. 
		Default: False		
	r_max: float, optional
		Cluster radius in units of r_500. Determines the line 
		of sight extent used during the projection. Default: 5.0
	norm_planck: bool, optional
		If set to True, the Planck Y_500 M_500 relation is
		used to re-normalize the cluster pressure profile.
		Default: False 

	Returns
	-------
	compton: float array
		Unitless radial compton-y profile.
	'''

	r_500 = m500_2_r500(M_500, z, factor = 500) * 1e6 * pc
	compton = []
	for x in r:
		if x <= r_max*r_500:
		    integral = quad(lambda rr: gnfw(rr, z, M_500, p, alpha_p_prime = alpha_p_prime)*rr / np.sqrt(rr**2 - x**2), x, r_max*r_500)
		    compton.append(thomson/m_e/c**2 * 2*integral[0])
		else:
		    compton.append(0)

	compton = np.array(compton)

	if norm_planck is True:
		compton *= Y_500_planck(M_500, z) / Y_500_sph(M_500, z, p=p)

	return(compton)


def simulate_cluster(M_500, z, p = "Arnaud", alpha_p_prime = False, map_size = 10, 
		     pixel_size = 1.5, dx = 0, dy = 0, interpol = 1000, fwhm = None, 
		     r_max = 5.0, r_min = 1e-3, bins = 1000, norm_planck = False):
	'''Computes a Compton-y map of a galaxy cluster at with mass
	M_500 at redshift z by numerically projecting a GNFW 
	electron pressure model.

	Parameters
	----------
	M_500: float
		Mass of the cluster enclosed within r_500
	z: float
		Cluster redshift
	p: float array or string, optional
		GNFW parameter values. If the values are provided 
		as an array, the order is P_0, c_500, gamma, alpha, 
		and beta. Alternatively, p can be set to 'Arnaud' 
		or 'Planck', in which case the best-fist parameters
		from Arnaud et al. (2010) and Planck intermediate 
		results V (2013) are used. Default: 'Arnaud'
	alpha_p_prime: bool, optional
		If set to True, variation of the slope of the power law
		in the normalization of the pressure profile as defined 
		in Eq. (9) of Arnaud et al. (2010) is applied. 
		Default: False		
	map_size: int, optional
		Size of the map in degrees. Default: 10
	pixel_size: float, optional
		Pixel size in arcmin. Default: 1.5
	dx: float, optional
		Offset of cluster center from image center along 
		x-axis in pixels. Default: 0
	dy: float, optional
		Offset of cluster center from image center along 
		y-axis in pixels. Default: 0
	interpol: int, optional
		Number of bins used for the computation of the initial 
		y-profile. The pixel values of the map are then obtained 
		through interpolation. Default: 1000
	fwhm: float, optional
		FWHM of Gaussian kernel in arcmin with which the 
		map will be convolved. Default: None
	r_max: float, optional
		Cluster radius in units of r_500. Determines the line 
		of sight extent used during the projection. Default: 5.0
	r_min: float, optional
		Lower bound of the cluster radius in units of r_500. 
		A lower bound for the projection is necessary due to 
		the central cusp of the GNFW model. Default: 1e-3
	bins: float, optional
		Number of bins used for the projection along the line 
		of sight. Default: 1000
	norm_planck: bool, optional
		If set to True, the Planck Y_500 M_500 relation is
		used to re-normalize the cluster pressure profile.
		Default: False 

	Returns
	-------
	cluster_map: 2D float array
		Unitless compton-y map of cluster.
	'''

	npix = int(map_size*60 / pixel_size)

	pixel_size_meters = pixel_size/60*np.pi/180 * cosmo.angular_diameter_distance(z).si.value

	YY, XX = np.indices((npix, npix))
	center = (npix//2+dx,npix//2+dy)
	r = np.sqrt((XX-center[0])**2 + (YY-center[1])**2)*pixel_size_meters

	r = r.reshape(npix*npix)

	if interpol is not None:
		r_interpol = np.linspace(np.min(r), np.max(r), interpol)
		y_interpol = gnfw_projected_fast(r_interpol, z, M_500, p, alpha_p_prime = alpha_p_prime, r_max, r_min, bins, norm_planck = norm_planck)
		cluster_map = np.interp(r, r_interpol, y_interpol)
	else:
		cluster_map = gnfw_projected_fast(r, z, M_500, p, alpha_p_prime = alpha_p_prime, r_max, r_min, bins, norm_planck = norm_planck)	

	cluster_map = cluster_map.reshape(npix,npix)

	if fwhm is not None:
		sigma = fwhm / (2*np.sqrt(2*np.log(2)))
		cluster_map = gaussian_filter(cluster_map, sigma=sigma/pixel_size, order=0, mode='wrap', truncate=20.0)

	return(cluster_map)


def simulate_cluster_beta(y_0, r_c, beta = 1, map_size = 10, pixel_size = 1.5, dx = 0, dy = 0, fwhm = None):
	'''Computes a Compton-y map of a galaxy cluster at with mass
	M_500 at redshift z by numerically projecting a beta-model 
	electron number density model. This is only valid is the 
	cluster is assumed to be isothermal.

	Parameters
	----------
	y_0: float
		Central Compton-y value
	r_c: float
		core radius of galaxy cluster in arcmin
	beta: float, optional
		value of exponent beta. Default: 1.0
	map_size: int, optional
		Size of the map in degrees. Default: 10
	pixel_size: float, optional
		Pixel size in arcmin. Default: 1.5
	dx: float, optional
		Offset of cluster center from image center along 
		x-axis in pixels. Default: 0
	dy: float, optional
		Offset of cluster center from image center along 
		y-axis in pixels. Default: 0
	fwhm: float, optional
		FWHM of Gaussian kernel in arcmin with which the 
		map will be convolved. Default: None

	Returns
	-------
	cluster_map: 2D float array
		Unitless compton-y map of cluster.
	'''

	npix = int(map_size*60 / pixel_size)

	YY, XX = np.indices((npix, npix))
	center = (npix//2+dx,npix//2+dy)
	r = np.sqrt((XX-center[0])**2 + (YY-center[1])**2)*pixel_size

	cluster_map	= beta_projected(r, y_0, r_c, beta)

	if fwhm is not None:
		sigma = fwhm / (2*np.sqrt(2*np.log(2)))
		cluster_map = gaussian_filter(cluster_map, sigma=sigma/pixel_size, order=0, mode='wrap', truncate=20.0)

	return(cluster_map)


def deproject(yy, func, compton=True, bins = 10**6):
	'''Deprojects a projected radial profile by inverting 
	the Abel intergral and assuming spherical symmetry.

	Parameters
	----------
	yy: float array
		radial distance from cluster center
	func: float array
		spherically projected profile to be de-projected
	compton: bool, optional
		If set to True, unit conversion factors for conversion 
		from Compton-y to electron pressure will be applied.
		The deprojected electron pressure will be given in J/m^3
	bins: int, optional
		Number of bins used for interpolation. Interpolating 
		points between the given profile data points will 
		result in a more accurate deprojection. Default: 10**6

	Returns
	-------
	yy: float array
		3D radial distance from cluster center
	result: float array
		Deprojected radial 3D profile
	'''

	result = np.zeros(len(yy)-1)
	delta = (yy[1]-yy[0])/bins
	F = make_interp_spline(yy, func, k=1)
	for i in np.arange(len(yy)-1):
		r = yy[i]
		y = np.linspace(yy[i]+delta, yy.max(), bins)
		dF = derivative(F, y, dx=delta)
		integrant = dF / np.sqrt(y**2-r**2)
		result[i] = -1*simps(integrant, y)/np.pi  
		
	if compton is True:
		result = result/thomson*m_e*c**2
		
	return(yy[:-1], result)	


def Y_500_sph(M_500, z, p="Arnaud", alpha_p_prime = False, r_max = 1.0, 
	      r_min = 0.0, arcmin = False):
	'''Computes the Comptonizaton parameter integrated in a sphere
	of radius r_max using a GNFW model.

	Parameters
	----------
	M_500: float
		Mass of the cluster enclosed within r_500
	z: float
		Cluster redshift
	p: float array or string, optional
		GNFW parameter values. If the values are provided 
		as an array, the order is P_0, c_500, gamma, alpha, 
		and beta. Alternatively, p can be set to 'Arnaud' 
		or 'Planck', in which case the best-fist parameters
		from Arnaud et al. (2010) and Planck intermediate 
		results V (2013) are used. Default: 'Arnaud'
	alpha_p_prime: bool, optional
		If set to True, variation of the slope of the power law
		in the normalization of the pressure profile as defined 
		in Eq. (9) of Arnaud et al. (2010) is applied. 
		Default: False		
	r_max: float, optional
		Upper bound for the cluster radius in units of r_500. 
		Default: 1.0
	r_min: float, optional
		Lower bound for the cluster radius in units of r_500.
		Default: 0.0
	arcmin: bool, optional
		If set to True, the angular diameter distance to the
		cluster is used to convert the value of Y_500 to units
		of 1/arcmin^2. Default: False

	Returns
	-------
	Y_500: float
		Comptonizaton parameter integrated in a sphere of 
		radius r_max in units of 1/Mpc^2

	'''

	r_500 = m500_2_r500(M_500, z, factor = 500) * 1e6 * pc
	integral = 4*np.pi*quad(lambda r: (gnfw(r, z, M_500, p, alpha_p_prime = alpha_p_prime)*r*r), r_min*r_500, r_max*r_500)[0]

	if arcmin is True:
		DA = cosmo.angular_diameter_distance(z).si.value
		unit = (np.radians(1/60.) * DA)**2
	else:
		unit = (1e6*pc)**2.

	Y_500 = thomson/m_e/c**2. * integral / unit

	return(Y_500)


def Y_500_cyl(M_500, z, R, p="Arnaud", alpha_p_prime = False, r_max = 5.0, 
	      r_min = 0.0, arcmin = False):
	'''Computes the Comptonizaton parameter integrated in a 
	cylindrical aperture of radius r_max using a GNFW model.

	Parameters
	----------
	M_500: float
		Mass of the cluster enclosed within r_500
	z: float
		Cluster redshift
	R: float
		Radius of the cylindrical cluster aperture in units 
		of r_500.
	p: float array or string, optional
		GNFW parameter values. If the values are provided 
		as an array, the order is P_0, c_500, gamma, alpha, 
		and beta. Alternatively, p can be set to 'Arnaud' 
		or 'Planck', in which case the best-fist parameters
		from Arnaud et al. (2010) and Planck intermediate 
		results V (2013) are used. Default: 'Arnaud'
	alpha_p_prime: bool, optional
		If set to True, variation of the slope of the power law
		in the normalization of the pressure profile as defined 
		in Eq. (9) of Arnaud et al. (2010) is applied. 
		Default: False		
	r_max: float, optional
		Cluster radius in units of r_500. Determines the line 
		of sight extent used during the integration. Default: 5.0
	r_min: float, optional
		Lower bound for the cluster radius in units of r_500.
		Default: 0.0
	arcmin: bool, optional
		If set to True, the angular diameter distance to the
		cluster is used to convert the value of Y_500 to units
		of 1/arcmin^2. Default: False

	Returns
	-------
	Y_500: float
		Comptonizaton parameter integrated in a cylindrical 
		aperture of radius r_max in units of 1/Mpc^2

	'''

	Y_500_spherical = Y_500_sph(M_500, z, p, r_max, r_min, arcmin = arcmin)

	r_500 = m500_2_r500(M_500, z, factor = 500) * 1e6 * pc

	integral = 4*np.pi*quad(lambda r: (gnfw(r, z, M_500, p, alpha_p_prime = alpha_p_prime)*r*np.sqrt(r**2-(R*r_500)**2)), R*r_500, r_max*r_500)[0]

	if arcmin is True:
		DA = cosmo.angular_diameter_distance(z).si.value
		unit = (np.radians(1/60.) * DA)**2
	else:
		unit = (1e6*pc)**2.

	Y_500 = Y_500_spherical - thomson/m_e/c**2. * integral / unit

	return(Y_500)


def T_e_profile(r, z, M_500, xx = None, yy = None, cool_core = True):
	'''Computes the radial 3D electron temperatue profile of a galaxy 
	cluster using the model presented by Vikhlinin et al. (2006). 
	The required value of the X-ray spectroscopic temperature T_x 
	is computed from the cluster mass M_500 using the M-T scaling 
	relation given by Reichert et al. (2011).

	Parameters
	----------
	r: float or float array
		Radial distance from the cluster center in meters
	z: float
		Cluster redshift
	M_500: float
		Mass of the cluster enclosed within r_500
	xx: float or float array
		x-coordinate. Used for projection only. Default: None
	yy: float or float array
		y-coordinate. Used for projection only. Default: None
	cool_core: bool, optional
		If set to True, the electron temperature profile will 
		featrue the usual cool core of the Vikhlinin et al. (2006)
		model. If set to False the cool core will be removed by 
		choosing an infinitesimally small cooling radius.

	Returns
	-------
	T: float or float array
		Radial 3D electron temperatue profile of a galaxy 
		cluster in units of keV.

	'''

	if r is None:
		r = np.sqrt(xx**2 + yy**2)	

	r_500 = m500_2_r500(M_500, z, factor = 500) * 1e6 * pc

	if cool_core is True:
		r_cool = 0.045*r_500
	else:
		r_cool = 1e-8*r_500

	E = cosmo.H(z).value/cosmo.H0.value
	T_x = ((M_500/1e14)/0.291 * (1/(E**(-1.04))))**(1/1.62)
	T = 1.35 * T_x * ((r/r_cool)**1.9 + 0.45) / ((r/r_cool)**1.9 + 1) / (1+ (r/(0.6*r_500))**2.)**0.45

	return(T)


def T_sz(r, z, M_500, p = "Arnaud", alpha_p_prime = False, r_max = 5, 
	 cool_core = True, norm_planck = False):
	'''Computes the radial 2D pressure-weighted electron temperatue 
	profile of a galaxy cluster using the 3D tempeature modele 
	presented by Vikhlinin et al. (2006) and the 3D pressure 
	model given by Arnaud et al. (2010). The required value of 
	the X-ray spectroscopic temperature T_x is computed from the 
	cluster mass M_500 using the M-T scaling relation given by 
	Reichert et al. (2011). The pressure-weighted electron 
	temperatue is a good approximation of the temperature that 
	is measured though the relativistic thermal Sunyaev-Zel'dovich 
	effect.

	Parameters
	----------
	r: float or float array
		Radial distance from the cluster center in meters
	z: float
		Cluster redshift
	M_500: float
		Mass of the cluster enclosed within r_500
	p: float array or string, optional
		GNFW parameter values. If the values are provided 
		as an array, the order is P_0, c_500, gamma, alpha, 
		and beta. Alternatively, p can be set to 'Arnaud' 
		or 'Planck', in which case the best-fist parameters
		from Arnaud et al. (2010) and Planck intermediate 
		results V (2013) are used. Default: 'Arnaud'
	alpha_p_prime: bool, optional
		If set to True, variation of the slope of the power law
		in the normalization of the pressure profile as defined 
		in Eq. (9) of Arnaud et al. (2010) is applied. 
		Default: False		
	r_max: float, optional
		Cluster radius in units of r_500. Determines the line 
		of sight extent used during the projection. Default: 5.0
	cool_core: bool, optional
		If set to True, the electron temperature profile will 
		featrue the usual cool core of the Vikhlinin et al. (2006)
		model. If set to False the cool core will be removed by 
		choosing an infinitesimally small cooling radius.
	norm_planck: bool, optional
		If set to True, the Planck Y_500 M_500 relation is
		used to re-normalize the cluster pressure profile.
		Default: False 

	Returns
	-------
	T: float or float array
		2D radial pressure-weighted electron temperatue profile of 
		a galaxy cluster in units of keV.

	'''

	r_500 = m500_2_r500(M_500, z, factor = 500) * 1e6 * pc
	temp = []
	for x in r:
		if x <= r_max*r_500:
			nom = quad(lambda y: gnfw(None, z, M_500, p, alpha_p_prime = alpha_p_prime, x, y)*T_e_profile(None, z, M_500, x, y, cool_core = cool_core), 0, np.sqrt((r_max*r_500)**2-x**2))[0]
			denom = quad(lambda y: gnfw(None, z, M_500, p, alpha_p_prime = alpha_p_prime, x, y), 0, np.sqrt((r_max*r_500)**2-x**2))[0]
			temp.append(nom/denom)
		else:
			temp.append(0)
	return(np.array(temp))



def T_sz_fast(r, z, M_500, p = "Arnaud", alpha_p_prime = False, r_max = 5, r_min = 1e-3, 
	      bins = 1000, cool_core = True, norm_planck = False):
	'''Computes the radial 2D pressure-weighted electron temperatue 
	profile of a galaxy cluster using the 3D tempeature modele 
	presented by Vikhlinin et al. (2006) and the 3D pressure 
	model given by Arnaud et al. (2010). The required value of 
	the X-ray spectroscopic temperature T_x is computed from the 
	cluster mass M_500 using the M-T scaling relation given by 
	Reichert et al. (2011). The pressure-weighted electron 
	temperatue is a good approximation of the temperature that 
	is measured though the relativistic thermal Sunyaev-Zel'dovich 
	effect. This code uses faster tabulated integration then the 
	otherwise identical function T_sz()

	Parameters
	----------
	r: float or float array
		Radial distance from the cluster center in meters
	z: float
		Cluster redshift
	M_500: float
		Mass of the cluster enclosed within r_500
	p: float array or string, optional
		GNFW parameter values. If the values are provided 
		as an array, the order is P_0, c_500, gamma, alpha, 
		and beta. Alternatively, p can be set to 'Arnaud' 
		or 'Planck', in which case the best-fist parameters
		from Arnaud et al. (2010) and Planck intermediate 
		results V (2013) are used. Default: 'Arnaud'
	alpha_p_prime: bool, optional
		If set to True, variation of the slope of the power law
		in the normalization of the pressure profile as defined 
		in Eq. (9) of Arnaud et al. (2010) is applied. 
		Default: False		
	r_max: float, optional
		Cluster radius in units of r_500. Determines the line 
		of sight extent used during the projection. Default: 5.0
	r_min: float, optional
		Lower bound of the cluster radius in units of r_500. 
		A lower bound for the projection is necessary due to 
		the central cusp of the GNFW model. Default: 1e-3
	bins: float, optional
		Number of bins used for the projection along the line 
		of sight. Default: 1000
	cool_core: bool, optional
		If set to True, the electron temperature profile will 
		featrue the usual cool core of the Vikhlinin et al. (2006)
		model. If set to False the cool core will be removed by 
		choosing an infinitesimally small cooling radius.
	norm_planck: bool, optional
		If set to True, the Planck Y_500 M_500 relation is
		used to re-normalize the cluster pressure profile.
		Default: False 

	Returns
	-------
	T: float or float array
		2D radial pressure-weighted electron temperatue profile of 
		a galaxy cluster in units of keV.

	'''

	r_500 = m500_2_r500(M_500, z, factor = 500) * 1e6 * pc
	y = r / r_500

	temp = []

	for yy in y:
		if yy <= r_max:
			if yy < r_min: 
				x_min = r_min 
			else: 
				x_min = 0 
			x = np.linspace(x_min,np.sqrt(r_max**2-yy**2),bins)
			r = np.sqrt(yy**2. + x**2.) * r_500
			integrant = gnfw(r, z, M_500, p, alpha_p_prime = alpha_p_prime)*T_e_profile(r, z, M_500, cool_core = cool_core)
			norm = gnfw(r, z, M_500, p, alpha_p_prime = alpha_p_prime)
			temp.append(simps(integrant, x*r_500) / simps(norm, x*r_500))
		else:
			temp.append(0)

	return(np.array(temp))


def tau_fast(r, z, M_500, p = "Arnaud", alpha_p_prime = False, r_max = 5, r_min = 1e-3, 
	     bins = 1000, cool_core = True, norm_planck = False):
	'''Computes the radial 2D integrated profile of the optical depth 
	of a galaxy cluster using the 3D pressure model given by 
	Arnaud et al. (2010) and the 3D tempeature modele presented by 
	Vikhlinin et al. (2006). The required value of the X-ray 
	spectroscopic temperature T_x is computed from the cluster 
	mass M_500 using the M-T scaling relation given by Reichert 
	et al. (2011).

	Parameters
	----------
	r: float or float array
		Radial distance from the cluster center in meters
	z: float
		Cluster redshift
	M_500: float
		Mass of the cluster enclosed within r_500
	p: float array or string, optional
		GNFW parameter values. If the values are provided 
		as an array, the order is P_0, c_500, gamma, alpha, 
		and beta. Alternatively, p can be set to 'Arnaud' 
		or 'Planck', in which case the best-fist parameters
		from Arnaud et al. (2010) and Planck intermediate 
		results V (2013) are used. Default: 'Arnaud'
	alpha_p_prime: bool, optional
		If set to True, variation of the slope of the power law
		in the normalization of the pressure profile as defined 
		in Eq. (9) of Arnaud et al. (2010) is applied. 
		Default: False		
	r_max: float, optional
		Cluster radius in units of r_500. Determines the line 
		of sight extent used during the projection. Default: 5.0
	r_min: float, optional
		Lower bound of the cluster radius in units of r_500. 
		A lower bound for the projection is necessary due to 
		the central cusp of the GNFW model. Default: 1e-3
	bins: float, optional
		Number of bins used for the projection along the line 
		of sight. Default: 1000
	cool_core: bool, optional
		If set to True, the electron temperature profile will 
		featrue the usual cool core of the Vikhlinin et al. (2006)
		model. If set to False the cool core will be removed by 
		choosing an infinitesimally small cooling radius.
	norm_planck: bool, optional
		If set to True, the Planck Y_500 M_500 relation is
		used to re-normalize the cluster pressure profile.
		Default: False 

	Returns
	-------
	tau: float or float array
		2D radial optical depth profile of a galaxy cluster.

	'''

	r_500 = m500_2_r500(M_500, z, factor = 500) * 1e6 * pc
	y = r / r_500

	tau = []

	for yy in y:
		if yy <= r_max:
			if yy < r_min: 
				x_min = r_min 
			else: 
				x_min = 0 
			x = np.linspace(x_min,np.sqrt(r_max**2-yy**2),bins)
			r = np.sqrt(yy**2. + x**2.) * r_500
			integrant = gnfw(r, z, M_500, p, alpha_p_prime = alpha_p_prime)/(T_e_profile(r, z, M_500, cool_core = cool_core)*1000*e)
			tau.append(2*thomson*simps(integrant, x*r_500))
		else:
			tau.append(0)

	return(np.array(tau))


def simulate_rel_cluster(M_500, z, p = "Arnaud", alpha_p_prime = False, map_size = 10, pixel_size = 1.5, 
			 dx = 0, dy = 0, interpol = 1000, fwhm = None, r_max = 5, r_min = 1e-3, bins = 1000, 
			 cool_core = True, norm_planck = False):
	'''Computes Compton-y, optical depth and pressure-weighted 
	temperature maps of a galaxy cluster at with mass
	M_500 at redshift z by numerically projecting and weighting
	the pressure model given by Arnaud et al. (2010) and the 
	electron temperature model presented by Vikhlinin et al. 
	(2006). The required value of the X-ray spectroscopic 
	temperature T_x is computed from the cluster mass M_500 
	using the M-T scaling relation given by Reichert et al. 
	(2011). 

	Parameters
	----------
	M_500: float
		Mass of the cluster enclosed within r_500
	z: float
		Cluster redshift
	p: float array or string, optional
		GNFW parameter values. If the values are provided 
		as an array, the order is P_0, c_500, gamma, alpha, 
		and beta. Alternatively, p can be set to 'Arnaud' 
		or 'Planck', in which case the best-fist parameters
		from Arnaud et al. (2010) and Planck intermediate 
		results V (2013) are used. Default: 'Arnaud'
	alpha_p_prime: bool, optional
		If set to True, variation of the slope of the power law
		in the normalization of the pressure profile as defined 
		in Eq. (9) of Arnaud et al. (2010) is applied. 
		Default: False		
	map_size: int, optional
		Size of the map in degrees. Default: 10
	pixel_size: float, optional
		Pixel size in arcmin. Default: 1.5
	dx: float, optional
		Offset of cluster center from image center along 
		x-axis in pixels. Default: 0
	dy: float, optional
		Offset of cluster center from image center along 
		y-axis in pixels. Default: 0
	interpol: int, optional
		Number of bins used for the computation of the initial 
		y-profile. The pixel values of the map are then obtained 
		through interpolation. Default: 1000
	fwhm: float, optional
		FWHM of Gaussian kernel in arcmin with which the 
		map will be convolved. Default: None
	r_max: float, optional
		Cluster radius in units of r_500. Determines the line 
		of sight extent used during the projection. Default: 5.0
	r_min: float, optional
		Lower bound of the cluster radius in units of r_500. 
		A lower bound for the projection is necessary due to 
		the central cusp of the GNFW model. Default: 1e-3
	bins: float, optional
		Number of bins used for the projection along the line 
		of sight. Default: 1000
	cool_core: bool, optional
		If set to True, the electron temperature profile will 
		featrue the usual cool core of the Vikhlinin et al. (2006)
		model. If set to False the cool core will be removed by 
		choosing an infinitesimally small cooling radius.
	norm_planck: bool, optional
		If set to True, the Planck Y_500 M_500 relation is
		used to re-normalize the cluster pressure profile.
		Default: False 

	Returns
	-------
	cluster_maps: float array
		3D array containing the simulated y-map, tau-map and T_SZ-map.
	'''

	npix = int(map_size*60 / pixel_size)

	pixel_size_meters = pixel_size/60*np.pi/180 * cosmo.angular_diameter_distance(z).si.value

	YY, XX = np.indices((npix, npix))
	center = (npix//2+dx,npix//2+dy)
	r = np.sqrt((XX-center[0])**2 + (YY-center[1])**2)*pixel_size_meters

	r = r.reshape(npix*npix)

	r_interpol = np.linspace(np.min(r), np.max(r), interpol)

	y_interpol = gnfw_projected_fast(r_interpol, z, M_500, p, alpha_p_prime = alpha_p_prime, r_max, r_min, bins, norm_planck=norm_planck)
	T_interpol = T_sz_fast(r_interpol, z, M_500, p, alpha_p_prime = alpha_p_prime, r_max, r_min, bins, cool_core = cool_core)
	tau_interpol = tau_fast(r_interpol, z, M_500, p, alpha_p_prime = alpha_p_prime, r_max, r_min, bins, cool_core = cool_core)
	
	y_map = np.interp(r, r_interpol, y_interpol).reshape(npix,npix)
	T_map = np.interp(r, r_interpol, T_interpol).reshape(npix,npix)
	tau_map = np.interp(r, r_interpol, tau_interpol).reshape(npix,npix)

	if fwhm is not None:
		sigma = fwhm / (2*np.sqrt(2*np.log(2)))
		T_map = gaussian_filter(tau_map*y_map, sigma=sigma/pixel_size, order=0, mode='wrap', truncate=10.0)
		y_map = gaussian_filter(y_map, sigma=sigma/pixel_size, order=0, mode='wrap', truncate=10.0)
		tau_map = gaussian_filter(tau_map, sigma=sigma/pixel_size, order=0, mode='wrap', truncate=10.0)
		index = y_map > 0
		T_map[index] = T_map[index]/y_map[index]

	cluster_maps =  np.array([y_map, tau_map, T_map])

	return(cluster_maps)


def M_500_planck(Y_500, z, alpha=1.79, beta=0.66, Y_star=-0.19, b=0.2):
	'''Computes the mass of a galaxy cluster at redshift z from its
	value using the 2015 Planck Y_500-M_500-scaling relation. 

	Parameters
	----------
	Y_500: float
		Cluster Y_500
	z: float
		Cluster redshift
	alpha: float, optional
		Scaling relation parameter. Default: 1.79
	beta: float, optional
		Scaling relation parameter. Default: 0.66
	Y_star: float, optional
		Scaling relation parameter. Default: -0.19
	b: float, optional
		Hydrostatic mass bias 1-b. Default: 0.2

	Returns
	-------
	M_500: float
		Cluster mass
	'''

	h_z = cosmo.H(z).value/70

	M_500 = (h_z**(-beta) * (Y_500 / 10**-4) / (10**Y_star * (0.7/0.7)**(-2+alpha)))**(1/alpha) / (1-b) * 6e14

	return(M_500)


def Y_500_planck(M_500, z, alpha=1.79, beta=0.66, Y_star=-0.19, b=0.2):
	'''Computes the Y_500 of a galaxy cluster at redshift z from its
	value using the 2015 Planck Y_500-M_500-scaling relation. 

	Parameters
	----------
	M_500: float
		Cluster mass
	z: float
		Cluster redshift
	alpha: float, optional
		Scaling relation parameter. Default: 1.79
	beta: float, optional
		Scaling relation parameter. Default: 0.66
	Y_star: float, optional
		Scaling relation parameter. Default: -0.19
	b: float, optional
		Hydrostatic mass bias 1-b. Default: 0.2

	Returns
	-------
	Y_500: float
		Integrated Compton parameter of the cluster
	'''

	h_z = cosmo.H(z).value/70
	
	Y_500 = 10**Y_star * (0.7/0.7)**(-2+alpha) * ((1-b)*M_500/6e14)**alpha * h_z**(beta) * 1e-4

	return(Y_500)


def r200r500(c, delta = 500):
	'''Computes the ratio of r_200 over another cluster fiducial radius
	(e.g. r_500) assuming an NFW profile withr a given concentration 
	parameter c.

	Parameters
	----------
	c: float
		Concentration parameter
	delta: float
		Average overdensity within a sphere with the radius of interest.
		Default: 500

	Returns
	-------
	ratio: float
		Ratio r_200 / r_delta
	'''    

	rho_0 = 200/3.*c**3./(np.log(1+c)-c/(1+c))
	r_delta = bisect(lambda r_x: rho_0*(np.log((1+r_x))-r_x/(1+r_x))/r_x**3 - (delta/3.), 0.1, 100)
	ratio = c/r_delta

	return(ratio)


def m200m500(c, delta = 500):
	'''Computes the ratio of M_200 over another cluster fiducial mass
	(e.g. M_500) assuming an NFW profile withr a given concentration 
	parameter c.

	Parameters
	----------
	c: float
		Concentration parameter
	delta: float
		Average overdensity within a sphere with the radius of interest.
		Default: 500

	Returns
	-------
	ratio: float
		Ratio M_200 / M_delta
	'''

	r_200 = r200r500(c, delta = delta)
	ratio = (200*r_200**3)/(delta)

	return(ratio)
