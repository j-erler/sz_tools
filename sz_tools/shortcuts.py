import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from scipy.optimize import curve_fit
from astropy.constants import c, h, k_B
from scipy.integrate import simps
from astropy.coordinates import SkyCoord

cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.7255)
T_CMB = cosmo.Tcmb0.si.value
c = c.si.value
h = h.si.value
k_B = k_B.si.value


def make_table(data, names = None):
	'''Creates a table from a multi-dimensional array.

	Parameters
	----------
	data: float array
		Data array
	names: string array, optional
		Optional column names. Default: None

	Returns
	-------
	table object.
	'''

	return(Table(data, names = names))
	

def write_file(file_name, data, names = None, overwrite=True):
	'''Writes a multi-dimensional array or table object to an ascii file
	with fixed-width columns and no delimiter.
       
	Parameters
	----------
	file_name: string
		Name of the ascii file
	data: float array or table object
		Data array or table
	names: string array, optional
		Optional column names. Default: None
	overwrite: bool, optional
		If set to True, files with identical names will be overwritten.
		Default: True
        
	Returns
	-------
	None
	'''

	ascii.write(data, file_name, format='fixed_width',  delimiter_pad=' ', 
	delimiter=None, fast_writer=False, overwrite=overwrite, names=names)
	return(None)


def read_file(file_name):
	'''Reads ascii files and returns data as table object.
	 
	Parameters
	----------
	file_name: string
		Name of the ascii file
	  
	Returns
	-------
	data: table object
		Table object containing the read data
	'''

	data = ascii.read(file_name)
	return(data)


def writefits(file_name, data, header=None):
	'''Writes a 2D array (usually an image) to a fits file.
	 
	Parameters
	----------
	file_name: string
		Name of the fits file
	data: float array
		2D array to be written
	header: fits header
		Optional fits header. A minimal header is created
		automatically if none is provided. Default: None    

	Returns
	-------
	None
	'''

	hdu = fits.PrimaryHDU()
	hdu.data = np.array(data, dtype=np.float32)
	if header is not None:	
		hdu.header = header
	hdu.writeto(file_name, overwrite=True)

	return(None)


def readfits(file_name):
	'''Reads fits files and returns data array and header.
	 
	Parameters
	----------
	file_name: string
		Name of the fits file
	  
	Returns
	-------
	data: float array
		Data array extracted from the fits file
	header: fits header
		Header of the fits file
	'''

	hdulist=fits.open(file_name)
	data = hdulist[0].data
	header = hdulist[0].header
	hdulist.close()

	return(data, header)


def hubble(z):
	'''Returnes the value of the Hubble constant at redshift z for a 
	 generic Lambda CDM cossmology with H0=70, Om0=0.3, Tcmb0=2.7255.
	 
	Parameters
	----------
	z: float
		Redshift
	  
	Returns
	-------
	H_z: float
		Value of hubble constant in km/s/Mpc
	'''

	H_z = cosmo.H(z).value
	return(H_z)


def angular_dist(z):
	'''Returnes the value of the angular diameter distance at redshift z to redshift 
	zero for a generic Lambda CDM cosmology with H0=70, Om0=0.3, Tcmb0=2.7255.

	Parameters
	----------
	z: float
		Redshift

	Returns
	-------
	D_A: float
		Value of angular diameter distance in Mpc
	'''

	D_A = cosmo.angular_diameter_distance(z).value
	return(D_A)


def luminosity_dist(z):
	'''Returnes the value of the luminosity distance at redshift z to redshift 
	zero for a generic Lambda CDM cosmology with H0=70, Om0=0.3, Tcmb0=2.7255.

	Parameters
	----------
	z: float
		Redshift

	Returns
	-------
	D_L: float
		Value of luminosity distance in Mpc
	'''

	D_L = cosmo.luminosity_distance(z).value
	return(D_L)


def dist(nx, ny, center=None, pixel_size = 1.0):
	'''Returnes a map of dimensions nx * ny that gives the radial distance to a given center 
	at each pixel

	Parameters
	----------
	nx: int
		Number of pixels along the first axis
	ny: int
		Number of pixels along the second axis
	center: float tuple, optional
		2D coordinates of the center to which the distance is computed. If set to None, the 
		center of the image (nx//2, ny//2) will be used Default: None
	pixel_size: float, optional
		Size of the pixels in the map in an arbitrary unit. Default: 1.0

	Returns
	-------
	R: float array
		2D map containing the radial distance to center at each pixel
	'''

	if center is None:
		center = (nx//2, ny//2)

	YY, XX = np.indices((nx,ny))
	R = np.sqrt((XX-center[0])**2 + (YY-center[1])**2)*pixel_size

	return(R)


def gaussian(x, A, mu, sigma=None, fwhm=None):
	'''Returns values of a Gaussian with center mu, width sigma and amplitude A
	evaluated at positions x.

	Parameters
	----------
	x: float or float array
		Values at which the functional values are computed
	A: float
		Amplitude of the Gaussian
	mu: float
		Center of the Gaussian
	sigma: float, optional
		Standard deviation of the Gaussian. Default: None
	fwhm: float, optional
		FWHM of the Gaussian, overwrites sigma. Default: None 

	Returns
	-------
	gauss: float or float array
		Computed functional values
	'''

	if fwhm is None and sigma is None:
		raise Exception("Either sigma or fwhm has to be provided")
	
	if fwhm is not None:
		sigma = fwhm / 2*np.sqrt(2*np.log(2))

	gauss = A*np.exp(-0.5*((x-mu)/sigma)**2)
	return(gauss)


def create_histogram(array, bins, plot=True, fit=False, log=False, xlabel="x", norm = None):
	'''Create a histogram with equally spaced bins using a provided data array. 

	Parameters
	----------
	array: float array
		Data array.
	bins: int
		Number of bins.
	plot: bool, optional
		If set to True, a plot will be created. Default: True
	fit: bool, optional
		If set to True, the histogram will be fitted with a Gaussian and the
		best-fit parameter values are returned. Default: False
	log: bool, optional
		If set to True, the histogram will be plotted with a logarithmic y-axis.
		Default: False
	xlabel: sting, optional
		X-label to be used when creating a plot. Default: "x"
	norm: string, optional
		If set to "pdf", the area of the histogram is normalized to one.
		If set to "unity", the amplitude of the histogram is normalized to one.
		Default: None

	Returns
	-------
	out: float array
		The default output is a float array with 3 columns containing the bin 
		centers, the histogram values and the poissonian histogram errors. If
		fit is set to True, the function will instead return the best-fit 
		parameters for a Gaussian fitted to the histogram.
	'''

	hist, edges = np.histogram(array, bins=bins)
	centers = (edges[0:-1]+edges[1:]) / 2

	if fit is True:
		p_opt, cov = curve_fit(gaussian, centers, hist, p0=(np.max(hist), np.mean(array), np.std(array)), sigma = np.sqrt(hist+1))
		p_opt[2] = abs(p_opt[2])

	if norm == "pdf":
		integral = simps(hist, centers)
		hist = hist / integral
	elif norm == "unity":
		hist = hist / np.max(hist)

	if (plot is True) & (log is False):
		plt.plot(centers, hist)
		plt.xlabel(xlabel)
		plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
		plt.ylabel("samples")
	elif (plot is True) & (log is True):
		plt.semilogy(centers, hist)
		plt.ylim(1)
		plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
		plt.xlabel(xlabel)
		plt.ylabel("samples")
	if fit is True:
		if plot is True:
			plt.plot(centers, gaussian(centers, p_opt[0], p_opt[1], p_opt[2]))
		out = p_opt
	else:
		out = np.array([centers, hist, np.sqrt(hist)])

	return(out)


def convert_units(freq, values, cmb2mjy=False, mjy2cmb=False, rj2mjy=False, mjy2rj=False, cmb2rj=False, rj2cmb=False):
	'''Convert observed signal at given frequencies to different units. 

	Parameters
	----------
	freq: float or float array
		Frequency in Hz.
	values: float or float array
		Measured signal.
	cmb2mjy: bool, optional
		If True, the input is assumed to be K_CMB, the output will be MJy/sr.
		Default: False
	mjy2cmb: bool, optional
		If True, the input is assumed to be MJy/sr, the output will be K_CMB.
		Default: False
	rj2mjy: bool, optional
		If True, the input is assumed to be K_RJ, the output will be MJy/sr.
		Default: False
	mjy2rj: bool, optional
		If True, the input is assumed to be MJy/sr, the output will be K_RJ.
		Default: False
	cmb2rj: bool, optional
		If True, the input is assumed to be K_CMB, the output will be K_RJ.
		Default: False
	rj2cmb: bool, optional
		If True, the input is assumed to be K_RJ, the output will be K_CMB.
		Default: False

	Returns
	-------
	converted_signal: float or float array
		Converted signal
	'''

	x = h * freq / k_B / T_CMB
    
	if cmb2mjy is True:
		conversion = 1e20 * 2*k_B**3.*T_CMB**2. / (h * c)**2. * x**4. * np.exp(x) / (np.exp(x)-1)**2.
	elif mjy2cmb is True:
		conversion = 1/(1e20 * 2*k_B**3.*T_CMB**2. / (h * c)**2. * x**4. * np.exp(x) / (np.exp(x)-1)**2.)
	elif rj2mjy is True:
		conversion = 1e20 * 2*freq**2.*k_B/c**2.
	elif mjy2rj is True:
		conversion = 1/(1e20 * 2*freq**2.*k_B/c**2.)
	elif cmb2rj is True:
		conversion = (k_B*T_CMB/h)**2. * x**4. * np.exp(x) / (np.exp(x)-1)**2. / freq**2.
	elif rj2cmb is True:
		conversion = 1/((k_B*T_CMB/h)**2. * x**4. * np.exp(x) / (np.exp(x)-1)**2. / freq**2.)        
	else:
		print("Not sure which units are given and what should be returned.")

	converted_signal = conversion * values

	return(converted_signal)


def mbb_spec(freq, A, T, beta, z = 0, f_0 = 857e9, K_CMB=False, K_RJ=False):
	'''Computes the flux density of a modified black body in MJy/sr. 

	Parameters
	----------
	freq: float or float array
		Frequency in Hz.
	A_dust: float
		Amplitude of the modified black body.
	T: float
		Dust temperature.
	beta: float
		Spectral index.
	f_0: float
		Pivot frequency in Hz.
	K_CMB: bool, optional
		The output will be given in units of K_CMB. Default: False
	K_RJ: bool, optional
		The output will be given in units of K_RJ. Default: False
	z: float
		Source redshift. Default: 0

	Returns
	-------
	dust: float or float array
		Flux density of a modified black body.
	'''

	dust = A * (freq*(1+z)/f_0)**(3.+beta) * (np.exp(h*f_0/k_B/T)-1) / (np.exp(h*freq*(1+z)/k_B/T)-1)

	if K_CMB is True:
		dust = convert_units(freq, dust, mjy2cmb=True)
	elif K_RJ is True:
		dust = convert_units(freq, dust, mjy2rj=True)

	return(dust)


def cov2corr(cov):
	'''Converts a given covariance matrix into a correlation matrix 

	Parameters
	----------
	cov: 2D float array
		Covariance matrix

	Returns
	-------
	corr: 2D float array
		Correlation matrix
	'''

	sqrt_var = np.sqrt(np.diagonal(cov))
	x_var, y_var = np.meshgrid(sqrt_var, sqrt_var)

	corr = cov / x_var / y_var

	return(corr)


def corr2cov(corr, var):
	'''Converts a given correlation matrix into a covariance matrix 

	Parameters
	----------
	corr: 2D float array
		Correlation matrix of n components
	var: 1D float array
		Variance of the n components

	Returns
	-------
	corr: 2D float array
		Covariance matrix
	'''
    
	x_var, y_var = np.meshgrid(np.sqrt(var), np.sqrt(var))

	cov = corr * x_var * y_var
		    
	return(cov)


def compact_error(array, interval=0.682):
	'''Computes the most compact credibility interval of a given
	MCMC sampled 1D probability density function. 

	Parameters
	----------
	array: 1D float array
		MCMC sampled 1D probability density function.
	interval: float, optional
		Credibility interval. Default: 0.68

	Returns
	-------
	credibility_interval: float array
		Two-element array containing the lower and upper bound of
		the determined credibility interval.
	'''
    
	n = len(array)
	array_sorted = np.sort(array)

	n_range = int(n*interval)

	error = array_sorted[n_range] - array_sorted[0]
	i_best = 0

	for i in np.arange(1, n-n_range):
		error_new = array_sorted[n_range+i] - array_sorted[i]
		if error_new < error:
			error = error_new
			i_best = i

	lower = array_sorted[i_best]
	upper = array_sorted[i_best+n_range]

	credibility_interval = np.array([lower, upper])

	return (credibility_interval)


def quantile(array, interval=0.68):
	'''Computes the credibility interval of a given
		MCMC sampled 1D probability density function. 

	Parameters
	----------
	array: 1D float array
		MCMC sampled 1D probability density function.
	interval: float, optional
		Credibility interval. Default: 0.68

	Returns
	-------
	credibility_interval: float array
		Two-element array containing the lower and upper bound of
		the determined credibility interval.
	'''
    
	n = len(array)
	array_sorted = np.sort(array)

	quantiles = np.array([0.5, 0.5+interval/2, 0.5-interval/2])
	indices = quantiles * n

	median = array_sorted[int(indices[0])]
	upper = array_sorted[int(indices[1])]
	lower = array_sorted[int(indices[2])]

	credibility_interval = np.array([lower, upper])

	return (credibility_interval)


def find_level(array, interval = 0.68):
	'''Computes the credibility interval of a given
		2D distribution 

	Parameters
	----------
	array: 2D float array
		2D probability density function.
	interval: float, optional
		Credibility interval. Default: 0.68

	Returns
	-------
	level: float
		Value along the z-axis of the distribution that will encompasses 
		the desired interval. Can be used to draw contours.
	'''

	array_flat = array.flatten()
	total = np.sum(array_flat)

	steps = np.sort(np.unique(array_flat))
	steps = steps[::-1]

	i=1
	integral = 0

	while integral < total*interval:
		index = np.where(array_flat > steps[i])[0]
		integral = np.sum(array_flat[index])
		i += 1

	level = steps[i]
	return(level)


def angle(nx, ny, center=None, rotate = 0, degrees = True):
	'''Returnes a map of dimensions nx * ny that gives the azimuth angle 
	of each pixel relative to a given center. The azimuth angle increases 
	counter-clockwise with the origin at the 12 o'clock position.

	Parameters
	----------
	nx: int
		Number of pixels along the first axis
	ny: int
		Number of pixels along the second axis
	center: float tuple, optional
		2D coordinates of the center to which the distance is computed. If 
		set to None, the center of the image (nx//2, ny//2) will be used 
		Default: None
	rotate: float, optional
		Applies a rotation in counter-clockwise direction. Has to be provided 
		in degrees. Default: 0
	degrees: bool, optional
		If True, the azimuth angle is returned in degrees, if False in radians.
		Default: True

	Returns
	-------
	phi: float array
		2D map containing the azimuth angle relative to center at each pixel.
	'''

	if center is None:
		center = (nx//2, ny//2)

	y, x = np.indices((nx,ny))
	y = y-center[0]
	x = x-center[1]

	r = np.sqrt(x**2 + y**2)
	if r.min() == 0:
		r[center] = np.nan

	phi = np.arcsin(y/r)
	if r.min() == 0:
		phi[center] = 0

	phi[x < 0] = np.pi - phi[x < 0] 
	phi[(y < 0) & (x >= 0)] = 2*np.pi + phi[(y < 0) & (x >= 0)] 

	phi = phi - (rotate+90)*np.pi/180
	phi[phi < 0] += 2*np.pi
	if r.min() == 0:
		phi[center] = 0

	if degrees is True:
		phi *= 180/np.pi

	return(phi)


def radial_profile(image, center = None, nbins = None, r_min = 0, r_max = None, 
               cone = None, weighted = False, log = False, check = False):
	'''Computes the radial profile of a map around a given center. The profile 
	can be computed by azimuthally averaging pixels in a given number of radial
	bins or by averaging pixels with identical (integer) distance to center. 

	Parameters
	----------
	image: 2D float array
		Image used for the computation of the radial profile
	center: float tuple, optional
		2D coordinates of the center to which the distance is computed. If 
		set to None, the center of the image (nx//2, ny//2) will be used 
		Default: None
	nbins: int
		Number of radial bins. If set to None, pixels with identical (integer) 
		distance to center will be averaged instead. Default: None
	r_min: float
		Inner boundary for radial bins in units of pixel. Default: 0
	r_max: float
		Outer boundary for radial bins in units of pixel. If None, r_max will be 
		computed such that the entire image area is used. Default: None
	cone: float array
		Tuple defining the lower and upper azimuth angle that span a cone. Has
		to be given in degrees. If None, the full circle is used. Default: None
	weighted: bool
		If set to True, the x-values of the radial profile will be weighted 
		with the signal. Otherwise the bin centers are used. Default: False
	log: bool
		If set to True, logarithmically spaced bins will be used Default: False
	check: bool
		If set to True, an image with identical dimensions to the input will be
		returned that illustrates the applied radial bins. Useful for debugging.
		Default: False
	Returns
	-------
	out: float array
		2D numpy array containing the x and y values of the radial profile.
	'''

	nx, ny = image.shape[0], image.shape[1]

	if center is None:
		center = (nx//2, ny//2)
	if cone is None:
		cone = np.array([0,360])

	r = dist(nx, ny, center)
	phi = angle(nx, ny, center[::-1], rotate=cone[0])
	cone -= cone[0]
	if cone[1] < 0:
		cone[1] += 360

	if nbins is not None:
		
		profile = np.zeros((3,nbins))
		control = np.zeros((nx,ny))
		
		if r_max is None:
		    if center[0] > (nx/2):
		        xrange = nx - center[0]
		    else:
		        xrange = center[0]
		    if center[1] > (ny/2):
		        yrange = ny - center[1]
		    else:
		        yrange = center[1]
		    r_max = np.sqrt(xrange**2 + yrange**2)
		
		if log is False:
		    bins = np.linspace(r_min,r_max,nbins+1)
		else:
		    if r_min == 0:
		        r_min = 1
		    bins = np.geomspace(r_min,r_max,nbins+1)
		
		for i in np.arange(nbins):
		    index = (r >= bins[i]) & (r < bins[i+1]) & (phi >= cone[0]) & (phi < cone[1])
		    signal = image[index]
		    profile[0,i] = np.sum(signal*r[index])/np.sum(signal)
		    profile[1,i] = np.mean(signal)
		    profile[2,i] = np.std(signal)
			
		    control[index] = i+1
		    
		if weighted is not True:
		    profile[0,:] = (bins[1:] + bins[0:-1])/2
		    
		if check is False:
		    out = profile
		else:
		    out = control
		    
	else:
		
		r = r.astype(np.int)
		r_uniq = np.unique(r)
		tbin = np.bincount(r.ravel(), image.ravel())
		nr = np.bincount(r.ravel())
		out = np.array([r_uniq, tbin / nr])
		    
	return(out)


def rebin(image, new_shape):
	'''Rebins a given 2D numpy array by averaging. 

	Parameter-s
	----------
	image: 2D float array
		Two-dimensional numpy array that is to be re-binned
	new shape: int tuple
		New dimensions for the provided array
	Returns
	-------
	new_image: float array
		Re-binned 2D array.
	'''

	shape = (new_shape[0], image.shape[0] // new_shape[0],
		 new_shape[1], image.shape[1] // new_shape[1])

	new_image = image.reshape(shape).mean(-1).mean(1)

	return(new_image)


def sample_sphere_uniform(n, mask = None, radec = True):
	'''Draws uniformly sampled tuples of coordinates on the sphere. 
	All-sky masks in the healpix format can be applied, in which case 
	masked areas will be excluded.

	Parameters
	----------
	n: int
		Number of data points to be drawn
	mask: float array, optional
		All-sky healpix mask. If a mask is used data points will 
		only be drawn in areas that are not masked. Default: None
	radec: bool
		Determines the coordinate system of the output. If True, 
		equatorial coordinates will be returned, i.e. RA, DEC (fk5). 
		If False, galactic coordinates are returned. Default: True
	Returns
	-------
	phi, theta
		longitude and latitude of sampled points in equatorial or 
		galactic coordinate system
	'''
    
	if mask is None:
		phi = 360 * np.random.random(n)
		theta = np.arccos(2*np.random.random(n) - 1)*180/np.pi - 90
	else:
		nside = hp.get_nside(mask)

		phi = np.zeros(n)
		theta = np.zeros(n)

		i = int(0)
		while i < n:
	    		phi_guess = 360 * np.random.random()
			theta_guess = np.arccos(2*np.random.random() - 1)*180/np.pi - 90

			index = hp.ang2pix(nside, phi_guess, theta_guess, lonlat = True)
			
			if mask[index] != 0:
			phi[i] = phi_guess
			theta[i] = theta_guess
			i += 1

	if radec is True:
		print("output will be fk5 coordinates (RA, DEC) for the equinox J2000")
		c = SkyCoord(phi, theta, frame='galactic', unit='deg')
		c_fk5 = c.transform_to('fk5')
		phi = c_fk5.ra.degree
		theta = c_fk5.dec.degree
	else:
		print("output will be galactic coordinates (glon, glat)")

	return(phi, theta)
