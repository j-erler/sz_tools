import numpy as np
import healpy as hp
from os.path import expanduser
import datetime
from astropy.io import fits
from astropy.io import ascii
from scipy import ndimage
import sz_tools as sz
import os

path = os.path.dirname( os.path.realpath( __file__ ) )

fwhm2sigma = 1/(2*np.sqrt(2*np.log(2)))

home = expanduser("~")

planck_path = home + "/SSD/Planck_maps/full_mission/"
ring1_path = home + "/SSD/Planck_maps/ringhalf_1/"
ring2_path = home + "/SSD/Planck_maps/ringhalf_2/"

planck_maps = {30:  'LFI_SkyMap_030_1024_R2.01_full.fits',
               44:  'LFI_SkyMap_033_1024_R2.01_full.fits',
               70:  'HFI_SkyMap_070_2048_R2.01_full.fits',
               100: 'HFI_SkyMap_100_2048_R2.02_full.fits',
               143: 'HFI_SkyMap_143_2048_R2.02_full.fits',
               217: 'HFI_SkyMap_217_2048_R2.02_full.fits',
               353: 'HFI_SkyMap_353_2048_R2.02_full.fits',
               545: 'HFI_SkyMap_545_2048_R2.02_full.fits',
               857: 'HFI_SkyMap_857_2048_R2.02_full.fits'}

ring1_maps = {30:  'LFI_SkyMap_030_1024_R2.00_full-ringhalf-1.fits',
              44:  'LFI_SkyMap_044_1024_R2.00_full-ringhalf-1.fits',
              70:  'LFI_SkyMap_070_2048_R2.00_full-ringhalf-1.fits',
              100: 'HFI_SkyMap_100_2048_R2.00_full-ringhalf-1.fits',
              143: 'HFI_SkyMap_143_2048_R2.00_full-ringhalf-1.fits',
              217: 'HFI_SkyMap_217_2048_R2.00_full-ringhalf-1.fits',
              353: 'HFI_SkyMap_353_2048_R2.00_full-ringhalf-1.fits',
              545: 'HFI_SkyMap_545_2048_R2.00_full-ringhalf-1.fits',
              857: 'HFI_SkyMap_857_2048_R2.00_full-ringhalf-1.fits'}

ring2_maps = {30:  'LFI_SkyMap_030_1024_R2.00_full-ringhalf-2.fits',
              44:  'LFI_SkyMap_044_1024_R2.00_full-ringhalf-2.fits',
              70:  'LFI_SkyMap_070_2048_R2.00_full-ringhalf-2.fits',
              100: 'HFI_SkyMap_100_2048_R2.00_full-ringhalf-2.fits',
              143: 'HFI_SkyMap_143_2048_R2.00_full-ringhalf-2.fits',
              217: 'HFI_SkyMap_217_2048_R2.00_full-ringhalf-2.fits',
              353: 'HFI_SkyMap_353_2048_R2.00_full-ringhalf-2.fits',
              545: 'HFI_SkyMap_545_2048_R2.00_full-ringhalf-2.fits',
              857: 'HFI_SkyMap_857_2048_R2.00_full-ringhalf-2.fits'}

data = ascii.read(path + "../data/NILC_bands.txt")
NILC_bands = np.array([data[:]['col1'], 
                       data[:]['col2'],
                       data[:]['col3'],
                       data[:]['col4'],
                       data[:]['col5'],
                       data[:]['col6'],
                       data[:]['col7'],
                       data[:]['col8'],
                       data[:]['col9'],
                       data[:]['col10'],])


def create_header(name, RA, DEC, npix, pixel_size):
	'''Creates a fits-compatible header. 

    Parameters
    ----------
	name: string
		name of the object
	RA: float
		Right acention of objects, fk5 coordinates are required
	DEC: float
		Declination of objects, fk5 coordinates are required
	pixel_size: float
		pixel size in arcmin
	
    Returns
    -------
	header: fits header
	'''

	today = str(datetime.date.today())
	c0 = fits.Card('SIMPLE', True, ' conforms to FITS standard')
	c1 = fits.Card('BITPIX', -32, ' array data type')
	c2 = fits.Card('NAXIS', 2, ' ')
	c3 = fits.Card('NAXIS1', npix, ' ')
	c4 = fits.Card('NAXIS2', npix, ' ')
	c5 = fits.Card('DATE', today, ' Creation date (CCYY-MM-DD) of FITS header')
	c6 = fits.Card('BUNIT', 'Compton-y', ' X-axis ')
	c7 = fits.Card('BAD_DATA', -1.6375E30, ' value for missing data')
	#
	c8 = fits.Card('RADECSYS', 'FK5', ' Celestial coordinate system')
	c9 = fits.Card('EQUINOX', 2000, ' Equinox of Ref. Coord.')
	c10 = fits.Card('PC1_1', 1.0, ' Degrees / Pixel')
	c11 = fits.Card('PC2_1', 0.0, ' Degrees / Pixel')
	c12 = fits.Card('PC1_2', 0.0, ' Degrees / Pixel')
	c13 = fits.Card('PC2_2', 1.0, ' Degrees / Pixel')
	#
	c14 = fits.Card('CTYPE1', 'RA---TAN', ' X-axis ')
	c15 = fits.Card('CRVAL1', RA, ' Origin coordinate')
	c16 = fits.Card('CRPIX1', (npix+1)/2., ' Origin pixel index (1 based)')
	c17 = fits.Card('CDELT1', -pixel_size/60.0, ' Degrees/pixel')
	#
	c18 = fits.Card('CTYPE2', 'DEC--TAN', ' Y-axis ')
	c19 = fits.Card('CRVAL2', DEC, ' Origin coordinate')
	c20 = fits.Card('CRPIX2', (npix+1)/2., ' Origin pixel index (1 based)')
	c21 = fits.Card('CDELT2', pixel_size/60.0, ' Degrees/pixel')
	#
	c22 = fits.Card('LONPOLE', 180.0 , ' Native longitude of Celestial pole')
	c23 = fits.Card('LATPOLE', 0.0, ' Celestial latitude of native pole')
	c24 = fits.Card('EXTEND', True, ' ')
	#
	header = fits.Header([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24])

	return(header)


def project_maps(name = None, RA = None, DEC = None, allsky_map = None, in_file = None, map_size = 10, 
pixel_size = 1.5, smooth = None, planck=None, MILCA = False, NILC = False, out_path = None, 
same_units = False, same_res = False):
	'''Creates gnomic projections around given sky coordinates from a healpix-compatible all-sky map. 

    Parameters
    ----------
	name: string array, optional
		name of objects, will be used as file name if files are written. Default:None
	RA: float array, optional
		Right acention of objects, fk5 coordinates are required. Default:None
	DEC: float array, optional
		Declination of objects, fk5 coordinates are required. Default:None
	allsky_map: float_array, optional
		all-sky map in healpix ring-ordered format. To be used as source map. Default:None
	in_file: string, optional
		file for batch-processing. Has to contain three columns: name, RA and DEC
		replaces name, RA and DEC is set. Default:None
	map_size: float, optional
		size of the desired projected map in degree, map will be square. Default: 10
	pixel_size: float, optional
		pixel size of the desired projected map in arcmin. Default: 1.5
	smooth: float, optional
		fwhm of gaussian kernel for smoothing of output maps. Default: None
	planck: int array, optional
		list of Planck bands in GHz to be used as source files. Default:None
	MILCA: Bool, optional
		if set the Planck MILCA y-map will be used as input. Default: False
	NILC: Bool, optional
		if set the Planck NILC y-map will be used as input. Default: False
	out_path: string, optional
		name of the output directory. Default: None
	same_units: bool, optional
		if changed to True all Planck maps will be provided in units of K_CMB.
		Default: False
	same_res: bool, optional
		if changed to True all Planck maps will be provided with the resolution 
		of the lowest-frequency channel. Default: False
	constrained: sting or float array
		defines additional spectral constraints to be used for the computation of
		ILC weights. If set to 'cmb', the cmb spectrum will be used.
	
    Returns
    -------
	output: array
		single image or data cube containing the projected maps. 
		If out_path is set, one or several files will be written
	'''

	if in_file is not None:
		data = ascii.read(in_file)
		name = np.array(data[:]['col1'])
		RA = np.array(data[:]['col2'])
		DEC = np.array(data[:]['col3'])

	npix = int(map_size*60 / pixel_size)
	nclusters = len(name)

	if planck is None:

		output = np.zeros((nclusters, npix, npix))
		
		if MILCA is True:
			allsky_map = hp.fitsfunc.read_map(home + '/SSD/Planck_maps/COM_CompMap_YSZ_R2.00/milca_ymaps.fits')
		if NILC is True:
			allsky_map = hp.fitsfunc.read_map(home + '/SSD/Planck_maps/COM_CompMap_YSZ_R2.00/nilc_ymaps.fits')

		for i in np.arange(nclusters):
				projected_map = hp.gnomview(allsky_map, coord=('G','C'), rot=(RA[i],DEC[i]), return_projected_map=True, xsize=npix, reso=pixel_size, no_plot=True)
				if smooth is not None:
					projected_map = ndimage.gaussian_filter(projected_map, sigma=smooth*fwhm2sigma/pixel_size, order=0, mode = "reflect", truncate = 10)
				output[i,:,:] = projected_map
	else:
		
		nf = len(planck)
			
		output = np.zeros((nclusters, nf, npix, npix))

		for f in np.arange(nf):
			file_name = planck_path + planck_maps[planck[f]]
			allsky_map = hp.fitsfunc.read_map(file_name)
			for i in np.arange(nclusters):
				projected_map = hp.gnomview(allsky_map, coord=('G','C'), rot=(RA[i],DEC[i]), return_projected_map=True, xsize=npix, reso=pixel_size, no_plot=True)
				if smooth is not None:
					projected_map = ndimage.gaussian_filter(projected_map, sigma=smooth*fwhm2sigma/pixel_size, order=0, mode = "reflect", truncate = 10)

				if same_units is True:
					if planck[f] == 545:
						projected_map /= sz.planck_uc(545)
		
					if planck[f] == 857:
						projected_map /= sz.planck_uc(857)

				if same_res is True and f != 0:
					kernel = np.sqrt(sz.planck_beams(planck[0])**2 - sz.planck_beams(planck[f])**2)
					print(sz.planck_beams(planck[0]), sz.planck_beams(planck[f]), kernel*fwhm2sigma/pixel_size)
					projected_map = ndimage.gaussian_filter(projected_map, sigma=kernel*fwhm2sigma/pixel_size, order=0, mode = "reflect", truncate = 10)

				output[i,f,:,:] = projected_map
	
	if out_path is not None:
		for i in np.arange(nclusters):
			header = create_header(name[i], RA[i], DEC[i], npix, pixel_size)
			hdu = fits.PrimaryHDU()

			if planck is None:
				hdu.data = np.float32(output[i,:,:])
			else:
				hdu.data = np.float32(output[i,:,:,:])

			hdu.header = header
			hdu.writeto(out_path + name[i] + ".fits", overwrite=True)

	return(output)


def ilc_windows(scales, nside, l_max = 2, silent = True):
	'''Computes allsky-ILC spatial window functions from the difference of gaussians. 
	All scales are conserved.

    Parameters
    ----------
	scales: float array
		FWHM of gaussians that define the scales for the decomposition.
		Have to be provided in decending order.
	nside: array
		Healpy nside parameter of the allsky maps.
	l_max: int
		Defines the maximum ell via lmax = l_max*nside - 1.
		The maximum allowed value is 3. Default: 2
	silent: bool
		Prints the sum of all windows as a diagnostic. All scales are conserved
		if all numbers are 1. Default: True
	
    Returns
    -------
	bands: 2D array
		Spherical-Harmonic window functions to be used for spatial decomposition.
	'''

	lmax = l_max*nside-1

	n_scales = len(scales)+1
	windows = np.zeros((n_scales+1, lmax+1))
	windows[n_scales,:] = np.ones((lmax+1))
	bands = np.zeros((n_scales, lmax+1))
	for i in np.arange(1, n_scales):
		windows[i,:] = hp.sphtfunc.gauss_beam(scales[i-1]/60*np.pi/180, pol=False, lmax = lmax)
		bands[i-1,:] = windows[i,:]-windows[i-1,:]
    	#print([i-1, int(scales[i-1]), int(scales[i-2])])
    
	bands[n_scales-1,:] = windows[n_scales,:]-windows[n_scales-1,:]    

	if silent is not True:
		control = np.sum(bands, 0)
		print("mininmum: ", np.min(control), "maximum: ", np.max(control), "mean: ", np.mean(control))

	return(bands)


def remove_offset(data, median = True, mean = False, hist = False):
	'''Removes offset from ILC maps.

    Parameters
    ----------
	data: float array
		ILC map
	median: bool, optional
		Subtracts the meadian of the data. Default: True
	mean: bool, optional
		Subtracts the mean of the data. Generally not recommended.
		Default: False
	hist: bool, optional
		Fits a gaussian to the histogram of the
		data and subtracts the best-fit center.
		Default: False
	
    Returns
    -------
	data: array
		Offset-corrected ILC map.
	'''

	if median is True:
		data = data - np.median(data)
	elif mean is True:
		data = data - np.mean(data)
	elif hist is True:
		fit_results = sz.create_histogram(data, np.sqrt(np.size(data)), fit=True, plot=False);
		data = data - fit_results[2]

	return(data)


def run_ilc(data, F, e = None, mask = None):
	'''Runs the internal linear combination (ilc) algorithm on a multi-freqency 
	dataset using given spectral constraints to obtain an estimate of the 
	amplitude of the desired signal. 

    Parameters
    ----------
	data: 2d array
		Multi-frequency data set. 2d images have to be flattened.
		The dimensions have to be n_freq x n_pix
	F: array
		Spectral constraints for the ilc algorithm. If contaminants
		are constrained as well, the dimensions have to be 
		n_components x n_freq
	e: array, optional
		If multible spectral components are constrained, e gives the
		responce of the ilc weights to the individual spectra
	mask: array, optional
		Flattened data mask. The mask will be used during the computation
		of the data covariance matrix and later applied to the output
	
    Returns
    -------
	ilc_result: array
		Optimal estimate of the signal amplitude for the given spectrum
	'''

	if mask is not None:
		not_masked = np.where(mask != 0)[0]
		cov_matrix = np.cov(data[:,not_masked])
	else:
		cov_matrix = np.cov(data)

	cov_inverted = np.linalg.inv(cov_matrix)

	if e is None:
		w = F @ cov_inverted/(F @ cov_inverted @ F)
		print('ilc responce: ', w @ F)
	else:
		w = e @ np.linalg.inv(F @ cov_inverted @ np.transpose(F)) @ F @ cov_inverted
		for i in np.arange(len(e)):
			print('ilc responce ' + str(i) + ': ', w @ F[i,:])

	ilc_result = w @ data

	if mask is not None:
		ilc_result *= mask

	return(ilc_result)


def ilc_scales(data, F, scales, pixel_size, responce = None, mask = None):
	'''Performes a spatial decomposition of the input maps and runs an internal linear 
	combination algorithm on each spatial slice. Returns the sum of all output slices. 

    Parameters
    ----------
	data: 2d array
		Multi-frequency data set. image cube of dimensions n_freq x n_pix x n_pix
	F: array
		Spectral constraints for the ilc algorithm. If contaminants
		are constrained as well, the dimensions have to be 
		n_components x n_freq
	scales: array
		Array defining the spatial scales for the decomposition. The spatial 
		decomposition is achived by computing the differences of smoothed images.
		Each scale corresponds to a Gaussian Kernel.
	responce: array, optional
		If multible spectral components are constrained, e gives the
		responce of the ilc weights to the individual spectra
	mask: array, optional
		Flattened data mask. The mask will be used during the computation
		of the data covariance matrix and later applied to the output
	
    Returns
    -------
	ilc_result: array
		Optimal estimate of the signal amplitude for the given spectrum
	'''
	
	nscales = len(scales)

	nf = data.shape[0]
	npix = data.shape[1]

	output_slices = np.zeros((nscales+1,npix**2))

	for i in np.arange(nscales+1):

		print([i, "/", nscales])
		data_slice = np.zeros((nf, npix, npix))

		for f in np.arange(nf):
			if i < nscales:
				if i == 0:
					scale1 = data[f,:,:]
				else:
					scale1 = ndimage.gaussian_filter(data[f,:,:], sigma=scales[i-1]*fwhm2sigma/pixel_size, order=0, mode = "constant", truncate = 10)
				scale2 = ndimage.gaussian_filter(data[f,:,:], sigma=scales[i]*fwhm2sigma/pixel_size, order=0, mode = "constant", truncate = 10)
				data_slice[f,:,:] = (scale1 - scale2)
			else:
				data_slice[f,:,:] = ndimage.gaussian_filter(data[f,:,:], sigma=scales[i-1]*fwhm2sigma/pixel_size, order=0, mode = "constant", truncate = 10)

		output_slices[i,:] = run_ilc(data_slice.reshape(nf, npix**2), F, e = responce, mask = mask)

	output = np.sum(output_slices, 0).reshape(npix, npix)
	
	return(output)


def ilc(name = None, RA = None, DEC = None, in_file = None, map_size = 10, pixel_size = 1.5, maps = None,
        freq = None, planck = None, scales = None, tsz = True, T_e = 0, cmb = False, 
        constrained = None, mask = None, smooth = None, out_path = None):
	'''Computes an ILC map. The function was written with Planck data in mind, but can also handle synthetic 
	data and data from future surveys. The ILC algorithm is written is pixel space and thus all maps have to 
	be smoothed to the same spatial resolution. The result can be improved by spatialy decomposing the input 
	maps and running the ILC algorithm on each spatial scale separatly. For this, several modes are available.

    Parameters
    ----------
	name: string array, optional
		Name of objects, will be used as file name if files are written. Default:None
	RA: float array, optional
		Right acention of objects, fk5 coordinates are required. Default:None
	DEC: float array, optional
		Declination of objects, fk5 coordinates are required. Default:None
	in_file: string, optional
		File for batch-processing. Has to contain three columns: name, RA and DEC
		replaces name, RA and DEC is set. Default:None
	map_size: float, optional
		Size of the desired projected map in degree, map will be square. Default: 10
	pixel_size: float, optional
		Pixel size of the desired projected map in arcmin. Default: 1.5
	maps: float array, optional
		Cube containing multifrequency maps as input for the ILC algorithm.
		The dimensions have to be nf x npix_x x npix_y. Default: None
	freq: float array, optional
		An array specifying the frequency bands of the input maps. Default: None
	planck: int array, optional
		List of Planck bands in GHz to be used as source files. Default:None
	scales: float array, optional
		Defines the gaussian windows to be used to spatially decompose the the maps.
		The windows are computed from the difference of pairs of gaussians, the FWHMs in arcmin
		of which are specified here. Default: None
	tsz: bool, optional
		If set to True, the function will use the tSZ spectrum to return an ILC y-map. Default: True
	T_e: float, optional
		Electron temperature to be used for the computation of the tSZ spectrum. The temperature will
		be assigned to the full map, so use with caution. Default: 0
	cmb:
		If set to True, the function will use the cmb spectrum to return a CMB map. Default: False
	constrained: string or float array, optional
		Additional spectral constraints for the ILC algorithm. If set to 'cmb', the cmb spectrum will
		be used to minimize cmb residuals. Choosing 'tsz' will remove tSZ residuals. Alternatively, 
		constrained can be a float array containing an arbitrary SED.
	mask: array, optional
		Flattened data mask. The mask will be used during the computation
		of the data covariance matrix and later applied to the output
	smooth: float, optional
		FWHM of gaussian kernel for smoothing of output maps. Default: None
	outfile: sting, optional
		Path and file name for data output. The output will be stored as a healpy .fits file.
		Default: None

    Returns
    -------
	output: float array
		Returns an ILC map.
	'''

	if scales is not None:
		if scales == 'default':
			scales = np.array([15,25,40,65,105,170,275])
			#scales = np.array([15,25,40,55,70,90,110,130,150,200]) * pixel_size
			scales = np.sqrt(scales**2 - 9.66**2)

	if planck is not None:
	
		maps = project_maps(name = name, RA = RA, DEC = DEC, in_file = in_file, 
						map_size = map_size, pixel_size = pixel_size, 
						smooth = smooth, planck = planck, out_path = out_path, 
						same_units = True, same_res = True)
	
	else:

		maps = maps.reshape(1, maps.shape[1], maps.shape[2], maps.shape[3])
		
	nc = maps.shape[0]
	nf = maps.shape[1]
	npix = maps.shape[2]

	if mask is not None:
		mask = mask.reshape(npix**2)

	output = np.zeros((nc, npix, npix))		

	if in_file is not None:
		data = ascii.read(in_file)
		name = np.array(data[:]['col1'])
		RA = np.array(data[:]['col2'])
		DEC = np.array(data[:]['col3'])

	if tsz is True:
		if planck is not None:
			spectrum = sz.tsz_spec_planck(planck, 1, T_e = T_e)
		else:
			spectrum = sz.tsz_spec(freq, 1, T_e = T_e)
	if cmb is True:
		spectrum = np.ones(nf)

	if constrained is not None:
		if constrained == 'cmb' or constrained == 'CMB':
			F = np.array([spectrum, np.ones(nf)])
		elif constrained == 'tsz' or constrained == 'tSZ':
			if planck is not None:
				F = np.array([spectrum, sz.tsz_spec_planck(planck, 1, T_e = T_e)])
			else:
				F = np.array([spectrum, sz.tsz_spec(freq, 1, T_e = T_e)])
		else:
			F = np.concatenate([spectrum.reshape(1,nf), constrained])
		responce = np.concatenate([np.ones((1)), np.zeros((F.shape[0]-1))])
	else:
		F = np.array(spectrum)
		responce = None

	for i in np.arange(nc): 

		data = maps[i,:,:,:]

		if scales is None:

			result = run_ilc(data.reshape(nf, npix**2), F, e = responce, mask = mask).reshape(npix, npix)

		else:

			result = ilc_scales(data, F, scales, pixel_size, responce = responce, mask = mask)

		result = remove_offset(result, median = True)	

		output[i,:,:] = result

		if out_path is not None:

			hdu = fits.PrimaryHDU()
			hdu.data = np.float32(result)
			if RA is not None and DEC is not None:
				header = create_header(name[i], RA[i], DEC[i], npix, pixel_size)
				hdu.header = header
			hdu.writeto(out_path + name[i] + "_y" + ".fits", overwrite=True)

	return(output)


def ilc_allsky(allsky_maps = None, freq = None, nside = 2048, planck = None, decompose = None, 
               field_nside = 2, T_e = 0, l_max = 2, tsz = True, constrained = None, cmb = False, 
					mask = None, iter = 0, ring1 = False, ring2 = False, outfile = None):
	'''Computes an allsky-ILC map. The function was written with Planck data in mind, 
	but can also handle synthetic data and data from future surveys. The ILC algorithm is 
	written is pixel space and thus all maps have to be smoothed to the same spatial resolution. 
	The result can be improved by spatialy decomposing the input maps and running the ILC 
	algorithm on each spatial scale separatly. For this, several modes are available, some of 
	which use spatial bands of the MILCA and NILC algorithms of the Planck collaboration.

    Parameters
    ----------
	allsky_maps: float array, optional
		A n_freq x n_pix array containing all-sky maps in different frequency bands.
		All maos have to be given at the same spatial resolution. Default: None
	freq: float array, optional
		An array specifying the frequency bands of the input maps. Default: None
	nside: array, optional
		Healpy nside parameter of the allsky maps. Default: 2048
	planck: int array, optional
		List of Planck bands in GHz to be used as source files. Default:None
	decompose: float array or string, optional
		Defines the gaussian windows to be used to spatially decompose the the all-sky maps.
		The windows are computed from the difference of pairs of gaussians, the FWHMs in arcmin
		of which are specified here. Besides giving an array of values for the FWHMs, setting 
		decompose to 'default', 'NILC' or 'MILCA' uses pre-fefind windows. Default: None
	field_nside: int array, optional
		Defines the number of fields the sky will be tesselated in for the computation of the
		covariance matrix. This is done using the healpy nested pixel-indexing scheme. 
		The values have the be valid healpy nside parameters. In case spatial decomposition is used, 
		the number of field_nsides has to be n_scales+1. If one of the pre-defined modes for the
		decomposition is used field_nside will be assigned automatically. Default: 2
	T_e: float, optional
		Electron temperature to be used for the computation of the tSZ spectrum. The temperature will
		be assigned to the full sky, so use with caution. Default: 0
	l_max: int, optional
		Defines the highest ell to be used for the all-sky map processing. To be multiplied with nside.
		healpy only supports a maximum value of 3*nside-1, so only choose either 2 or 3. Default: 2

	tsz: bool, optional
		If set to True, the function will use the tSZ spectrum to return an ILC y-map. Default: True
	cmb:
		If set to True, the function will use the cmb spectrum to return a CMB map. Default: False
	constrained: string or float array, optional
		Additional spectral constraints for the ILC algorithm. If set to 'cmb', the cmb spectrum will
		be used to minimize cmb residuals. Choosing 'tsz' will remove tSZ residuals. Alternatively, 
		constrained can be a float array containing an arbitrary SED.
	mask: array, optional
		Flattened data mask. The mask will be used during the computation
		of the data covariance matrix and later applied to the output
	iter: int, optional
		Number if iterations to be used while processing the all-sky maps.
		Higher values will reduce numerical errors. Healpy default is 3.
		Default: 0
	ring1: bool, optional
		If set to True, the Planck Ringhalf1 maps are used as input: Default: False	
	ring2: bool, optional
		If set to True, the Planck Ringhalf1 maps are used as input: Default: False	
	outfile: sting, optional
		Path and file name for data output. The output will be stored as a healpy .fits file.
		Default: None

    Returns
    -------
	output: float array
		Returns a ILC all-sky map in the healpy format.
	'''

	npix = hp.pixelfunc.nside2npix(nside)
	lmax = l_max*nside-1

	if planck is not None:

		nf = len(planck)
		allsky_maps = np.zeros((nf,npix))

		for f in np.arange(nf):

			if ring1 is True:
				file_name = ring1_path + ring1_maps[planck[f]]
			elif ring2 is True:
				file_name = ring2_path + ring2_maps[planck[f]]
			else:
				file_name = planck_path + planck_maps[planck[f]]
			allsky_map = hp.fitsfunc.read_map(file_name)

			if planck[f] == 30 or planck[f] == 44:
				allsky_map = hp.pixelfunc.ud_grade(allsky_map, nside, order_in = 'RING')

			if planck[f] == 545:
				allsky_map /= sz.planck_uc(545)
	
			if planck[f] == 857:
				allsky_map /= sz.planck_uc(857)

			if f != 0:
				print("Smoothing map:", planck[f])
				kernel = np.sqrt(sz.planck_beams(planck[0])**2 - sz.planck_beams(planck[f])**2) / 60 * np.pi/180
				allsky_map = hp.sphtfunc.smoothing(allsky_map, fwhm = kernel, iter = iter, lmax = lmax)
			
			if decompose is None:
				allsky_maps[f,:] = hp.pixelfunc.reorder(allsky_map, r2n = True)
			else:
				allsky_maps[f,:] = allsky_map

		del allsky_map

	else:
		nf = len(freq)

	if tsz is True:
		if planck is not None:
			spectrum = sz.tsz_spec_planck(planck, 1, T_e = T_e)
		else:
			spectrum = sz.tsz_spec(freq, 1, T_e = T_e)
	if cmb is True:
		spectrum = np.ones(nf)

	if constrained is not None:
		if constrained == 'cmb' or constrained == 'CMB':
			F = np.array([spectrum, np.ones(nf)])
		elif constrained == 'tsz' or constrained == 'tSZ':
			if planck is not None:
				F = np.array([spectrum, sz.tsz_spec_planck(planck, 1, T_e = T_e)])
			else:
				F = np.array([spectrum, sz.tsz_spec(freq, 1, T_e = T_e)])
		else:
			F = np.concatenate([spectrum.reshape(1,nf), constrained])
		responce = np.concatenate([np.ones((1)), np.zeros((F.shape[0]-1))])
	else:
		F = np.array(spectrum)
		responce = None

	output = np.zeros(npix)

	if decompose is not None:

		if decompose == 'milca':
			windows = None
		elif decompose == 'nilc':
			windows = NILC_bands
			field_nside = np.array([1,2,2,2,2,4,4,4,8,16])
		elif decompose == 'default':
			scales = np.array([1280,640,320,160,80,40,20,10,5])
			windows = ilc_windows(scales, nside, silent = True)
			field_nside = np.array([2,2,2,2,2,2,2,2,2,2])
		else:
			windows = ilc_windows(decompose, nside, silent = True)

		n_scales = windows.shape[0]
		filtered_maps = np.zeros((nf, npix))

		for i in np.arange(n_scales):
			for j in np.arange(nf):
				filtered_maps[j,:] = hp.pixelfunc.reorder(hp.sphtfunc.smoothing(allsky_maps[j,:], beam_window = windows[i,:], iter = iter, lmax = lmax), r2n = True)
			
			nfields = hp.pixelfunc.nside2npix(field_nside[i])
			pix_per_field = int(npix/nfields)
			fields = np.arange(0, nfields) * pix_per_field

			for k in np.arange(nfields-1):
				ilc_result = run_ilc(filtered_maps[:,fields[k]:fields[k+1]], F, e = responce, mask = mask)
				ilc_result = remove_offset(ilc_result, median = True)	
				output[fields[k]:fields[k+1]] += ilc_result

	else:
		nfields = hp.pixelfunc.nside2npix(field_nside)
		pix_per_field = int(npix/nfields)
		fields = np.arange(0, nfields) * pix_per_field

		for k in np.arange(nfields-1):
			ilc_result = run_ilc(allsky_maps[:, fields[k]:fields[k+1]], F, e = responce, mask = mask)
			ilc_result = remove_offset(ilc_result, median = True)	
			output[fields[k]:fields[k+1]] += ilc_result
	
	output = np.float32(hp.pixelfunc.reorder(output, n2r = True))

	if outfile is not None:
		hp.fitsfunc.write_map(outfile, output, overwrite = True)

	return(output)
