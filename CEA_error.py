import numpy as np
import sunpy.map
from sunpy.coordinates import frames
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy import ndimage
from coord_transform import hmi_disambig
from CEA_transform import *

#img2sph.pro
def img2sph(xi,eta,lonc=None,latc=None,asd=None,pa=None,opt_out=None):
    """Convert image coordinate (xi,eta) to Stonyhurst
    coordinate (lon,lat)

    Params:
        Image coordinates in unit of apparent solar radius.

    Optional Params:
        lonc,latc: Disk center longitude and latitude
        asd: apparent solar radius
        pa: p-angle

    Returns:
        Stonyhurst longitude and latitude in radians.

    Optional Output:
        rho,sig: angle with respect to observed solar center
        mu: cosine between observation and local normal
        chi: position angle on image measured westward from north
    """
    # Check optional parameters and assign values if needed
    if not lonc:
         lonc = 0.0
    if not latc:
        latc = 0.0
    if not asd:
        asd = 4.7026928e-3        #970 arcsec
    if not pa:
        pa = 0.0

    r = np.sqrt(xi**2 + eta**2)
    idx_r1 = np.where(r <= 0)
    idx_r2 = np.where(r >= 1)
    r[idx_r1] = np.nan
    r[idx_r2] = np.nan

    chi = np.arctan(xi/eta) + pa

    idx_c1 = np.where(chi > 2*np.pi)
    for i in range(idx_c1[0].shape[0]):
        x = idx_c1[1][i]
        y = idx_c1[0][i]
        if chi[y,x] > 2*np.pi:
            chi[y,x] -= 2*np.pi

    idx_c2 = np.where(chi < 0)
    for i in range(idx_c2[0].shape[0]):
        x = idx_c2[1][i]
        y = idx_c2[0][i]
        if chi[y,x] < 0:
            chi[y,x] += 2*np.pi

    sig = np.arctan(r*np.tan(asd))
    rho = np.arcsin(np.sin(sig)/np.sin(asd)) - sig
    idx_s = np.where(sig > asd)
    sig[idx_s] = np.nan
    mu = np.cos(rho + sig)

    sinr = np.sin(rho)
    cosr = np.cos(rho)
    sinlat = np.sin(latc)*cosr + np.cos(latc)*sinr*np.cos(chi)
    coslat = np.sqrt(1 - sinlat**2)

    lat = np.arcsin(sinlat)
    sinlon = sinr*np.sin(chi)/np.cos(lat)
    lon = np.arcsin(sinlon)

    idx_lo = np.where(cosr < np.sin(lat)*np.sin(latc))
    lon[idx_lo] = np.pi - lon[idx_lo]
    lon += lonc

    idx_lo = np.where(lon < 0)
    for i in range(idx_lo[0].shape[0]):
        x = idx_lo[1][i]
        y = idx_lo[0][i]
        if lon[y,x] < 0:
            lon[y,x] += 2*np.pi

    idx_lo = np.where(lon >= 2*np.pi)
    for i in range(idx_lo[0].shape[0]):
        x = idx_lo[1][i]
        y = idx_lo[0][i]
        if lon[y,x] >= 2*np.pi:
            lon[y,x] -= 2*np.pi

    return lat,lon

#bvec_errorprop.pro
def bvec_errorprop(header,fld,inc,azi,err_fld,err_inc,err_azi,cc_fi,cc_fa,cc_ia):
    """Convert vector field and covariance matrix components in
    field/inclination/azimuth into variance of Bp, Bt, Br"""
    # Get parameters from header
    crpix1 = header['CRPIX1']
    crpix2 = header['CRPIX2']
    cdelt1 = header['CDELT1']
    cdelt2 = header['CDELT2']
    crval1 = header['CRVAL1']
    crval2 = header['CRVAL2']
    rsun_obs = header['RSUN_OBS']	#solar disk radius in arcsec
    crota2 = header['CROTA2']		#neg p-angle
    crlt_obs = header['CRLT_OBS']	#disk center latitude

    nx0 = fld.shape[1]
    ny0 = fld.shape[0]

    # Get longitude/latitude
    xi = np.zeros((ny0,nx0))
    eta = np.zeros((ny0,nx0))
    for i in range(nx0):
        xi[:,i] = ((i + 1 - crpix1)*cdelt1 + crval1)/rsun_obs
    for j in range(ny0):
        eta[j,:] = ((j + 1 - crpix2)*cdelt2 + crval2)/rsun_obs

    lat,lon = img2sph(xi,eta,lonc=0.0,latc=np.radians(crlt_obs),
                      asd=np.radians(rsun_obs/3.6e3),pa=np.radians(-1*crota2))

    latc = np.radians(crlt_obs)
    lonc = 0.0
    pAng = np.radians((-1.0) * crota2)

    a11 = (-np.sin(latc)*np.sin(pAng)*np.sin(lon - lonc)
           + np.cos(pAng)*np.cos(lon - lonc))
    a12 = (np.sin(latc)*np.cos(pAng)*np.sin(lon - lonc)
           + np.sin(pAng)*np.cos(lon - lonc))
    a13 = (-np.cos(latc)*np.sin(lon - lonc))
    a21 = (-np.sin(lat)*(np.sin(latc)*np.sin(pAng)*np.cos(lon - lonc)
           + np.cos(pAng)*np.sin(lon - lonc))
           - np.cos(lat)*np.cos(latc)*np.sin(pAng))
    a22 = (np.sin(lat)*(np.sin(latc)*np.cos(pAng)*np.cos(lon - lonc)
           - np.sin(pAng)*np.sin(lon - lonc))
           + np.cos(lat)*np.cos(latc)*np.cos(pAng))
    a23 = (-np.cos(latc)*np.sin(lat)*np.cos(lon - lonc)
           + np.sin(latc)*np.cos(lat))
    a31 = (np.cos(lat)*(np.sin(latc)*np.sin(pAng)*np.cos(lon - lonc)
           + np.cos(pAng)*np.sin(lon - lonc))
           - np.sin(lat)*np.cos(latc)*np.sin(pAng))
    a32 = (-np.cos(lat)*(np.sin(latc)*np.cos(pAng)*np.cos(lon - lonc)
           - np.sin(pAng)*np.sin(lon - lonc))
           + np.sin(lat)*np.cos(latc)*np.cos(pAng))
    a33 = (np.cos(lat)*np.cos(latc)*np.cos(lon - lonc)
           + np.sin(lat)*np.sin(latc))

    # Sine/cosine
    sin_inc = np.sin(inc)
    cos_inc = np.cos(inc)
    sin_azi = np.sin(azi)
    cos_azi = np.cos(azi)

    # Covariance
    var_fld = err_fld * err_fld
    var_inc = err_inc * err_inc
    var_azi = err_azi * err_azi
    cov_fi = err_fld * err_inc * cc_fi
    cov_fa = err_fld * err_azi * cc_fa
    cov_ia = err_inc * err_azi * cc_ia

    # Partial derivatives
    dBp_dfld = (-a11*sin_inc*sin_azi + a12*sin_inc*cos_azi + a13*cos_inc)
    dBp_dinc = (-a11*cos_inc*sin_azi + a12*cos_inc*cos_azi - a13*sin_inc)*fld
    dBp_dazi = (-a11*sin_inc*cos_azi - a12*sin_inc*sin_azi)*fld

    dBt_dfld = (-a21*sin_inc*sin_azi + a22*sin_inc*cos_azi + a23*cos_inc)*(-1)
    dBt_dinc = (-a21*cos_inc*sin_azi + a22*cos_inc*cos_azi - a23*sin_inc)*fld*(-1)
    dBt_dazi = (-a21*sin_inc*cos_azi - a22*sin_inc*sin_azi)*fld*(-1)

    dBr_dfld = (-a31*sin_inc*sin_azi + a32*sin_inc*cos_azi + a33*cos_inc)
    dBr_dinc = (-a31*cos_inc*sin_azi + a32*cos_inc*cos_azi - a33*sin_inc)*fld
    dBr_dazi = (-a31*sin_inc*cos_azi - a32*sin_inc*sin_azi)*fld

    # Final variances
    var_bp = (dBp_dfld*dBp_dfld*var_fld
              + dBp_dinc*dBp_dinc*var_inc
              + dBp_dazi*dBp_dazi*var_azi
              + 2*dBp_dfld*dBp_dinc*cov_fi
              + 2*dBp_dfld*dBp_dazi*cov_fa
              + 2*dBp_dinc*dBp_dazi*cov_ia)

    var_bt = (dBt_dfld*dBt_dfld*var_fld
              + dBt_dinc*dBt_dinc*var_inc
              + dBt_dazi*dBt_dazi*var_azi
              + 2*dBt_dfld*dBt_dinc*cov_fi
              + 2*dBt_dfld*dBt_dazi*cov_fa
              + 2*dBt_dinc*dBt_dazi*cov_ia)

    var_br = (dBr_dfld*dBr_dfld*var_fld
              + dBr_dinc*dBr_dinc*var_inc
              + dBr_dazi*dBr_dazi*var_azi
              + 2*dBr_dfld*dBr_dinc*cov_fi
              + 2*dBr_dfld*dBr_dazi*cov_fa
              + 2*dBr_dinc*dBr_dazi*cov_ia)

    return var_bp,var_bt,var_br

#bvecerr2cea.pro
def bvecerr2cea(fld_path,inc_path,azi_path,fld_err_path,
                inc_err_path,azi_err_path,cc_fld_inc_path,
                cc_fld_azi_path,cc_inc_azi_path,disambig_path,
                do_disambig=True,phi_c=None,lambda_c=None, nx=None, ny=None,
                dx=None,dy=None,xyz=None):
    """Converts cutout vector field uncertainties to CEA maps

    Params:
        File names of field, inclination, azimuth, their respective
        errors, their respective correlation coefficient maps, and the
        disambiguation.

    Optional Params:
        phi_c: phi center coordinate (in degrees)
        lambda_c: lambda center coordinate (in degrees)
        nx: image size along x-axis
        ny: image size along y-axis
        dx: pixel size along x-axis
        dy: pixel size along y-axis

    Returns:
    Sunpy maps of Bp, Bt, and Br uncertainty.
    """
    #Read in input data
    fld_map = sunpy.map.Map(fld_path)
    inc_map = sunpy.map.Map(inc_path)
    azi_map = sunpy.map.Map(azi_path)
    disamb_map = sunpy.map.Map(disambig_path)

    err_fld_map = sunpy.map.Map(fld_err_path)
    err_inc_map = sunpy.map.Map(inc_err_path)
    err_azi_map = sunpy.map.Map(azi_err_path)

    cc_fi_map = sunpy.map.Map(cc_fld_inc_path)
    cc_fa_map = sunpy.map.Map(cc_fld_azi_path)
    cc_ia_map = sunpy.map.Map(cc_inc_azi_path)

    header = fld_map.meta
    field = fld_map.data
    inclination = np.radians(inc_map.data)
    azimuth = azi_map.data
    disambig = disamb_map.data

    err_fld = err_fld_map.data
    err_inc = np.radians(err_inc_map.data)
    err_azi = np.radians(err_azi_map.data)

    cc_fi = cc_fi_map.data
    cc_fa = cc_fa_map.data
    cc_ia = cc_ia_map.data

    if do_disambig:
        azimuth = np.radians(hmi_disambig(azimuth,disambig,2))
    else:
        azimuth = np.radians(azimuth)

    # Check whether or not input image sizes match
    nx0 = field.shape[1]
    ny0 = field.shape[0]

    if (field.shape != inclination.shape or field.shape != azimuth.shape or
        field.shape != err_fld.shape or field.shape != err_inc.shape or
        field.shape != err_azi.shape or field.shape != cc_fi.shape or
        field.shape != cc_fa.shape or field.shape != cc_ia.shape):
        print('Input image sized do not match.')
        return

    # Check that output params exist in header
    keys = ['crlt_obs','crln_obs','crota2','rsun_obs','cdelt1',
            'crpix1','crpix2']
    for idx,key in enumerate(keys):
        try:
            header[key]
        except KeyError:
            print('Keyword '+key+' missing')
            return

    # Get x and y coordinate center from header
    try:
        maxlon = header['LONDTMAX']
        minlon = header['LONDTMIN']
    except KeyError:
        print('No x center')
        return

    try:
        maxlat = header['LATDTMAX']
        minlat = header['LATDTMIN']
    except KeyError:
        print('No y center')
        return

    # Check optional parameters and assign values if needed
    if not phi_c:
        phi_c = (maxlon + minlon) / 2.0 + header['CRLN_OBS']
    if not lambda_c:
        lambda_c = (maxlat + minlat) / 2.0

    if not dx:
        dx = 3e-2
    else:
        dx = np.abs(dx)
    if not dy:
        dy = 3e-2
    else:
        dy = np.abs(dy)

    if not nx:
        nx = int(np.around(np.around((maxlon - minlon) * 1e3)/1e3/dx))
    if not ny:
    	ny = int(np.around(np.around((maxlat - minlat) * 1e3)/1e3/dy))

    # Get variance of Bp, Bt, Br
    #TODO: bvec_errorprop output
    var_bp,var_bt,var_br = bvec_errorprop(header,field,inclination,azimuth,
                                          err_fld,err_inc,err_azi,cc_fi,cc_fa,cc_ia)

    # Find coordinates of CEA pixels in image
    xi,eta,lat,lon = find_cea_coord(header,phi_c,lambda_c,nx,ny,dx,dy)

    # Perform sampling
    f_bperr_map = ndimage.map_coordinates(var_bp,[eta,xi],order=1)
    f_bterr_map = ndimage.map_coordinates(var_bt,[eta,xi],order=1)
    f_brerr_map = ndimage.map_coordinates(var_br,[eta,xi],order=1)

    err_bp = np.sqrt(f_bperr_map)
    err_bt = np.sqrt(f_bterr_map)
    err_br = np.sqrt(f_brerr_map)

    header_map = prep_hd(header,phi_c,lambda_c,nx,ny,dx,dy)

    bperr_map = sunpy.map.Map(err_bp, header_map)
    bterr_map = sunpy.map.Map(err_bt, header_map)
    brerr_map = sunpy.map.Map(err_br, header_map)

    return bperr_map,bterr_map,brerr_map
