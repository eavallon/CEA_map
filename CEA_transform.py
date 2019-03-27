import numpy as np
import sunpy.map
from sunpy.coordinates import frames
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy import ndimage
from coord_transform import hmi_disambig

#plane2sphere.pro
def plane2sphere(x,y,latc,lonc):
    """Convert (x,y) of a CEA map to Stonyhurst/Carrington
    coordinates (lat,lon)

    Params:
        x,y: coordinate of CEA map pixel relative to map
             reference point (usually center) in radians.

        latc,lonc: Stonyhurst latitude and longitude of map
                   reference point (usually center) in radians.

    Returns:
        Stonyhurst latitude and longitude of the map pixel in radians.
    """
    if np.abs(y) > 1:
        lat = np.nan
        lon = np.nan
        return lat,lon
    else:
        pass

    coslatc = np.cos(latc)
    sinlatc = np.sin(latc)

    cosphi = np.sqrt(1.0 - y**2)
    lat = np.arcsin((y*coslatc) + (cosphi*np.cos(x)*sinlatc))

    if np.cos(lat) == 0:
        test = 0.0
    else:
        test = cosphi*np.sin(x)/np.cos(lat)

    lon = np.arcsin(test) + lonc

    x0 = x
    if np.abs(x0) > np.pi/2.0:
        while x0 > np.pi/2.0:
            lon = np.pi - lon
            x0 = x0 - np.pi

        while x0 < -np.pi/2.0:
            lon = -np.pi - lon
            x0 = x0 + np.pi

    return lat,lon

#sphere2img.pro
def sphere2img(lat,lon,latc,lonc,xcen,ycen,rSun,peff,hemi_out=False):
    """Convert Stonyhurst lat,lon to xi,eta in
    heliocentric-cartesian coordinates.

    Params:
        lat,lon: latitude and longitude of desired pixel in radians.
        latc,lonc: latitude and longitude of disk center in radians.
        xcen,ycen: disk center in pixels.
        rSun: radius of Sun in arbitraty units.
        peff: p-angle
        hemi_out: whether or not to output hemisphere of farside.

    Returns:
        Coordinate on image (xi,eta) in units of rSun
        and hemisphere of farside (optional output).
    """
    # Correction of finite distance (1AU)
    sin_asd = 0.004660
    cos_asd = 0.99998914

    last_latc = 0.0
    cos_latc = 1.0
    sin_latc = 0.0

    if latc != last_latc:
        sin_latc = np.sin(latc)
        cos_latc = np.cos(latc)
        last_latc = latc

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    cos_lat_lon = cos_lat*np.cos(lon-lonc)

    cos_cang = sin_lat*sin_latc + cos_latc*cos_lat_lon
    if cos_cang < 0.0:
        hemisphere = 1
    else:
        hemisphere = 0

    r = rSun*cos_asd/(1.0 - cos_cang*sin_asd)
    xr = r*cos_lat*np.sin(lon - lonc)
    yr = r*(sin_lat*cos_latc - sin_latc*cos_lat_lon)

    cospa = np.cos(peff)
    sinpa = np.sin(peff)
    xi = xr*cospa - yr*sinpa
    eta = xr*sinpa + yr*cospa

    xi = xi + xcen
    eta = eta + ycen

    if hemi_out == True:
        return xi,eta,hemisphere
    else:
        return xi,eta

#find_cea_coord.pro
def find_cea_coord(header,phi_c,lambda_c,nx,ny,dx,dy):
    """Convert the index to CCD coordinate (xi,eta)"""
    nx = int(nx)
    ny = int(ny)

    # Array of CEA coords
    x = np.zeros((ny,nx))
    y = np.zeros((ny,nx))

    for i in range(nx):
        x[:,i] = np.radians((i-(nx-1)/2)*dx)
    for j in range(ny):
        y[j,:] = np.radians((j-(ny-1)/2)*dy)

    # Relevant header values
    rSun = header['rsun_obs']/header['cdelt1']     #solar radius in pixels
    disk_latc = np.radians(header['CRLT_OBS'])
    disk_lonc = np.radians(header['CRLN_OBS'])
    disk_xc = header['CRPIX1'] - 1                 #disk center wrt lower left of patch
    disk_yc = header['CRPIX2'] - 1
    pa = np.radians(header['CROTA2']*-1)

    latc = np.radians(lambda_c)
    lonc = np.radians(phi_c) - disk_lonc

    # Convert coordinates
    lat = np.zeros((ny,nx))
    lon = np.zeros((ny,nx))
    for i in range(nx):
        for j in range(ny):
            lat0,lon0 = plane2sphere(x[j,i],y[j,i],latc,lonc)
            lat[j,i] = lat0
            lon[j,i] = lon0

    xi = np.zeros((ny,nx))
    eta = np.zeros((ny,nx))
    for i in range(nx):
        for j in range(ny):
            xi0,eta0 = sphere2img(lat[j,i],lon[j,i],disk_latc,0.0,disk_xc,disk_yc,rSun,pa)
            xi[j,i] = xi0
            eta[j,i] = eta0

    return xi,eta,lat,lon

#img2heliovec.pro
def img2heliovec(bxImg,byImg,bzImg,lon,lat,lonc,latc,pAng):
    """Convert from image coordinates to Heliocentric spherical coordinates."""
    a11 = -np.sin(latc)*np.sin(pAng)*np.sin(lon - lonc) + np.cos(pAng)*np.cos(lon - lonc)
    a12 =  np.sin(latc)*np.cos(pAng)*np.sin(lon - lonc) + np.sin(pAng)*np.cos(lon - lonc)
    a13 = -np.cos(latc)*np.sin(lon - lonc)
    a21 = -np.sin(lat)*(np.sin(latc)*np.sin(pAng)*np.cos(lon - lonc) + np.cos(pAng)*np.sin(lon - lonc)) - np.cos(lat)*np.cos(latc)*np.sin(pAng)
    a22 =  np.sin(lat)*(np.sin(latc)*np.cos(pAng)*np.cos(lon - lonc) - np.sin(pAng)*np.sin(lon - lonc)) + np.cos(lat)*np.cos(latc)*np.cos(pAng)
    a23 = -np.cos(latc)*np.sin(lat)*np.cos(lon - lonc) + np.sin(latc)*np.cos(lat)
    a31 =  np.cos(lat)*(np.sin(latc)*np.sin(pAng)*np.cos(lon - lonc) + np.cos(pAng)*np.sin(lon - lonc)) - np.sin(lat)*np.cos(latc)*np.sin(pAng)
    a32 = -np.cos(lat)*(np.sin(latc)*np.cos(pAng)*np.cos(lon - lonc) - np.sin(pAng)*np.sin(lon - lonc)) + np.sin(lat)*np.cos(latc)*np.cos(pAng)
    a33 =  np.cos(lat)*np.cos(latc)*np.cos(lon - lonc) + np.sin(lat)*np.sin(latc)

    bxHelio = a11 * bxImg + a12 * byImg + a13 * bzImg
    byHelio = a21 * bxImg + a22 * byImg + a23 * bzImg
    bzHelio = a31 * bxImg + a32 * byImg + a33 * bzImg

    return bxHelio,byHelio,bzHelio

#prep_hd.pro
def prep_hd(header,phi_c,lambda_c,nx,ny,dx,dy):
    """Prepare header for CEA maps."""
    header_out = {}

    # Keywords to get from original header
    keys_hd = ['TELESCOP', 'INSTRUME', 'WAVELNTH', 'CAMERA','DATE',
               'DATE_S','DATE-OBS','T_OBS','T_REC','TRECEPOC',
               'TRECSTEP','TRECUNIT','HARPNUM','DSUN_OBS','DSUN_REF',
               'RSUN_REF','CRLN_OBS','CRLT_OBS','CAR_ROT','OBS_VR',
               'OBS_VW','OBS_VN','RSUN_OBS','QUALITY','QUAL_S','QUALLEV1']

    for key in keys_hd:
        header_out[key] = header[key]

    # Add new keywords
    header_out['NAXIS'] = 2
    header_out['NAXIS1'] = nx
    header_out['NAXIS2'] = ny

    header_out['CUNIT1'] = 'degree'
    header_out['CUNIT2'] = 'degree'

    header_out['CRPIX1'] = (nx - 1) / 2 + 1
    header_out['CRPIX2'] = (ny - 1) / 2 + 1
    header_out['CRVAL1'] = phi_c
    header_out['CRVAL2'] = lambda_c
    header_out['CDELT1'] = dx
    header_out['CDELT2'] = dy
    header_out['CTYPE1'] = 'CRLN-CEA'
    header_out['CTYPE2'] = 'CRLT-CEA'
    header_out['CROTA2'] = 0.0

    header_out['WCSNAME'] = 'Carrington Heliographic'
    header_out['BUNIT'] = 'Mx/cm^2'

    return header_out

#bvec2cea.pro
def bvec2cea(fld_path,inc_path,azi_path,disambig_path,do_disambig=True,
                phi_c=None,lambda_c=None, nx=None, ny=None,
                dx=None,dy=None,xyz=None):
    """Converts SHARP cutout vector field to CEA map.

    Params:
        fld_path: path to field file
        inc_path: path to inclination file
        azi_path: path to azimuth file
        disambig_path: path to disambig file
        do_disambig: If True perform disambiguation. Default=True

    Optional Params:
        phi_c: phi center coordinate (in degrees)
        lambda_c: lambda center coordinate (in degrees)
        nx: image size along x-axis
        ny: image size along y-axis
        dx: pixel size along x-axis
        dy: pixel size along y-axis

    Returns:
        Bp, Bt, and Br fits images
    """
    #Read in input data
    fld_map = sunpy.map.Map(fld_path)
    inc_map = sunpy.map.Map(inc_path)
    azi_map = sunpy.map.Map(azi_path)
    disamb_map = sunpy.map.Map(disambig_path)

    header = fld_map.meta
    field = fld_map.data
    inclination = np.radians(inc_map.data)
    azimuth = azi_map.data
    disambig = disamb_map.data

    if do_disambig == True:
        azimuth = np.radians(hmi_disambig(azimuth,disambig,2))
    else:
        azimuth = np.radians(azimuth)

    # Check that all images are the same shape
    if (field.shape != inclination.shape or field.shape != azimuth.shape):
        print('Input image sized do not match.')
        return

    # Check that output params exist in header
    keys = ['crlt_obs','crln_obs','crota2','rsun_obs','cdelt1',
            'crpix1','crpix2','LONDTMAX','LONDTMIN','LATDTMAX','LATDTMIN']
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

    try:
        maxlat = header['LATDTMAX']
        minlat = header['LATDTMIN']
    except KeyError:
        print('No y center')
        return

    # Check optional parameters and assign values if needed
    if phi_c == None:
        phi_c = ((maxlon + minlon) / 2.0) + header['CRLN_OBS']
    if lambda_c == None:
        lambda_c = (maxlat + minlat) / 2.0

    if dx == None:
        dx = 3e-2
    else:
        dx = np.abs(dx)
    if dy == None:
        dy = 3e-2
    else:
        dy = np.abs(dy)

    if nx == None:
        nx = int(np.around(np.around((maxlon - minlon) * 1e3)/1e3/dx))
    if ny == None:
    	ny = int(np.around(np.around((maxlat - minlat) * 1e3)/1e3/dy))

    # Find CCD native coords and Stonyhurst coords
    xi,eta,lat,lon = find_cea_coord(header,phi_c,lambda_c,nx,ny,dx,dy)

    bx_img = - field * np.sin(inclination) * np.sin(azimuth)
    by_img = field * np.sin(inclination) * np.cos(azimuth)
    bz_img = field * np.cos(inclination)

    # Get field in (xi,eta) coordinates
    bx_map = ndimage.map_coordinates(bx_img,[eta,xi])
    by_map = ndimage.map_coordinates(by_img,[eta,xi])
    bz_map = ndimage.map_coordinates(bz_img,[eta,xi])

    # Vector transformation
    disk_lonc = 0.0
    disk_latc = np.radians(header['crlt_obs'])
    pa = np.radians(header['crota2']*-1)

    bp,bt,br = img2heliovec(bx_map,by_map,bz_map,lon,lat,disk_lonc,disk_latc,pa)
    if xyz == None:
        bt *= -1.0

    header_map = prep_hd(header,phi_c,lambda_c,nx,ny,dx,dy)

    bp_map = sunpy.map.Map(bp, header_map)
    bt_map = sunpy.map.Map(bt, header_map)
    br_map = sunpy.map.Map(br, header_map)

    return bp_map,bt_map,br_map
