import numpy as np
import sunpy.map
from sunpy.coordinates import frames
from astropy.coordinates import SkyCoord
import astropy.units as u

#Translate coordinate transformation code from IDL to Python

#hmi_disambig.pro
def hmi_disambig(azimuth,disambig,method):
    """Combine HMI disambiguation result with azimuth.

    Params:
        azimuth: a numpy array. dtype float64
        disambig: a numpy array. dtype int16
        method: int corresponding to the disambig method

    Returns:
        Azimuth array with disambiguation result applied.
    """
    #Check if azimuth and disambig have the same dimensions
    if azimuth.shape != disambig.shape:
        print('Dimensions of two images do not agree.')

    #Check method
    if (method < 0 or method >2):
        method = 2
        print('Invalid disambiguation method. Set to default method = 2')

    #Perform disambiguation
    disambig = disambig/(2**method)
    index = np.where(disambig % 2 != 0)
    azimuth[index] = azimuth[index] + 180.0

    return azimuth

#hmi_b2ptr.pro
def hmi_b2ptr(field_map,bvec,header,lonlat_out=False):
    """Convert HMI vector field in native components
    (field, inclination,azimuth w.r.t. plane of sky)
    into spherical coordinates (zonal Bp, meridional Bt, radial Br).
    Written for HMI full disk and SHARP data.

    Params:
        bvec: a 3D array containing the field, inclination, and azimuth
        header: the header information

    Returns:
        The transformed HMI vector field. If lonlat is set to True,
        the Stonyhurst longitude and latitude will also be returned.
    """
    sz = bvec.shape
    nx = sz[2]
    ny = sz[1]
    nz = sz[0]

    if (nx != header['naxis1'] or ny != header['naxis2'] or nz != 3):
        print('Incorrect input data dimensions.')

    #Convert bvec to B_xi, B_eta, B_zeta
    field = bvec[0]
    gamma = np.radians(bvec[1])
    psi = np.radians(bvec[2])

    b_xi = -field * np.sin(gamma) * np.sin(psi)
    b_eta = field * np.sin(gamma) * np.cos(psi)
    b_zeta = field * np.cos(gamma)

    #Get helioprojective coordinates from pixel coordinates
    xpx, ypx = np.meshgrid(*[np.arange(v.value) for v in field_map.dimensions]) * u.pix
    hpc_coords = field_map.pixel_to_world(xpx, ypx)

    #Get Stonyhurst longitude/latitude in radians from helioprojective coordinates
    hg_coords = hpc_coords.transform_to(frames.HeliographicStonyhurst)
    phi = hg_coords.lon.to(u.rad)
    lam = hg_coords.lat.to(u.rad)

    lonlat = np.empty((2,ny,nx))
    lonlat[0] = phi
    lonlat[1] = lam

    #Get conversion matrix, according to Eq (1) in Gary & Hagyard (1990)
    #See Eq (7)(8) in Sun (2013) for implementation
    b = np.radians(header['crlt_obs'])      #b-angle, disk center latitude
    p = np.radians(-header['crota2'])       #p-angle, negative of CROTA2

    k11 = np.cos(lam) * (np.sin(b) * np.sin(p) * np.cos(phi) + np.cos(p) * np.sin(phi)) - np.sin(lam) * np.cos(b) * np.sin(p)
    k12 = -np.cos(lam) * (np.sin(b) * np.cos(p) * np.cos(phi) - np.sin(p) * np.sin(phi)) + np.sin(lam) * np.cos(b) * np.cos(p)
    k13 = np.cos(lam) * np.cos(b) * np.cos(phi) + np.sin(lam) * np.sin(b)
    k21 = np.sin(lam) * (np.sin(b) * np.sin(p) * np.cos(phi) + np.cos(p) * np.sin(phi)) + np.cos(lam) * np.cos(b) * np.sin(p)
    k22 = -np.sin(lam) * (np.sin(b) * np.cos(p) * np.cos(phi) - np.sin(p) * np.sin(phi)) - np.cos(lam) * np.cos(b) * np.cos(p)
    k23 = np.sin(lam) * np.cos(b) * np.cos(phi) - np.cos(lam) * np.sin(b)
    k31 = -np.sin(b) * np.sin(p) * np.sin(phi) + np.cos(p) * np.cos(phi)
    k32 = np.sin(b) * np.cos(p) * np.sin(phi) + np.sin(p) * np.cos(phi)
    k33 = -np.cos(b) * np.sin(phi)

    #Output, (Bp,Bt,Br) is identical to (Bxh, -Byh, Bzh)
    #in Gary & Hagyard (1990), see Appendix in Sun (2013)

    bptr = np.empty((3,ny,nx))

    bptr[0] = k31 * b_xi + k32 * b_eta + k33 * b_zeta
    bptr[1] = k21 * b_xi + k22 * b_eta + k23 * b_zeta
    bptr[2] = k11 * b_xi + k12 * b_eta + k13 * b_zeta

    return bptr
    if lonlat_out:
        return bptr,lonlat
