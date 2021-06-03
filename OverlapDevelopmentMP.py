import multiprocess
import geopandas
import shapely
import time
import numpy as np

def extendBound(bound,direction='up',method='nearestLeadingDigit',scale=10):
    '''Extend bound to next 'round' number
    
    Parameters
    ----------
    bound: float or float castable number or a list thereof
    direction: {'up','down',nonzero number} or a list of these values indicating the direction to round in
    method: str describing the extension method
        'nearestLeadingDigit': Bound is nearest numbers with leading digit followed by zeros
        'nearestPower': Bound is nearest integer power of scale (scale must be > 1).  For negative numbers, the sign and direction are reversed, the extension performed, then the sign of the result is reversed back.
        'nearestMultiple': Bound is nearest multiple of scale (scale must be > 0)
        'round': Bound is rounded using the default method
    scale: numeric as described in method options or a list thereof
    
    Returns
    -------
    float: the extended bound
    
    Notes
    -----
    All inputs, if not single-valued, must be list-castable and of equal length
    If all inputs are single-valued, the output is a float, otherwise it is a list of floats
    '''
    import numpy as np
    
    # Check and adjust the length of inputs
    unlist = False
    try:
        bound = list(bound)
    except:
        try:
            bound = [bound]
            unlist = True
        except:
            print("Input 'bound' must be numeric or convertible to list type.")
            return None
    try:
        if type(direction)==str:
            direction = [direction]
        direction = list(direction)
    except:
        try:
            direction = [direction]
        except:
            print("Input 'direction' must be a string or nonzero number or convertible to list type.")
            return None
    try:
        if type(method)==str:
            method = [method]
        method = list(method)
    except:
        try:
            method = [method]
        except:
            print("Input 'method' must be a string or convertible to list type.")
            return None
    try:
        scale = list(scale)
    except:
        try:
            scale = [scale]
        except:
            print("Input 'scale' must be numeric or convertible to list type.")
            return None
    inputs = [bound, direction, method, scale]
    lengths = [len(i) for i in inputs]
    set_lengths = set(lengths)
    max_len = max(set_lengths)
    set_lengths.remove(1)
    if len(set_lengths)>1:
        print('Inputs must be of the same length or of length one.  See help(extendBound)')
        return None
    if max_len>1: # can this be converted to a looped statement?
        if len(bound)==1:
            bound = bound*max_len
        if len(direction)==1:
            direction = direction*max_len
        if len(method)==1:
            method = method*max_len
        if len(scale)==1:
            scale = scale*max_len
        unlist = False

    # If multiple methods are specified, recursively call this function for each method and reassemble results
    if len(bound)>1 and len(set(method))>1:
        ret = np.array([None for b in bound])
        for m in list(set(method)):
            ind = np.where(np.array(method)==m)
            ret[ind] = extendBound(list(np.array(bound)[ind]),list(np.array(direction)[ind]),m,list(np.array(scale)[ind]))
        return list(ret)
    
    # Convert direction to a logical array roundup
    try:
        roundup = [True if d=='up' else False if d=='down' else True if float(d)>0 else False if float(d)<0 else None for d in direction]
    except:
        print('direction must be "up", "down", or a non-negative number')
        return None
    if any([r==None for r in roundup]):
        print('direction must be "up", "down", or a non-negative number')
        return None
    
    # Cases for multiple methods handled above, return to string method
    method = method[0]
    
    # Execute the conversions
    if method=='nearestLeadingDigit':
        iszero = np.array(bound)==0
        isnegative = np.array(bound) < 0
        offsets = np.logical_xor(roundup, isnegative)
        power = [0 if z else np.floor(np.log10(abs(b))) for b, z in zip(bound, iszero)]
        firstdigit = [abs(b)//np.power(10,p) for b, p in zip(bound, power)]
        exceeds = [abs(b)>f*np.power(10,p) for b, f, p in zip(bound, firstdigit, power)]
        newbound = [abs(b) if not t else (f+o)*np.power(10,p) for b, t, n, f, o, p in zip(bound, exceeds, isnegative, firstdigit, offsets, power)]
        newbound = [-n if t else n for n, t in zip(newbound,isnegative)]
    elif method=='nearestPower':
        try:
            scale = [float(s) for s in scale]
            if any([s<=1 for s in scale]):
                print('scale should be greater than 1')
                return None
        except ValueError:
            print('scale should be a number or list of numbers greater than 1')
            return None
        isnegative = np.array(bound) < 0
        offsets = np.logical_xor(roundup, isnegative)
        roundfuns = [np.ceil if o else np.floor for o in offsets]
        newbound = [0 if b==0 else np.power(s, r(np.log10(abs(b))/np.log10(s))) for b, r, s in zip(bound,roundfuns,scale)]
        newbound = [-n if t else n for n, t in zip(newbound,isnegative)]
    elif method=='nearestMultiple':
        try:
            scale = [float(s) for s in scale]
            if any([s<=0 for s in scale]):
                print('scale should be greater than 0')
                return None
        except ValueError:
            print('scale should be a number or list of numbers greater than 0')
            return None
        roundfuns = [np.ceil if r else np.floor for r in roundup]
        newbound = [s*(r(b/s)) for b, r, s in zip(bound,roundfuns,scale)]
    elif method=='round':
        roundfuns = [np.ceil if r else np.floor for r in roundup]
        newbound = [f(b) for b, f in zip(bound, roundfuns)]
    else:
        print('Invalid method, see help(extendBound)')
        return None
    return newbound[0] if unlist else newbound

def countCoordinates(geom):
    '''Count the total coordinates in the geometry
    
    Parameters
    ----------
    geom: shapely.geometry.Polygon or shapely.geometry.MultiPolygon (consisting only of shapely.geometry.Polygon componenet)
    
    Returns
    -------
    int: number of coordinates in the geometry
    
    Notes
    -----
    Each geometry element repeats the first point as the last point, and these are included in the count
    '''
    if geom.type=='MultiPolygon':
        count = [countCoordinates(g) for g in geom]
    elif geom.type=='Polygon':
        count = [len(geom.exterior.coords) + sum([len(i.coords) for i in geom.interiors])]
    else:
        print(f'Geometry is not strictly Polygon or MultiPolygon: {geom.type}')
    return sum(count)

def maxSubCoordinates(geom):
    '''Find the largest number of coordinates among components of geometry
    
    Parameters
    ----------
    geom: shapely.geometry.Polygon or shapely.geometry.MultiPolygon (consisting only of shapely.geometry.Polygon componenet)
    
    Returns
    -------
    int: number of coordinates in the largest geometry
    
    Notes
    -----
    Each geometry element repeats the first point as the last point, and these are included in the count
    '''
    if geom.type=='MultiPolygon':
        count = [countCoordinates(g) for g in geom]
    elif geom.type=='Polygon':
        count = [len(geom.exterior.coords) + sum([len(i.coords) for i in geom.interiors])]
    else:
        print(f'Geometry is not strictly Polygon or MultiPolygon: {geom.type}')
    return max(count)

def cleanGeometry(geom):
    '''Coerces geometry into a Polygon or MultiPolygon
    
    Selects only components of input shapely geometry that are non-empty
    Polygons or MultiPolygons and returns:
        an empty GeometryCollection if there are no selected elements
        the non-empty element if there is one selected element
        a multipolygon if there are multiple selected elements
    
    Requires shapely
    TODO: add case of nested GeometryCollections
    '''
    if geom.is_empty:
        return shapely.geometry.Polygon([])
    if geom.type in {'Polygon','MultiPolygon'}:
        return geom
    if not geom.type in {'GeometryCollection'}:
        return shapely.geometry.Polygon([])
    # A GeometryCollection contains only Points, LineStrings, or Polygons...
    polys = [g for g in list(geom) if g.type in {'Polygon','MultiPolygon'} and not g.is_empty]
    if len(polys)==0:
        return shapely.geometry.Polygon([])
    elif len(polys)==1:
        return polys[0]
    else:
        polys = [g if g.type=='Polygon' else list(g) for g in polys]
        return shapely.geometry.MultiPolygon(polys)
    return None # This should never be reached

def katana(geometry, threshold, count=0):
    '''Recursively split a geometry until the size of each component is below a threshold
    
    Parameters
    ----------
    geometry: shapely Polygon or MultiPolygon
    threshold: int>3, maximum number of coordinates in each returned geometry
    count: int>=0, internal variable tracking the recursion depth
    
    Returns
    -------
    list of shapely Polygons, the union of which comprises the input geometry, with each Polygon having at most threshold coordinates
    
    Notes
    -----
    Depends on shapely
    Recursion limit hard-coded to 250 (could be made dynamic with sys.getrecursionlimit()-3)
    Originally from # https://snorfalorpagus.net/blog/2016/03/13/splitting-large-polygons-for-faster-intersections/
        
    TODO
    ----
    Check that shapely.geometry.GeometryCollection will be handled properly
    
    
    License
    -------
    The following code is released under the BSD 2-clause license:

    Copyright (c) 2016, Joshua Arnott
    Modifications Copyright (c) 2021, Daniel Nezich

    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

        Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
        Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    '''
    bounds = geometry.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    ncoords = countCoordinates(geometry) # added DN 2021-05-12
    # if max(width, height) <= threshold or count == 250: # removed DN 2021-05-12
    if ncoords <= threshold or count == 250: # added DN 2021-05-12, can find the recursion limit and subtract 3 for equating with count with e.g.: import sys / print(sys.getrecursionlimit())
        # either the polygon is smaller than the threshold, or the maximum
        # number of recursions has been reached
        return [geometry] if isinstance(geometry, shapely.geometry.Polygon) else list(geometry)
    if height >= width:
        # split left to right
        a = shapely.geometry.box(bounds[0], bounds[1], bounds[2], bounds[1]+height/2)
        b = shapely.geometry.box(bounds[0], bounds[1]+height/2, bounds[2], bounds[3])
    else:
        # split top to bottom
        a = shapely.geometry.box(bounds[0], bounds[1], bounds[0]+width/2, bounds[3])
        b = shapely.geometry.box(bounds[0]+width/2, bounds[1], bounds[2], bounds[3])
    result = []
    for d in (a, b,):
        c = geometry.intersection(d)
        if not isinstance(c, shapely.geometry.GeometryCollection):
            c = [c]
        for e in c:
            if isinstance(e, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
                result.extend(katana(e, threshold, count+1))
    if count > 0:
        return result
    # convert multipart into singlepart
    final_result = [] # this will be a list of polygons
    for g in result:
        if isinstance(g, shapely.geometry.MultiPolygon):
            final_result.extend(g)
        else:
            final_result.append(g)
    return final_result

def katanaByParts(geom, vertex_limit=10000):
    '''Break geom into a list of shapely Polygons each with at most vertex_limit vertices.
    
    Parameters
    ----------
    geom: shapely Polygon or MultiPolygon
    vertex_limit: int>3, maximum number of vertices per output Polygon
    
    Returns
    -------
    list of shapely Polygon where each Polygon has at most vertex_limit vertices
    
    Notes
    -----
    The union of all returned Polygons is equal to the input geom.
    Cuts only the pieces of geom with vertex count greater than vertex_limit.
    Might be improved with an approach similar to intersectionBySjoinCut to avoid excessive looping and appending
    '''
    geomarr = [geom] if geom.type=='Polygon' else list(geom)
    newgeoms = []
    [newgeoms.extend(katana(g,vertex_limit)) if countCoordinates(g)>vertex_limit else newgeoms.append(g) for g in geomarr]
    return newgeoms

def intersectionBySjoin(geom1, geom2):
    '''Fast intersection of two geometries using spatial indexing.
    
    Expands each input geometry to an array of polygons, performs a
    spatial join to find potential overlaps between these polygons, finds
    the actual intersections between these polygons, and reassembles the full
    overlap by unary union.  Useful when the number of
    vertices in a geometry is very large (e.g. > 100k) but the number of
    vertices in each polygon comprising the geometry is moderate (e.g. <= 10k)
    and when performance degradation occurs with the standard shapely
    intersection.
    
    Parameters
    ----------
    geom1: shapely Polygon or MultiPolygon, or array of shapely Polygons
    geom2: shapely Polygon or MultiPolygon, or array of shapely Polygons
    
    Returns
    -------
    shapely Polygon or Multipolygon of the overlap of geom1 and geom2
    '''
    geom1arr = geom1 if type(geom1)==list else [geom1] if geom1.type=='Polygon' else list(geom1)
    geom2arr = geom2 if type(geom2)==list else [geom2] if geom2.type=='Polygon' else list(geom2)
    geom1gdf = geopandas.GeoDataFrame(geometry=geom1arr)
    geom2gdf = geopandas.GeoDataFrame(geometry=geom2arr)
    gdf_joined = geopandas.tools.sjoin(geom1gdf,geom2gdf,how='inner')
    gdf_joined['index_left'] = gdf_joined.index
    gdf_joined = gdf_joined.loc[:,['index_left','index_right']].reset_index(drop=True)
    gdf_joined['Overlap Geometry'] = geom1gdf.geometry.iloc[gdf_joined['index_left']].intersection(geom2gdf.geometry.iloc[gdf_joined['index_right']],align=False).values
    gdf_joined.set_geometry('Overlap Geometry',inplace=True)
    return shapely.ops.unary_union(gdf_joined.geometry)

def intersectGDF(gdf1, keyfield1, gdf2, keyfield2, areas_in=None, verbosity=1, area_epsg=6931, apply_buffer=False, threshold=None, vertex_limit=10000, check_inputs=True):
    '''Find all intersections between geometries in two geodataframes
    
    Parameters
    ----------
    gdf1: GeoDataFrame (must match crs of gdf2, and have valid geometries if apply_buffer is False)
    keyfield1: column name in gdf1 which uniquely identifies each row and will be used to label the results
    gdf2: GeoDataFrame (must match crs of gdf1, and have valid geometries if apply_buffer is False)
    keyfield2: column name in gdf2 which uniquely identifies each row and will be used to label the results
    areas_in: list of lists of overlap areas between geometries in gdf1 and gdf2, with dimensions [gdf2.shape[0]][gdf1.shape[0]], intersections will be calculated only for geometry pairs with nonzero entries (default None)
    verbosity: int, detail level of reporting during execution: 0=none, 1+=announce computation sections (default 1)
    area_epsg: int, convert to this epsg for area calculation (default 6931)
    apply_buffer: bool, use .buffer(0) to coerce valid geometries in gdf1 and gdf2 (often works, sometimes corrupts geometry) (default False)
    threshold: float or None, a float value discards overlaps that area less than threshold fraction of the largest overlap for each geometry in gdf2 (default None)
    vertex_limit: int>3, limit geometry elements to at most this number of vertices when performing internal calculations (default 10000)
    check_inputs: bool, set to False to circumvent input checks (crs match, geometry validity, areas_in shape) for speedup (default True)
    
    Returns
    -------
    GeoDataFrame: each row contains an overlap geometry between one row in gdf1 and one row in gdf2, with the following fields:
        keyfield1: entry from gdf1[keyfield1]
        keyfield1+' Index': index of row from gdf1
        keyfield2: entry from gdf2[keyfield2]
        keyfield2_' Index': index of row from gdf2
        'Overlap Geometry': intersection as shapely Polygon or MultiPolygon
        'Overlap Area': area of intersection geometry in units of area_epsg coordinate reference system
        'Overlap Ratio': fraction of area that this geometry represents out of all intersection geometries involving the source row in gdf2
    
    Notes
    -----
    Relies on geopandas, numpy as np, and time
    Returns None on input error after printing error description
    
    TODO:
    check that operation is correct with input gdf index not reset to range(0,N)
    change area_epsg logic to choose epsg based on input geometry location/extent (potentially cutting geometry to evaluate area in multiple crs where those sections are more valid)
    '''
    if verbosity>=1: print("Beginning overlap determination")
    begin_time = time.time()
    start_time = time.time()
    
    if check_inputs:
        # Ensure coordinate reference systems match
        if gdf1.crs!=gdf2.crs:
            print(f"Input coordinate reference systems must be the same: gdf1.crs={gdf1.crs}, gdf2.crs={gdf2.crs}")
            return None

        # Ensure geometries are valid
        n_invalid1 = gdf1.shape[0]-sum(gdf1.geometry.is_valid)
        n_invalid2 = gdf2.shape[0]-sum(gdf2.geometry.is_valid)
        if not apply_buffer and (n_invalid1>0 or n_invalid2>0):
            print(f"Input geometries must be valid: {n_invalid1} invalid geometries in gdf1, {n_invalid2} invalid geometries in gdf2")
            return None

        # Ensure areas_in is the proper size
        if not areas_in is None:
            N2 = len(areas_in)
            if N2!=gdf2.shape[0]:
                print(f"Input areas_in must have length {gdf2.shape[0]}, actual length is {N2}")
                return None
            N1 = len(areas_in[0])
            irreg = not all([len(a)==N1 for a in areas_in])
            if len(areas_in)!=gdf2.shape[0] or N1!=gdf1.shape[0] or irreg:
                print(f"Input areas_in must have size [{gdf2.shape[0]}][{gdf1.shape[0]}], actual shape is [{N2}][{N1}]{' with irregular second dimension length' if irreg else ''}")
                return None
    
    end_gmt = time.gmtime(time.time()-start_time)
    if verbosity>=1: print(f'{end_gmt.tm_yday-1}d{time.strftime("%H:%M:%S",end_gmt)}'+(' Input checks passed' if check_inputs else ' Input checks bypassed'))
    
    # Ensure we are computing with copies of the dataframes
    start_time = time.time()
    gdf1 = gdf1.copy(deep=True)
    gdf2 = gdf2.copy(deep=True)
    end_gmt = time.gmtime(time.time()-start_time)
    if verbosity>=1: print(f'{end_gmt.tm_yday-1}d{time.strftime("%H:%M:%S",end_gmt)}  Input dataframes copied')
    
    # Apply buffer to ensure geometries are valid (optional, avoids overlap topological errors, risks occasionally discarding portions of the geometry)
    if apply_buffer:
        start_time = time.time()
        gdf1.geometry = gdf1.geometry.buffer(0)
        end_gmt = time.gmtime(time.time()-start_time)
        if verbosity>=1: print(f'{end_gmt.tm_yday-1}d{time.strftime("%H:%M:%S",end_gmt)}  Polygon buffering for {keyfield1} completed')
        start_time = time.time()
        gdf2.geometry = gdf2.geometry.buffer(0)
        end_gmt = time.gmtime(time.time()-start_time)
        if verbosity>=1: print(f'{end_gmt.tm_yday-1}d{time.strftime("%H:%M:%S",end_gmt)}  Polygon buffering for {keyfield1} completed')
    
    if areas_in is None:
        # Perform a spatial join to find which geometries potentially overlap
        start_time = time.time()
        gdf_joined = geopandas.tools.sjoin(gdf1,gdf2,how='inner')
        end_gmt = time.gmtime(time.time()-start_time)
        if verbosity>=1: print(f'{end_gmt.tm_yday-1}d{time.strftime("%H:%M:%S",end_gmt)}  Spatial join completed, {gdf_joined.shape[0]} potential overlaps')
    else:
        # Use areas_in to construct potential overlaps
        start_time = time.time()
        gdf_joined = geopandas.GeoDataFrame(crs=gdf1.crs)
        tmp = np.nonzero(areas_in)
        gdf_joined[keyfield1] = gdf1.iloc[tmp[1],:][keyfield1].values
        gdf_joined[keyfield2] = gdf2.iloc[tmp[0],:][keyfield2].values
        gdf_joined.index = gdf1.iloc[tmp[1],:].index.values
        gdf_joined['index_right'] = gdf2.iloc[tmp[0],:].index.values
        end_gmt = time.gmtime(time.time()-start_time)
        if verbosity>=1: print(f'{end_gmt.tm_yday-1}d{time.strftime("%H:%M:%S",end_gmt)}  Interpretation of areas_in for overlaps completed, {gdf_joined.shape[0]} overlaps obtained')
    
    # Create a column indicating if the geometry needs to be cut and a column with the cut geometry (as array of polygons)
    start_time = time.time()
    gdf1['Do Cut'] = gdf1.geometry.apply(maxSubCoordinates)>vertex_limit
    gdf2['Do Cut'] = gdf2.geometry.apply(maxSubCoordinates)>vertex_limit
    gdf1['Cut Array Geometry'] = [katanaByParts(g,vertex_limit) if d else [g] if g.type=='Polygon' else list(g) for d, g in zip(gdf1['Do Cut'].values, gdf1.geometry.values)]
    gdf2['Cut Array Geometry'] = [katanaByParts(g,vertex_limit) if d else [g] if g.type=='Polygon' else list(g) for d, g in zip(gdf2['Do Cut'].values, gdf2.geometry.values)]
    end_gmt = time.gmtime(time.time()-start_time)
    if verbosity>=1: print(f'{end_gmt.tm_yday-1}d{time.strftime("%H:%M:%S",end_gmt)}  Geometry cutting completed, {gdf1["Do Cut"].sum()} in gdf1 and {gdf2["Do Cut"].sum()} in gdf2')
    
    # Format the resultant geodataframe for convenience in subsequent calculation
    start_time = time.time()
    indexfield1 = keyfield1+' Index'
    indexfield2 = keyfield2+' Index'
    gdf_joined[indexfield2] = gdf_joined['index_right']
    gdf_joined[indexfield1] = gdf_joined.index
    gdf_joined = gdf_joined.loc[:,[indexfield1,keyfield1,indexfield2,keyfield2]].reset_index(drop=True)
    end_gmt = time.gmtime(time.time()-start_time)
    if verbosity>=1: print(f'{end_gmt.tm_yday-1}d{time.strftime("%H:%M:%S",end_gmt)}  Output GeoDataFrame formatting completed')
    
    # Overlap Part 1: Calculate the actual overlap for each potential overlap, in sections based on the 'Do Cut' properties of the input GDFs
    start_time = time.time()
    indb_cut = gdf1.loc[gdf_joined[indexfield1],'Do Cut'].values | gdf2.loc[gdf_joined[indexfield2],'Do Cut'].values
    gdf_joined.loc[~indb_cut,'Overlap Geometry'] = gdf1.geometry.loc[gdf_joined.loc[~indb_cut,indexfield1]].intersection(gdf2.geometry.loc[gdf_joined.loc[~indb_cut,indexfield2]],align=False).values
    end_gmt = time.gmtime(time.time()-start_time)
    if verbosity>=1: print(f'{end_gmt.tm_yday-1}d{time.strftime("%H:%M:%S",end_gmt)}  Unmodified geomety overlaps calculated, {sum(~indb_cut)} total')
    
    # Overlap Part 2: Perform a sjoin intersection if cutting is involved (Polygon array input version)
    start_time = time.time()
    gdf_joined['Overlap Geometry'].iloc[np.nonzero(indb_cut)] = [intersectionBySjoin(g1,g2) for g1, g2 in zip(gdf1.loc[gdf_joined.loc[indb_cut,indexfield1],'Cut Array Geometry'],gdf2.loc[gdf_joined.loc[indb_cut,indexfield2],'Cut Array Geometry'])]
    gdf_joined.set_geometry('Overlap Geometry',inplace=True)
    end_gmt = time.gmtime(time.time()-start_time)
    if verbosity>=1: print(f'{end_gmt.tm_yday-1}d{time.strftime("%H:%M:%S",end_gmt)}  Cut geometry overlaps calculated, {sum(indb_cut)} total')
    
    # Overlap Cleanup: Remove empty and non-polygon geometries
    start_time = time.time()
    init_len = gdf_joined.shape[0]
    gdf_joined.geometry = [cleanGeometry(g) for g in gdf_joined.geometry]
    gdf_joined = gdf_joined.drop(np.where(gdf_joined.geometry.is_empty)[0]).reset_index(drop=True)
    end_gmt = time.gmtime(time.time()-start_time)
    if verbosity>=1: print(f'{end_gmt.tm_yday-1}d{time.strftime("%H:%M:%S",end_gmt)}  Empty overlaps removed, {init_len-gdf_joined.shape[0]} total')
        
    # Calculate overlap area
    start_time = time.time()
    gdf_joined['Overlap Area'] = gdf_joined.to_crs(epsg=area_epsg).area
    end_gmt = time.gmtime(time.time()-start_time)
    if verbosity>=1: print(f'{end_gmt.tm_yday-1}d{time.strftime("%H:%M:%S",end_gmt)}  Areas computed')
    
    # Downselect to relevant geometries
    if not threshold is None:
        start_time = time.time()
        a_ratio = gdf_joined['Overlap Area']/gdf_joined.groupby([keyfield2])['Overlap Area'].sum().loc[gdf_joined[keyfield2]].values
        ind_drop = np.where(a_ratio<threshold)[0]
        gdf_joined = gdf_joined.drop(index=ind_drop).reset_index(drop=True)
        end_gmt = time.gmtime(time.time()-start_time)
        if verbosity>=1: print(f'{end_gmt.tm_yday-1}d{time.strftime("%H:%M:%S",end_gmt)}  Area threshold applied, {gdf_joined.shape[0]} overlaps retained, {len(indb_cut)-gdf_joined.shape[0]} overlaps dropped')
    
    # Add area ratio column (ratio is with respect to all overlaps sharing the same keyfield2)
    gdf_joined[keyfield2+' Overlap Ratio'] = gdf_joined['Overlap Area']/gdf_joined.groupby([keyfield2])['Overlap Area'].sum().loc[gdf_joined[keyfield2]].values
    
    # Return results
    end_gmt = time.gmtime(time.time()-begin_time)
    if verbosity>=1: print(f'{end_gmt.tm_yday-1}d{time.strftime("%H:%M:%S",end_gmt)}  Total execution time upon completion')
    return gdf_joined, time.time()-begin_time