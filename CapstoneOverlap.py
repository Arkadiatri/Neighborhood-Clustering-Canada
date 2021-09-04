import multiprocess
import numpy
import pandas
import geopandas
import shapely
import time
import ipypb # Lightweight progress bar, source copied from GitHub
import dill
import gzip

__all__ = [
    'countCoordinates',
    'maxSubCoordinates',
    'cleanGeometry',
    'katana',
    'katanaByParts',
    'intersectionBySjoin',
    'intersectGDF',
    'intersectGDFMP',
    'convertToOldResults',
    'loadResults_',
    'loadResults',
    'saveResults_',
    'saveResults',
    'loadComputeSave'
]

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
        return None
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

def intersectGDF(gdf1, keyfield1, gdf2, keyfield2, areas_in=None, verbosity=1, area_epsg=None, apply_buffer=False, threshold=None, vertex_limit=10000, check_inputs=True):
    '''Find all intersections between geometries in two geodataframes
    
    Parameters
    ----------
    gdf1: GeoDataFrame (must match crs of gdf2, and have valid geometries if apply_buffer is False)
    keyfield1: column name in gdf1 which uniquely identifies each row and will be used to label the results
    gdf2: GeoDataFrame (must match crs of gdf1, and have valid geometries if apply_buffer is False)
    keyfield2: column name in gdf2 which uniquely identifies each row and will be used to label the results
    areas_in: list of lists of overlap areas between geometries in gdf1 and gdf2, with dimensions [gdf2.shape[0]][gdf1.shape[0]], intersections will be calculated only for geometry pairs with nonzero entries (default None)
    verbosity: int, detail level of reporting during execution: 0=none, 1+=announce computation sections (default 1)
    area_epsg: int or None, convert to this epsg for area calculation, None means use crs of gdf1/gdf2 (default None)
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
    Relies on geopandas, numpy, and time
    Returns None on input error after printing error description
    
    TODO:
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
        
        # Apply default area_epsg if needed
        if area_epsg is None:
            area_epsg = gdf1.crs.to_epsg()
    
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
        tmp = numpy.nonzero(areas_in)
        gdf_joined[keyfield1] = gdf1.iloc[tmp[1],:][keyfield1].values
        gdf_joined[keyfield2] = gdf2.iloc[tmp[0],:][keyfield2].values
        gdf_joined.index = gdf1.iloc[tmp[1],:].index.values
        gdf_joined['index_right'] = gdf2.iloc[tmp[0],:].index.values
        end_gmt = time.gmtime(time.time()-start_time)
        if verbosity>=1: print(f'{end_gmt.tm_yday-1}d{time.strftime("%H:%M:%S",end_gmt)}  Interpretation of areas_in for overlaps completed, {gdf_joined.shape[0]} overlaps obtained')
    
    # Create a column indicating if the geometry needs to be cut and a column with the cut geometry (as array of polygons)
    start_time = time.time()
    gdf1['Do Cut'] = gdf1.geometry.apply(countCoordinates)>vertex_limit
    gdf2['Do Cut'] = gdf2.geometry.apply(countCoordinates)>vertex_limit
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
    gdf_joined['Overlap Geometry'] = None
    gdf_joined['Overlap Geometry'].iloc[numpy.nonzero(~indb_cut)] = gdf1.geometry.loc[gdf_joined.loc[~indb_cut,indexfield1]].intersection(gdf2.geometry.loc[gdf_joined.loc[~indb_cut,indexfield2]],align=False).values # see note in Part 2
    end_gmt = time.gmtime(time.time()-start_time)
    if verbosity>=1: print(f'{end_gmt.tm_yday-1}d{time.strftime("%H:%M:%S",end_gmt)}  Unmodified geomety overlaps calculated, {sum(~indb_cut)} total')
    
    # Overlap Part 2: Perform a sjoin intersection if cutting is involved (Polygon array input version)
    start_time = time.time()
    gdf_joined['Overlap Geometry'].iloc[numpy.nonzero(indb_cut)] = [intersectionBySjoin(g1,g2) for g1, g2 in zip(gdf1.loc[gdf_joined.loc[indb_cut,indexfield1],'Cut Array Geometry'],gdf2.loc[gdf_joined.loc[indb_cut,indexfield2],'Cut Array Geometry'])] # Could not use loc for assignment because it results in a Series instead of a DataFrame when there is a single True value in indb_cut
    gdf_joined.set_geometry('Overlap Geometry',inplace=True)
    end_gmt = time.gmtime(time.time()-start_time)
    if verbosity>=1: print(f'{end_gmt.tm_yday-1}d{time.strftime("%H:%M:%S",end_gmt)}  Cut geometry overlaps calculated, {sum(indb_cut)} total')
    
    # Overlap Cleanup: Remove empty and non-polygon geometries
    start_time = time.time()
    init_len = gdf_joined.shape[0]
    gdf_joined.geometry = [cleanGeometry(g) for g in gdf_joined.geometry]
    # gdf_joined.geometry = [g if g.is_valid else g.buffer(0) for g in gdf_joined.geometry] # Dubious necessity... perform this cleanup externally with makeValidByBuffer(gdf) if needed
    gdf_joined = gdf_joined.drop(numpy.where(gdf_joined.geometry.is_empty)[0]).reset_index(drop=True)
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
        ind_drop = numpy.where(a_ratio<threshold)[0]
        gdf_joined = gdf_joined.drop(index=ind_drop).reset_index(drop=True)
        end_gmt = time.gmtime(time.time()-start_time)
        if verbosity>=1: print(f'{end_gmt.tm_yday-1}d{time.strftime("%H:%M:%S",end_gmt)}  Area threshold applied, {gdf_joined.shape[0]} overlaps retained, {len(indb_cut)-gdf_joined.shape[0]} overlaps dropped')
    
    # Add area ratio column (ratio is with respect to all overlaps sharing the same keyfield2)
    gdf_joined[keyfield2+' Overlap Ratio'] = gdf_joined['Overlap Area']/gdf_joined.groupby([keyfield2])['Overlap Area'].sum().loc[gdf_joined[keyfield2]].values
    
    # Return results
    end_gmt = time.gmtime(time.time()-begin_time)
    if verbosity>=1: print(f'{end_gmt.tm_yday-1}d{time.strftime("%H:%M:%S",end_gmt)}  Total execution time upon completion')
    return gdf_joined, time.time()-begin_time

def intersectGDFMP(gdf1, keyfield1, gdf2, keyfield2, areas_in=None, verbosity=1, area_epsg=6931, apply_buffer=False, threshold=None, vertex_limit=10000, check_inputs=True, num_process=None, num_block=None):
    '''Multiprocess overlap between geometries in two geodataframes
    
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
    num_process: int or None, number of processes to use for calculation (coerced up to 1 and down to the number of CPUs in the system, default is number of CPUs in the system minus two) (default None)
    num_block: int or None, number of rows of gdf2 to send to each process (coerced up to 1 and down to gdf2.shape[0], default is approximately gdf2.shape[0]/(3 times the number of CPUs in the system)) (default None)
    
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
    Relies on multiprocess, pandas, geopandas, numpy, and time
    Returns None on input error after printing error description
    '''
    if verbosity>=1: print('Executing intersectGDFMP')
    begin_time = time.time()

    # Input checks
    NCPU = multiprocess.cpu_count()
    N2 = gdf2.shape[0]

    if check_inputs:
        if num_process is None:
            num_process = max(1, NCPU-2) # empirical factor for circa 2020 multicore computers, prevents lockup by leaving processes available for the system and other applications
        else:
            try:
                num_process = int(num_process)
            except (ValueError, TypeError):
                print(f'Input Error: num_process must be an integer (there are {NCPU} CPUs detected, input was: {num_process})')
                return None
            if num_process < 0:
                tmp = num_process
                num_process = max(1, NCPU-num_process)
            else:
                num_process = min(NCPU, num_process)
        if verbosity>=1: print(f'  num_process set to {num_process} (of {NCPU})')

        if num_block is None:
            num_block = N2
            if num_process>1:
                num_block = max(1,num_block//(3*num_process)) # empirical factor based on Canadian census CFSA and DA catographic geometry overlap calculation
        else:
            try:
                num_block = int(num_block)
            except (ValueError, TypeError):
                print(f'Input Error: num_block must be an integer (input was: {num_block})')
                return None
            num_block = max(1,min(num_block,N2))
        if verbosity>=1: print(f'  num_block set to {num_block} (of {N2})')

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

        # Checks performed here, no need to do so in intersectGDF
        check_inputs = False 

    NDIV = N2//num_block+(0 if N2%num_block==0 else 1)
    ITERIND = lambda ind: slice(ind*num_block,((ind+1)*num_block if (ind+1)*num_block<N2 else N2))
    results = [None]*NDIV

    start_time = time.time()
    completed = 0
    with multiprocess.Pool(num_process) as pool:
        print(f'Generating pool, P={num_process}, N={N2}, DIV={num_block}, NDIV={NDIV}')
        ret = [pool.apply_async(intersectGDF,(gdf1,keyfield1,gdf2.iloc[ITERIND(ind),:],keyfield2,areas_in if areas_in is None else areas_in[ITERIND(ind)],verbosity,area_epsg,apply_buffer,threshold,vertex_limit,check_inputs)) for ind in range(NDIV)]
        print('Processing pool')
        for i in ipypb.track(range(NDIV)): # Alternative: initialize pb and call next(pb) in loop, instead of having a while loop to process all updates since last loop
            while True:
                indb_finished = [r.ready() for r in ret]
                indb_empty = [r is None for r in results]
                indb_update = [f and e for f, e in zip(indb_finished, indb_empty)]
                if any(indb_update):
                    ind_update = indb_update.index(True)
                    results[ind_update] = ret[ind_update].get(999) # Set finite timeout so that it returns properly
                    completed += 1
                    if completed%10==0 or completed==1 or completed==(NDIV):
                        print(f'Finished {completed}/{NDIV}, wall time {time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))}')
                    break
                time.sleep(1) # Make the infinite loop run slower; might be improved with explicit async

    wall_time = time.time()-start_time
    processor_time = sum([r[1] for r in results])
    gdf_joined = pandas.concat([r[0] for r in results]).sort_values(keyfield2,ignore_index=True)
    print(f'Pool processing concluded, process count {sum([r!=None for r in results])}/{NDIV}, wall time {time.strftime("%H:%M:%S", time.gmtime(wall_time))}, processor time {time.strftime("%H:%M:%S", time.gmtime(processor_time))}, processor:wall ratio {processor_time/wall_time:.3}x')
    print(f'Total time: {time.strftime("%H:%M:%S", time.gmtime(time.time()-begin_time))}')
    return gdf_joined

def convertToOldResults(gdf_joined):
    '''Convert gdf_joined to (gdf_union, areas) format for compatibility with older code'''
    indexfield1, keyfield1, indexfield2, keyfield2 = gdf_joined.columns[0:4]
    areas = np.zeros((len(gdf_joined[keyfield2].unique()),len(gdf_joined[keyfield1].unique())))
    for i, j, a in zip(gdf_joined[indexfield2],gdf_joined[indexfield1],gdf_joined['Overlap Area']):
        areas[i][j] = a
    areas = areas.tolist()
    gdf_union = gdf_joined.iloc[gdf_joined.groupby([keyfield2])['Overlap Area'].idxmax().values,:].reset_index(drop=True)
    gdf_union = gdf_union.sort_values(indexfield2).reset_index(drop=True)
    gdf_union = gdf_union[['Overlap Geometry',keyfield1,keyfield2]].rename_geometry('geometry')
    return gdf_union, areas

def loadResults_(name,tuples,fileformat='db',compress=False):
    '''Loads variables from files
    
    Parameters
    ----------
    name: str, file name base (including directory if desired)
    tuples: list of tuples (varname, suffix),
        varname: str, the key of the output dict where the data will be stored
        suffix: str, the string appended to name to generate a full file name
    fileformat: str, suffix to save the file with (do not include period)
    compress: bool, True to zip results (appends '.gz' to filename)
    
    Returns
    -------
    None if an error was encountered, or
    Tuple the length of tuples containing for each element of tuples:
        None if there was an error, or
        the variable loaded from file at the same position from tuples
    
    Notes
    -----
    Files read in binary format with optional gzip encoding
    This function is the complement to saveResults_()
    
    TODO
    ----
    Add option to change save format (text vs. binary)
    Make fileformat select the save format
    Remove n from tuples, it is not used as the return order is same as tuples
    '''
    if type(name)!=str:
        print('Error: name must be a string')
        return None
    if type(fileformat)!=str:
        print('Error: fileformat must be a string')
        return None
    
    ret = []
    for n, s in tuples:
        fn = name+s+'.'+fileformat+('.gz' if compress else '')
        try:
            with open(fn,'rb') as file:
                ret.append(dill.loads(gzip.decompress(file.read()) if compress else file.read()))
        except (FileNotFoundError, IOError) as e:
            ret.append(None)
            print(f'ERROR on file read: {e}')
    return tuple(ret)

def loadResults(name):
    '''Loads variables 'gdf_joined', 'gdf_union', and 'areas' from zipped files
    
    Parameters
    ----------
    name: str containing the base name of the files
    
    Returns
    -------
    None if an error was encountered, or
    Tuple the length of tuples containing:
        None if there was an error, or
        the variable loaded from file at the same position from tuples
    
    Notes
    -----
    File names area <name>_<variable>.db.gz and are in gzip dill binary format
    Include path in name
    '''
    tuples = [('gdf_joined','_joined'),
              ('gdf_union','_union'),
              ('areas','_areas')]
    
    return loadResults_(name,tuples,fileformat='db',compress=True)

def saveResults_(name,tuples,fileformat='db',compress=False):
    '''Saves variables to files
    
    Parameters
    ----------
    name: str, file name base (including directory if desired)
    tuples: list of tuples (varname, suffix),
        var: <any>, the variable to be output to file
        suffix: str, the string appended to name to generate a full file name
    fileformat: str, suffix to save the file with (do not include period)
    compress: bool, True to zip results (appends '.gz' to filename)
    
    Returns
    -------
    None if an error was encountered, or
    Tuple the same length as tuples containing return codes:
        0 Failure
        1 Success
    
    Notes
    -----
    Files written in binary format
    Files are created if they do not already exist
    Files are overwritten if they already exist

    TODO
    ----
    Make fileformat determine save format
    '''
    # Input checking
    if type(name)!=str:
        print('Error: name must be a string')
        return None
    if type(fileformat)!=str:
        print('Error: fileformat must be a string')
        return None
    
    # Make the directory if it does not exist
    path = os.path.dirname(name)
    if path!='':
        os.makedirs(path, exist_ok=True)
    
    ret = []
    for v, s in tuples:
        fn = name+s+'.'+fileformat+('.gz' if compress else '')
        try:
            with open(fn,'wb+') as file:
                file.write(gzip.compress(dill.dumps(v)) if compress else dill.dumps(v))
                ret.append(1)
        except IOError as e:
            ret.append(0)
            print(f'ERROR on file write: {e}')
    return tuple(ret)

def saveResults(name, gdf_joined, gdf_union, areas):
    '''Saves variables 'gdf_joined', gdf_union', and 'areas' to zipped files
    
    Parameters
    ----------
    name: str, file name base (including directory if desired)
    gdf_joined: geodataframe of all geographic overlaps, produced by intersectGDF()
    gdf_union: geodataframe of primary geographic overlaps, produced by convertToOldResults()
    areas: list of lists of overlap areas, produced by convertToOldResults()
    
    Returns
    -------
    None if an error was encountered, or
    Tuple the same length as tuples containing return codes:
        0 Failure
        1 Success
    
    Notes
    -----
    File names area <name><variable>.db.gz and are in gzip dill binary format
    Include path in name
    '''
    tuples = [(gdf_joined,'_joined'),
              (gdf_union,'_union'),
              (areas,'_areas')]
    
    return saveResults_(name,tuples,fileformat='db',compress=True)

def loadComputeSave(gdf1, key1, gdf2, key2, loadname=None, savename=None, verbosity=1, area_epsg=6931, apply_buffer=False, threshold=None):
    '''Returns the overlap of geometries, defaulting to file versions if possible
    
    Parameters
    ----------
    gdf1: GeoDataFrame (must match crs of gdf2, will be utilized for vectorized overlap calculation)
    keyfield1: column name in gdf1 which uniquely identifies each row and will be used to label the results
    gdf2: GeoDataFrame (must match crs of gdf1, will be iterated over for overlap calculation)
    keyfield2: column name in gdf2 which uniquely identifies each row and will be used to label the results
    loadname: str or None, base name of files to load data from (None -> 'DEFAULT'), see saveResults() (default None)
    savename: str or None, base name of files to save data to (None -> loadname), see loadResults() (default None)
    verbosity: int, detail level of reporting during execution: 0=none, 1=10-100 updates (default 1)
    area_epsg: int or None, convert to this epsg for area calculation, use None for crs of gdf1/gdf2 (default None)
    apply_buffer: bool, apply a zero buffer to each geometry to ensure validity(default False)
    threshold: float or None, a float value discards overlaps that area less than threshold fraction of the largest overlap for each geometry in gdf2 (default None)
    
    Returns
    -------
    gdf_joined: GeoDataFrame containing each overlap between gdf1 and gdf2 geometries as a row 
    gdf_union: GeoDataFrame containing columns of nonzero overlap geometries, corresponding gdf1[keyfield1], and corresponding gdf2[keyfield2], where only one value of gdf1[keyfield1] is selected which is the one with maximum overlap area
    areas: List of pandas Series of overlap areas; len(areas)=gdf2.shape[0], len(areas[i])=gdf1.shape[0]
    
    Notes
    -----
    gdf1 and gdf2 must be set to the same crs
    gdf1 and gdf2 must contain only valid geometries (use check_inputs=True to catch this case before calculation or apply_buffer=True to convert to valid geometries)
    '''
    verbosity = 1
    
    if savename is None:
        savename = loadname if not loadname is None else 'DEFAULT'
    
    ret = None if loadname is None else loadResults(loadname)
    recompute = False
    saveresults = False
    if ret is None:
        recompute = True
        saveresults = True
    else:
        gdf_joined, gdf_union, areas = ret
        if gdf_joined is None:
            if areas is None: # Recompute
                recompute = True
                saveresults = True
            else:             # Reconstruct from areas
                print("Overlaps will be recomputed based on loaded variable 'areas'")
                gdf_joined = intersectGDF(gdf1, key1, gdf2, key2, areas_in=areas, verbosity=verbosity, area_epsg=area_epsg, apply_buffer=apply_buffer, threshold=threshold, vertex_limit=10000, check_inputs=True)
                gdf_union, areas = convertToOldResults(gdf_joined)
                saveresults = True
        else:
            print("Overlaps loaded from file")

    if recompute:
        print("Overlaps must be computed")
        gdf_joined = intersectGDF(gdf1, key1, gdf2, key2, areas_in=None, verbosity=verbosity, area_epsg=area_epsg, apply_buffer=apply_buffer, threshold=threshold, vertex_limit=10000, check_inputs=True)
        gdf_union, areas = convertToOldResults(gdf_joined)
    
    if saveresults:
        saveResults(savename, gdf_joined, gdf_union, areas)
        print("Variables saved to file at "+savename)

    return gdf_joined, gdf_union, areas