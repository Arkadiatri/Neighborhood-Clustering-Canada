import multiprocess
import difflib

completed = 0

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

def intersectGDF(gdf1, keyfield1, gdf2, keyfield2, verbosity=1, area_epsg=6931, gdf1b=None, gdf2b=None):
    '''Find the overlap matrix of two geodataframe geometries
    
    Parameters
    ----------
    gdf1: GeoDataFrame (must match crs of gdf2, will be utilized for vectorized overlap calculation)
    keyfield1: column name in gdf1 which uniquely identifies each row and will be used to label the results
    gdf2: GeoDataFrame (must match crs of gdf1, will be iterated over for overlap calculation)
    keyfield2: column name in gdf2 which uniquely identifies each row and will be used to label the results
    verbosity: int, detail level of reporting during execution: 0=none, 1=10-100 updates, 2=update every loop and announce exceptions
    area_epsg: int, convert to this epsg for area calculation
    gdf1b: Buffered gdf1, to be used in case of failed overlap with gdf1, if None use gdf1.buffer(0)
    gdf2b: Buffered gdf2, to be used in case of failed overlap with gdf2, if None use gdf2.buffer(0)
    
    Returns
    -------
    gdf_union: Geodataframe containing columns of nonzero overlap geometries, corresponding gdf1[keyfield1], and corresponding gdf2[keyfield2],  here only one value of gdf1[keyfield1] is selected which is the one with maximum overlap area
    times: List of execution times for each overlap calculation; len(times)=gdf2.shape[0]
    areas: List of pandas Series of overlap areas; len(areas)=gdf2.shape[0], len(areas[i])=gdf1.shape[0], units given by area_epsg
    
    Notes
    -----
    gdf1 and gdf2 must be set to the same crs
    Iterates over gdf2, which should have the larger number of rows of {gdf1,gdf2} in order to minimize required memory (assuming geometries are of roughly equal size)
    Uses GeoDataFrame.buffer(0) to correct geometries
    '''
    # Initialize the return variables
    gdf_union = geopandas.GeoDataFrame()
    times = []
    areas = []

    # Ensure we are computing with copies of the dataframes (though this should not be necessary as gdf1, gdf2 are never updated)
    gdf1 = gdf1.copy(deep=True)
    gdf2 = gdf2.copy(deep=True)
    
    # Create new dataframes to hold buffered geometries
    #  Buffered geometries can often prevent topology errors during overlap calculation, at the risk of discarding portions of the geometry
    if gdf1b is None:
        start_time = time.time()
        gdf1b = gdf1.copy(deep=True)
        gdf1b['Geometry'] = gdf1b['Geometry'].buffer(0)
        if verbosity>=1: print(f'Polygon conversion for {keyfield1} completed in {time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))}')

    if gdf2b is None:
        start_time = time.time()
        gdf2b = gdf2.copy(deep=True)
        gdf2b['Geometry'] = gdf2b['Geometry'].buffer(0)
        if verbosity>=1: print(f'Polygon conversion for {keyfield2} completed in {time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))}')

    exceptioncount = 0
    start_time = time.time()
    verbosecount = 1 if verbosity>=2 else max(1,extendBound(gdf2.shape[0]/100,'up','nearestPower',10)) # Display update message after a power of 10 computations such that the total number of updates is between 10 and 100
    for i in range(gdf2.shape[0]):
        loop_start = time.time()
        try:
            gdf_tmp = gdf1['Geometry'].intersection(gdf2['Geometry'].iloc[i]) # Attempt using original geometries
        except (shapely.errors.TopologicalError): # This sometimes occurs
            if verbosity>=2: print(f'Handling exception at index {i}')
            exceptioncount += 1
            gdf_tmp = gdf1b['Geometry'].intersection(gdf2b['Geometry'].iloc[i]) # Use fallback buffered geometries

        areas.append(gdf_tmp.to_crs(epsg=area_epsg).area)
        ind = np.argmax(areas[-1]) # The FSA boundaries respect DA boundaries according to the , so there is just one FSA associated with each DA, which should have significantly larger area than any other.
        gdf_tmp = geopandas.GeoSeries(gdf_tmp.iloc[ind],crs=gdf1['Geometry'].crs)
        gdf_tmp = geopandas.GeoDataFrame(geometry=gdf_tmp,crs=gdf_tmp.crs)
        gdf_tmp[keyfield1] = gdf1[keyfield1].iloc[ind]
        gdf_tmp[keyfield2] = gdf2[keyfield2].iloc[i]
        gdf_union = gdf_union.append(gdf_tmp,ignore_index=True)
        loop_end = time.time()
        times.append(loop_end-loop_start)
        if verbosity>=1 and (i%verbosecount==0 or i+1==gdf2.shape[0]):
            print(f'Processed row {i+1}/{gdf2.shape[0]}, {gdf_tmp.shape[0]} overlap found in {loop_end-loop_start:.1f} sec, {gdf_union.shape[0]} overlaps total in {time.strftime("%H:%M:%S", time.gmtime(loop_end-start_time))}, total exceptions {exceptioncount}')
    if verbosity>=1: print('Overlap processing complete')
    return gdf_union, times, areas

def intersectGDFareas(gdf1, keyfield1, gdf2, keyfield2, areas_in=None, verbosity=1, area_epsg=6931, gdf1b=None, gdf2b=None):
    '''Find the overlap matrix of two geodataframe geometries
    
    Parameters
    ----------
    gdf1: GeoDataFrame (must match crs of gdf2, will be utilized for vectorized overlap calculation)
    keyfield1: column name in gdf1 which uniquely identifies each row and will be used to label the results
    gdf2: GeoDataFrame (must match crs of gdf1, will be iterated over for overlap calculation)
    keyfield2: column name in gdf2 which uniquely identifies each row and will be used to label the results
    areas_in: list of lists of overlap areas between geometries in gdf1 and gdf2, with dimensions [gdf2.shape[0]][gdf1.shape[0]]
    verbosity: int, detail level of reporting during execution: 0=none, 1=10-100 updates, 2=update every loop and announce exceptions
    area_epsg: int, convert to this epsg for area calculation
    gdf1b: Buffered gdf1, to be used in case of failed overlap with gdf1, if None use gdf1.buffer(0)
    gdf2b: Buffered gdf2, to be used in case of failed overlap with gdf2, if None use gdf2.buffer(0)
    
    Returns
    -------
    gdf_union: Geodataframe containing columns of nonzero overlap geometries, corresponding gdf1[keyfield1], and corresponding gdf2[keyfield2], where only one value of gdf1[keyfield1] is selected which is the one with maximum overlap area
    times: List of execution times for each overlap calculation; len(times)=gdf2.shape[0]
    areas: List of pandas Series of overlap areas; len(areas)=gdf2.shape[0], len(areas[i])=gdf1.shape[0]
    
    Notes
    -----
    gdf1 and gdf2 must be set to the same crs
    Iterates over gdf2, which should have the larger number of rows of {gdf1,gdf2} in order to minimize required memory (assuming geometries are of roughly equal size)
    Uses GeoDataFrame.buffer(0) to correct geometries
    areas must be supplied, 
    '''
    # Initialize the return variables
    gdf_union = geopandas.GeoDataFrame()
    times = []
    areas = []

    # Ensure we are computing with copies of the dataframes (though this should not be necessary as gdf1, gdf2 are never updated)
    gdf1 = gdf1.copy(deep=True)
    gdf2 = gdf2.copy(deep=True)
    
    # Create new dataframes to hold buffered geometries
    #  Buffered geometries can often prevent topology errors during overlap calculation, at the risk of discarding portions of the geometry
    if gdf1b is None:
        start_time = time.time()
        gdf1b = gdf1.copy(deep=True)
        gdf1b['Geometry'] = gdf1b['Geometry'].buffer(0)
        if verbosity>=1: print(f'Polygon conversion for {keyfield1} completed in {time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))}')

    if gdf2b is None:
        start_time = time.time()
        gdf2b = gdf2.copy(deep=True)
        gdf2b['Geometry'] = gdf2b['Geometry'].buffer(0)
        if verbosity>=1: print(f'Polygon conversion for {keyfield2} completed in {time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))}')

    exceptioncount = 0
    start_time = time.time()
    verbosecount = 1 if verbosity>=2 else max(1,extendBound(gdf2.shape[0]/100,'up','nearestPower',10)) # Display update message after a power of 10 computations such that the total number of updates is between 10 and 100
    for i in range(gdf2.shape[0]):
        loop_start = time.time()
        gdf1ind = np.array(areas_in[i])>0 # ADDED for areas
        try:
            gdf_tmp = gdf1.loc[gdf1ind,'Geometry'].intersection(gdf2['Geometry'].iloc[i]) # Attempt using original geometries # ADDED for areas ".loc[gdf1ind," # NOTE: relies on gdf1
        except (shapely.errors.TopologicalError): # This sometimes occurs
            if verbosity>=2: print(f'Handling exception at index {i}')
            exceptioncount += 1
            gdf_tmp = gdf1b.loc[gdf1ind,'Geometry'].intersection(gdf2b['Geometry'].iloc[i]) # Use fallback buffered geometries
            
        areas_tmp = np.zeros(gdf1.shape[0])
        areas_tmp[gdf1ind] = gdf_tmp.to_crs(epsg=area_epsg).area
        areas.append(areas_tmp)
        ind = np.argmax(gdf_tmp.to_crs(epsg=area_epsg).area) # CHANGED for areas (areas[-1]) # The FSA boundaries respect DA boundaries according to the , so there is just one FSA associated with each DA, which should have significantly larger area than any other.
        gdf_tmp = geopandas.GeoSeries(gdf_tmp.iloc[ind],crs=gdf1['Geometry'].crs)
        gdf_tmp = geopandas.GeoDataFrame(geometry=gdf_tmp,crs=gdf_tmp.crs)
        gdf_tmp[keyfield1] = gdf1[keyfield1].iloc[np.where(gdf1ind)[0][ind]] # ADDED for areas "np.where(gdf1ind)[ind]," to replace [ind]
        gdf_tmp[keyfield2] = gdf2[keyfield2].iloc[i]
        gdf_union = gdf_union.append(gdf_tmp,ignore_index=True)
        loop_end = time.time()
        times.append(loop_end-loop_start)
        if verbosity>=1 and (i%verbosecount==0 or i+1==gdf2.shape[0]):
            print(f'Processed row {i+1}/{gdf2.shape[0]}, {gdf_tmp.shape[0]} overlap found in {loop_end-loop_start:.1f} sec, {gdf_union.shape[0]} overlaps total in {time.strftime("%H:%M:%S", time.gmtime(loop_end-start_time))}, total exceptions {exceptioncount}')
    if verbosity>=1: print('Overlap processing complete')
    return gdf_union, times, areas

def getDiff(in1,in2):
    print('starting getDiff')
    import difflib
    ret = difflib.ndiff(in1.to_wkt().split(','), in2.to_wkt().split(','))
    ret = list(ret)
    #ompleted += 1
    #print('Completed {completed}/{gdf1.shape[0]}')
    return ret

def getDiff2(in1,in2,dispind):
    print('starting getDiff')
    import difflib
    import time
    start_time = time.time()
    ret = difflib.ndiff(in1.to_wkt().split(','), in2.to_wkt().split(','))
    ret = list(ret)
    calc_time = time.time()-start_time
    return (ret, dispind, calc_time)

def dmpDiff(in1,in2,dispind):
    import time
    import diff_match_patch as dmp_module
    start_time = time.process_time()
    
    dmp = dmp_module.diff_match_patch()
    diff = dmp.diff_main(in1.to_wkt(),in2.to_wkt())
    diff_time = time.process_time()
    
    dmp.diff_cleanupSemantic(diff)
    clean_time = time.process_time()

    return (diff, dispind, (diff_time-start_time, clean_time-diff_time))

def dmpDiffLine(in1,in2,dispind):
    import time
    import diff_match_patch as dmp_module
    
    start_time = time.process_time()

    fn_toline = lambda s: s.replace(',','\n')
    fn_tostr = lambda s: [ss.replace('\n',',') for ss in s]
    
    text1 = fn_toline(in1.to_wkt())
    text2 = fn_toline(in2.to_wkt())
    
    dmp = dmp_module.diff_match_patch()
    a = dmp.diff_linesToChars(text1, text2)
    lineText1 = a[0]
    lineText2 = a[1]
    lineArray = a[2]
    diff = dmp.diff_main(lineText1, lineText2, False)
    lineArray = fn_tostr(lineArray)
    dmp.diff_charsToLines(diff, lineArray)
    
    print(diff)
    
    diff_time = time.process_time()
    
    #dmp.diff_cleanupSemantic(diff)
    
    clean_time = time.process_time()

    return (diff, dispind, (diff_time-start_time, clean_time-diff_time))


'''
Polite spawning of threads that reinitialize components that may otherwise be shared (e.g. logging):

from multiprocessing import get_context

def your_func():
    with get_context("spawn").Pool() as pool:
        # ... everything else is unchanged
'''

'''
Line profiling
https://mortada.net/easily-profile-python-code-in-jupyter.html

@profile
def slow_function(a, b, c):

Instead of needing to decorate, use this:
%load_ext line_profiler

Example of single line:
%timeit get_farthest(trace, origin)

Use a line profile to examine calls within another
%lprun -f get_farthest get_farthest(trace, origin)
%lprun -f get_distances get_farthest(trace, origin)
%lprun -f haversine get_farthest(trace, origin)
'''

def getDiffs():
    completed = 0
    results = []
    NUM_PROCESSES = 10
    print('in getDiffs')
    with multiprocess.Pool(NUM_PROCESSES) as pool:
        print('begin multiprocess')
        ret = [pool.apply_async(getDiff,(gdf1['Geometry'].iloc[ind],gdf1b['Geometry'].iloc[ind])) for ind in range(gdf1.shape[0])]
        print('multiprocess defined')
        for r in ret:
            print('result agglomeration')
            results.append(r.get())
            print('end result agglomeration item')
        return results