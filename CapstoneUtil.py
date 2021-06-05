import numpy
import geopandas
import ogr
import shapely
import time
import folium

__all__ = [
    'GMLtoGDF_CA',
    'fillCity',
    'addToCity',
    'selectRegion',
    'getGDFBounds',
    'updateCity',
    'getCityByName',
    'getCityGroup',
    'getCityBounds',
    'extendBound',
    'extendBounds',
    'mapCitiesAdjacent',
    'makeValidByBuffer',
    'makeValidByOGR'
    'normalizeRows',
    'thresholdAreas',
    'normalizeAreas',
    'maximumAreas',
    'mergeBounds',
    'boundsCenter',
    'swapXY'
]

def GMLtoGDF_CA(filename):
    gdf = geopandas.read_file(filename)
    gdf = gdf.set_crs(epsg=3347) # Needed only for FSA file, the others are 3347 and parsed correctly by geopandas, and the pdf in the zip file has the same projection parameters (FSA vs. DA, ADA, CT)
    gdf['Area'] = gdf.geometry.to_crs(epsg=6931).area # Equal-area projection # MODIFY THIS to account for validity regions of each geometry
    gdf['Centroid'] = gdf.geometry.centroid
    gdf['Centroid'] = gdf['Centroid'].to_crs(epsg=4326) # Only the set geometry is converted with gdf.to_crs(); all other geometry-containing columns must be converted explicitly; here we convert all columns explicitly
    gdf['Centroid Latitude'] = gdf['Centroid'].geometry.y
    gdf['Centroid Longitude'] = gdf['Centroid'].geometry.x
    gdf.drop(columns = 'Centroid', inplace=True) # Because WKT Point cannot be serialized to JSON, we drop the Centroid column and keep only its float components
    return gdf

def fillCity(city, gdf, fieldname, method='equals', names=None):
    '''Populates the city dict using data from gdf, selection method, and names
    
    Expects city keys: name, group
    Populates city keys: gdf, geojson, data, centroid, bounds, selection
        geojson contains keys DAUID, Geometry, Area
        centroid and bounds use [latitude,longitude] format
    
    fieldname must be a column name in gdf used for selection
    method determines how the fieldname entry is used for row selection, with respect to the city name
        'equals': names is None (uses city name) or a list of strings at least one of which must match fieldname exactly
        'contains': names is None (uses city name) or a list of strings at least one of which must appear in fieldname
    
    Will overwrite existing entries
    '''
    if names==None:
        names = [city['name']]
    gdf_new = selectRegion(gdf, fieldname, method, names)
    updateCity(city, gdf_new)
    city['selection'] = f'{fieldname} {method} {names}'

def addToCity(city,gdf):
    '''Augments the city dict with gdf
    
    Expects city keys: name, group
    Populates city keys: gdf, geojson, data, centroid, bounds, selection
        geojson contains keys DAUID, Geometry, Area
        centroid and bounds use [latitude,longitude] format
    
    Adds to existing entries
    '''
    gdf_new = city['gdf'].append(gdf).drop_duplicates()
    updateCity(city, gdf_new)
    city['selection'] = city['selection'] + ', with manual addition'
    
def selectRegion(gdf, fieldname, method='equals', names=None):
    '''Selects rows from gdf using method to compare entries in fieldname to names
    
    fieldname must be a column name in gdf used for selection
    method determines how the fieldname entry is used for row selection, with respect to the city name
        'equals': names is None (uses city name) or a list of strings at least one of which must match fieldname exactly
        'contains': names is None (uses city name) or a list of strings at least one of which must appear in fieldname
    '''
    if names==None:
        print('error: names must be a string or list of strings')
        return None
    if not type(names)==list:
        if not type(names)==str:
            print('error: names must be a string or list of strings')
            return None
        names = [names]
    
    select = False
    for name in names:
        if method=='equals':
            select = select | (gdf[fieldname]==name)
        elif method=='contains':
            select = select | gdf[fieldname].str.contains(name)
        else:
            print("error: method must be 'equals' or 'contains'.")
            return
    return gdf.loc[select,:]
    
def getGDFBounds(gdf):
    '''Calculates the bounds of all geometries in gdf'''
    # tmp = gdf.geometry.unary_union.envelope.boundary.coords.xy
    # bounds = [[min(tmp[1]),min(tmp[0])],[max(tmp[1]),max(tmp[0])]]
    # return bounds
    tmp = gdf.geometry.total_bounds
    return [[tmp[1],tmp[0]],[tmp[3],tmp[2]]]

def updateCity(city, gdf):
    '''Populates the city dict using data from gdf
    
    Expects city keys: name, group
    Populates city keys: gdf, geojson, data, centroid, bounds
        geojson contains gdf keys except Centroid
        centroid and bounds use [latitude,longitude] format
    
    Will overwrite existing entries
    '''
    city['gdf'] = gdf
    city['geojson'] = gdf.to_crs(epsg=4326).to_json() # Folium requires this coordinate reference system
    #city['data'] = gdf.drop(columns=['Geometry']).copy(deep=True)
    tmp = gdf.to_crs(epsg=4326).geometry.unary_union # Folium requires this coordinate reference system
    city['centroid'] = [tmp.centroid.y,tmp.centroid.x]
    tmp = tmp.envelope.boundary.coords.xy
    city['bounds'] = [[min(tmp[1]),min(tmp[0])],[max(tmp[1]),max(tmp[0])]]

def getCityByName(cityname):
    '''Returns the first city dict that matches cityname exactly'''
    for city in cities:
        if city['name']==cityname:
            return city
    return None

def getCityGroup(groupname):
    '''Returns the subset of cities where group matches groupname exactly'''
    ret = []
    for city in cities:
        if city['group']==groupname:
            ret.append(city)
    return ret

def getCityBounds(cities, featurename):
    '''Returns the global bounds of column featurename across all cities
    
    Parameters
    ----------
    cities: dict or list of dicts as defined above
    featurename: str column name in cities[i]['gdf'] for which bounds will be obtained
    
    Returns
    -------
    (min, max) value across all cities
    '''
    if type(cities)==dict:
        cities = [cities]
    bounds = [[],[]]
    for city in cities:
        bounds[0].append(city['gdf'][featurename].min())
        bounds[1].append(city['gdf'][featurename].max())
    bounds[0] = min(bounds[0])
    bounds[1] = max(bounds[1])
    return bounds

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
        ret = numpy.array([None for b in bound])
        for m in list(set(method)):
            ind = numpy.where(numpy.array(method)==m)
            ret[ind] = extendBound(list(numpy.array(bound)[ind]),list(numpy.array(direction)[ind]),m,list(numpy.array(scale)[ind]))
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
        iszero = numpy.array(bound)==0
        isnegative = numpy.array(bound) < 0
        offsets = numpy.logical_xor(roundup, isnegative)
        power = [0 if z else numpy.floor(numpy.log10(abs(b))) for b, z in zip(bound, iszero)]
        firstdigit = [abs(b)//numpy.power(10,p) for b, p in zip(bound, power)]
        exceeds = [abs(b)>f*numpy.power(10,p) for b, f, p in zip(bound, firstdigit, power)]
        newbound = [abs(b) if not t else (f+o)*numpy.power(10,p) for b, t, n, f, o, p in zip(bound, exceeds, isnegative, firstdigit, offsets, power)]
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
        isnegative = numpy.array(bound) < 0
        offsets = numpy.logical_xor(roundup, isnegative)
        roundfuns = [numpy.ceil if o else numpy.floor for o in offsets]
        newbound = [0 if b==0 else numpy.power(s, r(numpy.log10(abs(b))/numpy.log10(s))) for b, r, s in zip(bound,roundfuns,scale)]
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
        roundfuns = [numpy.ceil if r else numpy.floor for r in roundup]
        newbound = [s*(r(b/s)) for b, r, s in zip(bound,roundfuns,scale)]
    elif method=='round':
        roundfuns = [numpy.ceil if r else numpy.floor for r in roundup]
        newbound = [f(b) for b, f in zip(bound, roundfuns)]
    else:
        print('Invalid method, see help(extendBound)')
        return None
    return newbound[0] if unlist else newbound

def extendBounds(bounds,method='nearestLeadingDigit',scale=10):
    if bounds[0]>bounds[1]:
        print('bounds must be ordered from least to greatest')
        return None    
    return extendBound(bounds,direction=['down','up'],method=method,scale=scale)

def mapCitiesAdjacent(cities,propertyname,title='',tooltiplabels=None):
    '''Displays cities in separate adjacent maps
    
    Cities dict as above
    Displayed as choropleth keyed to propertyname
    Header displays title and colormap labeled with keyed propertyname
    Tooltips pop up on mouseover showing properties listed in tooltiplabels
    
    Inspired by https://gitter.im/python-visualization/folium?at=5a36090a03838b2f2a04649d
    
    Assumes propertyname is identical in gdf and geojson
    '''
    if (not type(cities)==list) and type(cities)==dict:
        cities = [cities]
    
    f = bre.Figure()
    div_header = bre.Div(position='absolute',height='10%',width='100%',left='0%',top='0%').add_to(f)

    map_header = folium.Map(location=[0,0],control_scale=False,zoom_control=False,tiles=None,attr=False).add_to(div_header)
    div_header2 = bre.Div(position='absolute',height='10%',width='97%',left='3%',top='0%').add_to(div_header)
    html_header = '''<h3 align="left" style="font-size:16px;charset=utf-8"><b>{}</b></h3>'''.format(title)
    div_header2.get_root().html.add_child(folium.Element(html_header))

    vbounds = getCityBounds(cities,propertyname)
    vbounds[0] = 0
    vbounds = extendBounds(vbounds,'nearestLeadingDigit')
    
    cm_header = LinearColormap(
        colors=['yellow', 'orange', 'red'],
        index=None,
        vmin=vbounds[0],
        vmax=vbounds[1],
        caption=propertyname
        ).add_to(map_header) # .to_step(method='log', n=5, data=?), log has log labels but linear color scale

    for i, city in enumerate(cities):
        div_map = bre.Div(position='absolute',height='80%',width=f'{(100/len(cities))}%',left=f'{(i*100/len(cities))}%',top='10%').add_to(f)

        city_map = folium.Map(location=city['centroid'], control_scale=True)
        div_map.add_child(city_map)
        title_html = '''<h3 align="center" style="font-size:16px;charset=utf-8"><b>{}</b></h3>'''.format(city['name'])
        city_map.get_root().html.add_child(folium.Element(title_html))

        city_map.fit_bounds(city['bounds'])

        m = folium.GeoJson(
                    city['geojson'],
                    style_function=lambda feature: {
                        'fillColor': cm_header.rgb_hex_str(feature['properties'][propertyname]),
                        'fillOpacity': 0.8,
                        'color':'black',
                        'weight': 1,
                        'opacity': 0.2,
                    },
                    name=f'Choropleth_{i}'
                ).add_to(city_map)
        if not tooltiplabels==None:
            m.add_child(folium.features.GeoJsonTooltip(tooltiplabels))
    
    return display(f)

def makeValidByBuffer(gdf, verbose=False):
    '''Returns a copy of the dataframe (or list thereof) with zero buffer applied to invalid geometries
    
    Requires: geopandas, time'''
    unlist=False
    if type(gdf)!=list:
        gdf = [gdf]
        unlist = True
    ret = []
    for i, g in enumerate(gdf):
        if verbose: t_start = time.time()
        gb = g.copy(deep=True)
        indb_invalid = ~gb.geometry.is_valid
        gb.geometry.loc[indb_invalid] = gb.geometry.loc[indb_invalid].buffer(0)
        ret.append(gb)
        if verbose:
            print(f'GDF at index {i} corrected {sum(indb_invalid)} invalid '
                  f'geometries in {time.strftime("%H:%M:%S",time.gmtime(time.time()-t_start))}')
    if unlist:
        ret = ret[0]
    return ret

def makeValidByOGR(gdf, verbose=False):
    '''Returns a copy of the geodataframe (or list thereof) passed through ogr MakeValid'''
    unlist=False
    if type(gdf)!=list:
        gdf = [gdf]
        unlist = True
    ret = []
    for i, g in enumerate(gdf):
        if verbose: t_start = time.time()
        gb = g.copy(deep=True)
        indb_invalid = ~gb.geometry.is_valid
        gb.geometry.loc[indb_invalid] = [
            shapely.wkb.loads(ogr.CreateGeometryFromWkb(g.to_wkb())
                              .MakeValid().ExportToWkb())
            for g in gb.geometry.loc[indb_invalid]]
        ret.append(gb)
        if verbose:
            print(f'GDF at index {i} corrected {sum(indb_invalid)} invalid geometries in {time.strftime("%H:%M:%S",time.gmtime(time.time()-t_start))}')
    if unlist:
        ret = ret[0]
    return ret

def normalizeRows(areas, aslist=False):
    '''Normalizes each row of the input by its sum (numpy.array)'''
    a = np.array(areas)
    a = a / np.sum(a,axis=1)[:, np.newaxis]
    return a.tolist() if aslist else a

def thresholdAreas(areas, threshold=1e-6, aslist=False):
    '''Zeros out elements of areas that are below the threshold (numpy.array(float))'''
    a = np.array(areas)
    a_ratio = normalizeRows(areas)
    a[a_ratio<threshold] = 0
    return a.tolist() if aslist else a

def normalizeAreas(areas, threshold=1e-6, aslist=False):
    '''Thresholds then normalizes areas (numpy.array(float))'''
    a = thresholdAreas(areas, threshold)
    a = normalizeRows(a)
    return a.tolist() if aslist else a

def maximumAreas(areas, aslist=False):
    '''Zeros out non-maximum elements in each row of areas (numpy.array(int))
    Test with e.g.: (max(np.sum(r,axis=1)), min(np.sum(r,axis=1)), np.sum(r))'''
    a = np.array(areas)
    a_max = np.max(a,axis=1)
    r = (a >= a_max[:,np.newaxis]).astype(int)
    return r.tolist() if aslist else r

def mergeBounds(bounds1,bounds2):
    '''Returns the bounding box encompassing two bounding boxes'''
    return [min(bounds1[0],bounds2[0]), min(bounds1[1],bounds2[1]), max(bounds1[2],bounds2[2]), max(bounds1[3],bounds2[3])]

def boundsCenter(bounds):
    '''Return the center coordinates of a bounding box 4-tuple'''
    return [(bounds[0]+bounds[2])/2,(bounds[1]+bounds[3])/2]

def swapXY(coords):
    '''Swaps the latitude and longitude in a bounding box 4-tuple'''
    return [coords[1],coords[0],coords[3],coords[2]]