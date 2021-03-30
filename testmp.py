import multiprocess
import difflib

completed = 0

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