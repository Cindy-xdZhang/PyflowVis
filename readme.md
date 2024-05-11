![alt text](assets/misc/image-1.png)


# Cpp 
1.Write cpp functions in 'Cppmodules ' folder to export cpp functions to python(using pybind)
2.For pure cpp program using visual studio etc., write it in  'CppProjects' folder.



# TODO list General
0. optc optimize di term +ci term(most important thing!)
1. serialize (load/save)  vector field from pyflowVis to optc cpp framework
2. add Lic and texture manager for the plane.
3. add mouse click intersection with pickable object
4. add pathline?
5. language UI.
6. reference frame transformation

# TODO list for NLpFLOWvis
1. one intermediate representation for select, intersect, reference frame transforamtion
2. finetune 

## possible optimize 
1. multiple thread rendering using glMultiDrawArrays
2. vertex array object operate geometry,indices,texture as list, could using numpy.array for fast operation.

## profile result 

###    Ordered by: cumulative time
Mon Apr  8 16:52:59 2024    profile\profile.txt
         10002475 function calls (9838355 primitive calls) in 22.923 seconds
   Ordered by: Ordered by: cumulative time
   List reduced from 5303 to 30 due to restriction <30>
| ncalls     | tottime | percall | cumtime | percall | filename:lineno(function)                            |
| ---------- | ------- | ------- | ------- | ------- | ---------------------------------------------------- |
| 1          | 0.000   | 0.000   | 19.228  | 19.228  | main.py:116(main)                                    |
| 1          | 0.013   | 0.013   | 18.378  | 18.378  | VisualizationEngine.py:77(MainLoop)                  |
| 974        | 0.008   | 0.000   | 13.769  | 0.014   | Object.py:671(draw_all)                              |
| 20         | 0.001   | 0.000   | 11.396  | 0.570   | _jit_internal.py:890(_overload)                      |
| 974        | 0.003   | 0.000   | 9.414   | 0.010   | VertexArrayObject.py:318(draw)                       |
| 211        | 0.107   | 0.001   | 8.153   | 0.039   | VertexArrayObject.py:325(updateVectorGlyph)          |
| 13571      | 1.919   | 0.000   | 7.681   | 0.001   | VertexArrayObject.py:169(appendConeWithoutCommit)    |
| 173        | 0.008   | 0.000   | 6.418   | 0.037   | __init__.py:1(<module>)                              |
| 4870       | 0.056   | 0.000   | 5.548   | 0.001   | VertexArrayObject.py:123(draw)                       |
| 13571      | 1.543   | 0.000   | 3.651   | 0.000   | VertexArrayObject.py:137(appendCircleWithoutCommit)  |
| 4870       | 0.025   | 0.000   | 3.586   | 0.001   | shaderManager.py:178(setUniformScope)                |
| 54287      | 1.145   | 0.000   | 3.258   | 0.000   | numeric.py:1468(cross)                               |
| 974        | 0.003   | 0.000   | 3.218   | 0.003   | VertexArrayObject.py:393(draw)                       |
| 9740       | 0.049   | 0.000   | 2.858   | 0.000   | shaderManager.py:205(Use)                            |
| 61921      | 0.071   | 0.000   | 2.802   | 0.000   | latebind.py:35(__call__)                             |
| 9740       | 0.008   | 0.000   | 2.654   | 0.000   | shaderManager.py:119(check_for_updates)              |
| 9740       | 0.016   | 0.000   | 2.646   | 0.000   | fileMonitor.py:45(update_files)                      |
| 19480      | 0.017   | 0.000   | 2.630   | 0.000   | fileMonitor.py:37(checkModified)                     |
| 19482      | 0.034   | 0.000   | 2.614   | 0.000   | fileMonitor.py:31(get_last_modified)                 |
| 151432     | 2.612   | 0.000   | 2.612   | 0.000   | error.py:208(glCheckError)                           |
| 974        | 0.191   | 0.000   | 2.551   | 0.003   | opengl.py:260(render)                                |
| 697/98     | 0.002   | 0.000   | 2.156   | 0.022   | {built-in method builtins.__import__}                |
| 162861     | 0.578   | 0.000   | 1.809   | 0.000   | numeric.py:1393(moveaxis)                            |
| 24940      | 1.771   | 0.000   | 1.771   | 0.000   | {built-in method nt.stat}                            |
| 4870       | 0.021   | 0.000   | 1.760   | 0.000   | shaderManager.py:201(setUnforms)                     |
| 23376      | 0.876   | 0.000   | 1.737   | 0.000   | shaderManager.py:147(setUniform)                     |
| 5          | 0.000   | 0.000   | 1.640   | 0.328   | functional.py:1(<module>)                            |
| 20553      | 1.347   | 0.000   | 1.347   | 0.000   | {built-in method nt._path_exists}                    |
| 19482      | 0.026   | 0.000   | 1.299   | 0.000   | <frozen genericpath>:65(getmtime)                    |
| 20292/2176 | 0.028   | 0.000   | 1.128   | 0.001   | <frozen importlib._bootstrap>:1390(_handle_fromlist) |

###    Ordered by: internal time
Mon Apr  8 16:52:59 2024    profile\profile.txt; 10002475 function calls (9838355 primitive calls) in 22.923 seconds
   Ordered by: internal time
   | ncalls          | tottime | percall | cumtime | percall | filename:lineno(function)                           |
   | --------------- | ------- | ------- | ------- | ------- | --------------------------------------------------- |
   | 151432          | 2.612   | 0.000   | 2.612   | 0.000   | error.py:208(glCheckError)                          |
   | 13571           | 1.919   | 0.000   | 7.681   | 0.001   | VertexArrayObject.py:169(appendConeWithoutCommit)   |
   | 24940           | 1.771   | 0.000   | 1.771   | 0.000   | {built-in method nt.stat}                           |
   | 13571           | 1.543   | 0.000   | 3.651   | 0.000   | VertexArrayObject.py:137(appendCircleWithoutCommit) |
   | 20553           | 1.347   | 0.000   | 1.347   | 0.000   | {built-in method nt._path_exists}                   |
   | 54287           | 1.145   | 0.000   | 3.258   | 0.000   | numeric.py:1468(cross)                              |
   | 23376           | 0.876   | 0.000   | 1.737   | 0.000   | shaderManager.py:147(setUniform)                    |
   | 1466/0          | 0.684   | 0.000   | 0.000   |         | {built-in method builtins.exec}                     |
   | 325722          | 0.666   | 0.000   | 0.949   | 0.000   | numeric.py:1330(normalize_axis_tuple)               |
   | 1               | 0.578   | 0.578   | 0.578   | 0.578   | {built-in method pygame.display.set_mode}           |
   | 162861          | 0.578   | 0.000   | 1.809   | 0.000   | numeric.py:1393(moveaxis)                           |
   | 16630           | 0.427   | 0.000   | 0.958   | 0.000   | wrapper.py:856(wrapperCall)                         |
   | 95009           | 0.355   | 0.000   | 0.650   | 0.000   | linalg.py:2383(norm)                                |
   | 122724          | 0.258   | 0.000   | 0.258   | 0.000   | {built-in method numpy.array}                       |
   | 1210886/1210326 | 0.245   | 0.000   | 0.245   | 0.000   | {built-in method builtins.len}                      |
   | 2264            | 0.204   | 0.000   | 0.204   | 0.000   | {built-in method nt._getfinalpathname}              |
   | 974             | 0.191   | 0.000   | 2.551   | 0.003   | opengl.py:260(render)                               |
   | 862353/861994   | 0.185   | 0.000   | 0.185   | 0.000   | {built-in method builtins.isinstance}               |

