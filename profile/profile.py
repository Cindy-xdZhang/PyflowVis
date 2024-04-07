import pstats
 
import pstats

p = pstats.Stats("profile\profile.txt")

# strip_dirs(): Remove irrelevant path information
# sort_stats(): Sorting, supports the same ways as mentioned above
# print_stats(): Print analysis results, can specify the number of lines to print
 
# This would yield the same result as running cProfile.run("test()") directly
# p.strip_dirs().sort_stats(-1).print_stats()
 
# Sort by function name, only print the top 3 lines of functions. The parameter can also be a decimal, indicating the top percentage of functions information to print 
p.strip_dirs().sort_stats("cumtime").print_stats(30)
p.strip_dirs().sort_stats("tottime").print_stats(30)
p.strip_dirs().sort_stats("ncalls").print_stats(30)
# Sort by cumulative time and function name, only display the top 50%
# p.strip_dirs().sort_stats("cumulative", "name").print_stats(0.2)
 
# If you want to know which functions called sum_num, only display the top 50%
# p.print_callers(0.5, "sum_num")
 
# To see which functions the test() function called
# p.print_callees("test")
