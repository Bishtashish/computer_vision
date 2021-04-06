import pstats
p = pstats.Stats('ashish.txt')
p.sort_stats('cumulative').print_stats(10)