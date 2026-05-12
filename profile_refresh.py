import cProfile
from scripts.refresh_data import refresh

# Mock time.sleep to avoid it blocking the profile
import time
time.sleep = lambda x: None

cProfile.run('refresh("CHA")', 'refresh_stats.prof')

import pstats
p = pstats.Stats('refresh_stats.prof')
p.sort_stats('cumtime').print_stats(20)
