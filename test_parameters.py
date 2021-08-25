
import numpy as np

adaptive = 'RKM'            # adaptive stepsize algorithm (None, 'RKM', 'ERK', 'SDRK')
method = 'RK4'              # code label of numerical method (see dictionaries in butcher_table.py, multistep_table.py)
root = 'newton_fast'        # root solver method for implicit routines ('fixed_point', 'newton', 'newton_fast')

dt0 = 1                     # initial step size (proposal if adaptive != None)
eps = 1.e-8                 # error tolerance parameter
norm = None                 # type of norm used for error control (e.g. None (2), 1, np.inf)

error_type = 'absolute'     # type of numerical error to plot (absolute, relative)
average = True              # average error over time interval (True) or take max error (False)

interpolate = False         # option to interpolate numerical solution (not ready)
plot_variables = 2          # number of variables to plot




