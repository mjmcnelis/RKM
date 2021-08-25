
import numpy as np

parameters = {'adaptive':			'RKM',
			  'method':				'RK4',
			  'dt_initial':			1,
			  'max_steps':			100000,
			  'eps':				1.e-8,
			  'norm':				None,
			  'dt_min':				1.e-7,
			  'dt_max':				1,
			  'low':				0.2,
			  'high':				1.5,
			  'root':				'newton_fast',
			  'iterations':			2,
			  'eps_root':			1.e-6,
			  'interpolate':		False,
			 }

#-------------------------------------------------------------------------------------------------------|
#				|																						|
#  				|	adaptive time step algorithm (None, str)											|
#   adaptive    |																						|
#				|	None, 'RKM', 'ERK', 'SDRK'															|
#				|																						|
#-------------------------------------------------------------------------------------------------------|
#				|																						|
#  				|	code label of numerical method (str)												|
#	 method		|																						|
#				|	E1, H2, RK4, F45, etc. (see dictionaries in butcher_table.py, multistep_table.py)	|
#				|																						|
#-------------------------------------------------------------------------------------------------------|
#				|																						|
#  				|	initial time step (float)															|
#  dt_initial   |																						|
#				|	note: if adaptive != None, then it's a proposed value								|
#				|																						|
#-------------------------------------------------------------------------------------------------------|
#				|																						|
#   max_steps   |	max number of time steps (int)														|
#				|																						|
#-------------------------------------------------------------------------------------------------------|
#				|																						|
#      eps      |	tolerance parameter for error control (float)										|
#				|																						|
#-------------------------------------------------------------------------------------------------------|
#				|																						|
#  				|	type of norm used for error control (None, int, np.inf)								|
#     norm      |																						|
#				|	None, 1, np.inf																		|
#				|																						|
#-------------------------------------------------------------------------------------------------------|
#				|																						|
#	 dt_min 	|	min time step allowed (float)													    |
#				|																						|
#-------------------------------------------------------------------------------------------------------|
#				|																						|
#    dt_max     |	max time step allowed (float)													    |
#				|																						|
#-------------------------------------------------------------------------------------------------------|
#				|																						|
#      low      |	lower bound for time step variation (float)											|
#				|																						|
#-------------------------------------------------------------------------------------------------------|
#				|																						|
#     high      |	upper bound for time step variation (float)											|
#				|																						|
#-------------------------------------------------------------------------------------------------------|
#				|																						|
#       		|	root-finder method for implicit routines (str)										|
#     root      |																						|
#				|	'fixed_point', 'newton', 'newton_fast'												|
#				|																						|
#-------------------------------------------------------------------------------------------------------|
#				|																						|
#  iterations 	|	max number of root-finder iterations (int >= 1)										|
#				|																						|
#-------------------------------------------------------------------------------------------------------|
#				|																						|
#   eps_root    |	tolerance parameter for root-finder convergence (float)								|
#				|																						|
#-------------------------------------------------------------------------------------------------------|
#				|																						|
# 				|	option to interpolate numerical solution for dense output (bool)					|
#  interpolate  |																						|
#				|	True, False																			|
#				|																						|
#-------------------------------------------------------------------------------------------------------|




