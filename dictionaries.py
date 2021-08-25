#!/usr/bin/env python3

from butcher_table import (standard_runge_kutta_dict,
						   embedded_runge_kutta_dict)

from multistep_table import (adams_explicit_dict,
							 adams_implicit_dict,
							 adams_predictor_corrector_dict,
							 backward_differentiation_formula_dict,
							 numerical_differentiation_formula_dict)


ode_method_dict = standard_runge_kutta_dict.copy()				# combine dictionaries
ode_method_dict.update(embedded_runge_kutta_dict)
ode_method_dict.update(adams_explicit_dict)
ode_method_dict.update(adams_implicit_dict)
ode_method_dict.update(adams_predictor_corrector_dict)
ode_method_dict.update(backward_differentiation_formula_dict)
ode_method_dict.update(numerical_differentiation_formula_dict)


