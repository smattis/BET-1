import numpy as np

class NoList(exception):
	pass

class LengthNotMatching(exception):
	pass

class sample_set(object):
    def __init__(self, dim):
        self._dim = dim 
        self._values = None
        self._volumes = None
        self._probabilities = None
        self._jacobians = None
        self._error_estimates = None
        pass
    

    def get_dim(self):
        return self._dim

    def set_values(self, input):
        """
        Sets values. input is a list or 1D array
        """
        self._values = values
        pass

    def get_values(self):
        """
        Returns value
        """
        return self._values

    def set_volumes(self, volumes):
        self._volumes = volumes
        pass

    def get_volumes(self):
        return self._volumes

    def set_probabilities(self, probabilities):
        self._probabilities = probabilities
        pass

    def get_probabilities(self):
        return self._probabilities

    def set_jacobians(self, jacobians):
        self._jacobians = jacobians
        pass

    def get_jacobians(self):
        return self._jacobians = jacobians
    
    def set_error_estimates(self, error_estimates):
        self._error_estimates = error_estimates
        pass

    def get_error_estimates(self):
        return self._error_estimates

class sample_set(iterable):
    def __init__(self, input_list=None, output_list=None, input_domain=None, output_domain=None):
        self._input_list = input_list
        self._output_list = output_list

        
        if input_list and output_list:
            if len(input_list) != len(output_list):
		raise LengthNotMatching("Your length not matching")  

                
                
        
