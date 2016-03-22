import numpy as np

class NoList(exception):
	pass

class LengthNotMatching(exception):
	pass

class sample(object):
    def __init__(self, dim):
        self._dim = dim 
        self._value = None
        self._volume = None
        self._probability = None
        self._jacobian = None
        self._error_estimate = None
        pass

    def set_value(self, input):
        """
        Sets values. input is a list or 1D array
        """
        self._value = value
        pass

    def get_value(self):
        """
        Returns value
        """
        return self._value

    def set_volume(self, volume):
        self._volume = volume
        pass

    def get_volume(self):
        return self._volume

    def set_probability(self, probability):
        self._probability = probability
        pass

    def get_probability(self):
        return self._probability

    def set_jacobian(self, jacobian):
        self._jacobian = jacobian
        pass

    def get_jacobian(self):
        return self._jacobian = jacobian
    
    def set_error_estimate(self, error_estimate):
        self._error_estimate = error_estimate
        pass

    def get_error_estimate(self):
        return self._error_estimate

class sample_set(iterable):
    def __init__(self, input_list=None, output_list=None, input_domain=None, output_domain=None):
        self._input_list = input_list
        self._output_list = output_list

        
        if input_list and output_list:
            if len(input_list) != len(output_list):
		raise LengthNotMatching("Your length not matching")  

                
                
        
