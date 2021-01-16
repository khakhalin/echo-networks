class DataGenerator:

    def __init__(self, sys_type='lorenz', fromfile=False):
        """
        This is a class for chaotic time series data generation or reading it from file
        :param sys_type: string, type of dynamical system, lorenz
        :param fromfile: string, if time series recorded from file
        """
