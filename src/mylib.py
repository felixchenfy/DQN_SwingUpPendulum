
import numpy as np
import matplotlib.pyplot as plt


# get current time string
def get_time_str():
    import datetime
    t = datetime.datetime.now()
    s_time = str(t)[0:-7]
    return s_time

# save data and load data
class io(object):
    @staticmethod
    def savetxt(filename, data):
        _filename = filename + "_" + get_time_str() + ".txt"
        np.savetxt(_filename, data, delimiter=" ")

    @staticmethod
    def loadtxt(filename):
        return np.loadtxt(filename, delimiter=" ")
    '''
    test case:
    if __name__=="__main__":
        import numpy as np
        filename="foo.csv"
        a = np.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
        io.savetxt(filename, a)
        b=io.loadtxt(filename)
        print(b)
    '''

        