# -*- coding: utf-8 -*-
"""
Script designed to create testpoints for VC dimension tests, for any VC dimension and any inputspace dimension d.
Every set of 2 points is used to generate a d-dimensional hypercube, thus creating 2^d points per set.
Created on Thu Feb  6 15:19:29 2020

@author: Jochem
"""
#TODO: Add diamond generation for larger seperation
import numpy as np

# %% Generation itself
def generate_unsorted_points(input_dim, voltage_intervals = [-0.7, 0.3], num_levels = 2):
    # Auto generates a set of coordinates in N dimensional space (N=input_dim)
    # possibly with multiple levels per electrode interval (given by num_levels_per_electrode)
    # and possibly on a different voltage interval per electrode.
    # if voltage interval is 1D, same interval used for all electrodes. Dimension must be 2, min and max voltage.
    # if voltage interval is 2D, outer dimension must match input_dim and an interval per electrode is used
    # if voltage_interval is 3D, the script will basically repeated N times, where N is the outer dimension of voltage_intervals_per electrode.
    # If num_levels is 0D, same amount of levels for all electrodes.
    # If num_levels is 1D, custom level per electrode.

    # Reshape the voltage intervals to always be 3D shape.
    voltage_intervals = np.array(voltage_intervals)
    if voltage_intervals.ndim == 1:
        assert voltage_intervals.shape == (2,), "Voltage intervals not understood. Inner dimension must be of size 2: (min, max)."
        voltage_intervals = voltage_intervals.reshape(1,1,2)
        voltage_intervals = np.repeat(voltage_intervals, input_dim, axis=1)
        # final shape: (1,input_dim, 2).
    elif voltage_intervals.ndim == 2:
        assert voltage_intervals.shape == (input_dim, 2), "Voltage intervals not understood. See source code for examples. "
        voltage_intervals = voltage_intervals.reshape(1,input_dim,2)
        # final shape (1,input_dim, 2)
    assert voltage_intervals.ndim == 3 and voltage_intervals.shape[1:]==(input_dim,2)
    # Reshape num_levels to always be 1D, elements are the numbers, dimension is the electrode
    num_levels = np.array(num_levels, dtype=int)
    if num_levels.ndim == 0:
        num_levels = num_levels.reshape(1,)
        num_levels = np.repeat(num_levels, input_dim, axis=0)
    elif num_levels.ndim > 1:
        raise ValueError('Number of levels input not understood. See source code for valid inputs.')

    # Now our input has right format, continue to point generation
    coordinates = np.empty((input_dim,0))
    for electrode_intervals in voltage_intervals:
        vectors_encoded = np.zeros(input_dim, dtype=object)
        for i in range(input_dim):
            # unfortunately, we cannot use variable step size icw np.linspace, so we need to loop
            vectors_encoded[i] = np.linspace(electrode_intervals[i,0], electrode_intervals[i,1], num=num_levels[i])
        # Now get the meshgrid with the points
        if input_dim == 1:
            grid = np.meshgrid(vectors_encoded[0])
        elif input_dim == 2:
            grid = np.meshgrid(vectors_encoded[0], vectors_encoded[1])
        elif input_dim == 3:
            grid = np.meshgrid(vectors_encoded[0], vectors_encoded[1],
                               vectors_encoded[2])
        elif input_dim == 4:
            grid = np.meshgrid(vectors_encoded[0], vectors_encoded[1],
                               vectors_encoded[2], vectors_encoded[2])
        elif input_dim == 5:
            grid = np.meshgrid(vectors_encoded[0], vectors_encoded[1],
                               vectors_encoded[2], vectors_encoded[3],
                               vectors_encoded[4])
        elif input_dim == 6:
            grid = np.meshgrid(vectors_encoded[0], vectors_encoded[1],
                               vectors_encoded[2], vectors_encoded[3],
                               vectors_encoded[4], vectors_encoded[5])
        else:
            raise ValueError('Input dimension not yet implemented. Add to sourcecode.')
        coordinates = np.append(coordinates, np.reshape(grid, (input_dim, -1)), axis=1 )
    return coordinates

#%% ha
#def generate_unsorted_points(input_dim):
##    sets = [[-1.2, 0.6], [-0.6, 0.0]]  # must be X by 2. Old point:
#    sets = [[-1.2, 0.6]]
#    # check inputs
#    num_points = 2**input_dim * len(sets)  # get maximum number of points possible with the given sets
#    for i in sets:
#        if len(i) != 2:
#            raise ValueError('Program was only written to make hypercubes, so only 2 points per set! Change the input sets.')
#
#    # loop em up
#    samples = np.zeros([input_dim, num_points])
#    index = np.zeros(input_dim, dtype=int)
#    done = False
#    for s in sets:
#        # slightly ugly command to built a meshgrid with the required amount of dimensions
#        #coor = np.array([]);
#        if input_dim == 1:
#            coor = np.meshgrid(s)
#        elif input_dim == 2:
#            coor = np.meshgrid(s,s)
#        elif input_dim == 3:
#            coor = np.meshgrid(s,s,s)
#        elif input_dim == 4:
#            coor = np.meshgrid(s,s,s,s)
#        elif input_dim == 5:
#            coor = np.meshgrid(s,s,s,s,s)
#        elif input_dim == 6:
#            coor = np.meshgrid(s,s,s,s,s,s)
#        else:
#            raise ValueError('Input dimensions not implemented. Please add manually to code. \
#                             Or even better, fix this ugly hardcoding.')
#        dim = 0
#        for i in coor:
#            # flatten the numpy array such that we can paste it in the samples array
#            i = i.flatten()
#            if index[dim] + len(i) >= num_points:
#                # this means we have too many points, throw away some
#                i = i[0 : (num_points - index[dim])]
#                # also, we are done with filling after we have filled all these dimensions
#                done = True
#            samples[dim, index[dim] : index[dim]+len(i) ] = i
#            index[dim] = index[dim]+len(i)
#            dim +=1
#        if done:
#            break
#        # else, continue onto next set of points
#    return samples

def sort_points(coordinates):
    # First dimension defines the dimensional coordinate of the point, the second dimension defines the point number
    [dimension, number_of_points] = np.shape( coordinates )
    # create an array for distances which we will fill. NaN indicates we havent filled it.
    distance = np.empty([number_of_points, number_of_points])
    distance[:] = np.nan
    for n1 in range(number_of_points):
        for n2 in range(number_of_points):
            # we do not have to evaluate the lower triangle of the square matrix, that is double information, but okay.
            distance[n1,n2] = np.sum((coordinates[:, n1] - coordinates[:, n2] ) **2)
    # Now we know the distances between points, we can start ordering
    sorted_coordinates = np.zeros_like(coordinates)
    # Start intial two points
    [x,y] = flat_index_to_dimensional_index( np.nanargmax(distance), distance )
    sorted_coordinates[:,0] = coordinates[:,x]
    sorted_coordinates[:,1] = coordinates[:,y]
    distance[x,y] = np.nan  # ignore for the rest of the search
    distance[y,x] = np.nan  # diagonally symmetric
    sorted_set = [x,y]
    for n1 in range(2,number_of_points):
        # Find the distances between test points and the points already in our set. Then find the maximum distance of ( the minimum distances between points in our set and test point)
        selected_distance = np.nanmin( distance[sorted_set] , axis=0 )  # Find the distance between all points and the points closest to it in the sorted set
        index = np.nanargmax(selected_distance)  # Get the point which has the maximally far away from the point in the sorted set it is closest to
        distance[index, sorted_set] = np.nan
        distance[sorted_set, index] = np.nan
        sorted_set.append(index)
        #TODO: Update sorting to sort by a sum of maxed distances, instead of only the maximum closest distance as is used now
    return coordinates[:,sorted_set]


def flat_index_to_dimensional_index(flat_index, test_array):
    # Links the index of a flattened array (given by flat_index) to
    # the 2D index the un-flattened array (given by test_array) would have
    assert flat_index <= test_array.size, "Index too large for this array!"
    assert test_array.ndim == 2, "Test array must be 2 dimensional!"
    xmax = np.shape(test_array)[0]
    y = flat_index % xmax  # modulus
    x = int( flat_index / xmax )
    return [x,y]

def generate_sorted_points(vc_dim, input_dim, voltage_intervals, num_levels):
    unsorted_points = generate_unsorted_points(input_dim, voltage_intervals=voltage_intervals, num_levels=num_levels)
    if vc_dim > np.shape(unsorted_points)[1]:
        raise ValueError('Too many points requested! Either add generation sets in main file, or lower VC dimension.')
    sorted_points = sort_points( unsorted_points )
    # Now return only the selection of furthest away points which are required
    return sorted_points[:,0:vc_dim]

# %% Test code
# Make a rotating 3D plot which labels the points by the order thet are given by the sorted point generator
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    vc_dim = 16
    input_dim = 3
    num_levels = [3, 3, 4]
    voltage_intervals = [[-1.2, 0.6], [-0.7, 0.3],[-0.1, 0.1]]
    points = generate_sorted_points(vc_dim, input_dim, voltage_intervals, num_levels)

    if input_dim == 2:
        # Show results in a plot
        fig, ax = plt.subplots()
        ax.scatter(points[0], points[1])
        for i in range( len(points[0]) ):
            #denote which points is chosen first
            ax.annotate(str(i+1), (points[0][i], points[1][i]))
        plt.grid('on')
        plt.show()
    elif input_dim == 3:
        # First run %matplotlib auto in iPython kernel
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3,1.3)
        ax.set_zlim(-1.3,1.3)
        ax.scatter(points[0], points[1], points[2])
        for i in range( len(points[0]) ):
            #denote which points is chosen first
            ax.text(points[0][i], points[1][i], points[2][i], str(i), size=15, zorder=0)
        plt.grid('on')
        plt.show()
        for angle in range(0, 360):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(.001)


