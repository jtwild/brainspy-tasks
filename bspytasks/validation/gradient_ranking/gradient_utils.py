# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:05:10 2020
Exploration utilities to be used for gradient calculation.
Some parts are based on code from Unai in his (private) Device Exploration repo, some parts are purely new by me.
@author: Jochem
"""
# %% Importing packages
# General imports
import numpy as np
from tqdm import trange

# Imports from brainspy
from bspyproc.utils.pytorch import TorchUtils
from bspyproc.processors.processor_mgr import get_processor
from bspyproc.processors.simulation.surrogate import SurrogateModel

# %% Funciton definitions


def create_grid_manual(min_values, max_values, n_points=[5], grid_mode='full'):
    # Goal: create a grid containing voltage values for all electrodes,
    # ranging from min_value to max_value in steps of n_points.
    # n_points can optionally be of same size as min_values to have individual number of steps per input dimension
    # shape determines whether to output an array per input dimension ('full') (which can be more easily used for numerical gradient estimation by indexing),
    # or one array which can be fed into the processor more directly ('flat')

    # Check input data format and reformat if necessary, to get everything in same shape
    min_values = np.array(min_values)
    max_values = np.array(max_values)
    assert min_values.shape == max_values.shape, 'min_values and max_values should have same shape!'
    assert min_values.ndim == 1, 'Only implemented for one min_values dimension.'
    input_dim = min_values.size

    n_points = np.array(n_points)
    assert n_points.ndim == 1, 'n_points should have 1 dimension. If you want equal n_points for all dimensions, supply a list with one element.'
    if n_points.size == 1:
        n_points = np.repeat(n_points, input_dim, axis=0)

    # Now our data has the right shapes, continue to creation of grid
    # First create the value ranges
    value_ranges = np.zeros(input_dim, dtype=object)
    for i in range(input_dim):
        # unfortunately, we cannot use variable step size icw np.linspace, so we need to loop
        value_ranges[i] = np.linspace(min_values[i], max_values[i], n_points[i])

    # Now get the meshgrid with the points
    if input_dim == 1:
        grid = np.meshgrid(value_ranges[0])
    elif input_dim == 2:
        grid = np.meshgrid(value_ranges[0], value_ranges[1])
    elif input_dim == 3:
        grid = np.meshgrid(value_ranges[0], value_ranges[1],
                           value_ranges[2])
    elif input_dim == 4:
        grid = np.meshgrid(value_ranges[0], value_ranges[1],
                           value_ranges[2], value_ranges[3])
    elif input_dim == 5:
        grid = np.meshgrid(value_ranges[0], value_ranges[1],
                           value_ranges[2], value_ranges[3],
                           value_ranges[4])
    elif input_dim == 6:
        grid = np.meshgrid(value_ranges[0], value_ranges[1],
                           value_ranges[2], value_ranges[3],
                           value_ranges[4], value_ranges[5])
    elif input_dim == 7:
        grid = np.meshgrid(value_ranges[0], value_ranges[1],
                           value_ranges[2], value_ranges[3],
                           value_ranges[4], value_ranges[5],
                           value_ranges[6])
    else:
        raise ValueError('Input dimension not yet implemented. Add to sourcecode.')
    if grid_mode == 'flat':
        # flatten:
        grid = np.reshape(grid, (input_dim, -1)).T
    if grid_mode == 'full':
        # This gets rid of the array structture in the default meshgrid output
        # Here, the resulting array has a shape of (n_electrodes, n_data_points_in_elec0, n_data_points_in_elec1, etc, etc)
        # So: if you want to know the valyes for the first grid point, this is grid(:, 0,0,0,0,0,0)
        # TODO: test if this works for unevenly spaced arrays, it should
        shape = np.concatenate((np.array([input_dim]), n_points), axis=0)
        grid = np.reshape(grid, shape)
    return grid


def create_grid_automatic(model_data_path, n_points=[5], grid_mode='full'):
    # Goal: create a grid containing voltage values on the
    # min/max values of all electrodes based upon their min/max values in their SurrogateModel
    # could potentially be extended to automatic voltage ranges on certain electrodes by simply selecting the relevant electrodes.
    # Of course, a fixed voltage would need to be chosen for the other electrodes.
    model = SurrogateModel({'torch_model_dict': model_data_path})
    min_values, max_values = model.min_voltage, model.max_voltage
    grid = create_grid_manual(min_values, max_values, n_points, grid_mode)
    return grid


def get_outputs_gradients(model_data_path, grid, grid_mode='full', calc_mode='analytical'):
    # Goal: calculate the outputs and their correpsonding gradients
    # to implement later: add option for numerical gradient calculation based on grid spacing (delta output / delta input)
    # instead of analytical pytorch gradient

    # CHeck if grid_mode makes sense
    if grid_mode == 'flat':
        assert grid.ndim == 2, 'For flat grid input, grid should only have 2 dimensions: (n_datapoints, n_elec)'
    elif grid_mode == 'full':
        assert grid.ndim > 2, 'Expecting grid dimension > 2 for full mode, for ex. dimension 8(=7+1) for 7 electrodes'
    # Choose calculator mode
    if calc_mode == 'analytical':
        calculator = get_outputs_gradients_analytical
    elif calc_mode == 'numerical':
        calculator = get_outputs_gradients_numerical
    else:
        raise NotImplementedError(f'Unknown/unimplemented mode. Supplied mode was {calc_mode}')

    # Now loop over all grid points to get the output:
    model = SurrogateModel({'torch_model_dict': model_data_path})
    outputs, gradients = calculator(model, grid, grid_mode)
    return outputs, gradients


def get_outputs_gradients_analytical(model, grid, grid_mode):
    # Goal: loop over all input values to get analytical gradients (via pytorch backward method) and outputs.

    # Gradient calculation requires flat. To use multiple grid_modes, just check the current shape, rehsape once to flat shape, do calculation, and reshape to fulll if required
    if grid_mode == 'full':
        grid_shape = np.array(grid.shape)
        input_dim = grid_shape[0]
        grid = grid.reshape(input_dim, -1).T  # this basically changes the shape to flat. Will be changed to full again after calculation
        # if mode == flat, we do not need to rehsape

    # Initialize result arrays
    n_points = grid.shape[0]
    input_dim = grid.shape[1]
    outputs = np.full((n_points,), np.nan)
    gradients = np.full((n_points, input_dim), np.nan)
    # Loop to get results
    looper = trange(n_points, desc=' Device exploration progress', leave=True)
    for i in looper:
        single_input = format_input(grid[i, :])
        torch_output = model(single_input)
        outputs[i], gradients[i, :] = get_gradient_analytical(single_input, torch_output)

    # Reshape to get correct form if we are using full mode. If we use flat mode, shape is already correct
    if grid_mode == 'full':
        outputs_shape = grid_shape.copy()
        outputs_shape[0] = 1  # not 7 input electrodes, but just one output
        outputs = outputs.reshape(outputs_shape)
        gradients = gradients.reshape(grid_shape)

    return outputs, gradients


def get_gradient_analytical(input_data, output_data):
    # Copied and minimalized from Unai's work
    # Goal: get analytic gradient via pytorch method of a single data sample
    if input_data.grad is not None:
        input_data.grad.zero_()
    output_data.backward()

    gradients = TorchUtils.get_numpy_from_tensor(input_data.grad)
    detached_output = TorchUtils.get_numpy_from_tensor(output_data)

    return detached_output, gradients

def get_outputs_gradients_numerical(model, grid, grid_mode, return_difference = False):
    # Goal: get numerical gradients / output differences divided by input difference
    # Only works with full grid because there the differences are easy
    # Method: calculate delta output and delta input, calculate difference
    assert grid_mode == 'full', 'Numerical only works with full grid mode'
    n_grid_dims = grid.ndim - 1 #always one dimension for the different eelctrodes (the 0th dimension). THe remaining dimension is n_grid_dims, number of varied dimensions
    # Number of electrodes should be equal to first dimension of grid
    grad_shape = np.array(grid.shape)
    grad_shape[0] = n_grid_dims # because we can only set the gradient for parts that are varied in the grid.
    gradients = np.full(grad_shape, np.nan)

    # Get outputs
    outputs = get_outputs_on_grid(model, grid)
    # Calculate gradients for all varied dimension
    for i in range(n_grid_dims):
        # Place the relevant data on first dimension, perform calculation and go back to original shape
        grid = grid.swapaxes(1, i+1)
        outputs = outputs.swapaxes(1, i+1)
        gradients = gradients.swapaxes(1, i+1)
        # Perform calculation. The first dimension now contains the thing that we are differencing
        delta_output = outputs[0,1:] - grid[0, :-1] # only one output, so first output dimension size is 0. Take that 0th element.
        # Two possible outputs
        if return_difference:
            gradients[i,:] = delta_output # actually, not gradient but a difference in this case
        else:
            delta_input = grid[i,1:] - grid[i,:-1]
            gradients[i,:-1] = delta_output / delta_input # we cannot fill the entire grid because for the dimension we are changing we have one less data point after calculating differences
            raise Warning('Bug here. SOmehow axis 0 and 1 seems to be swapped? Axis 1 does not vary in the grid where required.')
        # Swap axes back
        grid = grid.swapaxes(1, i+1)
        outputs = outputs.swapaxes(1,i+1)
        gradients = gradients.swapaxes(1,i+1)
    return outputs, gradients

def get_outputs_on_grid(model, grid):
    outputs_shape = np.array(grid.shape)
    outputs_shape[0] = 1 # only one output value isntead of 7 input values
    n_elec = grid.shape[0]
    inputs = format_input( grid.reshape(n_elec, -1), requires_grad = False)
    outputs = TorchUtils.get_numpy_from_tensor( model(inputs.T) ).T  # model take sinputs with 7-dimensional electrode values on 2nd dimension, en n_samples on first dimension. SO take inverse.
    outputs = outputs.reshape(outputs_shape)
    return outputs


def format_input(data, requires_grad = True):
    # Copied from Unai's work
    # Goal: format input data to torch tensor.
    processed_data = TorchUtils.get_tensor_from_numpy(data)
    processed_data.requires_grad = requires_grad
    return processed_data

def averager(gradients, grid_mode):
    #Goal: average over all electrodes to get the average gradient per electrode
    assert grid_mode == 'full', 'Averager only knows how to work on full grid modes for now.'
    # electrodes are on 0th dimension, so average everything except the zeroth dimension
    avg_axes = tuple(range(1, gradients.ndim))

    averaged_gradients = np.nanmean(np.abs(gradients), axis=avg_axes)
    return averaged_gradients

#%% For testing the code
if __name__ == '__main__':
    model_data_path = "tmp/input/models/model_2020.pt"
#    min_voltage = [-0.7]*7
#    max_voltage = [0.3]*7
    n_points = [4]
    grid_mode = 'full'
    calc_mode = 'numerical'
    grid = create_grid_automatic(model_data_path, grid_mode=grid_mode, n_points=n_points)
#    grid = create_grid_manual(min_voltage, max_voltage, grid_mode = grid_mode, n_points = n_points)
    outputs, gradients = get_outputs_gradients(model_data_path, grid, grid_mode=grid_mode, calc_mode = calc_mode)
    avg_grad = averager(gradients, grid_mode)
