"""Crystal Orbital Overlap Population  calculation."""
__all__ = ['workflow_crystal_orbital_overlap_population']

import logging
import numpy as np
from nac.common import (
    number_spherical_functions_per_atom,
    retrieve_hdf5_data, is_data_in_hdf5)
from nac.integrals.multipole_matrices import compute_matrix_multipole
from nac.workflows.initialization import initialize
from nac.workflows.workflow_single_points import workflow_single_points
from scipy.constants import physical_constants
from qmflows.parsers.xyzParser import readXYZ

# Starting logger
LOGGER = logging.getLogger(__name__)


def workflow_crystal_orbital_overlap_population(config: dict):
    """Crystal Orbital Overlap Population  main function."""
    # Dictionary containing the general information
    config.update(initialize(config))

    # Checking hdf5 for eigenvalues and coefficients. If not present, they are
    # computed.
    check_hdf5_for_eigenvalues_coefficients(config)

    # Logger info
    LOGGER.info("Starting COOP calculation.")

    # Get eigenvalues and coefficients from hdf5
    atomic_orbitals, energies = get_eigenvalues_coefficients(config)

    # Converting the xyz-file to a mol-file
    mol = readXYZ(config["path_traj_xyz"])

    # Computing the indices of the atomic orbitals of the two selected
    # elements, and the overlap matrix that contains only elements related to
    # the two elements
    el_1_orbital_ind, el_2_orbital_ind, overlap_reduced = compute_overlap_and_atomic_orbitals(
        mol, config)

    # Compute the crystal orbital overlap population between the two selected
    # elements
    coop = compute_coop(
        atomic_orbitals,
        overlap_reduced,
        el_1_orbital_ind,
        el_2_orbital_ind)

    # Lastly, we save the COOP as a txt-file
    result_coop = print_coop(energies, coop)
    LOGGER.info("COOP calculation completed.")

    return result_coop


def check_hdf5_for_eigenvalues_coefficients(config: dict):
    """Checks if hdf5 contains the required eigenvalues and coefficients.
    If not, it runs the single point calculation."""
    path_coefficients = '{}/point_0/cp2k/mo/coefficients'.format(
        config["project_name"])
    path_eigenvalues = '{}/point_0/cp2k/mo/eigenvalues'.format(
        config["project_name"])

    predicate_1 = is_data_in_hdf5(config["path_hdf5"], path_coefficients)
    predicate_2 = is_data_in_hdf5(config["path_hdf5"], path_eigenvalues)

    if all((predicate_1, predicate_2)):
        LOGGER.info("Coefficients and eigenvalues already in hdf5.")
    else:
        # Call the single point workflow to calculate the eigenvalues and
        # coefficients
        LOGGER.info("Starting single point calculation.")
        workflow_single_points(config)


def get_eigenvalues_coefficients(config: dict):
    """Retrieves eigenvalues and coefficients from hdf5 file."""
    # Define paths to eigenvalues and coefficients hdf5
    path_coefficients = '{}/point_0/cp2k/mo/coefficients'.format(
        config["project_name"])
    path_eigenvalues = '{}/point_0/cp2k/mo/eigenvalues'.format(
        config["project_name"])

    # Retrieves eigenvalues and coefficients
    atomic_orbitals = retrieve_hdf5_data(
        config["path_hdf5"], path_coefficients)
    energies = retrieve_hdf5_data(config["path_hdf5"], path_eigenvalues)

    # Energies converted from Hartree to eV
    h2ev = physical_constants['Hartree energy in eV'][0]
    energies = energies * h2ev

    # Return atomic orbitals and energies
    return atomic_orbitals, energies


def compute_overlap_and_atomic_orbitals(mol: list, config: dict):
    """Computes the indices of the atomic orbitals of the two selected elements.
    Computes the overlap matrix, containing only the elements related to those two elements."""
    # Computing the overlap-matrix S
    overlap = compute_matrix_multipole(mol, config, 'overlap')

    # Computing number of spherical orbitals per atom
    sphericals = number_spherical_functions_per_atom(
        mol,
        'cp2k',
        config["cp2k_general_settings"]["basis"],
        config["path_hdf5"])

    # Getting the indices for the two selected elements
    element_1 = config["coop_elements"][0]
    element_2 = config["coop_elements"][1]

    element_1_index = [i for i, s in enumerate(mol) if element_1.lower() in s]
    element_2_index = [i for i, s in enumerate(mol) if element_2.lower() in s]

    # Making a list of the indices of the atomic orbitals for each of the two
    # elements
    atom_indices = np.zeros(len(mol) + 1, dtype='int')
    atom_indices[1:] = np.cumsum(sphericals)

    el_1_orbital_ind = [np.arange(sphericals[i]) +
                        atom_indices[i] for i in element_1_index]
    el_1_orbital_ind = np.reshape(el_1_orbital_ind, len(
        element_1_index) * sphericals[element_1_index[0]])

    el_2_orbital_ind = [np.arange(sphericals[i]) +
                        atom_indices[i] for i in element_2_index]
    el_2_orbital_ind = np.reshape(el_2_orbital_ind, len(
        element_2_index) * sphericals[element_2_index[0]])

    # Reduced overlap matrix, containing only the elements related to the
    # overlap between element_1 and element_2
    # First select all the rows that belong to element_1
    overlap_reduced = overlap[el_1_orbital_ind, :]
    # Then select from those rows the columns that belong to species element_2
    overlap_reduced = overlap_reduced[:, el_2_orbital_ind]

    # Return lists of indices of atomic orbitals, and the reduced overlap
    # matrix
    return el_1_orbital_ind, el_2_orbital_ind, overlap_reduced


def compute_coop(
        atomic_orbitals,
        overlap_reduced,
        el_1_orbital_ind,
        el_2_orbital_ind):
    """Defines the function that computes the crystal orbital overlap population,
    and applies it to each column of the coefficent matrix."""
    # Define a function to be applied to each column of the coefficient matrix
    def coop_func(
            atomic_orbitals,
            overlap_reduced,
            el_1_orbital_ind,
            el_2_orbital_ind):
        # Multiply each coefficient-product with the relevant overlap, and sum
        # everything
        return np.sum(
            np.tensordot(
                atomic_orbitals[el_1_orbital_ind],
                atomic_orbitals[el_2_orbital_ind],
                0) * overlap_reduced)

    # Call the function
    coop = np.apply_along_axis(
        coop_func,
        0,
        atomic_orbitals,
        overlap_reduced,
        el_1_orbital_ind,
        el_2_orbital_ind)

    # Return the calculated crystal orbital overlap population
    return coop


def print_coop(energies, coop):
    """Saves the COOP in a txt-file."""
    result_coop = np.zeros((len(coop), 2))
    result_coop[:, 0], result_coop[:, 1] = energies, coop
    np.savetxt('COOP.txt', result_coop)

    return result_coop
