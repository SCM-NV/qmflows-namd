

from itertools import (groupby, starmap)
from nac.basisSet.basisNormalization import compute_normalization_sphericals
from nac.common import (
    Matrix, Tensor3D, Vector, retrieve_hdf5_data, search_data_in_hdf5,
    store_arrays_in_hdf5, triang2mtx)
from nac.integrals.overlapIntegral import calcMtxOverlapP
from nac.integrals.nonAdiabaticCoupling import calculate_spherical_overlap
from nac.integrals.spherical_Cartesian_cgf import calc_transf_matrix
from os.path import join
from scipy import sparse
from typing import (Dict, List, Tuple)
import hashlib
import h5py
import logging
import numpy as np

# Get logger
logger = logging.getLogger(__name__)


def photo_excitation_rate(
        tensor_overlaps: Tensor3D, time_dependent_coeffs: Matrix,
        map_index_pyxaid_hdf5: Matrix, dt_au: float) -> Tuple:
    """
    Calculate the Electron transfer rate, using both adiabatic and nonadiabatic
    components, using equation number 8 from:
    J. AM. CHEM. SOC. 2005, 127, 7941-7951. Ab Initio Nonadiabatic Molecular
    Dynamics of the Ultrafast Electron Injection acrossthe Alizarin-TiO2
    Interface.
    The derivatives are calculated numerically using 3 points.

    :param tensor_overlaps: Overlap matrices at time t - dt, t and t + dt.
    :param time_dependent_coeffs: Time-dependentent coefficients
    at time t - dt, t and t + dt.
    :param map_index_pyxaid_hdf5: Index relation between the Excitations
    in PYXAID and the corresponding molecular orbitals store in the HDF5.
    :param dt_au: Delta time integration used in the dynamics.
    :returns: tuple containing both nonadiabatic and adiabatic components
    """
    # Rearrange the overlap matrix in the PYXAID order
    hole_indices, electron_indices = map_index_pyxaid_hdf5.transpose()

    # Overlaps for the electron and the hole pair
    electron_matrix_overlap = tensor_overlaps[1][
        electron_indices, electron_indices]
    hole_matrix_overlap = tensor_overlaps[1][hole_indices, hole_indices]

    # Density of the fragment
    electron_density = np.dot(
        electron_matrix_overlap, time_dependent_coeffs[1])
    hole_density = np.dot(hole_matrix_overlap, time_dependent_coeffs[1])

    # NonAdiabatic component
    coeff_derivatives = np.apply_along_axis(
        lambda v: (v[2] - v[0]) / (2 * dt_au), 0, time_dependent_coeffs)

    # Nonadiabatic components of the electron/hole transfer
    electron_nonadiabatic = np.dot(coeff_derivatives, electron_matrix_overlap)
    hole_nonadiabatic = np.dot(coeff_derivatives, hole_matrix_overlap)

    # Adiabatic component
    overlap_derv = np.apply_along_axis(
        lambda v: (v[2] - v[0]) / (2 * dt_au), 0, tensor_overlaps)

    # Adiabatic components of the electron/hole transfer
    electron_overlap_derv = overlap_derv[electron_indices, electron_indices]
    hole_overlap_derv = overlap_derv[hole_indices, hole_indices]

    electron_adiabatic = np.dot(
        time_dependent_coeffs[1], electron_overlap_derv)
    hole_adiabatic = np.dot(time_dependent_coeffs[1], hole_overlap_derv)

    rs = (electron_density, electron_nonadiabatic, electron_adiabatic,
          hole_density, hole_nonadiabatic, hole_adiabatic)

    return rs


def compute_overlaps_ET(
        project_name: str, molecules: List, basis_name: str,
        path_hdf5: str, dictCGFs: Dict, mo_paths_hdf5: List,
        fragment_indices: List, enumerate_from: int,
        package_name: int) -> List:
    """
    Given a list of molecular fragments compute the Overlaps in the
    molecular orbital basis for the different fragments and for all the
    molecular frames.
    """
    # Preprocess the indices of the atomic fragments
    fragment_indices = sanitize_fragment_indices(fragment_indices)

    # Matrix containing the lower and upper range for the CGFs of each atom
    # in order.
    indices_range_CGFs_spherical = create_indices_range_CGFs_spherical(
        molecules[0], dictCGFs)

    # Norms of the spherical CGfs
    dict_global_norms = compute_normalization_sphericals(dictCGFs)

    fragment_overlaps = []
    for k, vector_indices in enumerate(fragment_indices):
        logger.info("Computing Overlaps for molecular fragment: {}".format(k))
        # Extract atoms belonging to the fragment
        frames_fragment_atoms = [[mol[i] for i in vector_indices]
                                 for mol in molecules]
        # compute the transformation matrix from cartesian to spherical
        sparse_trans_mtx = compute_fragment_trans_mtx(
            path_hdf5, basis_name, frames_fragment_atoms[0], dict_global_norms,
            package_name)

        # create a Hash for the fragment
        fragment_hash = hashlib.md5(vector_indices.tostring()).hexdigest()
        logger.info("The overlaps for the molecular fragment number {} are going \
        to be stored in the hdf5 using the following hash: {}".format(k, fragment_hash))

        # Compute the indices of the MOs corresponding to the atoms in the
        # fragments
        indices_fragment_mos = compute_indices_fragments_mos(
            vector_indices, indices_range_CGFs_spherical)

        # Overlap matrix for the given fragment
        overlaps = compute_frames_fragment_overlap(
            project_name, frames_fragment_atoms, path_hdf5, mo_paths_hdf5,
            indices_fragment_mos, dictCGFs, sparse_trans_mtx, enumerate_from,
            fragment_hash)
        fragment_overlaps.append(overlaps)

    return fragment_overlaps


def compute_frames_fragment_overlap(
        project_name: str, frames_fragment_atoms: List, path_hdf5: str,
        mo_paths_hdf5: List, indices_fragment_mos: Vector, dictCGFs: Dict,
        sparse_trans_mtx: Matrix, enumerate_from: int,
        fragment_hash: int) -> List:
    """
    Compute all the overlap matrices for the frames of a molecular dynamics
    for a given fragment.
    """
    # Path to store the fragment overlap matrices in the HDF5
    range_points = range(
        enumerate_from, enumerate_from + len(frames_fragment_atoms))
    paths = [join(project_name, 'point_{}'.format(i),
                  'fragment_overlap_hash_{}'.format(fragment_hash))
             for i in range_points]

    return [compute_fragment_overlap(
        path_hdf5, p, fragment, path_mos, indices_fragment_mos, dictCGFs,
        sparse_trans_mtx)
        for p, fragment, path_mos in
        zip(paths, frames_fragment_atoms, mo_paths_hdf5)]


def compute_fragment_overlap(
        path_hdf5: str, path_overlap: str, fragment_atoms: List, path_mos: str,
        indices_fragment_mos: Vector, dictCGFs: Dict,
        trans_mtx: Matrix) -> str:
    """
    Compute the overlap matrix only for those atoms included in the fragment
    """
    if not search_data_in_hdf5(path_hdf5, path_overlap):
        logger.info("Computing overlap:{}".format(path_overlap))
        # Compute the overlap in the atomic basis
        dim_cart = trans_mtx.shape[1]
        overlap_AO = triang2mtx(
            calcMtxOverlapP(fragment_atoms, dictCGFs), dim_cart)
        # Read all the molecular orbital coefficients
        coefficients = retrieve_hdf5_data(path_hdf5, path_mos[1])
        # Number of Orbitals stored in the HDF5
        dim_y = coefficients.shape[1]
        # Extract the coefficients belonging to the fragment
        dim_x = indices_fragment_mos.size

        # Extract the MO Coefficients belonging to the fragment
        x_range = np.repeat(indices_fragment_mos, dim_y)
        y_range = np.tile(np.arange(dim_y), dim_x)
        fragment_coefficients = coefficients[x_range, y_range].reshape(dim_x, dim_y)

        # Compute the overlap in spherical coordinates
        overlap_spherical = calculate_spherical_overlap(
            trans_mtx, overlap_AO, fragment_coefficients, fragment_coefficients)
        store_arrays_in_hdf5(path_hdf5, path_overlap, overlap_spherical)
    else:
        logger.info("overlap:{} already on HDF5".format(path_overlap))

    return path_overlap


def create_indices_range_CGFs_spherical(
        molecule: List, dictCGFs: Dict) -> Matrix:
    """
    Creates a matrix containing the lower and upper(exclusive) indices
    of the CGFs for each atoms in order.
    """
    # Compute how many CGFs are in Spherical coordinates

    lens_CGFs_spherical = compute_lens_CGFs_sphericals(molecule, dictCGFs)
    ranges = np.empty((len(molecule), 2), dtype=np.int32)

    lower = 0
    for i, at in enumerate(molecule):
        upper = lower + lens_CGFs_spherical[at.symbol]
        ranges[i] = lower, upper
        lower = upper

    return ranges


def compute_lens_CGFs_sphericals(molecule: List, dictCGFs: Dict) -> Dict:
    """
    Calculate how many CGFs in sphericals are there per atom
    """
    # Number of CGFs per angular momenta in Cartesian coordinates
    CGFs_cartesians = {'S': 1, 'P': 3, 'D': 6, 'F': 10}

    # Number of CGFs per angular momenta in spherical coordinates
    CGFs_sphericals = {'S': 1, 'P': 3, 'D': 5, 'F': 7}

    # Unique labels
    labels = set(at.symbol for at in molecule)

    # Compute number of spherical CGFs per atoms
    lens_CGFs_spherical = {
        l: int(
            sum(len(list(vals)) * CGFs_sphericals[g] / CGFs_cartesians[g]
                for g, vals in groupby(dictCGFs[l], lambda cgf: cgf.orbType[0])))
        for l in labels}

    return lens_CGFs_spherical


def compute_fragment_trans_mtx(
        path_hdf5: str, basis_name: str, fragment_atoms: List,
        dict_global_norms: Dict, package_name: str) -> Matrix:
    """
    Calculate the cartesian to spherical transformation matrix for
    the given fragment
    """
    with h5py.File(path_hdf5, 'r') as f5:
        trans_mtx = calc_transf_matrix(
            f5, fragment_atoms, basis_name, dict_global_norms, package_name)

    return sparse.csr_matrix(trans_mtx)


def compute_indices_fragments_mos(
        vector_indices: Vector, indices_range_CGFs: Matrix) -> Vector:
    """
    Compute the indices of the CGFs of the fragments to extract from
    the coefficients stored in the HDF5.
    """
    # extract the ranges of CGFs for the fragments
    fragment_ranges = indices_range_CGFs[vector_indices]

    return np.concatenate(list(starmap(np.arange, fragment_ranges)))


def sanitize_fragment_indices(fragment_indices: List) -> Matrix:
    """
    transform to numpy array and start the indices from 0.

    :param fragment_indices: indices of the atoms belonging to the different
    molecular fragments.
    """
    # Convert the indices to a numpy array
    if isinstance(fragment_indices[0], list):
        fragment_indices = [np.array(xs, dtype=np.int) for xs in fragment_indices]
    else:
        fragment_indices = [np.array(fragment_indices, dtype=np.int)]

    # Shift the index 1 position to start from 0
    fragment_indices = [xs - 1 for xs in fragment_indices]

    return fragment_indices
