__author__ = "Felipe Zapata"

# ================> Python Standard  and third-party <==========
from .initialization import (
    create_map_index_pyxaid, read_swaps, read_time_dependent_coeffs)
from nac.schedule.components import calculate_mos
from nac.common import (
    Matrix, change_mol_units, femtosec2au,
    retrieve_hdf5_data)
from nac.schedule.scheduleET import (
    compute_overlaps_ET, photo_excitation_rate)
from nac.workflows.initialization import initialize
from noodles import (gather, schedule)
from qmflows import run
from qmflows.parsers import parse_string_xyz
from scipy import integrate

import logging
import numpy as np

from typing import (Dict, List)

# Get logger
logger = logging.getLogger(__name__)


def workflow_electron_transfer(workflow_settings: Dict):
    """
    Use a MD trajectory to calculate the Electron transfer rate.

    :param workflow_settings: Arguments to compute the oscillators see:
    `data/schemas/electron_transfer.json
    :returns: None
    """
    # Arguments to compute the orbitals and configure the workflow. see:
    # `data/schemas/general_settings.json
    config = workflow_settings['general_settings']

    # Dictionary containing the general configuration
    config.update(initialize(**config))

    # Point calculations Using CP2K
    mo_paths_hdf5 = calculate_mos(**config)

    # geometries in atomic units
    molecules_au = [change_mol_units(parse_string_xyz(gs))
                    for gs in config['geometries']]

    # Time-dependent coefficients
    path_time_coeffs = workflow_settings['path_time_coeffs']
    time_depend_coeffs = read_time_dependent_coeffs(path_time_coeffs)
    msg = "Reading time_dependent coefficients from: {}".format(
        path_time_coeffs)
    logger.info(msg)

    # compute_overlaps_ET
    scheduled_overlaps = schedule(compute_overlaps_ET)
    fragment_overlaps = scheduled_overlaps(
        config['project_name'], molecules_au, config['basis_name'], config['path_hdf5'],
        config['dictCGFs'], mo_paths_hdf5, workflow_settings['fragment_indices'],
        config['enumerate_from'], config['package_name'])

    # Delta time in a.u.
    dt = workflow_settings['dt']
    dt_au = dt * femtosec2au

    # Indices relation between the PYXAID active space and the orbitals
    # stored in the HDF5
    args_map_index = [workflow_settings[key] for key in
                      ['orbitals_range', 'pyxaid_HOMO', 'pyxaid_Nmin', 'pyxaid_Nmax']]
    map_index_pyxaid_hdf5 = create_map_index_pyxaid(*args_map_index)

    # Number of points in the pyxaid trajectory:
    # shape: (initial_conditions, n_points, n_states)
    n_points = len(config['geometries']) - 2

    # Read the swap between Molecular orbitals obtained from a previous
    # Coupling calculation
    swaps = read_swaps(['path_hdf5'], ['project_name'])

    # Electron transfer rate for each frame of the Molecular dynamics
    scheduled_photoexcitation = schedule(compute_photoexcitation)
    etrs = scheduled_photoexcitation(
        config['path_hdf5'], time_depend_coeffs, fragment_overlaps,
        map_index_pyxaid_hdf5, swaps, n_points, ['pyxaid_iconds'], dt_au)

    # Execute the workflow
    electronTransferRates, path_overlaps = run(
        gather(etrs, fragment_overlaps), folder=config['work_dir'])

    for i, mtx in enumerate(electronTransferRates):
        write_ETR(mtx, dt, i)

    write_overlap_densities(config['path_hdf5'], path_overlaps, swaps, dt)


def compute_photoexcitation(
        path_hdf5: str, time_dependent_coeffs: Matrix,
        paths_fragment_overlaps: List, map_index_pyxaid_hdf5: Matrix,
        swaps: Matrix, n_points: int, pyxaid_iconds: List,
        dt_au: float) -> List:
    """
    :param i: Electron transfer rate at time i * dt
    :param path_hdf5: Path to the HDF5 file that contains the
    numerical results.
    :param geometries: list of 3 contiguous molecular geometries
    :param time_depend_paths: mean value of the time dependent coefficients
    computed with PYXAID.
    param fragment_overlaps: Tensor containing 3 overlap matrices corresponding
    with the `geometries`.
    :param map_index_pyxaid_hdf5: map from PYXAID excitation to the indices i,j
    of the molecular orbitals stored in the HDF5.
    :param swaps: Matrix containing the crossing between the MOs during the
    molecular dynamics.
    :param n_points: Number of frames to compute the ETR.
    :param pyxaid_iconds: List of initial conditions
    :param dt_au: Delta time in atomic units
    :returns: promise to path to the Coupling inside the HDF5.
    """
    msg = "Computing the photo-excitation rate for the molecular fragments"
    logger.info(msg)

    results = []

    for paths_overlaps in paths_fragment_overlaps:
        overlaps = np.stack(retrieve_hdf5_data(path_hdf5, paths_overlaps))
        # Track the crossing between MOs
        for k, mtx in enumerate(np.rollaxis(overlaps, 0)):
            overlaps[k] = mtx[:, swaps[k]][swaps[k]]

        etr = np.stack(np.array([
            photo_excitation_rate(
                overlaps[i: i + 3],
                time_dependent_coeffs[j, i: i + 3],
                map_index_pyxaid_hdf5, dt_au)
            for i in range(n_points)]) for j in range(len(pyxaid_iconds)))

        etr = np.mean(etr, axis=0)

        results.append(etr)

    return np.stack(results)


def write_ETR(mtx, dt, i):
    """
    Save the ETR in human readable format
    """
    file_name = "electronTranferRates_fragment_{}.txt".format(i)
    header = 'electron_Density electron_ETR(Nonadiabatic Adibatic) hole_density hole_ETR(Nonadiabatic Adibatic)'
    # Density of Electron/hole
    density_electron = mtx[1:, 0]
    density_hole = mtx[1:, 3]
    # Integrate the nonadiabatic/adiabatic components of the electron/hole ETR
    int_elec_nonadia, int_elec_adia, int_hole_nonadia, int_hole_adia = [
        integrate.cumtrapz(mtx[:, k], dx=dt) for k in [1, 2, 4, 5]]

    # Join the data
    data = np.stack(
        (density_electron, int_elec_nonadia, int_elec_adia, density_hole,
         int_hole_nonadia, int_hole_adia), axis=1)

    # save the data in human readable format
    np.savetxt(file_name, data, fmt='{:^3}'.format('%e'), header=header)


def write_overlap_densities(
        path_hdf5: str, paths_fragment_overlaps: List, swaps: Matrix, dt: int=1):
    """
    Write the diagonal of the overlap matrices
    """
    logger.info("writing densities in human readable format")

    # Track the crossing between MOs
    for paths_overlaps in paths_fragment_overlaps:
        overlaps = np.stack(retrieve_hdf5_data(path_hdf5, paths_overlaps))
        for k, mtx in enumerate(np.rollaxis(overlaps, 0)):
            overlaps[k] = mtx[:, swaps[k]][swaps[k]]

    # Print to file the densities for each fragment on a given MO
    for ifrag, paths_overlaps in enumerate(paths_fragment_overlaps):
        # time frame
        frames = overlaps.shape[0]
        ts = np.arange(1, frames + 1).reshape(frames, 1) * dt
        # Diagonal of the 3D-tensor
        densities = np.diagonal(overlaps, axis1=1, axis2=2)
        data = np.hstack((ts, densities))
        # Save data in human readable format
        file_name = 'densities_fragment_{}.txt'.format(ifrag)
        np.savetxt(file_name, data, fmt='{:^3}'.format('%e'))
