"""Parallel FEM simulation using PyTorch DataLoader."""
#
#                                                                       Modules
# =============================================================================
import torch
from torch.utils.data import Dataset, DataLoader
from run_simulation import Simulation
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira Morais Pinto (rui_pinto@brown.edu)'
__credits__ = ['Rui B. M. Pinto', ]
__status__ = 'Development'
# =============================================================================
torch.set_default_dtype(torch.float64)
# =============================================================================
class SimulationDataset(Dataset):
    """Dataset for parallel FEM simulations.

    Each item represents a complete simulation configuration. Worker
    processes execute simulations independently and return results.
    """

    def __init__(self, configs):
        """Initialize dataset with simulation configurations.

        Args:
            configs (list): List of dicts, each containing parameters
                for Simulation.__init__
        """
        self.configs = configs

    def __len__(self):
        return len(self.configs)

    def __getitem__(self, idx):
        """Run simulation for given configuration index.

        Args:
            idx (int): Configuration index

        Returns:
            dict: Simulation results including patch_idx and
                simulation_data
        """
        config = self.configs[idx]
        sim = Simulation(**config)
        sim.run()

        return {
            'patch_idx': config['patch_idx'],
            'simulation_data': sim.simulation_data,
            'config': config
        }
# =============================================================================
def _collate_single(batch):
    """Unwrap single-item batch for DataLoader collation.
    
    With batch_size=1:

    - Worker calls __getitem__(idx) once → returns single simulation result
    - collate_fn receives [result] (list with one element)
    - _collate_single unpacks to return result directly
    
    With  batch_size=4:
    
    - Worker calls __getitem__ 4 times → 4 independent simulations run
        sequentially in that worker
    - collate_fn receives [result1, result2, result3, result4]
    """
    return batch[0]
# =============================================================================
def run_parallel_simulations(
    configs,
    num_workers=4,
    verbose=True
):
    """Execute multiple simulations in parallel using DataLoader.

    Args:
        configs (list): List of simulation configuration dicts
        num_workers (int): Number of parallel worker processes
        verbose (bool): Print progress information

    Returns:
        list: Simulation results from all workers
    """
    dataset = SimulationDataset(configs)

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=_collate_single,
        persistent_workers=False
    )

    results = []
    for i, result in enumerate(loader):
        if verbose and i % 100 == 0:
            print(f'Completed {i}/{len(configs)} simulations')
        results.append(result)

    return results
# =============================================================================
if __name__ == '__main__':
    filepath = '/Volumes/T7/material_patches_data/'

    configs = []
    for idx in range(0, 15):
        configs.append({
            'element_type': 'quad4',
            'material_behavior': 'elastoplastic_nlh',
            'num_increments': 100,
            'patch_idx': idx,
            'filepath': filepath,
            'mesh_nx': 3,
            'mesh_ny': 3,
            'mesh_nz': 1,
            'is_save': True,
            'is_red_int': True
        })

    results = run_parallel_simulations(
        configs,
        num_workers=8,
        verbose=True
    )

    print(f'Completed!')
