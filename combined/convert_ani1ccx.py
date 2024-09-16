import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from ase.neighborlist import NeighborList

class Convert(Dataset):
    def __init__(self, atoms: list, cutoff: float):
        super().__init__()
        self.atoms = atoms
        self.cutoff = cutoff

    def get(self, idx: int):
        properties = self.get_properties(idx)
        properties["_idx"] = np.array([idx], dtype=np.int32)
        return torchify_dict(properties)

    def len(self):
        return len(self.atoms)

    def get_properties(self, idx: int):
        outputs = {}
        edge_index = self._get_neighbors_ase(self.atoms[idx])
        outputs['edge_index'] = edge_index.astype(np.int64)
        outputs['pos'] = self.atoms[idx].get_positions().astype(np.float32)
        outputs['Z'] = self.atoms[idx].numbers.astype(np.int64)
        outputs['x'] = atoms_to_onehot_np(self.atoms[idx])
        outputs['energy'] = np.array([self.atoms[idx].get_potential_energy()]).astype(np.float32)
        return outputs

    def _get_neighbors_ase(self, atoms):
        # Initialize the neighbor list with a single cutoff distance
        neighbor_list = NeighborList(cutoffs=[self.cutoff] * len(atoms), self_interaction=False, bothways=True)
        neighbor_list.update(atoms)
        
        edge_indices = []

        for i in range(len(atoms)):
            indices, _ = neighbor_list.get_neighbors(i)
            for j in indices:
                edge_indices.append([i, j])

        edge_index = np.array(edge_indices).T

        return edge_index

def torchify_dict(data: dict):
    torch_properties = {}
    for pname, prop in data.items():
        if prop.dtype == np.int32:
            torch_properties[pname] = torch.IntTensor(prop)
        elif prop.dtype == np.int64:
            torch_properties[pname] = torch.LongTensor(prop)
        elif prop.dtype == np.float32:
            torch_properties[pname] = torch.FloatTensor(prop.copy())
        elif prop.dtype == np.float64:
            torch_properties[pname] = torch.DoubleTensor(prop.copy())
        else:
            raise ValueError(f"Invalid datatype {type(prop)} for property {pname}!")

    final_data = Data()
    for key, value in torch_properties.items():
        setattr(final_data, key, value)

    return final_data

def atoms_to_onehot_np(atom):
    # Define the fixed set of atom types
    atom_types = ['H', 'C', 'N', 'O']
    atom_number_to_index = {1: 0, 6: 1, 7: 2, 8: 3}  # Mapping of atomic numbers to indices
    
    # Get atomic numbers for the given atom
    atomic_numbers = atom.numbers
    
    # Initialize the one-hot encoded array
    one_hot_encoded = np.ones((len(atomic_numbers), len(atom_types)), dtype=np.float32)
    
    # Fill the one-hot encoded array based on atomic numbers
    for i, number in enumerate(atomic_numbers):
        if number in atom_number_to_index:
            one_hot_encoded[i, atom_number_to_index[number]] = 1.0
    
    # Add an extra dimension of ones
    extra_dimension = np.ones((one_hot_encoded.shape[0], 1), dtype=np.float32)
    one_hot_encoded_plus = np.hstack((one_hot_encoded, extra_dimension))
    
    return one_hot_encoded_plus.astype(np.float32)

