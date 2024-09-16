import numpy as np
import torch
from pymatgen.io.ase import AseAtomsAdaptor
from torch_geometric.data import Data, Dataset

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
        edge_index, offsets = self._get_neighbors_pymatgen(self.atoms[idx])
        outputs['edge_index'] = edge_index.astype(np.int64)
        outputs['pos'] = self.atoms[idx].get_positions().astype(np.float32)
        #outputs['Z'] = self.atoms[idx].numbers.astype(np.int64)
        outputs['cell_offset'] = offsets.astype(np.float32)
        outputs['unit_cell'] = self.atoms[idx].get_cell().astype(np.float32)
        outputs['x'] = atoms_to_onehot_np(self.atoms[idx])
        outputs['energy'] = np.array([self.atoms[idx].get_potential_energy()]).astype(np.float32)
        return outputs

    def _get_neighbors_pymatgen(self, atoms):
        struct = AseAtomsAdaptor.get_structure(atoms)
        _c_index, _n_index, _offsets, n_distance = struct.get_neighbor_list(
            r=self.cutoff, numerical_tol=0, exclude_self=True
        )
        _nonmax_idx = []
        for i in range(len(atoms)):
            idx_i = (_c_index == i).nonzero()[0]
            idx_sorted = np.argsort(n_distance[idx_i])
            _nonmax_idx.append(idx_i[idx_sorted])
        _nonmax_idx = np.concatenate(_nonmax_idx)

        _c_index = _c_index[_nonmax_idx]
        _n_index = _n_index[_nonmax_idx]
        _offsets = _offsets[_nonmax_idx]

        edge_index = np.vstack((_n_index, _c_index))
        cell_offsets = _offsets

        return edge_index, cell_offsets

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
    # Step 1: Get atomic numbers as a numpy array
    atomic_numbers = atom.numbers
    
    # Find unique atomic numbers and sort them
    unique_atomic_numbers = np.unique(atomic_numbers)
    
    # Map atomic numbers to sequential indices starting from 0
    mapping = {num: i for i, num in enumerate(unique_atomic_numbers)}
    mapped_atomic_numbers = np.vectorize(mapping.get)(atomic_numbers)
    
    # Step 2: One-hot encode the mapped atomic numbers
    num_atom_types = len(unique_atomic_numbers)
    one_hot_encoded = np.eye(num_atom_types)[mapped_atomic_numbers]

    # Step 3: Add an extra dimension (e.g., fifth dimension with zeros)
    extra_dimension = np.zeros((one_hot_encoded.shape[0], 1), dtype=np.float32)
    one_hot_encoded_plus = np.hstack((one_hot_encoded, extra_dimension))
    
    return one_hot_encoded_plus.astype(np.float32)

