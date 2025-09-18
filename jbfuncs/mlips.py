from jobflow import job
import json

@job
def sevenn_static_energy(structure_json, device):
    from sevenn.calculator import SevenNetCalculator
    from pymatgen.core.structure import Structure
    import os
    from pathlib import PurePath
    p = PurePath(os.getenv('SEVENN_CHECKPOINT'))
    atoms = Structure.from_dict(json.loads(structure_json)).to_ase_atoms()
    calc = SevenNetCalculator(p, modal='mpa', device=device)
    atoms.set_calculator(calc)
    return atoms.get_potential_energy()

@job
def sevenn_structure_optimize(structure_json, device):
    from sevenn.calculator import SevenNetCalculator
    from pymatgen.core.structure import Structure
    import os
    from pathlib import PurePath
    p = PurePath(os.getenv('SEVENN_CHECKPOINT'))
    atoms = Structure.from_dict(json.loads(structure_json)).to_ase_atoms()
    calc = SevenNetCalculator(p, modal='mpa', device=device)
    from ase.optimize import BFGS
    optimizer = BFGS
    atoms.set_calculator(calc)
    relax = optimizer(atoms)
    relax.run(fmax = 0.05, steps = 10000)
    Structure.from_ase_atoms(atoms).to_file('POSCAR')
    return {'structure':Structure.from_ase_atoms(atoms).to_json(), \
            'energy':atoms.get_potential_energy()}
