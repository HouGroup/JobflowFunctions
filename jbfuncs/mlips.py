from jobflow import job, Flow, Maker, Response
from pymatgen.core.structure import Structure
import json
from dataclasses import dataclass
from qtoolkit.core.data_objects import QResources
from jobflow_remote import submit_flow, set_run_config
from ase.calculators.calculator import Calculator
from typing import Callable, Dict, Any, Optional
from atomate2.vasp.jobs.core import BaseVaspMaker

@job
def mlip_job(structure):
    return structure.to_json()

@job
def mlip_to_vasp(structure_json, vasp_maker_json, key):
    import json
    structure = Structure.from_dict(json.loads(structure_json))
    job = BaseVaspMaker.from_dict(json.loads(vasp_maker_json)).make(structure)
    job.update_metadata({'key':key})
    return Response(replace = Flow([job]))

@dataclass
class SevennToVasp(Maker):
    name: str = 'sevenn_to_vasp'
    mlip_resources: Callable = None
    mlip_device: str = 'cpu'
    vasp_resources: Callable = None
    vasp_exec_config: Dict[str, Any] = None
    vasp_maker: Callable = None
    key: Dict[str, Any] = None

    def make(self, structure):
        job1 = sevenn_structure_optimize(structure.to_json(), 'cpu')
        job1 = set_run_config(job1, worker = 'std_worker',
                              resources = self.mlip_resources,
                              priority=10,
                              dynamic = False)
        
        job2 = mlip_to_vasp(job1.output['structure'],
                            self.vasp_maker.to_json(),
                            self.key)
        
        job2 = set_run_config(job2,
                              worker = 'std_worker',
                              exec_config={'pre_run':'module load VASP/6.4.3-optcell'},
                              resources = self.vasp_resources,
                              priority=0,
                              dynamic = True)
        
        return Flow([job1, job2])

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
