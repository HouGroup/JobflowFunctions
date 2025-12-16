import numpy as np
from pymatgen.core.structure import Structure
#from sevenn.calculator import SevenNetCalculator
from pymatgen.core.surface import SlabGenerator
from pymatgen.transformations.site_transformations \
import TranslateSitesTransformation, RemoveSitesTransformation
from pymatgen.transformations.standard_transformations \
import SupercellTransformation
from pymatgen.transformations.standard_transformations import RotationTransformation
from pymatgen.core.lattice import Lattice
from interfacemaster.cellcalc import get_pri_vec_inplane, get_right_hand
from interfacemaster.interface_generator import cross_plane
from interfacemaster.cellcalc import rot
import subprocess
import os
from dataclasses import dataclass
from jobflow import Flow, Response, job, Maker
from typing import Callable, Dict, Any, List
#from pymatgen.io.cif import CifWriter, CifParser
#from deepmd.calculator import DP
#calc=DP(model="/home/mzy/DP/LLZO/potential_file/LLZO-c.pb")
#按照最小的位移去移动
def shift_slab_to_origin(slab):
    coords = np.array([i.coords for i in slab if i.label == 'Zr'])
    lengths = np.linalg.norm(coords, axis = 1)
    shift = - coords[lengths == min(lengths)][0] + 1e-4
    tt = TranslateSitesTransformation(np.arange(len(slab)), shift, vector_in_frac_coords=False)
    return tt.apply_transformation(slab)
#第一，abc矩阵的读取需要注意，第二输出矩阵 = 原始矩阵的正交归一化版本的转置
def help_matrix(matrix):
    a,b,c = matrix.T
    c_h = np.cross(a,b)
    b_h = np.cross(c_h,a)
    return np.column_stack((a/np.linalg.norm(a), b_h/np.linalg.norm(b_h), c_h/np.linalg.norm(c_h)))
#得倒每一个点的距离
def get_dist_from_mp(structure):
    a, b, c = structure.lattice.matrix
    middle_point = (a+b+c)*1/2
    coords = structure.cart_coords
    return np.linalg.norm(coords - middle_point, axis=1)
#对晶体结构进行对称性分析，并添加一个名为 site_labels的位点属性，该属性包含每个原子的序号#对称等价位置标签和Wyckoff字母
def get_symmetrized_structure(structure):
    #from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    #spa = SpacegroupAnalyzer(structure)
    #sym_stct = spa.get_symmetrized_structure()
    #site_properties = [(sym_stct[i].label,sym_stct.wyckoff_letters[i]) for i in range(len(sym_stct))]
    return structure.add_site_property('site_labels', np.arange(len(structure)))
#将一个晶体结构绕Z轴旋转指定的角度（以弧度为单位），并返回旋转后的新结构
def get_rotated_structure(structure, R):
    new_lattice = (np.dot(R, structure.lattice.matrix.T)).T
    return Structure(lattice=new_lattice,
                     species=structure.species,
                     coords=structure.frac_coords,
                     site_properties=structure.site_properties,
                    )
#转化坐标系，改成绝对坐标，gpt说将a对应到x，旋转晶格
def align_ab_to_xy(structure):
    old_orient = help_matrix(structure.lattice.matrix.T)
    new_orient= np.eye(3)
    transform = np.linalg.inv(old_orient)
    new_matrix = np.dot(transform, structure.lattice.matrix.T)
    new_lattice = Lattice(new_matrix.T)
    return Structure(lattice=new_lattice,
                     species=structure.species,
                     coords=(np.dot(transform, structure.cart_coords.T)).T,
                     coords_are_cartesian=True,
                    site_properties=structure.site_properties)
#生成一个给定晶体结构和晶面指数（Miller index）的原始表面模型（primitive slab），并将其平移到坐标原点
def get_primitive_slab_hkl(structure, miller_index):
    B  = get_pri_vec_inplane(miller_index,
                             lattice=structure.lattice.matrix.T)
    n = cross_plane(lattice = structure.lattice.matrix.T,
                    n = np.cross(B[:,0], B[:,1]),
                    lim = 10,
                    tol = 0.1,
                    orthogonal = False)
    sup_cart = get_right_hand(np.column_stack((B,n)))
    scaling_matrix = np.dot(np.linalg.inv(LLZO.lattice.matrix.T), sup_cart).T
    st = SupercellTransformation(scaling_matrix=scaling_matrix)
    return shift_slab_to_origin(align_ab_to_xy(st.apply_transformation(structure)))

def redefine_cubic_lattice(structure):
    return Structure(lattice=np.eye(3) * structure.lattice.a,
                     species=structure.species,
                     coords=structure.cart_coords,
                     site_properties=structure.site_properties,
                     coords_are_cartesian = True
                    )

#将一个晶体结构复制成超晶格，使得超晶格的最小包围盒能够容纳一个半径为R的球体
import numpy as np
def replicate_to_sphere_size(structure, R):
    """
    make a supercell to reach a size enclosing a sphere
    with radius R
    """
    a, b, c = structure.lattice.matrix
    n1 = np.cross(b,c)
    h1 = abs(np.dot(a, n1)/np.linalg.norm(n1))
    n2 = np.cross(a,c)
    h2 = abs(np.dot(b, n2)/np.linalg.norm(n2))
    n3 = np.cross(a,b)
    h3 = abs(np.dot(c, n3)/np.linalg.norm(n3))
    dim1 = int(np.ceil(R*2/h1))
    dim2 = int(np.ceil(R*2/h2))
    dim3 = int(np.ceil(R*2/h3))
    st = SupercellTransformation(((dim1, 0, 0), (0, dim2, 0), (0, 0, dim3)))
    return st.apply_transformation(structure), [dim1, dim2, dim3]
#从一个超晶格结构中提取一个球形区域（半径为r）
def sphere_from_structure(structure, r):
    """
    extract a sphere from a supercell
    """
    distance_from_mp = get_dist_from_mp(structure)
    remove_indices = np.where(distance_from_mp > r)[0]
    rmt = RemoveSitesTransformation(remove_indices)
    structure = rmt.apply_transformation(structure)

    return structure
#固定远距离原子：将距离大于阈值rstar的原子设置为在所有方向上固定
#将两个晶体结构（st1和st2）沿Z轴方向组合成一个新的结构，并在它们之间添加指定的间隙（gap）
def combine_two_structures(st1, st2, gap):
    lattice = st1.lattice
    species = st1.species + st2.species

    middle_point_1 = np.dot([0.5,0.5,0.5], st1.lattice.matrix)
    middle_point_2 = np.dot([0.5,0.5,0.5], st2.lattice.matrix)
    
    coords = np.append(st1.cart_coords,
                       st2.cart_coords + middle_point_1 - middle_point_2 + [0,0,gap],
                       axis = 0)
    site_properties = {'site_labels': st1.site_properties['site_labels'] + st2.site_properties['site_labels']}
    return Structure(lattice=lattice,
                     species=species,
                     coords=coords,
                     coords_are_cartesian=True,
                    site_properties=site_properties)
#晶体结构（structure）转换到指定边长的立方晶格中，同时保持原子在空间中的实际位置不变（即笛卡尔坐标不变）
def to_cubic_lattice_structure(structure, cubic_a):
    a, b, c = structure.lattice.matrix
    middle_point = (a+b+c)*1/2
    lattice = Lattice.cubic(cubic_a)
    a,b,c = lattice.matrix
    new_middle_point = 1/2*(a+b+c)
    shift = new_middle_point - middle_point

    new_structure = Structure(lattice=lattice,
                     species=structure.species,
                     coords=structure.cart_coords,
                     coords_are_cartesian=True,
                     site_properties=structure.site_properties)
    tt = TranslateSitesTransformation(np.arange(len(new_structure)), new_middle_point-middle_point, False)

    return tt.apply_transformation(new_structure)

def save_site_properties_to_json(unique_sps, indices, filename="site_properties.json"):
    """
    保存为JSON格式，保留完整的数据结构
    """
    # 将numpy数组转换为Python原生类型
    def convert_to_serializable(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    data = {
        'unique_site_properties': convert_to_serializable(unique_sps),
        'atom_indices_by_type': convert_to_serializable(indices),
        'summary': {
            'total_unique_types': len(unique_sps),
            'total_atoms': sum(len(idx_list) for idx_list in indices)
        }
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

def add_surface_site_property(structure, rstar):
    dyn_mtx = np.array([[True, True, True]] * len(structure))
    dists = get_dist_from_mp(structure)
    dyn_mtx[dists > rstar] = [False, False, False]
    return structure.add_site_property('selective_dynamics', dyn_mtx)

def rotation_to_z(vector):
    """
    另一种实现方法：通过构建正交基
    
    Parameters:
    -----------
    vector : array-like, shape (3,)
        输入的三维向量
    
    Returns:
    --------
    R : ndarray, shape (3, 3)
        旋转矩阵
    """
    vector = np.array(vector, dtype=float)
    norm = np.linalg.norm(vector)
    
    if norm < 1e-10:
        raise ValueError("Input vector cannot be zero vector")
    
    v = vector / norm
    
    # 如果已经指向 z 轴方向
    if np.allclose(v, [0, 0, 1]):
        return np.eye(3)
    if np.allclose(v, [0, 0, -1]):
        return np.array([[-1, 0, 0],
                          [0, -1, 0],
                          [0, 0, 1]])
    
    # 构建新的正交基，其中 v 作为 z 轴
    # 选择一个与 v 不平行的向量
    if abs(v[0]) < 0.9:
        x_axis = np.array([1, 0, 0])
    else:
        x_axis = np.array([0, 1, 0])
    
    # Gram-Schmidt 正交化
    z_new = v  # 新基的 z 轴
    y_new = np.cross(z_new, x_axis)
    y_new = y_new / np.linalg.norm(y_new)
    x_new = np.cross(y_new, z_new)
    
    # 构建从标准基到新基的旋转矩阵
    # 注意：我们要的是从新基到标准基的旋转，所以转置
    R_new_to_std = np.column_stack([x_new, y_new, z_new])
    
    # 但我们需要的是将标准基的 z 轴旋转到 v 方向
    # 所以需要逆矩阵（对于旋转矩阵，逆矩阵等于转置）
    R = R_new_to_std.T
    
    return R

class SphereGenerator:
    def __init__(self, structure):
        self.structure = shift_slab_to_origin(get_symmetrized_structure(structure))
        
    def get_hemisphere(self, r, dr, dxdydz = [0,0,0], R=np.eye(3), normal = [0,0,1], up = True):
        supercell, dims = replicate_to_sphere_size(self.structure, r + dr)
        supercell = get_rotated_structure(supercell, R)
        dxdydz = [0.5 + dxdydz[0]/dims[0], 0.5 + dxdydz[1]/dims[1], 0.5 + dxdydz[2]/dims[2]]
        tt = TranslateSitesTransformation(np.arange(len(supercell)), dxdydz)
        supercell = tt.apply_transformation(supercell)
        sphere = sphere_from_structure(supercell, r)

        r_vecs = sphere.cart_coords - np.dot(sphere.lattice.matrix.T, [0.5,0.5,0.5])
        if up == True:
            remove_ids = np.where(np.dot(r_vecs, normal) >= 0)[0]
        else:
            remove_ids = np.where(np.dot(r_vecs, normal) < 0)[0]
        rmt = RemoveSitesTransformation(remove_ids)
        return rmt.apply_transformation(sphere)

class SphereGBGenerator:
    def __init__(self, strcture, r, dr):
        self.structure = strcture
        self.r = r
        self.dr = dr
        
    def get_sphere_GB(self, R,
                          normal,
                          dxdydz_1,
                          dxdydz_2,
                          gap):
        sphg_1 = SphereGenerator(self.structure)
        hemisphere_1 = sphg_1.get_hemisphere(self.r, self.dr, dxdydz_1, np.eye(3), normal, True)
        hemisphere_1 = get_rotated_structure(hemisphere_1, rotation_to_z(normal))
        #hemisphere_1.to_file('1_POSCAR')
        sphg_2 = SphereGenerator(self.structure)
        hemisphere_2 = sphg_2.get_hemisphere(self.r, self.dr, dxdydz_2, R, normal,  False)
        hemisphere_2 = get_rotated_structure(hemisphere_2, rotation_to_z(normal))
        #hemisphere_2.to_file('2_POSCAR')
        cb_structure = combine_two_structures(hemisphere_1, hemisphere_2, gap)
        return to_cubic_lattice_structure(cb_structure, 2*(self.r+self.dr))

def write_LAMMPS(
        lattice,
        atoms,
        elements,
        filename='lmp_atoms_file',
        orthogonal=False):
    """
    write LAMMPS input atom file file of a supercell
    """

    # ------------ 自己指定元素和类型映射 ------------
    # 想要 Li, La, Zr, O 分别是 1,2,3,4
    type_map = {"Li": 1, "La": 2, "Zr": 3, "O": 4}

    # 检查有没有元素不在映射里
    unknown = set(np.unique(elements)) - set(type_map.keys())
    if unknown:
        raise ValueError(f"这些元素没有在 type_map 里定义: {unknown}")

    # list of elements（按 type 顺序只是为了写文件 header 时好看）
    items = sorted(type_map.items(), key=lambda kv: kv[1])
    element_species = np.array([k for k, v in items])
    element_indices = np.array([v for k, v in items])

    # to Cartesian
    atoms = np.dot(lattice, atoms.T).T

    # 用映射直接生成 species_identifiers
    species_identifiers = np.array([type_map[es] for es in elements]).reshape(1, -1)

    # the atom ID
    IDs = np.arange(len(atoms)).reshape(1, -1) + 1

    # get the final format
    Final_format = np.concatenate((IDs.T, species_identifiers.T), axis=1)
    Final_format = np.concatenate((Final_format, atoms), axis=1)

    # define the box
    xhi, yhi, zhi = lattice[0][0], lattice[1][1], lattice[2][2]
    xlo, ylo, zlo = 0, 0, 0
    xy = lattice[:, 1][0]
    xz = lattice[:, 2][0]
    yz = lattice[:, 2][1]

    with open(filename, 'w', encoding="utf-8") as f:
        f.write(
            '#LAMMPS input file of atoms generated by interface_master. '
            'The elements are: ')
        for ei, es in zip(element_indices, element_species):
            f.write(f'{ei} {es} ')
        f.write(f'\n {len(atoms)} atoms \n \n')
        f.write(f'{len(element_species)} atom types \n \n')
        f.write(f'{xlo:.8f} {xhi:.8f} xlo xhi \n')
        f.write(f'{ylo:.8f} {yhi:.8f} ylo yhi \n')
        f.write(f'{zlo:.8f} {zhi:.8f} zlo zhi \n\n')
        if not orthogonal:
            f.write(f'{xy:.8f} {xz:.8f} {yz:.8f} xy xz yz \n\n')
        f.write('Atoms \n \n')
        np.savetxt(f, Final_format, fmt='%i %i %.16f %.16f %.16f')

import numpy as np

def extract_last_column(traj_file, output_file='atomic_energies.dat'):
    """从LAMMPS轨迹文件中提取最后一列（每原子能量）"""
    
    energies = []
    atom_ids = []
    
    with open(traj_file, 'r') as f:
        lines = f.readlines()
    
    # 查找原子数据开始位置
    for i, line in enumerate(lines):
        if 'ITEM: ATOMS' in line:
            # 找到原子数据开始行
            data_start = i + 1
            # 解析列标题
            columns = line.strip().split()[2:]  # 去掉"ITEM: ATOMS"
            print(f"轨迹文件包含的列: {columns}")
            break
    
    # 提取每原子能量（最后一列）
    for i in range(data_start, len(lines)):
        line = lines[i].strip()
        if line and not line.startswith('ITEM:'):  # 跳过空行和标题行
            parts = line.split()
            if len(parts) >= len(columns):  # 确保有足够的数据
                atom_id = int(parts[0])
                energy = float(parts[-1])  # 最后一列
                energies.append(energy)
                atom_ids.append(atom_id)
    energies = np.array(energies)
    return atom_ids, energies

def get_sectional_error_by_r(indices, dists, energies, cut_dists, perfect_energies):
    errors_by_r = []
    sectional_errors_by_r = []
    for i in range(len(indices)):
        errors_by_r.append([])
        for j in cut_dists:
            energies_this_element = energies[indices[i]]
            indices_in_r = np.where(dists[indices[i]] < j)[0]
            errors_by_r[i].append(np.sum((energies_this_element[indices_in_r] - perfect_energies[i])))
        sectional_errors_by_r.append(np.array(errors_by_r[i])/(np.pi*cut_dists**2)*16.02176634)
    total_sectional_errors = np.sum(sectional_errors_by_r, axis=0)
    return errors_by_r, sectional_errors_by_r,total_sectional_errors

def read_lammps_atoms(filename):
    import numpy as np
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    xlo = xhi = ylo = yhi = zlo = zhi = None
    for line in lines:
        if "xlo xhi" in line:
            p = line.split()
            xlo, xhi = float(p[0]), float(p[1])
        elif "ylo yhi" in line:
            p = line.split()
            ylo, yhi = float(p[0]), float(p[1])
        elif "zlo zhi" in line:
            p = line.split()
            zlo, zhi = float(p[0]), float(p[1])

    a = xhi - xlo
    b = yhi - ylo
    c = zhi - zlo

    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Atoms"):
            start_idx = i + 2
            break

    data = np.loadtxt(lines[start_idx:])  # id type x y z
    ids   = data[:, 0].astype(int)
    types = data[:, 1].astype(int)
    coords = data[:, 2:5]  # x y z，笛卡尔坐标

    return ids, types, coords, (a, b, c)

from jobflow import job

@job
def sample_gb_energy(crystal_structure,
                    check_point_file,
                    bulk_energy_traj_file,
                    sphere_R,
                    vaccum_thickness,
                    gb_r,
                    rot_axis,
                    rot_angle,
                    normal,
                    xyz_1,
                    xyz_2,
                    gap):
    
    #generate gb
    sgg = SphereGBGenerator(crystal_structure, sphere_R, vaccum_thickness)
    gb = sgg.get_sphere_GB(rot(rot_axis,rot_angle), normal, xyz_1, xyz_2, gap)
    
    #write lammps structure file
    lattice = np.array(gb.lattice.matrix)      # 和你函数里的 lattice 对应
    atoms = np.array(gb.frac_coords)           # 和你函数里的 atoms 对应
    elements = np.array([str(sp) for sp in gb.species])  # 和你函数里的 elements 对应
    write_LAMMPS(
        lattice=lattice,
        atoms=atoms,
        elements=elements,
        filename="gb.data",
        orthogonal=False
    )
    
    #gb_indices
    all_indices = np.array(gb.site_properties['site_labels'])
    indices =[]
    for i in range(192):
        indices.append(np.where(all_indices == i)[0])
    
    #write lammps input file
    lammps_input = f"""
# NANOPARTICLE MELTING
#------------------INITIALIZATION------------------
units           metal
atom_style      atomic
dimension       3
boundary        f f f
read_data       gb.data

mass 1 6.941
mass 2 138.9055
mass 3 91.224
mass 4 15.9994

#------------------FORCE FIELDS------------------
pair_style     deepmd {check_point_file}
pair_coeff     * *

neighbor        6.0 bin
neigh_modify   every 10 delay 0 check no

#------------------FIX OUTER ATOMS------------------
# 计算盒子中心
variable        cx equal (xlo+xhi)/2
variable        cy equal (ylo+yhi)/2
variable        cz equal (zlo+zhi)/2


# 定义计算能量的命令
compute energy all pe/atom
compute total_energy all reduce sum c_energy

# 输出设置
thermo_style custom step etotal pe ke temp press vol
comm_modify cutoff 24.00

#------------------ENERGY CALCULATION AND OUTPUT------------------

dump            atom_pe all custom 1 gb_energy.lammpstrj id element type x y z c_energy
dump_modify     atom_pe sort id element Li La Zr O

run             0  # 运行0步触发输出
"""

    with open('lammps.in','w') as f:
        f.write(lammps_input)

    #run lammps
    cmd = "lmp -i lammps.in -log log.lammps"
    proc = subprocess.Popen(
        cmd.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # 合并到 stdout
        text=True,               # 返回 str 而非 bytes
        bufsize=1,               # 行缓冲，便于实时打印
    )
    proc.wait()
    
    #read gb energy
    atom_ids, energies = extract_last_column(bulk_energy_traj_file)
    atom_ids1, energies1 = extract_last_column('gb_energy.lammpstrj')
    
    ids, types, coords, (a, b, c) = read_lammps_atoms("gb.data")

    center = np.array([a/2, b/2, c/2])           # 盒子中心
    dists = np.linalg.norm(coords - center, axis=1)
    errors_by_r, sectional_errors_by_r,total_sectional_errors = \
get_sectional_error_by_r(indices, dists, energies1, np.arange(3, gb_r), energies)
    return total_sectional_errors[-1]

#define optimizer
from skopt import Optimizer, gp_minimize

def cp_updt_dict(old_dict, up_dict):
    if old_dict == None:
        return up_dict
    new_dict = old_dict.copy()
    new_dict.update(up_dict)
    return new_dict

from skopt.space import Real
from tqdm.notebook import tqdm
import shutil

@dataclass
class SpheregbBOMaker(Maker):
    #BO args
    name: str = 'Sphere GB BO'
    trials: int = 10
    base_estimator: str = 'GP'
    acq_func: str = 'EI'
    acq_optimizer: str = 'lbfgs'
    random_state: int = 42
    metadata: Dict[str, Any] = None
    #GB args
    crystal_structure: Structure = None
    check_point_file: str = None
    bulk_energy_traj_file: str = None
    sphere_R: float = 50
    vaccum_thickness: float = 20
    gb_r: float = 40
    rot_axis: List = None
    rot_angle: float = 0
    normal: List = None
    
    def lammps_input(self):
        lammps_input = f"""
# NANOPARTICLE MELTING
#------------------INITIALIZATION------------------
units           metal
atom_style      atomic
dimension       3
boundary        f f f
read_data       gb.data

mass 1 6.941
mass 2 138.9055
mass 3 91.224
mass 4 15.9994

#------------------FORCE FIELDS------------------
pair_style     deepmd {self.check_point_file}
pair_coeff     * *

neighbor        6.0 bin
neigh_modify   every 10 delay 0 check no

#------------------FIX OUTER ATOMS------------------
# 计算盒子中心
variable        cx equal (xlo+xhi)/2
variable        cy equal (ylo+yhi)/2
variable        cz equal (zlo+zhi)/2


# 定义计算能量的命令
compute energy all pe/atom
compute total_energy all reduce sum c_energy

# 输出设置
thermo_style custom step etotal pe ke temp press vol
comm_modify cutoff 24.00

#------------------ENERGY CALCULATION AND OUTPUT------------------

dump            atom_pe all custom 1 gb_energy.lammpstrj id element type x y z c_energy
dump_modify     atom_pe sort id element Li La Zr O

run             0  # 运行0步触发输出
"""
        return lammps_input
    
    def sample(self, params):
        x1, y1, z1, x2, y2, z2, gap = params
        #generate gb
        sgg = SphereGBGenerator(self.crystal_structure, self.sphere_R, self.vaccum_thickness)
        gb = sgg.get_sphere_GB(rot(self.rot_axis, self.rot_angle),
                                    self.normal,
                                    [x1,y1,z1],
                                    [x2,y2,z2],
                                    gap)
        ##write lammps structure file
        lattice = np.array(gb.lattice.matrix)      # 和你函数里的 lattice 对应
        atoms = np.array(gb.frac_coords)           # 和你函数里的 atoms 对应
        elements = np.array([str(sp) for sp in gb.species])  # 和你函数里的 elements 对应
        write_LAMMPS(
            lattice=lattice,
            atoms=atoms,
            elements=elements,
            filename="gb.data",
            orthogonal=False
        )

        ##gb_indices
        all_indices = np.array(gb.site_properties['site_labels'])
        indices =[]
        for i in range(192):
            indices.append(np.where(all_indices == i)[0])
        
        #lammps input file
        lammps_input = self.lammps_input
        with open('lammps.in','w') as f:
            f.write(self.lammps_input())
        
        #run lammps
        cmd = "lmp -i lammps.in -log log.lammps"
        proc = subprocess.Popen(
            cmd.split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并到 stdout
            text=True,               # 返回 str 而非 bytes
            bufsize=1,               # 行缓冲，便于实时打印
        )
        proc.wait()
        
        #read gb energy
        atom_ids, energies = extract_last_column(self.bulk_energy_traj_file)
        atom_ids1, energies1 = extract_last_column('gb_energy.lammpstrj')
        
        ids, types, coords, (a, b, c) = read_lammps_atoms("gb.data")

        center = np.array([a/2, b/2, c/2])           # 盒子中心
        dists = np.linalg.norm(coords - center, axis=1)
        errors_by_r, sectional_errors_by_r,total_sectional_errors = \
    get_sectional_error_by_r(indices, dists, energies1, np.arange(3, self.gb_r), energies)

        shutil.move('gb.data', f'gb_{self.count}.data')
        self.count += 1

        return total_sectional_errors[-1]
    
    @job
    def make(self):
        self.count = 0
        def trial_with_progress(func, n_calls, *args, **kwargs):
            with tqdm(total = n_calls, desc = "BO optimizing") as rgst_pbar:
                def wrapped_func(*args, **kwargs):
                    result = func(*args, **kwargs)
                    rgst_pbar.update(1)
                    return result
            return gp_minimize(wrapped_func, search_space, n_calls = n_calls, n_random_starts = int(0.1 * n_calls), *args, **kwargs)
        
        search_space = [Real(0, self.crystal_structure.lattice.a, name = 'x1'),
                 Real(0, self.crystal_structure.lattice.b, name = 'y1'),
                 Real(0, self.crystal_structure.lattice.c, name = 'z1'),
                 Real(0, self.crystal_structure.lattice.a, name = 'x2'),
                 Real(0, self.crystal_structure.lattice.b, name = 'y2'),
                 Real(0, self.crystal_structure.lattice.c, name = 'z2'),
                 Real(0, 3, name = 'gap')]
                 
        result = trial_with_progress(self.sample, n_calls=self.trials, random_state=self.random_state)
        return {'x':result.x_iters, 'y':result.func_vals}
