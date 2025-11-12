from jobflow import job
import json
from InterOptimus.itworker import InterfaceWorker

@job(large_data = "data")
def interface_searching(film_conv, substrate_conv,
                        max_area = 50, max_length_tol = 0.03, max_angle_tol = 0.03,
                        film_max_miller = 4, substrate_max_miller = 4,
                        termination_ftol = 0.15, film_thickness = 12, substrate_thickness = 12, vacuum_over_film = 15,
                        set_relax_thicknesses = (3,3), relax_in_layers = False, fmax = 0.05, steps = 200,
                        device = 'cuda', discut = 0.8, ckpt_path = None,
                        BO_coord_bin_size = 0.25, BO_energy_bin_size = 0.2, BO_rms_bin_size = 0.3,
                        n_calls_density = 5, z_range = (0.5,3), calc = 'sevenn', strain_E_correction = True)
    iw = InterfaceWorker(film_conv, substrate_conv)
    
    iw.lattice_matching(max_area = max_area,
                    max_length_tol = max_length_tol,
                    max_angle_tol = max_angle_tol,
                    film_max_miller = film_max_miller,
                    substrate_max_miller = substrate_max_miller)
                    
    iw.ems.plot_unique_matches()
    
    iw.ems.plot_matching_data([f'{film_conv.reduced_formula}',
                               f'{substrate_conv.reduced_formula}'],
                               'polar.jpg',
                               show_millers = True,
                               show_legend = False)
                               
    iw.parse_interface_structure_params(termination_ftol = termination_ftol,
                                        film_thickness = film_thickness,
                                        substrate_thickness = substrate_thickness,
                                        vacuum_over_film=vacuum_over_film)
    
    iw.global_minimization(n_calls_density = 1,
                           z_range = (0.5, 3),
                           calc = 'sevenn',
                           strain_E_correction = True)
    
    return {"data": {"iw_results":iw.opt_results, "matches":iw.ems.all_matche_data}}
