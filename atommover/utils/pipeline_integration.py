
def get_awg_sequence_for_algorithm(algorithm_instance, atom_array, aod_settings, physical_params, do_ejection=False):
    """
    Run an algorithm on an AtomArray and return the corresponding AWG sequence.
    
    Args:
        algorithm_instance: Instance of an Algorithm class (e.g. PCFA())
        atom_array: AtomArray object with initial and target configuration
        aod_settings: AODSettings object
        physical_params: PhysicalParams object
        do_ejection: Whether to run ejection
        
    Returns:
        list[AWGBatch]: The sequence of RF commands
        tuple: (config, move_list, success_flag) from the algorithm
    """
    from atommover.utils.awg_control import RFConverter
    
    # 1. Run the algorithm to get moves
    config, move_list, success_flag = algorithm_instance.get_moves(atom_array, do_ejection=do_ejection)
    
    # 2. Convert moves to AWG commands
    converter = RFConverter(aod_settings, physical_params)
    awg_sequence = converter.convert_sequence(move_list)
    
    return awg_sequence, (config, move_list, success_flag)
