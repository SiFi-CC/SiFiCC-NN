def get_parameters(mode):
    if mode == "CC-4to1":
        # Parameters for the CC mode
        datasets = {
            "continuous": "OptimisedGeometry_4to1_Continuous_1.8e10protons_simv4",
            "0mm": "OptimisedGeometry_4to1_0mm_3.9e9protons_simv4",
            "5mm": "OptimisedGeometry_4to1_5mm_3.9e9protons_simv4",
            "10mm": "OptimisedGeometry_4to1_10mm_3.9e9protons_simv4",
            "m5mm": "OptimisedGeometry_4to1_minus5mm_3.9e9protons_simv4",
        }
        output_dimensions = {
            "classification": 1,
            "energy": 2,
            "position": 6,
        }
        dataset_name = "SimGraphSiPM"
        # DATASET_NEUTRONS = "OptimisedGeometry_4to1_0mm_gamma_neutron_2e9_protons"

    elif mode == "CM-4to1":
        # Parameters for the CM mode
        datasets = {
            "continuous": "SystemMatrix_CodedMaskHIT_simv5_linesource_0to29999",
            "spot1": "OptimisedGeometry_CodedMaskHIT_Spot1_1e10_protons_MK",
            "spot2": "OptimisedGeometry_CodedMaskHIT_Spot2_1e10_protons_MK",
            "spot3": "OptimisedGeometry_CodedMaskHIT_Spot3_1e10_protons_MK",
            "spot4": "OptimisedGeometry_CodedMaskHIT_Spot4_1e10_protons_MK",
            "spot5": "OptimisedGeometry_CodedMaskHIT_Spot5_1e10_protons_MK",
            "spot6": "OptimisedGeometry_CodedMaskHIT_Spot6_1e10_protons_MK",
            "spot7": "OptimisedGeometry_CodedMaskHIT_Spot7_1e10_protons_MK",
        }    
        output_dimensions = {
            "classification": 1,
            "energy": 1,
            "y-position": 1,
            "x-z-position": 385,
        }
        dataset_name = "CMSimGraphSiPM"
    elif mode == "CMbeamtime":
        datasets = {"run00582": "run00582_sifi",
            "run00583": "run00583_sifi",
            "run00584": "run00584_sifi",}
        """{
            "run00596": "run00596_sifi_1M_TESTING",
            "run00566": "run00566_sifi",
            "run00567": "run00567_sifi",
            "run00568": "run00568_sifi",
            "run00569": "run00569_sifi",
            "run00570": "run00570_sifi",
            "run00571": "run00571_sifi",
            "run00575": "run00575_sifi",
            "run00576": "run00576_sifi",
            "run00577": "run00577_sifi",
            "run00578": "run00578_sifi",
            "run00579": "run00579_sifi",
            "run00580": "run00580_sifi",
            "run00581": "run00581_sifi",
        }"""

            
        output_dimensions = {
            "classification": 1,
            "energy": 1,
            "y-position": 1,
            "x-z-position": 385,
        }
        dataset_name = "BeamTime"
    return datasets, output_dimensions, dataset_name
