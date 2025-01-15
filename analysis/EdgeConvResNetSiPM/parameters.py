mode = None


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
        # DATASET_NEUTRONS = "OptimisedGeometry_4to1_0mm_gamma_neutron_2e9_protons"

    elif mode == "CM-4to1":
        # Parameters for the CM mode
        datasets = {
            "continuous": "mergedTree",
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
            "position": 3,
        }
    return datasets, output_dimensions
