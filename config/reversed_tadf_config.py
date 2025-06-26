# config/reversed_tadf_config.py
"""Configuration for Reversed TADF analysis"""

# Excitation states to consider
EXCITATION_STATES = {
    'singlet': ['S1', 'S2', 'S3'],
    'triplet': ['T1', 'T2', 'T3', 'T4']
}

# Energy gap thresholds (in eV)
GAP_THRESHOLDS = {
    'small_gap': 0.1,      # Very small gap for efficient crossing
    'moderate_gap': 0.3,   # Moderate gap still allowing thermal activation
    'large_gap': 0.5,      # Large gap requiring special mechanisms
    'huge_gap': 1.0        # Very large gap, unlikely for TADF
}

# Reversed TADF mechanism criteria
MECHANISM_CRITERIA = {
    'hRISC': {
        'description': 'Hot Reverse ISC from T2 to S1',
        'conditions': {
            'S1-T2': {'max': 0.0},      # S1 must be lower than T2
            'T1-T2': {'max': 0.5},      # Small T1-T2 gap for efficient T1→T2
            'S1-T1': {'min': -0.5}      # Not too negative to maintain some coupling
        }
    },
    
    'inverted_gap': {
        'description': 'Inverted singlet-triplet gap',
        'conditions': {
            'S1-T1': {'max': -0.05}     # S1 significantly lower than T1
        }
    },
    
    'upper_triplet': {
        'description': 'Upper triplet state mediated RISC',
        'conditions': {
            'S1-T3': {'min': -0.2, 'max': 0.2},  # S1 near T3
            'T1-T3': {'max': 1.0}                # Accessible T3 from T1
        }
    },
    
    'multi_channel': {
        'description': 'Multiple RISC channels via S2',
        'conditions': {
            'S2-T1': {'min': -0.3, 'max': 0.3},  # S2 near T1
            'S2-S1': {'max': 1.0},               # Accessible S2 from S1
            'S1-T1': {'min': 0.0}                # Normal TADF also possible
        }
    },
    
    'solvent_stabilized': {
        'description': 'Solvent-stabilized CT state crossing',
        'conditions': {
            'CT-LE_gap': {'min': -0.3, 'max': 0.3},  # CT near LE states
            'dipole_change': {'min': 5.0}             # Large dipole moment change
        }
    }
}

# Spin-orbit coupling (SOC) elements to consider
SOC_ELEMENTS = {
    'heavy_atoms': ['Br', 'I', 'Se', 'Te', 'Au', 'Pt', 'Ir'],
    'moderate_SOC': ['S', 'Cl', 'P', 'Si'],
    'light_atoms': ['C', 'N', 'O', 'F', 'H']
}

# Rate constant estimation parameters
RATE_PARAMETERS = {
    'temperature': 300,  # K
    'kT': 0.025852,     # eV at 300K
    'SOC_scaling': {    # Relative SOC strength
        'heavy': 100,
        'moderate': 10,
        'light': 1
    }
}

# Molecular descriptor requirements
DESCRIPTOR_REQUIREMENTS = {
    'required': [
        'SMILES',
        'Molecule',
        'S1_energy',
        'T1_energy',
        'T2_energy'
    ],
    'optional': [
        'S2_energy',
        'T3_energy',
        'oscillator_strength',
        'dipole_moment',
        'HOMO_LUMO_gap',
        'spin_contamination'
    ]
}

# Visualization settings
VISUALIZATION_SETTINGS = {
    'energy_diagram': {
        'width': 10,
        'height': 8,
        'state_colors': {
            'S0': 'black',
            'S1': 'blue',
            'S2': 'lightblue',
            'T1': 'red',
            'T2': 'orange',
            'T3': 'yellow'
        },
        'transition_styles': {
            'allowed': 'solid',
            'forbidden': 'dashed',
            'weak': 'dotted'
        }
    },
    
    'gap_distribution': {
        'bins': 50,
        'alpha': 0.7,
        'show_threshold_lines': True
    }
}

# Export settings
EXPORT_SETTINGS = {
    'formats': ['csv', 'xlsx', 'json'],
    'include_structures': True,
    'include_energy_diagrams': True,
    'max_molecules_per_file': 1000
}