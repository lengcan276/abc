# config/excitation_settings.py
"""
Configuration settings for excited state calculations and analysis.
"""

EXCITATION_SETTINGS = {
    'method': 'wb97xd',
    'basis': 'def2-tzvp',
    
    'calculation_params': {
        'td_states': 10,  # Number of excited states to calculate
        'singlet_states': True,
        'triplet_states': True,
        'scf_convergence': 'tight',
        'integral_accuracy': 'ultrafine'
    },
    
    'screening_criteria': {
        'min_transition_similarity': 0.7,  # Minimum transition similarity
        'confidence_threshold': 0.7,       # Confidence threshold
        'gap_threshold': -0.005,          # -5 meV in eV
        'max_state_number': 6             # Maximum state number to consider (S6 or T6)
    },
    
    'systematic_corrections': {
        # Systematic error corrections based on benchmarking (eV)
        'S1-T1': -0.037,
        'S2-T4': -0.040,
        'S3-T4': -0.045,
        'default': -0.040
    },
    
    'reference_molecules': {
        # Reference molecules for validation
        'calicene': {'gap': -0.101, 'type': 'S3-T4'},
        '3ring_cn': {'gap': -0.028, 'type': 'S2-T4'},
        '5ring_nh2_3ring_cn_both': {'gap': -0.015, 'type': 'S1-T1'},
        '5ring_nme2_3ring_cn_in_con2': {'gap': -0.025, 'type': 'S1-T1'},
        '5ring_oh_3ring_cn_both': {'gap': -0.012, 'type': 'S1-T1'},
        'me_oh_cn': {'gap': -0.018, 'type': 'S1-T1'},
        '5ring_nh2_3ring_cn_in': {'gap': -0.020, 'type': 'S1-T1'}
    },
    
    'file_patterns': {
        # File naming patterns for different calculation types
        'singlet': [
            '{molecule}_s0_singlet_wb97xd.log',
            '{molecule}/excited.log',
            '{molecule}_singlet.log'
        ],
        'triplet': [
            '{molecule}_s0_triplet_wb97xd.log', 
            '{molecule}/triplet.log',
            '{molecule}_triplet.log'
        ]
    },
    
    'output_settings': {
        'save_detailed_analysis': True,
        'plot_energy_diagrams': True,
        'generate_summary_report': True,
        'export_format': ['csv', 'json', 'xlsx']
    }
}

# Additional configuration for visualization
VISUALIZATION_SETTINGS = {
    'plot_style': 'seaborn',
    'figure_size': (10, 8),
    'dpi': 300,
    'color_scheme': {
        'singlet': '#3498db',
        'triplet': '#e74c3c',
        'inverted': '#2ecc71',
        'normal': '#95a5a6'
    }
}

# Database connection settings (if needed)
DATABASE_SETTINGS = {
    'enabled': False,
    'host': 'localhost',
    'port': 5432,
    'database': 'tadf_research',
    'user': 'researcher',
    'password': ''  # Should be loaded from environment variable
}