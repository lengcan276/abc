# reversed_tadf_analyzer.py
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple

class ReversedTADFAnalyzer:
    """Analyzer for identifying and characterizing Reversed TADF candidates"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.gap_columns = {}
        
    def load_data(self) -> bool:
        """Load molecular properties data"""
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Loaded {len(self.df)} molecules")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def identify_gap_columns(self) -> Dict[str, List[str]]:
        """Identify all excitation gap columns in the dataset"""
        gap_patterns = {
            'S0-S1': ['s0_s1', 's1_energy', 'singlet_1'],
            'S0-S2': ['s0_s2', 's2_energy', 'singlet_2'],
            'S0-T1': ['s0_t1', 't1_energy', 'triplet_1'],
            'S0-T2': ['s0_t2', 't2_energy', 'triplet_2'],
            'S0-T3': ['s0_t3', 't3_energy', 'triplet_3'],
            'S1-T1': ['s1_t1', 'singlet_triplet_gap', 'delta_st'],
            'S1-T2': ['s1_t2', 's1_t2_gap'],
            'S1-T3': ['s1_t3', 's1_t3_gap'],
            'S2-T1': ['s2_t1', 's2_t1_gap'],
            'S2-T2': ['s2_t2', 's2_t2_gap'],
            'T1-T2': ['t1_t2', 'triplet_gap', 'delta_tt'],
            'T2-T3': ['t2_t3', 't2_t3_gap'],
            'S2-S1': ['s2_s1', 'singlet_gap', 'delta_ss']
        }
        
        self.gap_columns = {}
        for gap_type, patterns in gap_patterns.items():
            found_cols = []
            for col in self.df.columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in patterns):
                    found_cols.append(col)
            if found_cols:
                self.gap_columns[gap_type] = found_cols
                
        return self.gap_columns
    
    def analyze_reversed_tadf_mechanisms(self) -> Dict[str, pd.DataFrame]:
        """Analyze different Reversed TADF mechanisms"""
        mechanisms = {}
        
        # Mechanism 1: hRISC (hot RISC) - T2 to S1
        # Requires: S1 < T2 (negative S1-T2 gap) and small T1-T2 gap
        if 'S1-T2' in self.gap_columns and 'T1-T2' in self.gap_columns:
            s1_t2_col = self.gap_columns['S1-T2'][0]
            t1_t2_col = self.gap_columns['T1-T2'][0]
            
            hrisc_candidates = self.df[
                (self.df[s1_t2_col] < 0) &  # S1 lower than T2
                (self.df[t1_t2_col].abs() < 0.5)  # Small T1-T2 gap
            ].copy()
            
            mechanisms['hRISC'] = hrisc_candidates
            print(f"\nhRISC mechanism candidates: {len(hrisc_candidates)}")
            
        # Mechanism 2: Inverted singlet-triplet gap (S1 < T1)
        if 'S1-T1' in self.gap_columns:
            s1_t1_col = self.gap_columns['S1-T1'][0]
            inverted_st = self.df[self.df[s1_t1_col] < 0].copy()
            mechanisms['Inverted_ST'] = inverted_st
            print(f"Inverted S1-T1 gap candidates: {len(inverted_st)}")
            
        # Mechanism 3: Multi-state cascade (involving S2)
        if 'S2-T1' in self.gap_columns and 'S2-S1' in self.gap_columns:
            s2_t1_col = self.gap_columns['S2-T1'][0]
            s2_s1_col = self.gap_columns['S2-S1'][0]
            
            cascade_candidates = self.df[
                (self.df[s2_t1_col].abs() < 0.3) &  # S2 close to T1
                (self.df[s2_s1_col] < 1.0)  # S2-S1 gap not too large
            ].copy()
            
            mechanisms['Multi_state_cascade'] = cascade_candidates
            print(f"Multi-state cascade candidates: {len(cascade_candidates)}")
            
        # Mechanism 4: Upper triplet crossing (T3 or higher involved)
        if 'S1-T3' in self.gap_columns:
            s1_t3_col = self.gap_columns['S1-T3'][0]
            upper_triplet = self.df[self.df[s1_t3_col].abs() < 0.2].copy()
            mechanisms['Upper_triplet_crossing'] = upper_triplet
            print(f"Upper triplet crossing candidates: {len(upper_triplet)}")
            
        return mechanisms
    
    def calculate_rtisc_rates(self, molecules_df: pd.DataFrame) -> pd.DataFrame:
        """Estimate reverse ISC rates based on gap energies"""
        # Simplified estimation based on energy gap law
        # k_RISC ∝ exp(-ΔE/kT) * SOC²
        
        kT = 0.025  # kT at room temperature in eV
        
        if 'S1-T1' in self.gap_columns:
            gap_col = self.gap_columns['S1-T1'][0]
            molecules_df['k_RISC_relative'] = np.exp(-molecules_df[gap_col].abs() / kT)
            
        if 'S1-T2' in self.gap_columns:
            gap_col = self.gap_columns['S1-T2'][0]
            molecules_df['k_hRISC_relative'] = np.exp(-molecules_df[gap_col].abs() / kT)
            
        return molecules_df
    
    def summarize_candidates(self, mechanisms: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create a summary of all Reversed TADF candidates"""
        summary_data = []
        
        for mechanism, df in mechanisms.items():
            if len(df) > 0:
                for idx, mol in df.iterrows():
                    summary_data.append({
                        'Molecule': mol.get('Molecule', idx),
                        'Mechanism': mechanism,
                        'S1-T1_gap': mol.get(self.gap_columns.get('S1-T1', [''])[0], np.nan),
                        'S1-T2_gap': mol.get(self.gap_columns.get('S1-T2', [''])[0], np.nan),
                        'T1-T2_gap': mol.get(self.gap_columns.get('T1-T2', [''])[0], np.nan),
                        'SMILES': mol.get('SMILES', '')
                    })
        
        return pd.DataFrame(summary_data)
    
    def export_results(self, output_dir: str):
        """Export analysis results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Identify gap columns
        self.identify_gap_columns()
        
        # Analyze mechanisms
        mechanisms = self.analyze_reversed_tadf_mechanisms()
        
        # Save individual mechanism results
        for mechanism, df in mechanisms.items():
            if len(df) > 0:
                output_file = os.path.join(output_dir, f'reversed_tadf_{mechanism.lower()}.csv')
                df.to_csv(output_file, index=False)
                print(f"Saved {mechanism} candidates to {output_file}")
        
        # Create and save summary
        summary = self.summarize_candidates(mechanisms)
        if len(summary) > 0:
            summary_file = os.path.join(output_dir, 'reversed_tadf_summary.csv')
            summary.to_csv(summary_file, index=False)
            print(f"\nTotal Reversed TADF candidates: {len(summary)}")
            print(f"Summary saved to {summary_file}")
            
            # Print mechanism distribution
            print("\nMechanism distribution:")
            print(summary['Mechanism'].value_counts())


# Example usage
if __name__ == "__main__":
    analyzer = ReversedTADFAnalyzer(
        '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/extracted/molecular_properties_summary.csv'
    )
    
    if analyzer.load_data():
        analyzer.export_results(
            '/vol1/home/lengcan/cleng/Function_calling/test/0-ground_state_structures/0503/reverse_TADF_system_deepreseach_0617/data/extracted/reversed_tadf_analysis'
        )