# agents/insight_agent.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from collections import Counter

class InsightAgent:
    """
    Agent responsible for analyzing modeling and exploration results
    to generate quantum chemistry insights and explanations.
    """
    
    def __init__(self, modeling_results=None, exploration_results=None):
        """Initialize the InsightAgent with modeling and exploration results."""
        self.modeling_results = modeling_results
        self.exploration_results = exploration_results
        self.insights = {}
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging for the insight agent."""
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                           filename='../data/logs/insight_agent.log')
        self.logger = logging.getLogger('InsightAgent')
        
    def load_results(self, modeling_results=None, exploration_results=None):
        """Load modeling and exploration results."""
        if modeling_results:
            self.modeling_results = modeling_results
        if exploration_results:
            self.exploration_results = exploration_results
            
        if not self.modeling_results:
            self.logger.warning("No modeling results provided.")
        if not self.exploration_results:
            self.logger.warning("No exploration results provided.")
            
        return True
    
    def explain_feature(self, feature_name):
        """
        Generate an explanation for a specific molecular feature.
        """
        # Dictionary mapping feature names to explanations
        feature_explanations = {
            # Electronic properties
            'electron_withdrawing_effect': "A measure of the feature's ability to pull electron density away from the molecule's core. Higher values indicate stronger electron-withdrawing characteristics, which can stabilize frontier molecular orbitals.",
            'electron_donating_effect': "Represents the ability to donate electron density to the molecular system. Higher values indicate stronger electron-donating properties, potentially raising HOMO energy levels.",
            'net_electronic_effect': "The overall balance of electron-donating vs electron-withdrawing effects in the molecule. Positive values indicate net electron-donating character, while negative values indicate net electron-withdrawing character.",
            
            # Structural properties
            'estimated_conjugation': "Indicates the extent of π-conjugation in the molecular structure. Higher values suggest more extensive delocalization of electrons across the molecule, which affects excited state properties.",
            'planarity_index': "A measure of how flat or planar the molecule is likely to be. More planar molecules typically have better orbital overlap and more extensive conjugation.",
            'estimated_size': "Approximates the overall molecular size based on its components. Larger molecules may exhibit different excited state behaviors due to more distributed electron density.",
            
            # Physical properties
            'estimated_polarity': "Represents the expected polarity of the molecule based on its functional groups. Higher values indicate stronger dipole moments and potential for solvatochromic effects.",
            'estimated_hydrophobicity': "Indicates how hydrophobic (water-repelling) the molecule is likely to be. Positive values suggest higher hydrophobicity, while negative values suggest hydrophilicity.",
            
            # Quantum properties
            'homo': "Highest Occupied Molecular Orbital energy level (in eV). This represents the energy required to remove an electron from the molecule.",
            'lumo': "Lowest Unoccupied Molecular Orbital energy level (in eV). This represents the energy gained when an electron is added to the molecule.",
            'homo_lumo_gap': "The energy difference between HOMO and LUMO orbitals. This relates to the molecule's stability, reactivity, and optical properties.",
            'dipole': "The dipole moment of the molecule, indicating the separation of positive and negative charges. Higher values suggest more polarized electron distribution.",
            
            # Excited state properties
            's1_energy_ev': "Energy of the first singlet excited state (S1) in electron volts. This determines the wavelength of absorption/emission.",
            't1_energy_ev': "Energy of the first triplet excited state (T1) in electron volts. Important for phosphorescence and TADF mechanisms.",
            's1_t1_gap_ev': "Energy difference between S1 and T1 states. Negative values indicate reverse TADF candidates, where T1 is higher in energy than S1.",
            'oscillator_strength': "Relates to the probability of radiative transition between ground and excited states. Higher values indicate stronger absorption/emission.",
            
            # Substituent-specific features
            'has_cn': "Presence of cyano (CN) groups, which are strongly electron-withdrawing and can enhance conjugation.",
            'has_nh2': "Presence of amino (NH2) groups, which are electron-donating and can raise HOMO energy levels.",
            'has_oh': "Presence of hydroxyl (OH) groups, which are moderately electron-donating and can participate in hydrogen bonding.",
            'has_me': "Presence of methyl (CH3) groups, which have weak electron-donating effects and can influence molecular packing.",
            'has_f': "Presence of fluorine atoms, which are electronegative and can withdraw electron density.",
            'has_no2': "Presence of nitro groups, which are strongly electron-withdrawing and can lower LUMO energy levels.",
            'has_cf3': "Presence of trifluoromethyl groups, which are bulky and strongly electron-withdrawing.",
            
            # Ring-specific features
            'has_5ring': "Presence of five-membered rings, which introduce non-planarity and affect orbital energies differently than benzene rings.",
            'has_3ring': "Presence of three-membered rings (e.g., cyclopropyl), which introduce strain and localizing effects on electronic structure.",
            'has_7ring': "Presence of seven-membered rings, which create different geometries and can affect molecular flexibility.",
            
            # Positional features
            'has_in_group': "Substituents positioned toward the interior of the molecular structure, potentially creating steric effects.",
            'has_out_group': "Substituents positioned at the exterior of the molecular structure, often with less steric hindrance.",
            'has_both_groups': "Presence of substituents in both interior and exterior positions, creating complex electronic effects.",
            
            # Combined properties
            'homo_polarity': "Interaction between HOMO energy and molecular polarity, revealing how polarization affects electron-donating capability.",
            'lumo_electronic_effect': "Interaction between LUMO energy and net electronic effects, showing how electron-withdrawing groups influence electron acceptance.",
            'gap_conjugation': "Relationship between HOMO-LUMO gap and molecular conjugation, indicating how extended π-systems affect the energy gap.",
            'dipole_planarity': "Combined effect of dipole moment and molecular planarity, revealing how geometry affects charge separation.",
            'energy_per_size': "Energy normalized by molecular size, useful for comparing electronic properties across differently sized molecules."
        }
        
        # Clean feature name for matching
        clean_feature = feature_name.replace('_count', '').replace('total_', '')
        
        # Return explanation if available, or generate a generic one
        if clean_feature in feature_explanations:
            return feature_explanations[clean_feature]
        elif 'count' in feature_name:
            group = feature_name.replace('_count', '')
            return f"Number of {group} groups in the molecule, which affects the overall electronic and structural properties."
        else:
            return f"A molecular descriptor related to the {feature_name.replace('_', ' ')} properties of the molecule."
    
    def quantum_chemistry_explanation(self, feature_name, importance_value):
        """
        Generate quantum chemistry explanations for why a feature is important
        for predicting reverse TADF properties.
        """
        # Dictionary mapping features to quantum chemistry explanations
        quantum_explanations = {
            'electron_withdrawing_effect': [
                "Strong electron-withdrawing groups can stabilize the LUMO orbital while having less effect on the HOMO, narrowing the HOMO-LUMO gap.",
                "In reverse TADF, electron-withdrawing groups may selectively stabilize the singlet state more than the triplet, potentially leading to the unusual case where S1 is lower than T1.",
                "The spatial distribution of electron-withdrawing effects can create localized regions for singlet excitations while triplet states may remain more delocalized."
            ],
            'electron_donating_effect': [
                "Electron-donating groups typically raise the HOMO energy level, which can affect the energy ordering of excited states.",
                "In molecules with reverse TADF properties, electron-donating groups may selectively destabilize triplet states more than singlet states.",
                "The balance between donating and withdrawing effects creates unique frontier orbital distributions that can lead to unusual S1-T1 energy ordering."
            ],
            'estimated_conjugation': [
                "Higher conjugation typically reduces the exchange energy (K) between singlet and triplet states, which usually determines their energy difference.",
                "In reverse TADF, extensive conjugation may lead to spatially separated frontier orbitals, reducing the overlap integral critical for exchange interactions.",
                "The delocalization of π-electrons across the conjugated system affects the spatial distribution of excited states, potentially inverting the normal S1-T1 energy ordering."
            ],
            'planarity_index': [
                "Molecular planarity affects the overlap of frontier orbitals, which directly influences the exchange energy between singlet and triplet states.",
                "Non-planar structures can create twisted intramolecular charge transfer (TICT) states, where singlet states may be stabilized below triplet states.",
                "The degree of planarity impacts how rigidly the molecular orbitals are confined, affecting the energy splitting between different spin states."
            ],
            'estimated_polarity': [
                "Highly polar molecules can stabilize charge-transfer states, which are typically more pronounced in singlet than triplet excitations.",
                "The polarity-induced stabilization of singlet states can sometimes overcome the exchange energy advantage of triplet states, leading to reverse TADF.",
                "Molecular polarity affects solvent reorganization energies differently for S1 and T1 states, potentially contributing to their unusual energy ordering."
            ],
            'homo_lumo_gap': [
                "Smaller HOMO-LUMO gaps generally indicate more easily accessible excited states, but the specific impact on S1 vs T1 states varies.",
                "In reverse TADF, unusual HOMO-LUMO spatial distributions may lead to different configurations for singlet and triplet states.",
                "The gap between frontier orbitals affects the contribution of double excitations, which can be critical for determining the relative energies of S1 and T1."
            ],
            'net_electronic_effect': [
                "The balance of electron-donating and withdrawing effects determines orbital energy levels and their spatial distribution.",
                "In molecules exhibiting reverse TADF, a carefully balanced electronic effect can create the unusual condition where configuration interaction stabilizes S1 below T1.",
                "The net electronic effect influences how charge is distributed in excited states, potentially creating conditions where singlet states experience greater stabilization than triplets."
            ],
            'dipole': [
                "Higher molecular dipole moments can stabilize charge-transfer character in excited states, affecting S1 and T1 energies differently.",
                "In reverse TADF systems, strong dipole moments may preferentially stabilize singlet states with significant charge-transfer character.",
                "The ground state dipole orientation relative to excited state transitions can create selective energetic advantages for singlet over triplet states."
            ]
        }
        
        # Determine importance level
        importance_level = "high" if importance_value > 0.1 else "moderate" if importance_value > 0.05 else "low"
        
        # Generate explanation
        if feature_name in quantum_explanations:
            explanations = quantum_explanations[feature_name]
            
            if importance_level == "high":
                return explanations[0] if len(explanations) >= 1 else "This feature strongly influences the S1-T1 energy ordering in potential reverse TADF molecules."
            elif importance_level == "moderate":
                return explanations[1] if len(explanations) >= 2 else "This feature has a notable effect on the electronic properties that determine S1-T1 gap direction."
            else:
                return explanations[2] if len(explanations) >= 3 else "This feature contributes to the subtle electronic effects that can lead to reverse TADF behavior."
        else:
            # Generic explanation based on feature name
            if "electron" in feature_name or "homo" in feature_name or "lumo" in feature_name:
                return f"This electronic property affects the distribution and energy of frontier orbitals, potentially contributing to the unusual S1-T1 energy ordering in reverse TADF materials."
            elif "ring" in feature_name or "planar" in feature_name or "size" in feature_name:
                return f"This structural feature influences the spatial arrangement of molecular orbitals, which can affect the exchange interaction between singlet and triplet states."
            elif "polarity" in feature_name or "dipole" in feature_name:
                return f"This property relates to charge distribution, which can selectively stabilize certain excited states and potentially lead to reverse TADF characteristics."
            else:
                return f"This molecular descriptor influences the electronic structure in ways that may contribute to the unusual ordering of excited states in reverse TADF materials."
    
    def analyze_classification_results(self):
        """Analyze classification model results and generate insights."""
        if not self.modeling_results or 'classification' not in self.modeling_results:
            self.logger.error("No classification results to analyze.")
            return None
            
        classification = self.modeling_results['classification']
        if not classification:
            self.logger.error("Empty classification results.")
            return None
            
        print("Analyzing classification model results...")
        
        # Extract key information
        accuracy = classification.get('accuracy', 0)
        features = classification.get('features', [])
        importance = classification.get('importance', pd.DataFrame())
        
        # Generate insights
        insights = {
            'model_quality': {
                'accuracy': accuracy,
                'interpretation': "excellent" if accuracy > 0.9 else "good" if accuracy > 0.8 else "moderate" if accuracy > 0.7 else "poor"
            },
            'key_features': [],
            'quantum_explanations': []
        }
        
        # Analyze top features
        top_features = importance.head(5)['Feature'].tolist() if not importance.empty else []
        
        for feature in top_features:
            feature_importance = importance[importance['Feature'] == feature]['Importance'].values[0] if feature in importance['Feature'].values else 0
            
            insights['key_features'].append({
                'name': feature,
                'importance': feature_importance,
                'explanation': self.explain_feature(feature)
            })
            
            insights['quantum_explanations'].append({
                'name': feature,
                'explanation': self.quantum_chemistry_explanation(feature, feature_importance)
            })
            
        self.insights['classification'] = insights
        return insights
    
    def analyze_regression_results(self):
        """Analyze regression model results and generate insights."""
        if not self.modeling_results or 'regression' not in self.modeling_results:
            self.logger.error("No regression results to analyze.")
            return None
            
        regression = self.modeling_results['regression']
        if not regression:
            self.logger.error("Empty regression results.")
            return None
            
        print("Analyzing regression model results...")
        
        # Extract key information
        r2 = regression.get('r2', 0)
        rmse = regression.get('rmse', 0)
        features = regression.get('features', [])
        importance = regression.get('importance', pd.DataFrame())
        
        # Generate insights
        insights = {
            'model_quality': {
                'r2': r2,
                'rmse': rmse,
                'interpretation': "excellent" if r2 > 0.9 else "good" if r2 > 0.7 else "moderate" if r2 > 0.5 else "poor"
            },
            'key_features': [],
            'quantum_explanations': []
        }
        
        # Analyze top features
        top_features = importance.head(5)['Feature'].tolist() if not importance.empty else []
        
        for feature in top_features:
            feature_importance = importance[importance['Feature'] == feature]['Importance'].values[0] if feature in importance['Feature'].values else 0
            
            insights['key_features'].append({
                'name': feature,
                'importance': feature_importance,
                'explanation': self.explain_feature(feature)
            })
            
            insights['quantum_explanations'].append({
                'name': feature,
                'explanation': self.quantum_chemistry_explanation(feature, feature_importance)
            })
            
        self.insights['regression'] = insights
        return insights
    
    def analyze_exploration_results(self):
        """Analyze exploration results and generate insights."""
        if not self.exploration_results:
            self.logger.error("No exploration results to analyze.")
            return None
            
        print("Analyzing exploration results...")
        
        # Extract key information
        neg_molecules = self.exploration_results.get('analysis_results', {}).get('neg_molecules', [])
        top_diff_features = self.exploration_results.get('analysis_results', {}).get('top_diff_features', [])
        
        # Generate insights
        insights = {
            'reverse_tadf_candidates': len(neg_molecules),
            'structural_patterns': [],
            'feature_explanations': []
        }
        
        # Analyze structural patterns
        for feature in top_diff_features[:5]:  # Top 5 differentiating features
            insights['structural_patterns'].append({
                'name': feature,
                'explanation': self.explain_feature(feature)
            })
            
            insights['feature_explanations'].append({
                'name': feature,
                'explanation': self.quantum_chemistry_explanation(feature, 0.1)  # Assume moderate importance
            })
            
        self.insights['exploration'] = insights
        return insights
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive markdown report with all insights."""
        if not self.insights:
            # Run analyses if not already done
            self.analyze_classification_results()
            self.analyze_regression_results()
            self.analyze_exploration_results()
            
        print("Generating comprehensive insight report...")
        
        # Create reports directory
        report_dir = '../data/reports'
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(report_dir, 'reverse_tadf_insights_report.md')
        
        with open(report_path, 'w') as f:
            # Title
            f.write("# Reverse TADF Molecular Design Insights\n\n")
            
            # Introduction
            f.write("## Introduction\n\n")
            f.write("This report presents key insights into the molecular design principles for reverse Thermally Activated Delayed Fluorescence (TADF) materials. ")
            f.write("Reverse TADF occurs when the first triplet excited state (T1) has higher energy than the first singlet excited state (S1), ")
            f.write("contrary to the typical ordering of these states. This unique property enables specialized photophysical processes with applications in advanced optoelectronic devices.\n\n")
            
            # Model Performance
            f.write("## Predictive Model Performance\n\n")
            
            if 'classification' in self.insights:
                class_quality = self.insights['classification']['model_quality']
                f.write(f"### Classification Model (Negative vs Positive S1-T1 Gap)\n\n")
                f.write(f"* **Accuracy**: {class_quality['accuracy']:.2f}\n")
                f.write(f"* **Performance Assessment**: {class_quality['interpretation'].title()}\n\n")
                
            if 'regression' in self.insights:
                reg_quality = self.insights['regression']['model_quality']
                f.write(f"### Regression Model (S1-T1 Gap Value)\n\n")
                f.write(f"* **R² Score**: {reg_quality['r2']:.2f}\n")
                f.write(f"* **RMSE**: {reg_quality['rmse']:.4f} eV\n")
                f.write(f"* **Performance Assessment**: {reg_quality['interpretation'].title()}\n\n")
                
            # Key Molecular Descriptors
            f.write("## Key Molecular Descriptors for Reverse TADF\n\n")
            
            # Combine insights from both models
            all_features = []
            
            if 'classification' in self.insights:
                class_features = self.insights['classification']['key_features']
                all_features.extend([(f['name'], f['importance'], f['explanation'], 'classification') for f in class_features])
                
            if 'regression' in self.insights:
                reg_features = self.insights['regression']['key_features']
                all_features.extend([(f['name'], f['importance'], f['explanation'], 'regression') for f in reg_features])
                
            # Sort by importance and remove duplicates
            all_features.sort(key=lambda x: x[1], reverse=True)
            unique_features = []
            feature_names = set()
            
            for feature in all_features:
                if feature[0] not in feature_names:
                    unique_features.append(feature)
                    feature_names.add(feature[0])
                    
            # Write feature descriptions
            for i, (name, importance, explanation, model_type) in enumerate(unique_features[:8]):  # Top 8 unique features
                f.write(f"### {i+1}. {name.replace('_', ' ').title()}\n\n")
                f.write(f"**Importance Score**: {importance:.4f} ({model_type} model)\n\n")
                f.write(f"{explanation}\n\n")
                
            # Quantum Chemistry Explanations
            f.write("## Quantum Chemistry Insights\n\n")
            f.write("The unusual ordering of S1 and T1 states in reverse TADF materials can be explained through several quantum mechanical principles:\n\n")
            
            # Get quantum explanations from classification model (typically more interpretable)
            quantum_explanations = []
            if 'classification' in self.insights:
                quantum_explanations = self.insights['classification']['quantum_explanations']
                
            elif 'regression' in self.insights:
                quantum_explanations = self.insights['regression']['quantum_explanations']
                
            # Write quantum explanations
            for i, exp in enumerate(quantum_explanations[:5]):  # Top 5 explanations
                f.write(f"### {exp['name'].replace('_', ' ').title()}\n\n")
                f.write(f"{exp['explanation']}\n\n")
                
            # Design Principles
            f.write("## Molecular Design Principles for Reverse TADF\n\n")
            f.write("Based on our analysis, we recommend the following design strategies for developing reverse TADF materials:\n\n")
            
            # Generate design principles based on key features
            design_principles = []
            
            # Check if we have exploration insights
            if 'exploration' in self.insights:
                structural_patterns = self.insights['exploration']['structural_patterns']
                
                for pattern in structural_patterns:
                    name = pattern['name'].replace('has_', '').replace('_', ' ')
                    
                    if name in ['cn', 'no2', 'cf3']:
                        design_principles.append(f"Incorporate strong electron-withdrawing groups (such as -{name.upper()}) to tune frontier orbital energies")
                    elif name in ['nh2', 'oh', 'me']:
                        design_principles.append(f"Utilize electron-donating groups (such as -{name.upper()}) to modulate excited state energy levels")
                    elif name in ['5ring', '3ring', '7ring']:
                        design_principles.append(f"Explore non-hexagonal ring structures (like {name.replace('ring', '-membered rings')}) to create unique geometric constraints")
                        
            # Add principles based on quantum features
            electronic_principle_added = False
            conjugation_principle_added = False
            polarity_principle_added = False
            
            for feature, importance, _, _ in unique_features:
                if ('electron' in feature) and not electronic_principle_added:
                    design_principles.append("Balance electron-donating and electron-withdrawing groups to create the specific frontier orbital distribution needed for reverse TADF")
                    electronic_principle_added = True
                    
                elif ('conjugation' in feature or 'planarity' in feature) and not conjugation_principle_added:
                    design_principles.append("Optimize the degree of π-conjugation and molecular planarity to control orbital overlap and exchange interactions")
                    conjugation_principle_added = True
                    
                elif ('polarity' in feature or 'dipole' in feature) and not polarity_principle_added:
                    design_principles.append("Consider molecular polarity and charge separation to selectively stabilize singlet excited states")
                    polarity_principle_added = True
                    
            # Add general principles if we don't have enough
            general_principles = [
                "Target reduced exchange energy through spatial separation of HOMO and LUMO"
                "Design molecules with charge-transfer character in excited states to influence S1-T1 energy splitting",
                "Consider the effects of conformational flexibility on excited state energetics",
                "Explore heteroatom substitution patterns to fine-tune orbital energies"
            ]
            
            # Add general principles if needed
            for principle in general_principles:
                if len(design_principles) >= 6:  # Limit to 6 principles
                    break
                if principle not in design_principles:
                    design_principles.append(principle)
                    
            # Write design principles
            for principle in design_principles[:6]:
                f.write(f"* {principle}\n")
                
            # Conclusions
            f.write("\n## Conclusion\n\n")
            f.write("Reverse TADF materials represent an intriguing class of compounds with unique photophysical properties. ")
            f.write("Our analysis reveals that the unusual S1-T1 energy ordering is influenced by a complex interplay of electronic, structural, and quantum mechanical factors. ")
            f.write("By carefully tuning molecular features such as electronic effects, conjugation patterns, and structural arrangements, ")
            f.write("it is possible to design materials that exhibit this rare phenomenon. The identified descriptors and design principles ")
            f.write("provide a valuable guide for the rational development of next-generation reverse TADF materials for advanced optoelectronic applications.\n")
            
        print(f"Comprehensive insight report generated: {report_path}")
        return report_path
    
    def run_insight_pipeline(self, modeling_results=None, exploration_results=None):
        """Run the complete insight generation pipeline."""
        self.load_results(modeling_results, exploration_results)
        
        # Analyze results
        classification_insights = self.analyze_classification_results()
        regression_insights = self.analyze_regression_results()
        exploration_insights = self.analyze_exploration_results()
        
        # Generate comprehensive report
        report_path = self.generate_comprehensive_report()
        
        return {
            'classification_insights': classification_insights,
            'regression_insights': regression_insights,
            'exploration_insights': exploration_insights,
            'report': report_path
        }
