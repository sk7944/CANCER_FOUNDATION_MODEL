#!/usr/bin/env python3
"""
RPPA Data Mapping Guide for TCGA Format
========================================

This module provides detailed guidance on how RPPA protein names are formatted
in TCGA data and how to map user protein names to these formats.

TCGA RPPA Format: ProteinName[-phosphoSite]-Vendor-Validation
- Vendor codes: R (Rabbit), M (Mouse), C (Chicken)
- Validation codes: V (Validated), E (Evaluated), C (Caution), QC (Quality Control)

Example mappings:
- AKT → AKT-R-V, AKT-M-E, etc.
- p-AKT-S473 → Akt_pS473-R-V, Akt_pS473-M-E, etc.
- EGFR → EGFR-R-V, EGFR-M-C, etc.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple

class RPPAMappingGuide:
    """Guide for mapping user RPPA data to TCGA format"""
    
    def __init__(self):
        self.tcga_rppa_examples = self._load_tcga_examples()
        
    def _load_tcga_examples(self) -> Dict[str, List[str]]:
        """Load example TCGA RPPA protein names"""
        
        # Real examples from TCGA RPPA data
        examples = {
            # Non-phosphorylated proteins
            'EGFR': ['EGFR-R-V', 'EGFR-M-E', 'EGFR-R-C'],
            'HER2': ['HER2-M-V', 'HER2-R-E'],
            'PTEN': ['PTEN-R-V', 'PTEN-M-C'],
            'p53': ['p53-R-V', 'p53-M-E', 'p53-R-C'],
            'Rb': ['Rb-M-V', 'Rb-R-E'],
            'GAPDH': ['GAPDH-M-C', 'GAPDH-R-V'],
            'c-Kit': ['c-Kit-R-V', 'c-Kit-M-E'],
            'c-Met': ['c-Met-M-QC', 'c-Met-R-V'],
            'Bak': ['Bak-R-E'],
            'Bim': ['Bim-R-V'],
            'ATM': ['ATM-R-V'],
            'Chk2': ['Chk2-M-E', 'Chk2-R-C'],
            'JNK2': ['JNK2-R-C'],
            'S6': ['S6-R-E'],
            'Stathmin': ['Stathmin-R-V'],
            
            # Phosphorylated proteins
            'p-AKT-S473': ['Akt_pS473-R-V', 'Akt_pS473-M-E'],
            'p-AKT-T308': ['Akt_pT308-R-V', 'Akt_pT308-M-C'],
            'p-mTOR-S2448': ['mTOR_pS2448-R-V'],
            'p-S6-S235/S236': ['S6_pS235_S236-R-V', 'S6_pS235_S236-M-E'],
            'p-GSK3-S9': ['GSK3_pS9-R-V', 'GSK3-alpha-beta_pS21_S9-R-V'],
            'p-ERK-T202/Y204': ['p44_42_MAPK-R-V', 'Erk1_2_pT202_Y204-R-E'],
            'p-EGFR-Y1068': ['EGFR_pY1068-R-V', 'EGFR_pY1068-M-E'],
            'p-EGFR-Y1173': ['EGFR_pY1173-R-V'],
            'p-HER2-Y1248': ['HER2_pY1248-R-V', 'HER2_pY1248-M-E'],
            'p-c-Met-Y1235': ['c-Met_pY1235-R-V'],
            'p-Chk2-T68': ['Chk2_pT68-R-C', 'Chk2_pT68-R-E'],
            'p-Rb-S807/S811': ['Rb_pS807_S811-R-V', 'Rb_pS807_S811-M-E'],
            'p-STAT3-Y705': ['STAT3_pY705-R-V'],
            'p-ER-alpha-S118': ['ER-alpha_pS118-R-V'],
            'p-MEK1-S217/S221': ['MEK1_pS217_S221-R-V'],
            'p-JNK-T183/Y185': ['JNK_pT183_pY185-R-C', 'JNK_pT183_pT185-R-V'],
            'p-C-Raf-S338': ['C-Raf_pS338-R-C'],
            'p-Shc-Y317': ['Shc_pY317-R-V'],
            'p-NDRG1-T346': ['NDRG1_pT346-R-V'],
            'p-ACC-S79': ['ACC_pS79-R-V'],
            'p-TAZ-S89': ['TAZ_pS89-R-C'],
            'p-4E-BP1-T37': ['4E-BP1_pT37-R-V', '4E-BP1_pT37_T46-R-V'],
            
            # Cleaved proteins
            'Cleaved-Caspase-3': ['Caspase-3-R-C', 'Cleaved_Caspase_3-R-V'],
            'Cleaved-PARP': ['PARP-Ab-3-R-C', 'Cleaved_PARP-R-V'],
            
            # Protein complexes/subunits
            'Complex-II-subunit30': ['Complex-II_subunit30-M-V'],
            'Oxphos-complex-V-subunitb': ['Oxphos-complex-V_subunitb-M-E'],
            '14-3-3-epsilon': ['14-3-3_epsilon-M-C'],
            '53BP1': ['53BP1-R-E'],
            
            # Other proteins
            'Annexin-1': ['Annexin_1-M-E'],
            'Transglutaminase': ['Transglutaminase-M-V'],
            'eEF2': ['eEF2-R-V'],
            'PKM2': ['PKM2-R-C'],
            'YB-1': ['YB-1-R-V'],
            'XRCC1': ['XRCC1-R-E'],
            'IGFBP2': ['IGFBP2-R-V'],
            'PYGL': ['PYGL-R-E'],
            'Dvl3': ['Dvl3-R-V'],
            'Bap1c-4': ['Bap1c-4-M-E'],
            'ERCC5': ['ERCC5-R-C'],
            'RBM15': ['RBM15-R-V'],
            'CD26': ['CD26-R-V'],
            'PDCD1': ['PDCD1-M-E']
        }
        
        return examples
    
    def show_mapping_guide(self):
        """Display comprehensive RPPA mapping guide"""
        
        print("=" * 80)
        print("📋 TCGA RPPA PROTEIN NAME MAPPING GUIDE")
        print("=" * 80)
        print()
        print("🔬 TCGA RPPA Format: ProteinName[-phosphoSite]-Vendor-Validation")
        print()
        print("📊 Vendor Codes:")
        print("   • R = Rabbit antibody")
        print("   • M = Mouse antibody")
        print("   • C = Chicken antibody")
        print()
        print("📊 Validation Codes:")
        print("   • V = Validated")
        print("   • E = Evaluated")
        print("   • C = Caution")
        print("   • QC = Quality Control")
        print()
        print("-" * 80)
        print("💡 IMPORTANT NOTES FOR USERS:")
        print("-" * 80)
        print()
        print("1. ONE PROTEIN → MULTIPLE TCGA ENTRIES")
        print("   Your single protein measurement may map to multiple TCGA features")
        print("   due to different antibodies (vendor codes).")
        print()
        print("   Example: 'EGFR' in your data may map to:")
        print("   - RPPA_EGFR-R-V_value")
        print("   - RPPA_EGFR-M-E_value")
        print("   - RPPA_EGFR-R-C_value")
        print()
        print("2. PHOSPHORYLATION NOTATION")
        print("   User format: p-AKT-S473 or pAKT(S473)")
        print("   TCGA format: Akt_pS473-R-V")
        print()
        print("3. HANDLING DUPLICATE MAPPINGS")
        print("   Options:")
        print("   a) Use the same value for all vendor variants (recommended)")
        print("   b) Use the most validated version (suffix -V)")
        print("   c) Average across multiple measurements if available")
        print()
        print("-" * 80)
        print("📝 EXAMPLE MAPPINGS:")
        print("-" * 80)
        print()
        
        # Show examples
        for user_name, tcga_names in list(self.tcga_rppa_examples.items())[:15]:
            print(f"{user_name:20s} → {', '.join(tcga_names)}")
        
        print("\n... and more")
        print()
        print("-" * 80)
        print("🔧 RECOMMENDED USER DATA FORMAT:")
        print("-" * 80)
        print()
        print("CSV/TSV file with:")
        print("• Rows: Patient/Sample IDs")
        print("• Columns: Protein names (e.g., EGFR, p-AKT-S473, HER2)")
        print("• Values: log2(normalized protein abundance)")
        print()
        print("Example:")
        print("Sample_ID,EGFR,p-AKT-S473,HER2,PTEN")
        print("Patient_1,1.23,-0.45,0.67,-1.23")
        print("Patient_2,0.89,0.12,-0.34,0.45")
        print()
    
    def create_mapping_table(self, user_proteins: List[str]) -> pd.DataFrame:
        """
        Create a mapping table showing how user proteins map to TCGA format
        
        Args:
            user_proteins: List of user protein names
            
        Returns:
            DataFrame with mapping information
        """
        
        mapping_data = []
        
        for protein in user_proteins:
            # Clean protein name
            clean_name = protein.strip().replace(' ', '-')
            
            # Check if it's in our examples
            if clean_name in self.tcga_rppa_examples:
                tcga_forms = self.tcga_rppa_examples[clean_name]
            else:
                # Try to find partial matches
                tcga_forms = []
                for example_protein, example_forms in self.tcga_rppa_examples.items():
                    if clean_name.upper() in example_protein.upper() or example_protein.upper() in clean_name.upper():
                        tcga_forms.extend(example_forms)
                
                if not tcga_forms:
                    # Generate possible forms
                    if 'p-' in clean_name or '_p' in clean_name:
                        # Phosphorylated
                        base = clean_name.split('p-')[0] if 'p-' in clean_name else clean_name.split('_p')[0]
                        tcga_forms = [f"{base}_pSite-R-V", f"{base}_pSite-M-E"]
                    else:
                        # Non-phosphorylated
                        tcga_forms = [f"{clean_name}-R-V", f"{clean_name}-M-E", f"{clean_name}-R-C"]
            
            for tcga_form in tcga_forms:
                mapping_data.append({
                    'User_Protein': protein,
                    'TCGA_Feature': f"RPPA_{tcga_form}_value",
                    'TCGA_Cox_Feature': f"RPPA_{tcga_form}_cox",
                    'Vendor': tcga_form.split('-')[-2] if '-' in tcga_form else 'Unknown',
                    'Validation': tcga_form.split('-')[-1] if '-' in tcga_form else 'Unknown'
                })
        
        return pd.DataFrame(mapping_data)
    
    def generate_duplication_strategy(self, user_proteins: List[str]) -> Dict[str, str]:
        """
        Generate strategy for handling protein duplication across vendor codes
        
        Returns:
            Dictionary with recommendations
        """
        
        strategy = {
            'approach': 'duplicate_values',
            'description': 'Duplicate user protein values across all vendor variants',
            'implementation': """
# For each user protein value, create multiple TCGA features:
for user_protein in user_data.columns:
    protein_value = user_data[user_protein]
    
    # Find all TCGA variants for this protein
    tcga_variants = find_tcga_variants(user_protein)
    
    # Duplicate the value across all variants
    for tcga_feature in tcga_variants:
        processed_data[tcga_feature] = protein_value
        processed_data[tcga_feature.replace('_value', '_cox')] = 0  # Cox placeholder
            """,
            'alternative_approaches': [
                {
                    'name': 'use_primary_vendor',
                    'description': 'Use only the most common vendor variant (usually R-V)'
                },
                {
                    'name': 'weighted_average',
                    'description': 'If multiple measurements exist, use weighted average'
                },
                {
                    'name': 'missing_for_others',
                    'description': 'Use actual value for one variant, mark others as missing'
                }
            ]
        }
        
        return strategy

def main():
    """Example usage and demonstration"""
    
    guide = RPPAMappingGuide()
    
    # Show the guide
    guide.show_mapping_guide()
    
    # Example user proteins
    print("\n" + "=" * 80)
    print("📊 EXAMPLE MAPPING FOR USER DATA")
    print("=" * 80)
    
    user_proteins = [
        'EGFR',
        'p-AKT-S473',
        'HER2',
        'PTEN',
        'p-mTOR-S2448',
        'GAPDH'
    ]
    
    print("\nUser proteins:", user_proteins)
    print("\nMapping to TCGA format:")
    
    mapping_df = guide.create_mapping_table(user_proteins)
    print(mapping_df.to_string())
    
    # Show duplication strategy
    print("\n" + "=" * 80)
    print("🔄 DUPLICATION STRATEGY")
    print("=" * 80)
    
    strategy = guide.generate_duplication_strategy(user_proteins)
    print(f"\nRecommended approach: {strategy['approach']}")
    print(f"Description: {strategy['description']}")
    print("\nImplementation example:")
    print(strategy['implementation'])
    
    print("\nAlternative approaches:")
    for alt in strategy['alternative_approaches']:
        print(f"  • {alt['name']}: {alt['description']}")

if __name__ == "__main__":
    main()