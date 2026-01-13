#!/usr/bin/env python3
"""
Protein Name Mapping Utility for RPPA Data
==========================================

This module provides comprehensive protein name mapping for RPPA (Reverse Phase Protein Array) data.
Handles various antibody names, phosphorylation states, and protein aliases commonly used in proteomics.

Usage:
    from src.utils.protein_mapper import ProteinMapper
    
    mapper = ProteinMapper()
    mapped_names = mapper.map_protein_names(user_protein_list)
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

class ProteinMapper:
    """Comprehensive protein name mapping for RPPA data"""
    
    def __init__(self, custom_mappings_file: Optional[str] = None, logger=None):
        """
        Initialize protein mapper
        
        Args:
            custom_mappings_file: Optional custom protein mappings JSON file
            logger: Optional logger instance
        """
        self.logger = logger or self._setup_logger()
        
        # Load built-in mappings
        self.protein_mappings = self._create_comprehensive_protein_mappings()
        
        # Load custom mappings if provided
        if custom_mappings_file:
            self._load_custom_mappings(custom_mappings_file)
        
        self.logger.info(f"✅ ProteinMapper initialized with {len(self.protein_mappings)} mappings")
    
    def _setup_logger(self):
        """Setup default logger"""
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
    
    def _create_comprehensive_protein_mappings(self) -> Dict[str, str]:
        """Create comprehensive protein name mappings"""
        
        # Core protein mappings with common aliases
        core_mappings = {
            # Tumor suppressors
            'p53': ['TP53', 'P53', 'p53', 'Tp53', 'tumor_protein_p53'],
            'p21': ['CDKN1A', 'P21', 'p21', 'CIP1', 'WAF1'],
            'p27': ['CDKN1B', 'P27', 'p27', 'KIP1'],
            'Rb': ['RB1', 'RB', 'Rb', 'pRb', 'retinoblastoma'],
            'PTEN': ['PTEN', 'Pten', 'MMAC1', 'TEP1'],
            
            # Oncogenes and growth factors
            'c-Myc': ['MYC', 'C-MYC', 'c-Myc', 'MYCC', 'c_Myc'],
            'EGFR': ['EGFR', 'EGF-R', 'ERBB1', 'HER1'],
            'HER2': ['ERBB2', 'HER2', 'Her2', 'NEU', 'HER-2'],
            'HER3': ['ERBB3', 'HER3', 'Her3', 'HER-3'],
            'IGF1R': ['IGF1R', 'IGF-1R', 'IGF1_Receptor'],
            
            # PI3K/AKT/mTOR pathway
            'AKT': ['AKT1', 'PKB', 'AKT', 'Akt', 'PKB_alpha'],
            'mTOR': ['MTOR', 'mTOR', 'FRAP1', 'RAFT1'],
            'p70S6K1': ['RPS6KB1', 'S6K1', 'p70S6K1', 'p70_S6K1'],
            'S6': ['RPS6', 'S6', 'S6_ribosomal'],
            '4E-BP1': ['EIF4EBP1', '4E-BP1', '4EBP1', 'PHAS-I'],
            'GSK3': ['GSK3B', 'GSK3', 'GSK3-beta', 'GSK3_beta'],
            'FOXO3A': ['FOXO3', 'FOXO3A', 'FKHRL1'],
            
            # MAPK pathway
            'ERK': ['MAPK1', 'ERK2', 'ERK', 'p42MAPK'],
            'MEK': ['MAP2K1', 'MEK1', 'MEK', 'MAPKK1'],
            'RAF': ['RAF1', 'RAF', 'c-RAF', 'CRAF'],
            'JNK': ['MAPK8', 'JNK', 'JNK1', 'SAPK1'],
            'p38': ['MAPK14', 'p38', 'p38alpha', 'MAPK14'],
            
            # Cell cycle
            'Cyclin_D1': ['CCND1', 'Cyclin_D1', 'CyclinD1', 'CYCLIN_D1'],
            'Cyclin_E1': ['CCNE1', 'Cyclin_E1', 'CyclinE1', 'CYCLIN_E1'],
            'CDK4': ['CDK4', 'Cdk4', 'CDK-4'],
            'CDK6': ['CDK6', 'Cdk6', 'CDK-6'],
            
            # Apoptosis
            'Bcl-2': ['BCL2', 'Bcl-2', 'BCL-2', 'bcl2'],
            'Bax': ['BAX', 'Bax', 'bcl2_associated_X'],
            'Bad': ['BAD', 'Bad', 'BCL2_antagonist'],
            'Cleaved_Caspase_3': ['CASP3', 'Caspase_3', 'Cleaved_Caspase_3', 'CC3'],
            'PARP': ['PARP1', 'PARP', 'Cleaved_PARP'],
            
            # Adhesion and EMT
            'E-Cadherin': ['CDH1', 'E-Cadherin', 'E_Cadherin', 'ECAD'],
            'N-Cadherin': ['CDH2', 'N-Cadherin', 'N_Cadherin', 'NCAD'],
            'beta-Catenin': ['CTNNB1', 'CTNBB1', 'beta-Catenin', 'β-Catenin', 'beta_Catenin'],
            'Vimentin': ['VIM', 'Vimentin'],
            'Snail': ['SNAI1', 'Snail'],
            'Slug': ['SNAI2', 'Slug'],
            'ZEB1': ['ZEB1', 'Zeb1'],
            'Twist': ['TWIST1', 'Twist'],
            
            # DNA damage response
            'ATM': ['ATM', 'Atm'],
            'ATR': ['ATR', 'Atr'],
            'Chk1': ['CHEK1', 'Chk1', 'CHK1'],
            'Chk2': ['CHEK2', 'Chk2', 'CHK2'],
            'BRCA1': ['BRCA1', 'Brca1'],
            'p53BP1': ['TP53BP1', '53BP1', 'p53BP1'],
            
            # Metabolism
            'ACC': ['ACACA', 'ACC', 'ACC1', 'Acetyl_CoA_Carboxylase'],
            'FASN': ['FASN', 'Fatty_Acid_Synthase'],
            'LDHA': ['LDHA', 'LDH-A', 'Lactate_Dehydrogenase_A'],
            
            # Immune/Inflammation
            'NF-kB-p65': ['RELA', 'NFkB', 'NF-kB', 'p65', 'NFKB'],
            'IkB-alpha': ['NFKBIA', 'IkB', 'IκBα', 'IKBA'],
            'STAT3': ['STAT3', 'Stat3'],
            'STAT5': ['STAT5A', 'STAT5', 'Stat5'],
        }
        
        # Phosphorylation state mappings
        phospho_mappings = {
            # AKT pathway phosphorylations
            'p-AKT-S473': ['AKT_pS473', 'pAKT_S473', 'p-Akt_S473', 'AKT1_pS473'],
            'p-AKT-T308': ['AKT_pT308', 'pAKT_T308', 'p-Akt_T308', 'AKT1_pT308'],
            'p-mTOR-S2448': ['MTOR_pS2448', 'p-mTOR_S2448', 'pmTOR_S2448'],
            'p-p70S6K1-T389': ['RPS6KB1_pT389', 'p-p70S6K1_T389', 'pp70S6K1_T389'],
            'p-S6-S235_S236': ['RPS6_pS235_S236', 'p-S6_S235_236', 'pS6_S235_236'],
            'p-4E-BP1-T37_T46': ['EIF4EBP1_pT37_T46', 'p-4EBP1_T37_46'],
            'p-GSK3-S21_S9': ['GSK3B_pS9', 'p-GSK3_S9', 'pGSK3_S9'],
            
            # MAPK phosphorylations
            'p-ERK-T202_Y204': ['MAPK1_pT202_Y204', 'p-ERK_T202_Y204', 'pERK'],
            'p-MEK-S217_S221': ['MAP2K1_pS217_S221', 'p-MEK_S217_221'],
            'p-p38-T180_Y182': ['MAPK14_pT180_Y182', 'p-p38_T180_Y182'],
            'p-JNK-T183_Y185': ['MAPK8_pT183_Y185', 'p-JNK_T183_Y185'],
            
            # Cell cycle phosphorylations
            'p-Rb-S807_S811': ['RB1_pS807_S811', 'p-Rb_S807_811', 'pRb'],
            
            # DNA damage phosphorylations
            'p-ATM-S1981': ['ATM_pS1981', 'p-ATM_S1981', 'pATM'],
            'p-Chk1-S345': ['CHEK1_pS345', 'p-Chk1_S345', 'pChk1'],
            'p-Chk2-T68': ['CHEK2_pT68', 'p-Chk2_T68', 'pChk2'],
            'p-p53-S15': ['TP53_pS15', 'p-p53_S15', 'pp53_S15'],
            
            # EGFR family phosphorylations
            'p-EGFR-Y1068': ['EGFR_pY1068', 'p-EGFR_Y1068', 'pEGFR_Y1068'],
            'p-HER2-Y1248': ['ERBB2_pY1248', 'p-HER2_Y1248', 'pHER2'],
            
            # Apoptosis phosphorylations
            'p-Bad-S112': ['BAD_pS112', 'p-Bad_S112', 'pBad_S112'],
            'p-Bcl-2-S70': ['BCL2_pS70', 'p-Bcl2_S70', 'pBcl2'],
            
            # STAT phosphorylations
            'p-STAT3-Y705': ['STAT3_pY705', 'p-STAT3_Y705', 'pSTAT3_Y705'],
            'p-STAT5-Y694': ['STAT5A_pY694', 'p-STAT5_Y694', 'pSTAT5'],
            
            # NFkB phosphorylations
            'p-NF-kB-p65-S536': ['RELA_pS536', 'p-NFkB_S536', 'pNFkB_p65_S536'],
            'p-IkB-alpha-S32': ['NFKBIA_pS32', 'p-IkB_S32', 'pIkB_S32'],
        }
        
        # Cleaved protein mappings
        cleaved_mappings = {
            'Cleaved_Caspase_3': ['CASP3', 'Cleaved_Caspase_3', 'CC3', 'Cl_Caspase_3'],
            'Cleaved_PARP': ['PARP1', 'Cleaved_PARP', 'Cl_PARP', 'c_PARP'],
            'Cleaved_Notch1': ['NOTCH1', 'Cleaved_Notch1', 'Cl_Notch1'],
        }
        
        # Combine all mappings
        all_mappings = {}
        
        # Add core mappings
        for canonical, aliases in core_mappings.items():
            for alias in aliases:
                all_mappings[alias.upper()] = canonical
                all_mappings[alias.lower()] = canonical
                all_mappings[alias] = canonical
        
        # Add phosphorylation mappings
        for canonical, aliases in phospho_mappings.items():
            for alias in aliases:
                all_mappings[alias.upper()] = canonical
                all_mappings[alias.lower()] = canonical
                all_mappings[alias] = canonical
        
        # Add cleaved protein mappings
        for canonical, aliases in cleaved_mappings.items():
            for alias in aliases:
                all_mappings[alias.upper()] = canonical
                all_mappings[alias.lower()] = canonical
                all_mappings[alias] = canonical
        
        return all_mappings
    
    def _load_custom_mappings(self, custom_file: str):
        """Load custom protein mappings from JSON file"""
        try:
            with open(custom_file, 'r') as f:
                custom_mappings = json.load(f)
            
            # Add custom mappings to existing ones
            for canonical, aliases in custom_mappings.items():
                for alias in aliases:
                    self.protein_mappings[alias.upper()] = canonical
                    self.protein_mappings[alias.lower()] = canonical
                    self.protein_mappings[alias] = canonical
            
            self.logger.info(f"✅ Loaded custom protein mappings from {custom_file}")
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to load custom mappings: {e}")
    
    def map_protein_names(self, protein_names: List[str], 
                         target_format: str = 'tcga_rppa') -> Dict[str, List[str]]:
        """
        Map user protein names to TCGA RPPA format names
        
        Args:
            protein_names: List of user protein names
            target_format: 'tcga_rppa' for TCGA format with vendor codes
            
        Returns:
            mapping: Dict of user_name -> List[tcga_names] (can be multiple forms)
        """
        mapping = {}
        unmapped = []
        
        for name in protein_names:
            # Clean the name
            clean_name = self._clean_protein_name(name)
            tcga_names = []
            
            # Try to map to TCGA RPPA format
            if target_format == 'tcga_rppa':
                tcga_names = self._map_to_tcga_rppa_format(name, clean_name)
            
            if tcga_names:
                mapping[name] = tcga_names
            else:
                # Try standard mapping
                if clean_name in self.protein_mappings:
                    base_name = self.protein_mappings[clean_name]
                    mapping[name] = [base_name]
                elif clean_name.upper() in self.protein_mappings:
                    base_name = self.protein_mappings[clean_name.upper()]
                    mapping[name] = [base_name]
                elif clean_name.lower() in self.protein_mappings:
                    base_name = self.protein_mappings[clean_name.lower()]
                    mapping[name] = [base_name]
                else:
                    # Try fuzzy matching
                    fuzzy_match = self._fuzzy_match_protein(clean_name)
                    if fuzzy_match:
                        mapping[name] = [fuzzy_match]
                    else:
                        unmapped.append(name)
        
        self.logger.info(f"📊 Mapped {len(mapping)}/{len(protein_names)} proteins")
        if unmapped:
            self.logger.warning(f"⚠️  Unmapped proteins: {unmapped[:5]}{'...' if len(unmapped) > 5 else ''}")
        
        return mapping
    
    def _map_to_tcga_rppa_format(self, original_name: str, clean_name: str) -> List[str]:
        """
        Map protein name to TCGA RPPA format with vendor codes
        
        TCGA format: ProteinName[-phosphoSite]-Vendor-Validation
        Vendor codes: R, M, C (antibody vendors)
        Validation codes: V, E, C, QC (validation status)
        
        Returns list of possible TCGA names since one protein can have multiple antibodies
        """
        tcga_names = []
        
        # Common vendor-validation combinations in TCGA
        vendor_combos = [
            'R-V', 'R-E', 'R-C',
            'M-V', 'M-E', 'M-C', 'M-QC',
            'C-V', 'C-E', 'C-C'
        ]
        
        # Extract phosphorylation info
        phospho_info = self.get_phosphorylation_info(original_name)
        
        # Determine base protein name
        base_protein = None
        
        # Check our mappings
        if clean_name.upper() in self.protein_mappings:
            base_protein = self.protein_mappings[clean_name.upper()]
        elif phospho_info['base_protein'].upper() in self.protein_mappings:
            base_protein = self.protein_mappings[phospho_info['base_protein'].upper()]
        
        # If no mapping found, use the original name
        if not base_protein:
            base_protein = phospho_info['base_protein'] if phospho_info['is_phosphorylated'] else clean_name
        
        # Generate TCGA format names
        if phospho_info['is_phosphorylated'] and phospho_info['phospho_sites']:
            # Phosphorylated protein
            for i, (residue, site) in enumerate(zip(phospho_info['phospho_residues'], phospho_info['phospho_sites'])):
                phospho_suffix = f"_p{residue}{site}"
                
                # Try different vendor combinations
                for combo in vendor_combos[:3]:  # Usually use first few combos
                    tcga_name = f"{base_protein}{phospho_suffix}-{combo}"
                    tcga_names.append(tcga_name)
        else:
            # Non-phosphorylated protein
            # Try different vendor combinations
            for combo in vendor_combos[:3]:  # Usually use first few combos
                tcga_name = f"{base_protein}-{combo}"
                tcga_names.append(tcga_name)
        
        # Also check for exact matches in known TCGA names
        # (This would require loading actual TCGA protein list)
        
        return tcga_names
    
    def _clean_protein_name(self, name: str) -> str:
        """Clean and standardize protein name"""
        # Remove common prefixes/suffixes
        clean_name = name.strip()
        
        # Remove antibody-specific suffixes
        clean_name = re.sub(r'_Ab\d*$', '', clean_name)
        clean_name = re.sub(r'_antibody$', '', clean_name, flags=re.IGNORECASE)
        
        # Standardize phosphorylation notation
        clean_name = re.sub(r'_p([STY])(\d+)', r'_pS\2', clean_name)
        clean_name = re.sub(r'-p([STY])(\d+)', r'_pS\2', clean_name)
        
        # Standardize separators
        clean_name = clean_name.replace('_', '-').replace(' ', '-')
        
        return clean_name
    
    def _fuzzy_match_protein(self, name: str) -> Optional[str]:
        """Attempt fuzzy matching for protein names"""
        name_upper = name.upper()
        
        # Try partial matches
        for mapped_name, canonical in self.protein_mappings.items():
            if name_upper in mapped_name.upper() or mapped_name.upper() in name_upper:
                if len(name_upper) > 2 and len(mapped_name) > 2:  # Avoid very short matches
                    return canonical
        
        return None
    
    def get_phosphorylation_info(self, protein_name: str) -> Dict[str, any]:
        """Extract phosphorylation information from protein name"""
        info = {
            'is_phosphorylated': False,
            'base_protein': protein_name,
            'phospho_sites': [],
            'phospho_residues': []
        }
        
        # Check for phosphorylation patterns
        phospho_patterns = [
            r'p-(.+?)[-_]([STY])(\d+)',  # p-Protein_S123
            r'(.+?)[-_]p([STY])(\d+)',   # Protein_pS123
            r'(.+?)[-_]p([STY]\d+)',     # Protein_pS123
        ]
        
        for pattern in phospho_patterns:
            match = re.search(pattern, protein_name, re.IGNORECASE)
            if match:
                info['is_phosphorylated'] = True
                if len(match.groups()) >= 3:
                    info['base_protein'] = match.group(1)
                    info['phospho_residues'].append(match.group(2))
                    info['phospho_sites'].append(match.group(3))
        
        return info
    
    def create_mapping_report(self, protein_names: List[str]) -> str:
        """Generate comprehensive mapping report"""
        mapping = self.map_protein_names(protein_names)
        
        # Categorize proteins
        mapped_proteins = list(mapping.keys())
        unmapped_proteins = [p for p in protein_names if p not in mapping]
        
        # Analyze phosphorylation
        phospho_proteins = []
        cleaved_proteins = []
        
        for name in mapped_proteins:
            if 'p-' in name or '_p' in name:
                phospho_proteins.append(name)
            elif 'Cleaved' in name or 'Cl_' in name:
                cleaved_proteins.append(name)
        
        report = f"""
🧬 Protein Mapping Report
========================
📊 Total proteins: {len(protein_names)}
📊 Successfully mapped: {len(mapped_proteins)} ({len(mapped_proteins)/len(protein_names)*100:.1f}%)
📊 Unmapped proteins: {len(unmapped_proteins)} ({len(unmapped_proteins)/len(protein_names)*100:.1f}%)

🔬 Protein Categories:
📊 Phosphorylated proteins: {len(phospho_proteins)}
📊 Cleaved proteins: {len(cleaved_proteins)}

✅ Example Mappings:
"""
        
        # Show example mappings
        for i, (user_name, canonical) in enumerate(list(mapping.items())[:10]):
            report += f"   {user_name} → {canonical}\n"
        
        if len(mapping) > 10:
            report += f"   ... and {len(mapping) - 10} more\n"
        
        if unmapped_proteins:
            report += f"\n⚠️  Unmapped Proteins:\n"
            for protein in unmapped_proteins[:10]:
                report += f"   {protein}\n"
            if len(unmapped_proteins) > 10:
                report += f"   ... and {len(unmapped_proteins) - 10} more\n"
        
        return report
    
    def export_mappings(self, output_file: str):
        """Export all protein mappings to JSON file"""
        # Organize mappings by canonical name
        organized_mappings = {}
        for alias, canonical in self.protein_mappings.items():
            if canonical not in organized_mappings:
                organized_mappings[canonical] = []
            if alias not in organized_mappings[canonical]:
                organized_mappings[canonical].append(alias)
        
        with open(output_file, 'w') as f:
            json.dump(organized_mappings, f, indent=2)
        
        self.logger.info(f"✅ Exported protein mappings to {output_file}")

def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Map protein names')
    parser.add_argument('--input', required=True, help='Input protein list file')
    parser.add_argument('--output', help='Output mapping file')
    parser.add_argument('--custom-mappings', help='Custom mappings JSON file')
    
    args = parser.parse_args()
    
    # Load protein names
    with open(args.input, 'r') as f:
        protein_names = [line.strip() for line in f if line.strip()]
    
    # Create mapper
    mapper = ProteinMapper(args.custom_mappings)
    
    # Generate report
    report = mapper.create_mapping_report(protein_names)
    print(report)
    
    # Save results if output specified
    if args.output:
        mapping = mapper.map_protein_names(protein_names)
        with open(args.output, 'w') as f:
            json.dump(mapping, f, indent=2)

if __name__ == "__main__":
    main()