#!/usr/bin/env python3
"""
Debug script to inspect PDB files and see what residues are actually present
"""
from pathlib import Path
from Bio.PDB import PDBParser

def inspect_pdb(pdb_path):
    """Inspect a PDB file and report all residues"""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('complex', str(pdb_path))

    print(f"\n{'='*70}")
    print(f"File: {pdb_path.name}")
    print(f"{'='*70}")

    # Get all residues organized by type
    protein_residues = []
    hetatm_residues = []
    water = []

    for model in structure:
        for chain in model:
            print(f"\nChain {chain.get_id()}:")
            for residue in chain:
                resname = residue.get_resname().strip()
                resid = residue.get_id()
                is_het = resid[0].strip()  # Hetero flag

                if resname == 'HOH':
                    water.append(resname)
                elif is_het:  # HETATM
                    hetatm_residues.append((resname, chain.get_id(), len(list(residue.get_atoms()))))
                    print(f"  HETATM: {resname} (chain {chain.get_id()}, {len(list(residue.get_atoms()))} atoms)")
                else:  # Regular protein
                    protein_residues.append(resname)

    print(f"\nSummary:")
    print(f"  Protein residues: {len(protein_residues)}")
    print(f"  HETATM residues: {len(hetatm_residues)}")
    print(f"  Water molecules: {len(water)}")

    if hetatm_residues:
        print(f"\nAll HETATM residues found:")
        for resname, chain, natoms in hetatm_residues:
            print(f"  - {resname} (chain {chain}, {natoms} atoms)")

    return hetatm_residues

def main():
    # Inspect first 5 failing PDB files
    dcoid_dir = Path("D-COID")

    failing_cases = [
        ("actives", "mini_complex_1a28_STR_A_active.pdb", "STR"),
        ("actives", "mini_complex_1a54_MDC_A_active.pdb", "MDC"),
        ("actives", "mini_complex_1a6v_NPC_H_active.pdb", "NPC"),
        ("actives", "mini_complex_1a6w_NIP_H_active.pdb", "NIP"),
        ("actives", "mini_complex_1a9u_SB2_A_active.pdb", "SB2"),
    ]

    print("Inspecting first 5 failing PDB files...")
    print("Looking for expected ligand codes in structures\n")

    for subdir, filename, expected_ligand in failing_cases:
        pdb_path = dcoid_dir / subdir / filename

        if not pdb_path.exists():
            print(f"\n[ERROR] File not found: {pdb_path}")
            continue

        print(f"\nExpected ligand code: {expected_ligand}")
        hetatm_residues = inspect_pdb(pdb_path)

        # Check if expected ligand is present
        found = any(resname == expected_ligand for resname, _, _ in hetatm_residues)
        if found:
            print(f"✓ Found expected ligand {expected_ligand}")
        else:
            print(f"✗ Expected ligand {expected_ligand} NOT FOUND")
            if hetatm_residues:
                print(f"  Maybe the ligand is actually: {[r[0] for r in hetatm_residues]}")

if __name__ == "__main__":
    main()
