#!/usr/bin/env python3
"""
Simple PDB inspector - no external dependencies needed
"""
from pathlib import Path
from collections import defaultdict

def inspect_pdb_simple(pdb_path, expected_ligand):
    """Inspect PDB file by reading HETATM records directly"""
    print(f"\n{'='*70}")
    print(f"File: {pdb_path.name}")
    print(f"Expected ligand: {expected_ligand}")
    print(f"{'='*70}")

    hetatm_residues = defaultdict(int)
    atom_lines = []

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('HETATM'):
                # PDB format: HETATM record
                # Columns 18-20: residue name
                resname = line[17:20].strip()

                if resname != 'HOH':  # Skip water
                    hetatm_residues[resname] += 1
                    atom_lines.append(line.strip())

    print(f"\nHETATM residues found (excluding water):")
    if hetatm_residues:
        for resname, count in sorted(hetatm_residues.items()):
            marker = "✓ MATCH!" if resname == expected_ligand else ""
            print(f"  - {resname}: {count} atoms {marker}")

        # Show first HETATM line as example
        if atom_lines:
            print(f"\nFirst HETATM line:")
            print(f"  {atom_lines[0]}")
    else:
        print("  [NONE FOUND]")

    # Check if expected ligand is present
    if expected_ligand in hetatm_residues:
        print(f"\n✓ Expected ligand '{expected_ligand}' IS present")
        return True
    else:
        print(f"\n✗ Expected ligand '{expected_ligand}' NOT FOUND")
        if hetatm_residues:
            print(f"  Available ligands: {list(hetatm_residues.keys())}")
        return False

def main():
    dcoid_dir = Path("D-COID")

    failing_cases = [
        ("actives", "mini_complex_1a28_STR_A_active.pdb", "STR"),
        ("actives", "mini_complex_1a54_MDC_A_active.pdb", "MDC"),
        ("actives", "mini_complex_1a6v_NPC_H_active.pdb", "NPC"),
        ("actives", "mini_complex_1a6w_NIP_H_active.pdb", "NIP"),
        ("actives", "mini_complex_1a9u_SB2_A_active.pdb", "SB2"),
    ]

    print("Inspecting first 5 failing PDB files...")
    print("Checking what HETATM records are actually in the files\n")

    found_count = 0
    for subdir, filename, expected_ligand in failing_cases:
        pdb_path = dcoid_dir / subdir / filename

        if not pdb_path.exists():
            print(f"\n[ERROR] File not found: {pdb_path}")
            continue

        if inspect_pdb_simple(pdb_path, expected_ligand):
            found_count += 1

    print(f"\n{'='*70}")
    print(f"Summary: {found_count}/{len(failing_cases)} expected ligands found")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
