# MC-PDFT-OPESf: Reaction Kinetics with Multireference Electronic Structure

**MC-PDFT-OPESf** couples multi-configuration pair-density functional theory (MC-PDFT) with on-the-fly enhanced sampling-flooding (OPESf) enhanced sampling via ASE, enabling efficient rate calculations for strongly correlated reactive systems.

## 📚 Requirements

To use MC-PDFT-OPESf, you'll need the following Python packages:
```bash
pip install pyscf ase
```
### PLUMED (with Python interface)
Build PLUMED with the Python interface and set the kernel path:
```bash
export PLUMED_KERNEL=/path/to/libplumedKernel.so
```

## Examples in the Paper

### Diels-Alder Reaction of Butadiene and Ethylene (MC-PDFT + OPESf)
**Inputs:** `da_opesf_plumed.dat`, `frame_da.xyz`  
**MC-PDFT Settings:** `active space = (6e, 6o)`, `on-top = tPBE`
```bash
python src/PDFT_OPESf.py --method mc-pdft --xyz data/diels_alder/frame_da.xyz --plumed data/diels_alder/da_opesf_plumed.dat --basis 'ccpvdz' --ontop 'tPBE' --n_act 6 --n_elec 6 --charge 0 --spin 0 --dt 0.0005 --T 900 --n_step 1000000
```

### SN2 Reaction of Methylchloride with Fluoride ion (KS-DFT + OPESf)
**Inputs:** `sn2_opesf_plumed.dat`, `frame_sn2.xyz`  
**KS-PDFT Settings:** `xc = b3lyp`, `basis = 6-31G*`
```bash
python src/PDFT_OPESf.py --method ks-dft --xyz data/sn2/frame_sn2.xyz --plumed data/sn2/sn2_opesf_plumed.dat --basis '6-31G*' --xc 'b3lyp' --charge -1 --spin 0 --dt 0.0005 -T 1200 --n_step 1000000
```

## 📜 Citation

If you use **MC-PDFT-OPESf** in your research or publications, please cite the following:

```bibtex

```
