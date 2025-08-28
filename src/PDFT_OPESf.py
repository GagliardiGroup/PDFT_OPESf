import numpy as np
from os.path import exists, isfile
from ase.io import read
from ase.calculators.calculator import Calculator, all_changes
from ase.units import fs, kJ, mol, nm, kB, Ha, Bohr
from ase.parallel import broadcast, world
from ase.md.nvtberendsen import NVTBerendsen
from ase.io.trajectory import Trajectory
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from pyscf import gto, dft, mcscf, mcpdft
from pyscf.lib import chkfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--xyz', type=str, default='frame_da.xyz')
parser.add_argument('--position_npy', type=str, default=None)
parser.add_argument('--momentum_npy', type=str, default=None)

# Bias and Dynamics Settings
parser.add_argument('--plumed', type=str, default='da_opesf_plumed.dat')
parser.add_argument('--n_step', type=int, default=1000000)
parser.add_argument('--dt', type=float, default=0.0005)
parser.add_argument('--T', type=float, default=900)

parser.add_argument('--method', type=str, default='ks-dft', choices=['ks-dft', 'mc-pdft'])

# KS-DFT Settings
parser.add_argument('--basis', type=str, default='ccpvdz')
parser.add_argument('--charge', type=int, default=0)
parser.add_argument('--spin', type=int, default=0)
parser.add_argument('--xc', type=str, default='b3lyp')

# MC-PDFT Settings
parser.add_argument('--ontop', type=str, default='tPBE')
parser.add_argument('--n_act', type=int, default=6)
parser.add_argument('--n_elec', type=int, default=6)

args = parser.parse_args()

timestep = args.dt
ps = 1000 * fs
T = args.T
steps = args.n_step
atoms = read(args.xyz)
plumed_dat = read_plumed_to_setup(args.plumed)


class Plumed(Calculator):
    """
    Sucerquia et al., J. Chem. Phys., 156.15, 154301 (2022)
    """
    implemented_properties = ['energy', 'forces']

    def __init__(self, calc, input, timestep, atoms=None, kT=1., log='',
                 restart=False, use_charge=False, update_charge=False):
        """
        Plumed calculator is used for simulations of enhanced sampling methods
        with the open-source code PLUMED (plumed.org).

        [1] The PLUMED consortium, Nat. Methods 16, 670 (2019)
        [2] Tribello, Bonomi, Branduardi, Camilloni, and Bussi,
        Comput. Phys. Commun. 185, 604 (2014)

        Parameters
        ----------
        calc: Calculator object
            It  computes the unbiased forces

        input: List of strings
            It contains the setup of plumed actions

        timestep: float
            Time step of the simulated dynamics

        atoms: Atoms
            Atoms object to be attached

        kT: float. Default 1.
            Value of the thermal energy in eV units. It is important for
            some methods of plumed like Well-Tempered Metadynamics.

        log: string
            Log file of the plumed calculations

        restart: boolean. Default False
            True if the simulation is restarted.

        use_charge: boolean. Default False
            True if you use some collective variable which needs charges. If
            use_charges is True and update_charge is False, you have to define
            initial charges and then this charge will be used during all
            simulation.

        update_charge: boolean. Default False
            True if you want the charges to be updated each time step. This
            will fail in case that calc does not have 'charges' in its
            properties.


        .. note:: For this case, the calculator is defined strictly with the
            object atoms inside. This is necessary for initializing the
            Plumed object. For conserving ASE convention, it can be initialized
            as atoms.calc = (..., atoms=atoms, ...)


        .. note:: In order to guarantee a proper restart, the user has to fix
            momenta, positions and Plumed.istep, where the positions and
            momenta corresponds to the last configuration in the previous
            simulation, while Plumed.istep is the number of timesteps
            performed previously. This can be done using
            ase.calculators.plumed.restart_from_trajectory.
        """

        from plumed import Plumed as pl

        if atoms is None:
            raise TypeError('plumed calculator has to be defined with the \
                             object atoms inside.')

        self.istep = 0
        Calculator.__init__(self, atoms=atoms)

        self.input = input
        self.calc = calc
        self.use_charge = use_charge
        self.update_charge = update_charge

        if world.rank == 0:
            natoms = len(atoms.get_positions())
            self.plumed = pl()

            ''' Units setup
            warning: inputs and outputs of plumed will still be in
            plumed units.

            The change of Plumed units to ASE units is:
            kjoule/mol to eV
            nm to Angstrom
            ps to ASE time units
            ASE and plumed - charge unit is in e units
            ASE and plumed - mass unit is in a.m.u units '''

            ps = 1000 * fs
            self.plumed.cmd("setMDEnergyUnits", mol / kJ)
            self.plumed.cmd("setMDLengthUnits", 1 / nm)
            self.plumed.cmd("setMDTimeUnits", 1 / ps)
            self.plumed.cmd("setMDChargeUnits", 1.)
            self.plumed.cmd("setMDMassUnits", 1.)

            self.plumed.cmd("setNatoms", natoms)
            self.plumed.cmd("setMDEngine", "ASE")
            self.plumed.cmd("setLogFile", log)
            self.plumed.cmd("setTimestep", float(timestep))
            self.plumed.cmd("setRestart", restart)
            self.plumed.cmd("setKbT", float(kT))
            self.plumed.cmd("init")
            for line in input:
                self.plumed.cmd("readInputLine", line)
        self.atoms = atoms

    def _get_name(self):
        return f'{self.calc.name}+Plumed'

    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        comp = self.compute_energy_and_forces(self.atoms.get_positions(),
                                              self.istep)
        energy, forces = comp
        self.istep += 1
        self.results['energy'], self. results['forces'] = energy, forces

    def compute_energy_and_forces(self, pos, istep):
        unbiased_energy = self.calc.get_potential_energy(self.atoms)
        unbiased_forces = self.calc.get_forces(self.atoms)

        if world.rank == 0:
            ener_forc = self.compute_bias(pos, istep, unbiased_energy)
        else:
            ener_forc = None
        energy_bias, forces_bias = broadcast(ener_forc)
        energy = unbiased_energy + energy_bias
        forces = unbiased_forces + forces_bias
        
        return energy, forces

    def compute_bias(self, pos, istep, unbiased_energy):
        self.plumed.cmd("setStep", istep)

        if self.use_charge:
            if 'charges' in self.calc.implemented_properties and \
               self.update_charge:
                charges = self.calc.get_charges(atoms=self.atoms.copy())

            elif self.atoms.has('initial_charges') and not self.update_charge:
                charges = self.atoms.get_initial_charges()

            else:
                assert not self.update_charge, "Charges cannot be updated"
                assert self.update_charge, "Not initial charges in Atoms"

            self.plumed.cmd("setCharges", charges)

        # Box for functions with PBC in plumed
        if self.atoms.cell:
            cell = np.asarray(self.atoms.get_cell())
            self.plumed.cmd("setBox", cell)

        self.plumed.cmd("setPositions", pos)
        self.plumed.cmd("setEnergy", unbiased_energy)
        self.plumed.cmd("setMasses", self.atoms.get_masses())
        forces_bias = np.zeros((self.atoms.get_positions()).shape)
        self.plumed.cmd("setForces", forces_bias)
        virial = np.zeros((3, 3))
        self.plumed.cmd("setVirial", virial)
        self.plumed.cmd("prepareCalc")
        self.plumed.cmd("performCalc")
        energy_bias = np.zeros((1,))
        self.plumed.cmd("getBias", energy_bias)
        return [energy_bias, forces_bias]

    def write_plumed_files(self, images):
        """ This function computes what is required in
        plumed input for some trajectory.

        The outputs are saved in the typical files of
        plumed such as COLVAR, HILLS """
        for i, image in enumerate(images):
            pos = image.get_positions()
            self.compute_energy_and_forces(pos, i)
        return self.read_plumed_files()

    def read_plumed_files(self, file_name=None):
        read_files = {}
        if file_name is not None:
            read_files[file_name] = np.loadtxt(file_name, unpack=True)
        else:
            for line in self.input:
                if line.find('FILE') != -1:
                    ini = line.find('FILE')
                    end = line.find(' ', ini)
                    if end == -1:
                        file_name = line[ini + 5:]
                    else:
                        file_name = line[ini + 5:end]
                    read_files[file_name] = np.loadtxt(file_name, unpack=True)

            if len(read_files) == 0:
                if exists('COLVAR'):
                    read_files['COLVAR'] = np.loadtxt('COLVAR', unpack=True)
                if exists('HILLS'):
                    read_files['HILLS'] = np.loadtxt('HILLS', unpack=True)
        assert not len(read_files) == 0, "There are not files for reading"
        return read_files

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.plumed.finalize()
        
def restart_from_trajectory(prev_traj, *args, prev_steps=None, atoms=None,
                            **kwargs):
    """This function helps the user to restart a plumed simulation
    from a trajectory file.

    Parameters
    ----------
    prev_traj : Trajectory object
        previous simulated trajectory

    prev_steps : int. Default steps in prev_traj.
        number of previous steps

    others :
       Same parameters of :mod:`~ase.calculators.plumed` calculator

    Returns
    -------
    Plumed calculator


    .. note:: prev_steps is crucial when trajectory does not contain
        all the previous steps.
    """
    atoms.calc = Plumed(*args, atoms=atoms, restart=True, **kwargs)

    with Trajectory(prev_traj) as traj:
        if prev_steps is None:
            atoms.calc.istep = len(traj) - 1
        else:
            atoms.calc.istep = prev_steps
        atoms.set_positions(traj[-1].get_positions())
        atoms.set_momenta(traj[-1].get_momenta())
    return atoms.calc

# === PySCF Calculator Classes ===

def init_geo(mf, atoms):
    # convert ASE structural information to PySCF information
    if atoms.pbc.any():
        cell = mf.cell.copy()
        cell.atom = atoms_from_ase(atoms)
        cell.a = atoms.cell.copy()
        cell.build()
        mf.reset(cell=cell.copy())
    else:
        mol = mf.mol.copy()
        mol.atom = atoms_from_ase(atoms)
        mol.build()
        mf.reset(mol=mol.copy())

class PySCF_KSDFT(Calculator):
    """
    PySCF_KSDFT calculator is used for ab-initio simulations with 
    single-reference electronic structure such as KS-DFT HF, CISD, 
    CCSD and MP2 implemented in the open-source code PySCF (pyscf.org).
    
    Taken from Jakob Kraus https://github.com/pyscf/pyscf/issues/624
    
    
    Units & conventions
    -------------------
    Energies from PySCF are in Hartree; gradients are in Hartree/Bohr.
    This calculator converts to ASE units in `calculate()`:
      - energy -> eV (via `Ha`)
      - forces -> eV/Å (via `Ha / Bohr`)
    """

    implemented_properties = ['energy','forces']
    
    
    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='PySCF_KSDFT', atoms=None, directory='.', **kwargs):
        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, directory, **kwargs)
        self.initialize(**kwargs)


    def initialize(self, mf=None):
        self.mf = mf

    def set(self, **kwargs):
        changed_parameters = Calculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()
    
    def calculate(self,atoms=None,properties=['energy','forces'],system_changes=all_changes):
        
        Calculator.calculate(self,atoms=atoms,properties=properties,system_changes=system_changes)
        
        init_geo(self.mf, atoms)

        if hasattr(self.mf,'_scf'):
            self.mf._scf.kernel()
            self.mf.__init__(self.mf._scf)
        self.mf.kernel()
        e = self.mf.e_tot
        
        self.results['energy'] = e * Ha
        
        gf = self.mf.nuc_grad_method()
        gf.verbose = self.mf.verbose
        gf.grid_response = True
        forces = -1. * gf.kernel() * (Ha / Bohr)
        totalforces = []
        totalforces.extend(forces)
        totalforces = np.array(totalforces)
        self.results['forces'] = totalforces


class PySCF_MCPDFT(Calculator):
    """
    PySCF_MCDFT calculator is used for ab-initio simulations with 
    multi-reference electronic structure such as MC-PDFT, L-PDFT, 
    CASSCF and SA-CASSCF implemented in the open-source code PySCF (pyscf.org).
    
    [1] Li Manni et al., J. Chem. Theor. Comput., 10, 3669 (2014).
    [2] Sand et al., J. Chem. Theor. Comput., 14, 126 (2018).
    [3] Scott et al., J. Chem. Phys., 154, 074108 (2021).
    
    Units & conventions
    -------------------
    Energies from PySCF are in Hartree; gradients are in Hartree/Bohr.
    This calculator converts to ASE units in `calculate()`:
      - energy -> eV (via `Ha`)
      - forces -> eV/Å (via `Ha / Bohr`)
    """

    implemented_properties = ['energy', 'forces']
    
    def __init__(self, atoms, basis, ontop, n_act, n_elec, charge, spin):
        """
        Build an MC-PDFT calculator with a scaner object.

        Parameters
        ----------
        atoms : ase.Atoms
        
        basis : str
            Basis set for all atoms.
            
        ontop : str
            On-top functional name for MC-PDFT (e.g., "tPBE", "MC23").
            
        n_act : int
            Number of active orbitals in the CASSCF active space.
            
        n_elec : int
            Number of active electrons in the CASSCF active space
        
        charge : int
            Total molecular charge
        
        spin : int
            2S. For a singlet, use 0; for a doublet, 1 etc.

        Checkpointing
        -------------
        mcscf_pickle.chk (optional)
            If present, used to load previous MO coefficients
        """
        Calculator.__init__(self, atoms=atoms)
        self.atoms = atoms
        self.mol = gto.M(
            atom=atoms_from_ase(self.atoms),
            basis=basis,
            spin=spin,
            unit="Angstrom",
            symmetry=False,
            charge=charge,
        )

        hf = self.mol.HF().density_fit()
        hf.run()

        self.mc = mcpdft.CASSCF(hf, ontop, n_act, n_elec, grids_level=4)
        self.mc.max_cycle = 2000
        self.mc.conv_tol = 1e-7
        self.mc.conv_tol_grad = 1e-4
        self.add_chkfile(self.mc)
        self.mc.kernel()

        self.scanner = self.mc.nuc_grad_method().as_scanner()

    def set_atoms(self, atoms):
        if self.atoms != atoms:
            self.atoms = atoms.copy()
            self.results = {}

    def add_chkfile(self, mc):
        mc.chkfile = "mcscf.chk"
        if isfile("mcscf_pickle.chk"):
            mo = chkfile.load("mcscf_pickle.chk", "mcscf/mo_coeff")
            mo = mcscf.project_init_guess(mc, mo, prev_mol=None)
            mc.mo_coeff = mo

    def calculate(
            self,
            atoms=None, properties=['energy','forces'], system_changes=all_changes):

        Calculator.calculate(self,atoms=atoms,properties=properties,system_changes=system_changes)

        self.set_atoms(atoms)
        self.mol.set_geom_(atoms_from_ase(self.atoms), unit="Angstrom")
        self.mol.build()

        etot, grad = self.scanner(self.mol)

        if not self.scanner.converged:
            raise RuntimeError('Gradients did not converge!')

        self.results['energy'] = etot * Ha
        forces = -1. * grad * (Ha / Bohr)
        totalforces = []
        totalforces.extend(forces)
        totalforces = np.array(totalforces)
        self.results['forces'] = totalforces        

def main():
    print("Starting...")
    global atoms
    if args.method == 'ks-dft':
        mol = gto.M(atom=atoms_from_ase(atoms), 
                    basis=args.basis, 
                    charge=args.charge, 
                    spin=args.spin, 
                    unit='Angstrom')
        mol.build()
        mf = dft.RKS(mol).density_fit()
        mf.xc = args.xc
        mf.kernel()
        
        atoms.calc = Plumed(calc=PySCF_KSDFT(mf=mf), 
                            input=plumed_dat, timestep=timestep*ps, atoms=atoms, kT=kB*T)

    elif args.method == 'mc-pdft':
        atoms.calc = Plumed(calc=PySCF_MCPDFT(atoms, args.basis, args.ontop, args.n_act, args.n_elec, args.charge, args.spin),
                            input=plumed_dat, timestep=timestep*ps, atoms=atoms, kT=kB*T)
    
    if args.position_npy is not None:
        atoms.set_positions(np.load(args.position_npy))
    
    if args.momentum_npy is not None:
        atoms.set_momenta(np.load(args.momentum_npy))
    else
        MaxwellBoltzmannDistribution(atoms,temperature_K=T)
    
    dyn = NVTBerendsen(atoms, timestep=timestep*ps, temperature_K=T, taut=1*ps,
                       trajectory='md.traj', logfile='md.log')
    
    product_reg = -0.18 # Diels-Alder 
    #product_reg = 0.025 # SN2
    for step in range(steps):
        dyn.run(1)
        if step > 1 and step % 10 == 0:
            if check_cv_from_colvar('COLVAR', product_reg):
                dyn.run(100)
                print("Stopping simulation: cv exceeded threshold.")
                break

    print("Ending...")

if __name__ == '__main__':
    main()

