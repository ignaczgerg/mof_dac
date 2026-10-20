#!/usr/bin/env python
# coding: utf-8

from typing import Any, Union, List, Dict, Optional, Tuple, Sequence
import subprocess
import json
import argparse
import pathlib
from pathlib import Path
from dscribe.descriptors import SOAP
import pymatgen
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifParser
import tempfile
import os
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
import warnings

parent_dir = os.path.dirname(os.path.abspath(__file__))

from contextlib import contextmanager
import sys
import numpy as np
import mendeleev
import shutil
import subprocess
import json
import signal


def parse_files_input(files: Sequence[Union[str, Path]], recursive: bool = False) -> List[Path]:
    """
    Parse a mixed list of files/directories into a de-duplicated list of CIF file paths.

    Supported inputs:
    - CIF files
    - Directories containing CIF files
    - Text files (.txt/.lst) containing one path per line
    """
    found: List[Path] = []

    for entry in files:
        path = Path(entry)

        if path.is_dir():
            iterator = path.rglob("*.cif") if recursive else path.glob("*.cif")
            found.extend(sorted(p for p in iterator if p.is_file()))
            continue

        if path.is_file() and path.suffix.lower() == ".cif":
            found.append(path)
            continue

        if path.is_file() and path.suffix.lower() in {".txt", ".lst"}:
            try:
                for raw in path.read_text().splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    listed = Path(line)
                    if listed.is_file() and listed.suffix.lower() == ".cif":
                        found.append(listed)
                    else:
                        warnings.warn(f"Skipping non-CIF or missing path listed in {path}: {line}")
            except Exception as exc:
                warnings.warn(f"Could not read list file {path}: {exc}")
            continue

        warnings.warn(f"Skipping unsupported or missing input path: {path}")

    unique: List[Path] = []
    seen = set()
    for p in found:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    return unique


class AtomicData:
    def __init__(self):
        self.elements = mendeleev.get_all_elements()
        self.ZToName = {el.atomic_number:el.symbol for el in self.elements}
        self.NameToZ = {el.symbol:el.atomic_number for el in self.elements}

# atomic_data = AtomicData()
atomic_data = AtomicData()

class JSONWithCommentsDecoder(json.JSONDecoder):
    """
    Custom JSON decoder to handle JSON files that include comments.
    Removes lines starting with '//' before parsing the JSON.
    """
    def __init__(self, **kw):
        super().__init__(**kw)

    def decode(self, s: str) -> Any:
        # Remove lines that are comments
        s = '\n'.join(l if not l.lstrip().startswith('//') else '' for l in s.split('\n'))
        return super().decode(s)

class TimeoutException(Exception):
    """Custom exception raised when the execution time exceeds the time limit."""
    pass

@contextmanager
def time_limit(seconds: int):
    """
    Context manager to limit the execution time of a block of code.
    
    This uses the SIGALRM signal to raise a `TimeoutException` if the execution
    of the block exceeds the specified time limit.

    Args:
        seconds (int): The time limit in seconds for the execution of the block.

    Raises:
        TimeoutException: If the code inside the block takes longer than the 
        specified time to execute.
    """
    
    def signal_handler(signum, frame):
        """Handles the SIGALRM signal by raising a TimeoutException."""
        raise TimeoutException("Timed out!")
    
    # Set the signal handler for the SIGALRM signal
    signal.signal(signal.SIGALRM, signal_handler)
    
    # Set an alarm to trigger after the specified number of seconds
    signal.alarm(seconds)
    
    try:
        # Execute the block of code within the time limit
        yield
    finally:
        # Disable the alarm once the block is complete (or if it times out)
        signal.alarm(0)

class NoOut:
    """
    Context manager that suppresses all standard output (stdout) and standard error (stderr).
    
    When used, this will redirect stdout and stderr to `/dev/null`, effectively 
    silencing any print statements or error messages during the block's execution.

    IMPORTANT: Warnings will also be suppressed. One should use this block only if warnings are correctly checked. 
               With Ipython/Jupyter kernels usage behavior may not be as expected.
               Is usefull for parallel calculations with warnings that could not be disabled.
    """
    
    def __enter__(self):
        """Redirects stdout and stderr to /dev/null to suppress output."""
        # Save the original stdout and stderr file descriptors
        self._stdout = os.dup(1)
        self._stderr = os.dup(2)
        
        # Open /dev/null for discarding output
        self._null = os.open(os.devnull, os.O_RDWR)
        
        # Redirect stdout and stderr to /dev/null
        os.dup2(self._null, 1)
        os.dup2(self._null, 2)

    def __exit__(self, *args):
        """Restores the original stdout and stderr after exiting the context."""
        # Restore the original stdout and stderr
        os.dup2(self._stdout, 1)
        os.dup2(self._stderr, 2)
        
        # Close the file descriptor for /dev/null
        os.close(self._null)

        # Also close the duplicated stdout and stderr file descriptors
        os.close(self._stdout)
        os.close(self._stderr)


### Species from calculated dataset
### Soap depends on number of species, so other species raise error in soap calculation
atoms_species_default = ['Ag', 'Al', 'As', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Br', 'C', 'Ca', 
                            'Cd', 'Ce', 'Cl', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Er', 'Eu', 'F',  
                            'Fe', 'Ga', 'Gd', 'Ge', 'H', 'Hf', 'Hg', 'Ho', 'I', 'In', 'Ir', 
                            'K', 'La', 'Li', 'Lu', 'Mg', 'Mn', 'Mo', 'N', 'Na', 'Nb', 'Nd',  
                            'Ni', 'Np', 'O', 'P', 'Pb', 'Pd', 'Pr', 'Pt', 'Pu', 'Rb', 'Re',  
                            'Rh', 'Ru', 'S', 'Sb', 'Sc', 'Se', 'Si', 'Sm', 'Sn', 'Sr', 'Tb',  
                            'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'V', 'W', 'Y', 'Yb', 'Zn', 'Zr']


class OutFileWithCheck:
    """
    Context manager to handle writing output to a file or standard output (stdout).
    
    This class checks if the output file already exists, optionally prevents overwriting, 
    and can redirect output to a file or the console (stdout).
    
    Args:
        outfile (Optional[Union[str, pathlib.Path]]): Path to the output file. 
            If None, output is directed to sys.stdout.
        rewrite (bool): If False and the file already exists, an exception is raised. 
            Defaults to False.
        add_time_mark (bool): Placeholder for a feature to add a timestamp to the output file. 
            Currently unused.
    """
    
    def __init__(self, outfile: Optional[Union[str, pathlib.Path]] = None, rewrite: bool = False, add_time_mark: bool = False):
        self.rewrite = rewrite
        self.outfile = outfile
        self.add_time_mark = add_time_mark  # Not implemented, but reserved for future use
    
    def __enter__(self):
        """
        Open the output file or redirect to stdout if outfile is None.
        Raises an exception if the file exists and `rewrite` is False.
        
        Returns:
            File object or sys.stdout.
        """
        if self.outfile is None:
            # If no outfile is specified, use stdout and do not close it after exit
            self.should_close = False
            self.out = sys.stdout
        else:
            # Check if the file already exists and whether we are allowed to overwrite it
            if not self.rewrite and os.path.exists(self.outfile):
                raise Exception(f"{self.outfile} already exists")
            
            # Open the file in write mode and mark it for closing after use
            self.should_close = True
            self.out = open(self.outfile, 'w')

        return self.out
    
    def __exit__(self, *args):
        """
        Close the file if it was opened, but leave stdout open.
        """
        if self.should_close:
            self.out.close()
    
    
class Converter:
                
    @staticmethod
    ### Model to convert VASP MOLDRAW FILES to CIF
    def mol_to_cif(inpfile: Union[str, pathlib.Path], outfile: Optional[Union[str, pathlib.Path]] = None, 
                rewrite: bool = False, charge_file: Optional[Union[str, pathlib.Path]] = None, datablock_name : str = 'image0') -> None:
        """
        Converts a molecular input file (MOLDRAW format) to a CIF (Crystallographic Information File) format.

        Args:
            inpfile (Union[str, pathlib.Path]): Path to the input molecular file.
            outfile (Optional[Union[str, pathlib.Path]]): Path to the output CIF file.
                If None, output is written to stdout.
            rewrite (bool): Whether to overwrite the output file if it already exists. Defaults to False.
            charge_file (Optional[Union[str, pathlib.Path]]): Path to a charge file. If provided,
                the charges will be included in the CIF output.

        Raises:
            Exception: If the output file already exists and `rewrite` is False.
        """
        
        with OutFileWithCheck(outfile=outfile, rewrite=rewrite) as fw:
            # Read the input molecular file
            with open(inpfile) as inp:
                lines = inp.readlines()
            
            # Read and process the charge file if provided
            if charge_file is not None:
                with open(charge_file) as f:
                    charges = [float(x.split()[1]) for x in f.readlines()[1:]]  # Skip header and parse charges
                charges = np.array(charges)
                charges = charges - charges.mean()  # Normalize charges by subtracting their mean
            
            # Extract cell dimensions from the input file (assumed to be on line 4)
            cell = [float(x) for x in lines[3].split()]
            
            # Find the start and end indices of atomic coordinate data
            start_end = [i for i, x in enumerate(lines) if x == 'COORD\n' or x == '-1 0 0 0\n']
            
            # Start writing CIF data
            print(f"data_{datablock_name}", file=fw)
            
            # Write cell dimensions and angles
            for value, name in zip(cell, ["_cell_length_a",
                                        "_cell_length_b",
                                        "_cell_length_c",
                                        "_cell_angle_alpha",
                                        "_cell_angle_beta",
                                        "_cell_angle_gamma"]):
                print(f"{name:20} {value:>12.5f}", file=fw)
            
            # Write symmetry information and atom data headers
            print(f'''
_symmetry_space_group_name_H-M    "P 1"
_symmetry_int_tables_number       1

loop_
    _symmetry_equiv_pos_as_xyz
    'x, y, z'

loop_
    _atom_site_type_symbol
    _atom_site_label
    _atom_site_fract_x
    _atom_site_fract_y
    _atom_site_fract_z
    _atom_site_occupancy
{"    _atom_site_charge" if charge_file is not None else ""}''', end="\n" if charge_file is not None else "", file=fw)
            
            # Write atomic coordinates and optional charges
            for idx, line in enumerate(lines[start_end[0]+1:start_end[1]]):
                atom_data = line.split()
                atom_type = atomic_data.ZToName[int(atom_data[0])]  # Get atom name from atomic number
                coords = [float(x) for x in atom_data[1:]]  # Extract fractional coordinates
                
                # Output with or without charges based on the presence of the charge file
                if charge_file is not None:
                    print(f"{atom_type:2}  {atom_type+str(idx+1):4} {coords[0]:12.8f}  {coords[1]:12.8f}  {coords[2]:12.8f}  1.000  {charges[idx]:15.10f}", file=fw)
                else:
                    print(f"{atom_type:2}  {atom_type+str(idx+1):4} {coords[0]:12.8f}  {coords[1]:12.8f}  {coords[2]:12.8f}  1.000", file=fw)

    @staticmethod
    def cif_to_cif(inpfile: Union[str, pathlib.Path], outfile: Optional[Union[str, pathlib.Path]], 
                rewrite: bool = False, verbose: bool = False, log_file: Optional[Union[str, pathlib.Path]] = None) -> None:
        """
        Converts a CIF file to a standardized CIF format using Open Babel (obabel).
        
        This is useful for converting non-standard CIF files to a format that other 
        modules or tools can read. The function uses the `obabel` command-line tool.
        
        Args:
            inpfile (Union[str, pathlib.Path]): The input CIF file path.
            outfile (Optional[Union[str, pathlib.Path]]): The output CIF file path. 
                If None, the output is written to stdout.
            rewrite (bool): If True, allows overwriting the output file if it exists. 
                Defaults to False.
            verbose (bool): If True, writes the obabel conversion logs to the console 
                or to a log file if `log_file` is specified. Defaults to False.
            log_file (Optional[Union[str, pathlib.Path]]): Path to a log file to write 
                verbose output. If None, logs are printed to stdout.
        
        Raises:
            Exception: If Open Babel (obabel) is not found or the conversion fails.
        """
        
        # Check if Open Babel (obabel) is installed
        path = shutil.which("obabel")
        if path is None:
            raise Exception("For CIF to CIF conversion, Open Babel (obabel) is required.")
        
        # Run the obabel command to convert CIF to CIF
        fbabel = subprocess.run(
            ["obabel", "-icif", "-ocif", str(inpfile)],
            capture_output=True
        )
        
        # Decode stdout and stderr from the subprocess run
        stdout, stderr = fbabel.stdout.decode('utf-8'), fbabel.stderr.decode('utf-8')
        
        # If verbose is True, output the stderr log (conversion log)
        if verbose:
            with OutFileWithCheck(log_file, rewrite=True) as logf:
                print(f"Open Babel conversion {inpfile} to {outfile}", file=logf) 
                print(stderr, file=logf)  # Write verbose logs to the log file or stdout
                print("-"*80, file=logf)
        
        # Check if the conversion was successful by searching for "1 molecule converted"
        if "1 molecule converted" not in stderr:
            raise Exception(f"Conversion failed: Open Babel did not convert the CIF file. Error:\n{stderr}")
        
        # Write the output to the specified file or stdout
        with OutFileWithCheck(outfile=outfile, rewrite=rewrite) as out:
            print(stdout, file=out)


class SoapCalculator:
    """
    A class to handle SOAP descriptor calculations for CIF files based on provided JSON configuration.

    Args:
        json_config (Union[str, Path]): Path to a JSON file containing SOAP configuration settings.
        properties_json (Union[str, Path]): Path to a JSON file containing atomic properties.
    """
    def __init__(self, json_config: Union[str, Path] = f"{parent_dir}/soap_config.json",
                 properties_json: Union[str, Path] = f"{parent_dir}/species_default_properties.json"):
        # Load settings from JSON configuration
        with open(json_config) as settings_file:
            self.settings = json.load(settings_file, cls=JSONWithCommentsDecoder)    

        # Load atomic properties from JSON
        with open(properties_json) as prop_file:
            self.properties = json.load(prop_file, cls=JSONWithCommentsDecoder)

        self.soaps = []
        try:
            # Initialize SOAP descriptors using the settings and atomic properties
            for setting in self.settings:
                setting['compression']["species_weighting"] = self.properties.get(setting['compression']["species_weighting"], None)
                self.soaps.append(SOAP(species=atoms_species_default, **setting))
        except Exception as e:
            raise Exception(f"SOAP initialization from JSON settings failed: {e}")
    
    def single_cif(self, filename: Union[str, Path], timelimit: int = 240, primitive_pymatgen_parser: bool = True, 
                   parent_name_use: bool = True, obabel_preconvertion: bool = True) -> Tuple[str, Union[np.ndarray, None]]:
        """
        Calculates SOAP descriptors for a single CIF file.

        Args:
            filename (Union[str, Path]): Path to the CIF file.
            timelimit (int): Maximum time allowed for the calculation. Defaults to 240 seconds.
            primitive_pymatgen_parser (bool): If True, use primitive parsing in Pymatgen. Defaults to True.
            parent_name_use (bool): If True, includes parent directory in the output name. Defaults to True.
            obabel_preconvertion (bool): If True, converts CIF using Open Babel before processing. Defaults to True.
        
        Returns:
            Tuple[str, Union[np.ndarray, None]]: A tuple where the 1st value is the output file name and the 2nd value is 
            the concatenated SOAP descriptors as a numpy array, or None in case of failure.
        """
        filename = Path(filename)
        # Determine the output name based on `parent_name_use`
        outname = f"{filename.parent.name}/{filename.name}" if parent_name_use and filename.parent.name!='' else f"{filename.name}"

        # Temporary file for Open Babel pre-conversion if enabled
        temp_name = None

        try:
            # Convert CIF using Open Babel if necessary
            if obabel_preconvertion:
                temp_name = next(tempfile._get_candidate_names()) + ".cif"
                Converter.cif_to_cif(filename, temp_name)
                filename = Path(temp_name)

            # Suppress output during the calculation
            with NoOut():
                try:
                    # Limit the time for SOAP calculation
                    with time_limit(timelimit):
                        soap_desc = []
                        
                        # Parse the CIF file using Pymatgen
                        cif_parser = CifParser(str(filename))
                        pymatgen_structure = cif_parser.parse_structures(primitive=primitive_pymatgen_parser)[0]
                        
                        # Convert the Pymatgen structure to ASE Atoms format
                        structure = AseAtomsAdaptor.get_atoms(pymatgen_structure)
                        
                        # Calculate SOAP descriptors for the structure using all initialized SOAPs
                        for soap in self.soaps:
                            soap_desc.append(soap.create(structure, n_jobs=1))

                        # Concatenate all SOAP descriptors and return in a dictionary
                        return (outname, np.hstack(soap_desc))
                    
                except Exception as e:
                    # Log the error or track for debugging if needed
                    print(f"{outname}. Error during SOAP calculation: {e}")
                    return (outname, None)  # Return None if SOAP calculation fails

        except Exception as e:
            # Handle any errors during the Open Babel conversion or SOAP calculation
            print(f"{outname}. Error during pre-conversion or calculation: {e}")
            return (outname, None)

        finally:
            # Clean up the temporary file if it was created
            if temp_name and os.path.exists(temp_name):
                try:
                    os.remove(temp_name)
                except Exception as cleanup_error:
                    print(f"{outname}. Error cleaning up temporary file: {cleanup_error}")


    def single_cif_spec_case(self, cif_content: str, timelimit: int = 240, 
                             primitive_pymatgen_parser: bool = True) -> Tuple[str, Union[np.ndarray, None]]:
        """
        Calculates SOAP descriptors for a single CIF file as a string content.

        Args:
            cif_content (str): String with CIF file content.
            timelimit (int): Maximum time allowed for the calculation. Defaults to 240 seconds.
            primitive_pymatgen_parser (bool): If True, use primitive parsing in Pymatgen. Defaults to True.
        
        Returns:
            Tuple[str, Union[np.ndarray, None]]: A tuple where the 1st value is the output file name and the 2nd value is 
            the concatenated SOAP descriptors as a numpy array, or None in case of failure.
        """        
        outname = "in_memory_cif"
        with NoOut():
            try:
                # Limit the time for SOAP calculation
                with time_limit(timelimit):
                    soap_desc = []
                    
                    # Parse the CIF file using Pymatgen
                    cif_parser = CifParser.from_str(cif_content)
                    pymatgen_structure = cif_parser.parse_structures(primitive=primitive_pymatgen_parser)[0]
                    
                    # Convert the Pymatgen structure to ASE Atoms format
                    structure = AseAtomsAdaptor.get_atoms(pymatgen_structure)
                    
                    # Calculate SOAP descriptors for the structure using all initialized SOAPs
                    for soap in self.soaps:
                        soap_desc.append(soap.create(structure, n_jobs=1))

                    # Concatenate all SOAP descriptors and return in a dictionary
                    return (outname, np.hstack(soap_desc))
                
            except Exception as e:
                # Log the error or track for debugging if needed
                print(f"{outname}. Error during SOAP calculation: {e}")
                return (outname, None)  # Return None if SOAP calculation fails
            
            
        

    #TODO: add an option make batch by batch executation to store and update result from tmp file
    #      during long executation executor could fail because of external troubleshooting.
    def multiple_cifs(self, files: List[Union[str, Path]], content_stings: bool = False, timelimit: int = 240, 
                      primitive_pymatgen_parser: bool = True, n_jobs: int = -1, joblib_verbosity: int = 7,
                      parent_name_use: Optional[bool] = None, obabel_preconvertion: bool = True, 
                      recursive: bool = False) -> Tuple[pd.DataFrame, List[Union[str, np.ndarray, None]]]:
        """
        Calculates SOAP descriptors for multiple CIF files in either serial or parallel mode.

        Args:
            files (List[str]): A list of file paths or directories to search for CIF files.
                Accepts individual file paths, directories, and text files containing lists of file paths.
            content_stings (bool) : Do file strings already present CIF content. Defaults to False.
            timelimit (int): Maximum time allowed for each calculation. Defaults to 240 seconds.
            primitive_pymatgen_parser (bool): Whether to use primitive parsing in Pymatgen. Defaults to True.
            n_jobs (int): Number of parallel jobs to run. Defaults to -1 (use all available cores).
            joblib_verbosity (int): Verbosity level for Joblib parallel execution. Defaults to 7.
            parent_name_use (Optional[bool]): If True, includes parent directory in the output name.
                If None, it's True for recursive searches, False otherwise. Defaults to None.
            obabel_preconvertion (bool): If True, converts CIF using Open Babel before processing. Defaults to True.
            recursive (bool): If True, searches directories recursively. Defaults to False.

        Returns:
            pd.DataFrame: A DataFrame containing results, with the output file names as columns and SOAP descriptors.
        """
        if content_stings:
            inputs = files
            if n_jobs == 1 or len(inputs) == 1:
                results = [self.single_cif_spec_case(inputfile, timelimit=timelimit, 
                                           primitive_pymatgen_parser=primitive_pymatgen_parser) 
                           for inputfile in inputs]
            else:
                # Parallel execution using Joblib
                results = Parallel(n_jobs=n_jobs, verbose=joblib_verbosity)(
                    delayed(self.single_cif_spec_case)(inputfile, timelimit=timelimit, 
                                             primitive_pymatgen_parser=primitive_pymatgen_parser)
                    for inputfile in inputs
                )
            # Convert the results into a DataFrame and return it
            return pd.DataFrame({x[0]:x[1] for x in results if x[1] is not None}).T, [x[1] for x in results if x[1] is None] 
            
        else:
            # Set parent_name_use based on whether recursion is used
            if parent_name_use is None:
                parent_name_use = True if recursive else False
    
            # Parse the input files or directories
            inputs = parse_files_input(files, recursive=recursive)
            # print(inputs)
            # Handle serial or parallel processing based on n_jobs
            if n_jobs == 1 or len(inputs) == 1:
                # Serial execution
                results = [self.single_cif(inputfile, timelimit=timelimit, 
                                           primitive_pymatgen_parser=primitive_pymatgen_parser,
                                           parent_name_use=parent_name_use, 
                                           obabel_preconvertion=obabel_preconvertion) 
                           for inputfile in inputs]
            else:
                # Parallel execution using Joblib
                results = Parallel(n_jobs=n_jobs, verbose=joblib_verbosity)(
                    delayed(self.single_cif)(inputfile, timelimit=timelimit, 
                                             primitive_pymatgen_parser=primitive_pymatgen_parser,
                                             parent_name_use=parent_name_use, 
                                             obabel_preconvertion=obabel_preconvertion)
                    for inputfile in inputs
                )
    
            # Convert the results into a DataFrame and return it
            return pd.DataFrame({x[0]:x[1] for x in results if x[1] is not None}).T, [x[0] for x in results if x[1] is None]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process CIF files and export SOAP descriptors to a CSV file."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing CIF files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV file path for SOAP descriptors.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for CIF files under input directory.",
    )
    parser.add_argument(
        "--json-config",
        type=Path,
        default=Path(f"{parent_dir}/soap_config.json"),
        help="Path to SOAP config JSON.",
    )
    parser.add_argument(
        "--properties-json",
        type=Path,
        default=Path(f"{parent_dir}/species_default_properties.json"),
        help="Path to species properties JSON.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel workers (joblib).",
    )
    parser.add_argument(
        "--timelimit",
        type=int,
        default=240,
        help="Per-file timeout in seconds.",
    )
    parser.add_argument(
        "--no-obabel-preconversion",
        action="store_true",
        help="Disable Open Babel CIF pre-conversion step.",
    )

    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_csv = args.output_csv.resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {input_dir}")

    calculator = SoapCalculator(
        json_config=args.json_config,
        properties_json=args.properties_json,
    )

    descriptors_df, failed_files = calculator.multiple_cifs(
        files=[str(input_dir)],
        timelimit=args.timelimit,
        n_jobs=args.n_jobs,
        recursive=args.recursive,
        parent_name_use=True,
        obabel_preconvertion=not args.no_obabel_preconversion,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    descriptors_df.to_csv(output_csv, index=True)

    print(f"Processed {len(descriptors_df)} CIF files successfully.")
    print(f"Saved SOAP descriptors to: {output_csv}")
    if failed_files:
        print(f"Failed to process {len(failed_files)} files.")


if __name__ == "__main__":
    main()
