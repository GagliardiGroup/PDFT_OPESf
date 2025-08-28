import numpy as np

def read_plumed_to_setup(filepath: str):
    """
    Read a PLUMED input file and return a list of one-line actions suitable
    for the ASE-PLUMED wrapper. Supports multi-line blocks delimited by ... / ... NAME.
    Comments (#) and blank lines are skipped.
    """
    lines = []
    with open(filepath, "r") as f:
        in_block = False
        buf = []
        for raw in f:
            s = raw.strip()
            if not s or s.startswith('#'):
                continue
            if s.endswith('...'):
                in_block = True
                buf.append(s[:-3].strip())
                continue
            if in_block:
                if s.startswith('...'):
                    in_block = False
                    lines.append(' '.join(buf))
                    buf = []
                else:
                    buf.append(s)
            else:
                lines.append(s)
    if buf:
        # Unclosed block fallback
        lines.append(' '.join(buf))
    return lines

def check_cv_from_colvar(filename, threshold=0.1):
    """
    Reads the COLVAR file and checks the last value of cv.

    Parameters:
    - filename: str, path to the COLVAR file.
    - threshold: float, the threshold value to stop the simulation.

    Returns:
    - bool: True if cv exceeds the threshold, False otherwise.
    """
    try:
        data = np.loadtxt(filename, comments="#")
        if data.size == 0:
            return False

        # Extract the last value of c1
        c1_value = data[-1, 3]  # assuming c1 is the 4th column (0-indexed)

        return c1_value > threshold
    except Exception as e:
        print(f"Error reading COLVAR file: {e}")
        return False

