import pandas as pd
import numpy as np

try: 
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem, RDKFingerprint
except:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rdkit-pypi"])
    
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys, AllChem, RDKFingerprint


def Smiles2Fing(smiles, fingerprint_type='MACCS'):
    ms_tmp = [Chem.MolFromSmiles(i) for i in smiles]
    ms_none_idx = [i for i in range(len(ms_tmp)) if ms_tmp[i] is None]
    
    ms = list(filter(None, ms_tmp))
    
    if fingerprint_type == 'MACCS':
        fingerprints = [np.array(MACCSkeys.GenMACCSKeys(i), dtype=int) for i in ms]
    elif fingerprint_type == 'Morgan':
        fingerprints = [np.array(AllChem.GetMorganFingerprintAsBitVect(i, 2, nBits=1024), dtype=int) for i in ms]
    elif fingerprint_type == 'RDKit':
        fingerprints = [np.array(RDKFingerprint(i), dtype=int) for i in ms]
    elif fingerprint_type == 'Layered':
        fingerprints = [np.array(AllChem.LayeredFingerprint(i), dtype=int) for i in ms]
    elif fingerprint_type == 'Pattern':
        fingerprints = [np.array(AllChem.PatternFingerprint(i), dtype=int) for i in ms]
    else:
        raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}")
    
    fingerprints_df = pd.DataFrame(fingerprints)
    
    # 컬럼명 생성 (예: maccs_1, maccs_2, ..., maccs_167)
    colname = [f'{fingerprint_type.lower()}_{i+1}' for i in range(fingerprints_df.shape[1])]
    fingerprints_df.columns = colname
    fingerprints_df = fingerprints_df.reset_index(drop=True)
    
    return ms_none_idx, fingerprints_df
