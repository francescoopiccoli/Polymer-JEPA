from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

def get_monomerA_fingerprint(smiles):
    # Convert SMILES to RDKit Mol and get Morgan fingerprint
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def get_most_similar_monomerA(test_monomerA, train_monomerAs):
    test_fp = get_monomerA_fingerprint(test_monomerA)
    max_sim = -1
    most_similar = None
    for mA in train_monomerAs:
        fp = get_monomerA_fingerprint(mA)
        if fp is None or test_fp is None:
            continue
        sim = DataStructs.TanimotoSimilarity(test_fp, fp)
        if sim > max_sim:
            max_sim = sim
            most_similar = mA
    return most_similar