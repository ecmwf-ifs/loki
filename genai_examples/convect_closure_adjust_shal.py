from MODD_CONVPAREXT import JCVEXB, JCVEXT

def convect_closure_adjust_shal(KLON, KLEV, PADJ, PUMF, PZUMF, PUER, PZUER, PUDR, PZUDR):
    """
    This routine adjusts the mass flux using the factor PADJ computed in CONVECT_CLOSURE.
    The computations are done at every model level starting from bottom.

    Args:
        KLON (int): horizontal dimension
        KLEV (int): vertical dimension
        PADJ (numpy array): mass adjustment factor
        PUMF (numpy array): updraft mass flux (kg/s)
        PZUMF (numpy array): initial value of PUMF
        PUER (numpy array): updraft entrainment (kg/s)
        PZUER (numpy array): initial value of PUER
        PUDR (numpy array): updraft detrainment (kg/s)
        PZUDR (numpy array): initial value of PUDR
    """

    for JK in range(1 + JCVEXB, KLEV - JCVEXT):
        PUMF[:, JK] = PZUMF[:, JK] * PADJ[:]
        PUER[:, JK] = PZUER[:, JK] * PADJ[:]
        PUDR[:, JK] = PZUDR[:, JK] * PADJ[:]
