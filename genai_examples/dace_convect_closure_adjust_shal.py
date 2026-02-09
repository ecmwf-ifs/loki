import dace
from MODD_CONVPAREXT import JCVEXB, JCVEXT
import numpy as np

@dace.program
def convect_closure_adjust_shal(KLON: int, KLEV: int, PADJ: np.ndarray, PUMF: np.ndarray, PZUMF: np.ndarray, PUER: np.ndarray, PZUER: np.ndarray, PUDR: np.ndarray, PZUDR: np.ndarray) -> None:
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
   for JK in dace.map[1 + JCVEXB: KLEV - JCVEXT]:
       with dace.tasklet:
           a = PADJ[:]
           b, c, d = PZUMF[:, JK], PZUER[:, JK], PZUDR[:, JK]
           """
           Adjusting mass flux, updraft entrainment and detrainment
           """
           PUMF[::, JK] <<= b * a
           PUER[::, JK] = c * a
           PUDR[::, JK] = d * a