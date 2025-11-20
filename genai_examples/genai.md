# Using Codestral as a code assistant

## Installation

A local instance of [codestral:22b](https://mistral.ai/news/codestral) has been installed on an EWC instance. 

Codestral is a 22B parameters LLM (Large Language Model) from [Mistral](https://mistral.ai) trained for coding tasks.

Here, Codestral is exposed through [ollama](ollama.ai) inference server for LLMs. Ollama exposes a CLI and a localhost API endpoint to query LLMs.

 ```bash
# query with CLI
ollama run codestral:22b "your-prompt" 
 ```

- VSCode integration :

We've made codestral available in VSCode with [Continue.dev](https://docs.continue.dev/) plugin. Continue.dev provides a chat in VSCode + commands for autocompletion. It's used with a local config 

```json
{
  "models": [
    {
      "title": "Codestral",
      "provider": "ollama",
      "model": "codestral:22b"
      }
  ],
}
```

- WIP -> API Endpoint :

The local installation is only accessible vi ssh, especially vscode remote ssh server. It's intended to expose a secure API Endpoint.


## Loki integration 

3 loki transformations has been created. 

- PythonGenAITransformation : transforming fortran to python
- ACCGenAITransformation : inserting OpenACC directives in a python script
- DaceGenAITransfomration : generating a Dace script from a python script (intended to be chained with the python transformation).

A generic transformation is also available : 

- PromptedGenAITransformation : it combines a command (ex : "Transform this routine to python" and a stringified fortran routine fgen(routine))

## Examples

### Raw Fortran Subroutine

```fortran 
!     ######spl
     SUBROUTINE CONVECT_CLOSURE_ADJUST_SHAL( CVPEXT, D, PADJ,        &
                                             PUMF, PZUMF, PUER, PZUER, PUDR, PZUDR  )
     USE YOMHOOK , ONLY : LHOOK, DR_HOOK, JPHOOK
!
USE MODD_CONVPAREXT, ONLY : CONVPAREXT
USE MODD_DIMPHYEX, ONLY: DIMPHYEX_T
!
IMPLICIT NONE
!
TYPE(CONVPAREXT),           INTENT(IN) :: CVPEXT
TYPE(DIMPHYEX_T),           INTENT(IN) :: D
REAL, DIMENSION(D%NIT),      INTENT(IN) :: PADJ     ! mass adjustment factor
!
!
REAL, DIMENSION(D%NIT,D%NKT), INTENT(INOUT) :: PUMF  ! updraft mass flux (kg/s)
REAL, DIMENSION(D%NIT,D%NKT), INTENT(INOUT) :: PZUMF ! initial value of  "
REAL, DIMENSION(D%NIT,D%NKT), INTENT(INOUT) :: PUER  ! updraft entrainment (kg/s)
REAL, DIMENSION(D%NIT,D%NKT), INTENT(INOUT) :: PZUER ! initial value of  "
REAL, DIMENSION(D%NIT,D%NKT), INTENT(INOUT) :: PUDR  ! updraft detrainment (kg/s)
REAL, DIMENSION(D%NIT,D%NKT), INTENT(INOUT) :: PZUDR ! initial value of  "
!
INTEGER :: IKB, IKE                 ! vert. loop bounds
INTEGER :: JK, JI                   ! vertical loop index
!
REAL(KIND=JPHOOK) :: ZHOOK_HANDLE
IF (LHOOK) CALL DR_HOOK('CONVECT_CLOSURE_ADJUST_SHAL',0,ZHOOK_HANDLE)
IKB  = 1 + CVPEXT%JCVEXB
IKE  = D%NKT - CVPEXT%JCVEXT
!
!
!*       1.     Adjust mass flux by the factor PADJ to converge to
!               specified degree of stabilization
!               ----------------------------------------------------
!
DO JK = IKB + 1, IKE
  DO JI = D%NIB, D%NIE
     PUMF(JI,JK)  = PZUMF(JI,JK)   * PADJ(JI)
     PUER(JI,JK)  = PZUER(JI,JK)   * PADJ(JI)
     PUDR(JI,JK)  = PZUDR(JI,JK)   * PADJ(JI)
   ENDDO
END DO
!
END SUBROUTINE CONVECT_CLOSURE_ADJUST_SHAL
```

### Fortran to Python transformation

- Prompt : "Translate this Fortran routine in python"

```python
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

```

### Python -> Dace

- Prompt : "Optimize this routine with dace" + routine python précédement générée.

```python 
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
```

### Insertion de directives OpenACC

- Prompt : "Add OpenACC directives to the previous routine"

```fortran
SUBROUTINE CONVECT_CLOSURE_ADJUST_SHAL(KLON, KLEV, PADJ, PUMF, PZUMF, PUER, PZUER, PUDR, PZUDR)
USE PARKIND1, ONLY: JPRB
! ... (other declarations and initializations)

!$acc parallel loop collapse(2) private(JK, i_PUMF_0) present(KLON, KLEV, PADJ, PZUMF, PZUER, PZUDR, PUMF, PUER, PUDR)
DO JK=1 + JCVEXB + 1, KLEV - JCVEXT
    DO i_PUMF_0=1, KLON
        !$acc loop vector
        PUMF(i_PUMF_0, JK) = PZUMF(i_PUMF_0, JK)*PADJ(i_PUMF_0)
        PUER(i_PUMF_0, JK) = PZUER(i_PUMF_0, JK)*PADJ(i_PUMF_0)
        PUDR(i_PUMF_0, JK) = PZUDR(i_PUMF_0, JK)*PADJ(i_PUMF_0)
    END DO
END DO
!$acc end parallel loop

END SUBROUTINE CONVECT_CLOSURE_ADJUST_SHAL
```

