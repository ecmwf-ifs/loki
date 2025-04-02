from loki.backend.fgen import fgen
from loki.transformations.genai.fortran2python import Fortran2PythonTransformation
import pytest

from loki import Subroutine, Module
from loki.ir import nodes as ir, FindNodes

def test_fortran2python_transformation():
    
    fortran_code = """
subroutine convect_satmixratio(xalpw, xbetaw, xgamw, &
    &xlvtt, xlstt, xcpv, xcpd, xci, xcl, &
    &ppres, pt, peps, pew, plv, pls, pcph)
    implicit none
    !
    !*       0.1   declarations of dummy arguments :
    !
    !
    real, intent(in) :: ppres   ! pressure
    real, intent(in) :: pt      ! temperature   
    real, intent(in) :: peps    ! xrd / xrv (ideally pre-computed in )
    !
    real, intent(out):: pew     ! vapor saturation mixing ratio
    real, intent(out):: plv     ! latent heat l_v    
    real, intent(out):: pls     ! latent heat l_s  
    real, intent(out):: pcph    ! specific heat c_ph   

    real :: zt      ! temperature   


    zt     = min( 400., max( pt, 10. ) ) ! overflow bound
    pew    = exp( xalpw - xbetaw / zt - xgamw * log( zt ) )
    pew    = peps * pew / ( ppres - pew )
    plv    = xlvtt + ( xcpv - xcl ) * ( zt - xtt ) ! compute l_v
    pls    = xlstt + ( xcpv - xci ) * ( zt - xtt ) ! compute l_i
    pcph   = xcpd + xcpv * pew
end subroutine convect_satmixratio
    """
      
    fsub = Subroutine.from_source(fortran_code, preprocess=True)
    F2PyT = Fortran2PythonTransformation()
    gen_python_code = F2PyT.transform_subroutine(fsub)
    
    print(gen_python_code)
    
    assert False
    