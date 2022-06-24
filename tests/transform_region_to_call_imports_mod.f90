MODULE transform_region_to_call_imports_mod
  IMPLICIT NONE
CONTAINS
  SUBROUTINE transform_region_to_call_imports (a, b)
    USE region_to_call_mod, ONLY: param, arr1, arr2
    INTEGER, INTENT(OUT) :: a(10), b(10)
    INTEGER :: j
    
!$loki region-to-call
    DO j=1,10
      a(j) = param
    END DO
!$loki end region-to-call
    
!$loki region-to-call
    DO j=1,10
      arr1(j) = j + 1
    END DO
!$loki end region-to-call
    
    arr2(:) = arr1(:)
    
!$loki region-to-call
    DO j=1,10
      b(j) = arr2(j) - a(j)
    END DO
!$loki end region-to-call
  END SUBROUTINE transform_region_to_call_imports
END MODULE transform_region_to_call_imports_mod
