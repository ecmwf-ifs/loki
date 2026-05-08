MODULE KERNEL_GEOM_EXPR_CALL_MOD
  USE GEOMETRY_MOD, ONLY: GEOMETRY_TYPE
  USE KERNEL_GEOM_EXPR_MOD, ONLY: KERNEL_GEOM_EXPR
  IMPLICIT NONE
CONTAINS

  SUBROUTINE kernel_geom_expr_call(ydgeometry)
    TYPE(GEOMETRY_TYPE), INTENT(IN) :: ydgeometry
    INTEGER :: kst, kend
    LOGICAL :: flag1

    CALL kernel_geom_expr(ydgeometry, kst, kend, flag1)
  END SUBROUTINE kernel_geom_expr_call

END MODULE KERNEL_GEOM_EXPR_CALL_MOD
