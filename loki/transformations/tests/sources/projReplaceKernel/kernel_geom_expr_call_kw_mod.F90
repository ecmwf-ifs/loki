MODULE KERNEL_GEOM_EXPR_CALL_KW_MOD
  USE GEOMETRY_MOD, ONLY: GEOMETRY_TYPE
  USE KERNEL_GEOM_EXPR_MOD, ONLY: KERNEL_GEOM_EXPR
  IMPLICIT NONE
CONTAINS

  SUBROUTINE kernel_geom_expr_call_kw(ydgeo)
    TYPE(GEOMETRY_TYPE), INTENT(IN) :: ydgeo
    INTEGER :: start_idx, end_idx
    LOGICAL :: flag_on

    CALL kernel_geom_expr(ydgeometry=ydgeo, kst=start_idx, kend=end_idx, flag1=flag_on)
  END SUBROUTINE kernel_geom_expr_call_kw

END MODULE KERNEL_GEOM_EXPR_CALL_KW_MOD
