module loki_debug

  use PARKIND1, only: JPIM, JPRB
  use YOMPHYDER, only: STATE_TYPE
  use YOECLDP, only: TECLDP

  implicit none

  type Variable
     character(len=:), allocatable :: name

     contains
       procedure :: write => write_variable
  end type Variable

  type, extends(Variable) :: Scalar
     class(*), pointer :: ptr

     contains
       procedure :: write => write_scalar
  end type Scalar

  type, extends(Variable) :: Array1D
     class(*), pointer :: ptr(:)

     contains
       procedure :: write => write_array_1d
  end type Array1D

  type, extends(Variable) :: Array2D
     class(*), pointer :: ptr(:,:)

     contains
       procedure :: write => write_array_2d
  end type Array2D

  type, extends(Variable) :: Array3D
     class(*), pointer :: ptr(:,:,:)

     contains
       procedure :: write => write_array_3d
  end type Array3D

  ! A wrapper type to allow allocation/assignment of variable arrays
  type Item
     class(Variable), pointer :: v
  end type Item

  ! Derived types with a list of component "items"
  type, extends(Variable) :: DerivedType
     class(*), pointer :: ptr
     class(Item), allocatable :: items(:)

     contains
       procedure :: write => write_derived_type
  end type DerivedType

  interface V
     module procedure scalar_int, scalar_real, scalar_logical
     module procedure array_real_1d, array_int_1d, array_logical_1d
     module procedure array_real_2d, array_real_3d, dt_state_type, dt_tecldp
  end interface V

  type StateDump
     ! A program/subroutine state containing variable items.
     !
     ! This container class allows repeated, indexed state dumps
     ! to record a progression or time series of variable states.
     class(Item), pointer :: items(:)
     character(len=:), allocatable :: fname_data, fname_info
     integer :: f_data = 666, f_info = 667

   contains
     procedure :: open => state_open
     procedure :: dump => state_dump_indexed
     procedure :: close => state_close
  end type StateDump

contains

  function scalar_int(i, name)
    type(Scalar) :: scalar_int
    integer(kind=JPIM), intent(in), target :: i
    character(len=*), optional :: name

    scalar_int%ptr => i

    if (present(name)) then
       scalar_int%name = name
    end if
  end function scalar_int

  function scalar_real(r, name)
    type(Scalar) :: scalar_real
    real(kind=JPRB), intent(in), target :: r
    character(len=*), optional :: name

    scalar_real%ptr => r

    if (present(name)) then
       scalar_real%name = name
    end if
  end function scalar_real

  function scalar_logical(i, name)
    type(Scalar) :: scalar_logical
    logical, intent(in), target :: i
    character(len=*), optional :: name

    scalar_logical%ptr => i

    if (present(name)) then
       scalar_logical%name = name
    end if
  end function scalar_logical

  function array_real_1d(arr, name)
    type(Array1D) :: array_real_1d
    real(kind=JPRB), dimension(:), target, intent(in) :: arr
    character(len=*), optional :: name

    array_real_1d%ptr => arr

    if (present(name)) then
       array_real_1d%name = name
    end if
  end function array_real_1d

  function array_int_1d(arr, name)
    type(Array1D) :: array_int_1d
    integer(kind=JPIM), dimension(:), target, intent(in) :: arr
    character(len=*), optional :: name

    array_int_1d%ptr => arr

    if (present(name)) then
       array_int_1d%name = name
    end if
  end function array_int_1d

  function array_logical_1d(arr, name)
    type(Array1D) :: array_logical_1d
    logical, dimension(:), target, intent(in) :: arr
    character(len=*), optional :: name

    array_logical_1d%ptr => arr

    if (present(name)) then
       array_logical_1d%name = name
    end if
  end function array_logical_1d

  function array_real_2d(arr, name)
    type(Array2D) :: array_real_2d
    real(kind=JPRB), dimension(:,:), target, intent(in) :: arr
    character(len=*), optional :: name

    array_real_2d%ptr => arr

    if (present(name)) then
       array_real_2d%name = name
    end if
  end function array_real_2d

  function array_real_3d(arr, name)
    type(Array3D) :: array_real_3d
    real(kind=JPRB), dimension(:,:,:), target, intent(in) :: arr
    character(len=*), optional :: name

    array_real_3d%ptr => arr

    if (present(name)) then
       array_real_3d%name = name
    end if
  end function array_real_3d

  function dt_state_type(state, name) result(derived)
    ! Constructor for derived dype STATE_TYPE
    type(DerivedType) :: derived
    type(STATE_TYPE), target, intent(in) :: state
    character(len=*), optional :: name

    derived%ptr => state
    allocate(derived%items(7))
    allocate(derived%items(1)%v, source=V(state%u, name=trim(name)//'%u'))
    allocate(derived%items(2)%v, source=V(state%v, name=trim(name)//'%v'))
    allocate(derived%items(3)%v, source=V(state%T, name=trim(name)//'%T'))
    allocate(derived%items(4)%v, source=V(state%o3, name=trim(name)//'%o3'))
    allocate(derived%items(5)%v, source=V(state%q, name=trim(name)//'%q'))
    allocate(derived%items(6)%v, source=V(state%a, name=trim(name)//'%a'))
    allocate(derived%items(7)%v, source=V(state%cld, name=trim(name)//'%cld'))

    if (present(name)) then
       derived%name = name
    end if
  end function dt_state_type

  function dt_tecldp(yrecldp, name) result(derived)
    ! Constructor for derived dype TECLDP
    type(DerivedType) :: derived
    type(TECLDP), target, intent(in) :: yrecldp
    character(len=*), optional :: name

    derived%ptr => yrecldp
    allocate(derived%items(124))
    allocate(derived%items(1)%v, source=V(yrecldp%LAERICEAUTO, name=trim(name)//'%LAERICEAUTO'))
    allocate(derived%items(2)%v, source=V(yrecldp%LAERICESED, name=trim(name)//'%LAERICESED'))
    allocate(derived%items(3)%v, source=V(yrecldp%LAERLIQAUTOCP, name=trim(name)//'%LAERLIQAUTOCP'))
    allocate(derived%items(4)%v, source=V(yrecldp%LAERLIQAUTOCPB, name=trim(name)//'%LAERLIQAUTOCPB'))
    allocate(derived%items(5)%v, source=V(yrecldp%LAERLIQAUTOLSP, name=trim(name)//'%LAERLIQAUTOLSP'))
    allocate(derived%items(6)%v, source=V(yrecldp%LAERLIQCOLL, name=trim(name)//'%LAERLIQCOLL'))
    allocate(derived%items(7)%v, source=V(yrecldp%LCLDBUDGET, name=trim(name)//'%LCLDBUDGET'))
    allocate(derived%items(8)%v, source=V(yrecldp%LCLDEXTRA, name=trim(name)//'%LCLDEXTRA'))
    allocate(derived%items(9)%v, source=V(yrecldp%NAECLBC, name=trim(name)//'%NAECLBC'))
    allocate(derived%items(10)%v, source=V(yrecldp%NAECLDU, name=trim(name)//'%NAECLDU'))
    allocate(derived%items(11)%v, source=V(yrecldp%NAECLOM, name=trim(name)//'%NAECLOM'))
    allocate(derived%items(12)%v, source=V(yrecldp%NAECLSS, name=trim(name)//'%NAECLSS'))
    allocate(derived%items(13)%v, source=V(yrecldp%NAECLSU, name=trim(name)//'%NAECLSU'))
    allocate(derived%items(14)%v, source=V(yrecldp%NAERCLD, name=trim(name)//'%NAERCLD'))
    allocate(derived%items(15)%v, source=V(yrecldp%NBETA, name=trim(name)//'%NBETA'))
    allocate(derived%items(16)%v, source=V(yrecldp%NCLDDIAG, name=trim(name)//'%NCLDDIAG'))
    allocate(derived%items(17)%v, source=V(yrecldp%NCLDTOP, name=trim(name)//'%NCLDTOP'))
    allocate(derived%items(18)%v, source=V(yrecldp%NSHAPEP, name=trim(name)//'%NSHAPEP'))
    allocate(derived%items(19)%v, source=V(yrecldp%NSHAPEQ, name=trim(name)//'%NSHAPEQ'))
    allocate(derived%items(20)%v, source=V(yrecldp%NSSOPT, name=trim(name)//'%NSSOPT'))
    allocate(derived%items(21)%v, source=V(yrecldp%RAMID, name=trim(name)//'%RAMID'))
    allocate(derived%items(22)%v, source=V(yrecldp%RAMIN, name=trim(name)//'%RAMIN'))
    allocate(derived%items(23)%v, source=V(yrecldp%RBETA, name=trim(name)//'%RBETA'))
    allocate(derived%items(24)%v, source=V(yrecldp%RBETAP1, name=trim(name)//'%RBETAP1'))
    allocate(derived%items(25)%v, source=V(yrecldp%RCCN, name=trim(name)//'%RCCN'))
    allocate(derived%items(26)%v, source=V(yrecldp%RCCNOM, name=trim(name)//'%RCCNOM'))
    allocate(derived%items(27)%v, source=V(yrecldp%RCCNSS, name=trim(name)//'%RCCNSS'))
    allocate(derived%items(28)%v, source=V(yrecldp%RCCNSU, name=trim(name)//'%RCCNSU'))
    allocate(derived%items(29)%v, source=V(yrecldp%RCL_AI, name=trim(name)//'%RCL_AI'))
    allocate(derived%items(30)%v, source=V(yrecldp%RCL_APB1, name=trim(name)//'%RCL_APB1'))
    allocate(derived%items(31)%v, source=V(yrecldp%RCL_APB2, name=trim(name)//'%RCL_APB2'))
    allocate(derived%items(32)%v, source=V(yrecldp%RCL_APB3, name=trim(name)//'%RCL_APB3'))
    allocate(derived%items(33)%v, source=V(yrecldp%RCL_AR, name=trim(name)//'%RCL_AR'))
    allocate(derived%items(34)%v, source=V(yrecldp%RCL_AS, name=trim(name)//'%RCL_AS'))
    allocate(derived%items(35)%v, source=V(yrecldp%RCL_BI, name=trim(name)//'%RCL_BI'))
    allocate(derived%items(36)%v, source=V(yrecldp%RCL_BR, name=trim(name)//'%RCL_BR'))
    allocate(derived%items(37)%v, source=V(yrecldp%RCL_BS, name=trim(name)//'%RCL_BS'))
    allocate(derived%items(38)%v, source=V(yrecldp%RCL_CDENOM1, name=trim(name)//'%RCL_CDENOM1'))
    allocate(derived%items(39)%v, source=V(yrecldp%RCL_CDENOM2, name=trim(name)//'%RCL_CDENOM2'))
    allocate(derived%items(40)%v, source=V(yrecldp%RCL_CDENOM3, name=trim(name)//'%RCL_CDENOM3'))
    allocate(derived%items(41)%v, source=V(yrecldp%RCL_CI, name=trim(name)//'%RCL_CI'))
    allocate(derived%items(42)%v, source=V(yrecldp%RCL_CONST1I, name=trim(name)//'%RCL_CONST1I'))
    allocate(derived%items(43)%v, source=V(yrecldp%RCL_CONST1R, name=trim(name)//'%RCL_CONST1R'))
    allocate(derived%items(44)%v, source=V(yrecldp%RCL_CONST1S, name=trim(name)//'%RCL_CONST1S'))
    allocate(derived%items(45)%v, source=V(yrecldp%RCL_CONST2I, name=trim(name)//'%RCL_CONST2I'))
    allocate(derived%items(46)%v, source=V(yrecldp%RCL_CONST2R, name=trim(name)//'%RCL_CONST2R'))
    allocate(derived%items(47)%v, source=V(yrecldp%RCL_CONST2S, name=trim(name)//'%RCL_CONST2S'))
    allocate(derived%items(48)%v, source=V(yrecldp%RCL_CONST3I, name=trim(name)//'%RCL_CONST3I'))
    allocate(derived%items(49)%v, source=V(yrecldp%RCL_CONST3R, name=trim(name)//'%RCL_CONST3R'))
    allocate(derived%items(50)%v, source=V(yrecldp%RCL_CONST3S, name=trim(name)//'%RCL_CONST3S'))
    allocate(derived%items(51)%v, source=V(yrecldp%RCL_CONST4I, name=trim(name)//'%RCL_CONST4I'))
    allocate(derived%items(52)%v, source=V(yrecldp%RCL_CONST4R, name=trim(name)//'%RCL_CONST4R'))
    allocate(derived%items(53)%v, source=V(yrecldp%RCL_CONST4S, name=trim(name)//'%RCL_CONST4S'))
    allocate(derived%items(54)%v, source=V(yrecldp%RCL_CONST5I, name=trim(name)//'%RCL_CONST5I'))
    allocate(derived%items(55)%v, source=V(yrecldp%RCL_CONST5R, name=trim(name)//'%RCL_CONST5R'))
    allocate(derived%items(56)%v, source=V(yrecldp%RCL_CONST5S, name=trim(name)//'%RCL_CONST5S'))
    allocate(derived%items(57)%v, source=V(yrecldp%RCL_CONST6I, name=trim(name)//'%RCL_CONST6I'))
    allocate(derived%items(58)%v, source=V(yrecldp%RCL_CONST6R, name=trim(name)//'%RCL_CONST6R'))
    allocate(derived%items(59)%v, source=V(yrecldp%RCL_CONST6S, name=trim(name)//'%RCL_CONST6S'))
    allocate(derived%items(60)%v, source=V(yrecldp%RCL_CONST7S, name=trim(name)//'%RCL_CONST7S'))
    allocate(derived%items(61)%v, source=V(yrecldp%RCL_CONST8S, name=trim(name)//'%RCL_CONST8S'))
    allocate(derived%items(62)%v, source=V(yrecldp%RCL_CR, name=trim(name)//'%RCL_CR'))
    allocate(derived%items(63)%v, source=V(yrecldp%RCL_CS, name=trim(name)//'%RCL_CS'))
    allocate(derived%items(64)%v, source=V(yrecldp%RCL_DI, name=trim(name)//'%RCL_DI'))
    allocate(derived%items(65)%v, source=V(yrecldp%RCL_DR, name=trim(name)//'%RCL_DR'))
    allocate(derived%items(66)%v, source=V(yrecldp%RCL_DS, name=trim(name)//'%RCL_DS'))
    allocate(derived%items(67)%v, source=V(yrecldp%RCL_DYNVISC, name=trim(name)//'%RCL_DYNVISC'))
    allocate(derived%items(68)%v, source=V(yrecldp%RCL_FAC1, name=trim(name)//'%RCL_FAC1'))
    allocate(derived%items(69)%v, source=V(yrecldp%RCL_FAC2, name=trim(name)//'%RCL_FAC2'))
    allocate(derived%items(70)%v, source=V(yrecldp%RCL_FZRAB, name=trim(name)//'%RCL_FZRAB'))
    allocate(derived%items(71)%v, source=V(yrecldp%RCL_FZRBB, name=trim(name)//'%RCL_FZRBB'))
    allocate(derived%items(72)%v, source=V(yrecldp%RCL_KA273, name=trim(name)//'%RCL_KA273'))
    allocate(derived%items(73)%v, source=V(yrecldp%RCL_KK_CLOUD_NUM_LAND, name=trim(name)//'%RCL_KK_CLOUD_NUM_LAND'))
    allocate(derived%items(74)%v, source=V(yrecldp%RCL_KK_CLOUD_NUM_SEA, name=trim(name)//'%RCL_KK_CLOUD_NUM_SEA'))
    allocate(derived%items(75)%v, source=V(yrecldp%RCL_KKAAC, name=trim(name)//'%RCL_KKAAC'))
    allocate(derived%items(76)%v, source=V(yrecldp%RCL_KKAAU, name=trim(name)//'%RCL_KKAAU'))
    allocate(derived%items(77)%v, source=V(yrecldp%RCL_KKBAC, name=trim(name)//'%RCL_KKBAC'))
    allocate(derived%items(78)%v, source=V(yrecldp%RCL_KKBAUN, name=trim(name)//'%RCL_KKBAUN'))
    allocate(derived%items(79)%v, source=V(yrecldp%RCL_KKBAUQ, name=trim(name)//'%RCL_KKBAUQ'))
    allocate(derived%items(80)%v, source=V(yrecldp%RCL_SCHMIDT, name=trim(name)//'%RCL_SCHMIDT'))
    allocate(derived%items(81)%v, source=V(yrecldp%RCL_X1I, name=trim(name)//'%RCL_X1I'))
    allocate(derived%items(82)%v, source=V(yrecldp%RCL_X1R, name=trim(name)//'%RCL_X1R'))
    allocate(derived%items(83)%v, source=V(yrecldp%RCL_X1S, name=trim(name)//'%RCL_X1S'))
    allocate(derived%items(84)%v, source=V(yrecldp%RCL_X2I, name=trim(name)//'%RCL_X2I'))
    allocate(derived%items(85)%v, source=V(yrecldp%RCL_X2R, name=trim(name)//'%RCL_X2R'))
    allocate(derived%items(86)%v, source=V(yrecldp%RCL_X2S, name=trim(name)//'%RCL_X2S'))
    allocate(derived%items(87)%v, source=V(yrecldp%RCL_X3I, name=trim(name)//'%RCL_X3I'))
    allocate(derived%items(88)%v, source=V(yrecldp%RCL_X3S, name=trim(name)//'%RCL_X3S'))
    allocate(derived%items(89)%v, source=V(yrecldp%RCL_X4I, name=trim(name)//'%RCL_X4I'))
    allocate(derived%items(90)%v, source=V(yrecldp%RCL_X4R, name=trim(name)//'%RCL_X4R'))
    allocate(derived%items(91)%v, source=V(yrecldp%RCL_X4S, name=trim(name)//'%RCL_X4S'))
    allocate(derived%items(92)%v, source=V(yrecldp%RCLCRIT, name=trim(name)//'%RCLCRIT'))
    allocate(derived%items(93)%v, source=V(yrecldp%RCLCRIT_LAND, name=trim(name)//'%RCLCRIT_LAND'))
    allocate(derived%items(94)%v, source=V(yrecldp%RCLCRIT_SEA, name=trim(name)//'%RCLCRIT_SEA'))
    allocate(derived%items(95)%v, source=V(yrecldp%RCLDIFF, name=trim(name)//'%RCLDIFF'))
    allocate(derived%items(96)%v, source=V(yrecldp%RCLDIFF_CONVI, name=trim(name)//'%RCLDIFF_CONVI'))
    allocate(derived%items(97)%v, source=V(yrecldp%RCLDMAX, name=trim(name)//'%RCLDMAX'))
    allocate(derived%items(98)%v, source=V(yrecldp%RCLDTOPCF, name=trim(name)//'%RCLDTOPCF'))
    allocate(derived%items(99)%v, source=V(yrecldp%RCLDTOPP, name=trim(name)//'%RCLDTOPP'))
    allocate(derived%items(100)%v, source=V(yrecldp%RCOVPMIN, name=trim(name)//'%RCOVPMIN'))
    allocate(derived%items(101)%v, source=V(yrecldp%RDENSREF, name=trim(name)//'%RDENSREF'))
    allocate(derived%items(102)%v, source=V(yrecldp%RDENSWAT, name=trim(name)//'%RDENSWAT'))
    allocate(derived%items(103)%v, source=V(yrecldp%RDEPLIQREFDEPTH, name=trim(name)//'%RDEPLIQREFDEPTH'))
    allocate(derived%items(104)%v, source=V(yrecldp%RDEPLIQREFRATE, name=trim(name)//'%RDEPLIQREFRATE'))
    allocate(derived%items(105)%v, source=V(yrecldp%RICEHI1, name=trim(name)//'%RICEHI1'))
    allocate(derived%items(106)%v, source=V(yrecldp%RICEHI2, name=trim(name)//'%RICEHI2'))
    allocate(derived%items(107)%v, source=V(yrecldp%RICEINIT, name=trim(name)//'%RICEINIT'))
    allocate(derived%items(108)%v, source=V(yrecldp%RKCONV, name=trim(name)//'%RKCONV'))
    allocate(derived%items(109)%v, source=V(yrecldp%RKOOPTAU, name=trim(name)//'%RKOOPTAU'))
    allocate(derived%items(110)%v, source=V(yrecldp%RLCRITSNOW, name=trim(name)//'%RLCRITSNOW'))
    allocate(derived%items(111)%v, source=V(yrecldp%RLMIN, name=trim(name)//'%RLMIN'))
    allocate(derived%items(112)%v, source=V(yrecldp%RNICE, name=trim(name)//'%RNICE'))
    allocate(derived%items(113)%v, source=V(yrecldp%RPECONS, name=trim(name)//'%RPECONS'))
    allocate(derived%items(114)%v, source=V(yrecldp%RPRC1, name=trim(name)//'%RPRC1'))
    allocate(derived%items(115)%v, source=V(yrecldp%RPRC2, name=trim(name)//'%RPRC2'))
    allocate(derived%items(116)%v, source=V(yrecldp%RPRECRHMAX, name=trim(name)//'%RPRECRHMAX'))
    allocate(derived%items(117)%v, source=V(yrecldp%RSNOWLIN1, name=trim(name)//'%RSNOWLIN1'))
    allocate(derived%items(118)%v, source=V(yrecldp%RSNOWLIN2, name=trim(name)//'%RSNOWLIN2'))
    allocate(derived%items(119)%v, source=V(yrecldp%RTAUMEL, name=trim(name)//'%RTAUMEL'))
    allocate(derived%items(120)%v, source=V(yrecldp%RTHOMO, name=trim(name)//'%RTHOMO'))
    allocate(derived%items(121)%v, source=V(yrecldp%RVICE, name=trim(name)//'%RVICE'))
    allocate(derived%items(122)%v, source=V(yrecldp%RVRAIN, name=trim(name)//'%RVRAIN'))
    allocate(derived%items(123)%v, source=V(yrecldp%RVRFACTOR, name=trim(name)//'%RVRFACTOR'))
    allocate(derived%items(124)%v, source=V(yrecldp%RVSNOW, name=trim(name)//'%RVSNOW'))

    if (present(name)) then
       derived%name = name
    end if
  end function dt_tecldp

  ! Type-specific output subroutines

  subroutine write_variable(this, f_data, f_info, index)
    class(Variable) :: this
    integer, intent(in) :: f_data, f_info
    integer, intent(in), optional :: index
  end subroutine write_variable

  subroutine write_scalar(this, f_data, f_info, index)
    class(Scalar) :: this
    integer, intent(in) :: f_data, f_info
    integer, intent(in), optional :: index
    character, parameter :: tab=char(9)
    integer :: idx

    if (present(index)) then
       idx = index
    else
       idx = -1
    end if

    select type(vptr => this%ptr)
    type is (integer(kind=JPIM))
       ! Integer scalar variable
       ! write(*, '(A,A,A,I4)') '[Loki-debug] V[scalar]::', this%name, ' => ', vptr
       write(f_info, '(A,A,A,A,A,A,I6)') this%name, tab, 'int32', tab, '1', tab, idx
       write(f_data) vptr

    type is (real(kind=JPRB))
       ! Real scalar variable
       ! write(*, '(A,A,A,F6.4)') '[Loki-debug] V[scalar]::', this%name, ' => ', vptr
       write(f_info, '(A,A,A,A,A,A,I6)') this%name, tab, 'float64', tab, '1', tab, idx
       write(f_data) vptr

    type is (logical)
       ! Integer scalar variable
       ! write(*, '(A,A,A,L2)') '[Loki-debug] V[scalar]::', this%name, ' => ', vptr
       write(f_info, '(A,A,A,A,A,A,I6)') this%name, tab, 'logical', tab, '1', tab, idx
       write(f_data) vptr

    class default
       write(*, '(A,A)') '[Loki-debug] Error::Could not write ', this%name
    end select
  end subroutine write_scalar

  subroutine write_array_1d(this, f_data, f_info, index)
    class(Array1D) :: this
    integer, intent(in) :: f_data, f_info
    integer, intent(in), optional :: index
    character, parameter :: tab=char(9)
    integer :: idx

    if (present(index)) then
       idx = index
    else
       idx = -1
    end if

    select type(vptr => this%ptr)
    type is (real(kind=JPRB))
       ! write(*, '(A,A,A,I6,A)') '[Loki-debug] V[array]::', this%name, '(', size(this%ptr), ' )'
       write(f_info, '(A,A,A,A,I6,A,I6)') this%name, tab, 'float64', tab, size(vptr), tab, idx
       write(f_data) vptr

    type is (integer(kind=JPIM))
       ! write(*, '(A,A,A,I6,A)') '[Loki-debug] V[array]::', this%name, '(', size(this%ptr), ' )'
       write(f_info, '(A,A,A,A,I6,A,I6)') this%name, tab, 'int32', tab, size(vptr), tab, idx
       write(f_data) vptr

    type is (logical)
       ! write(*, '(A,A,A,I6,A)') '[Loki-debug] V[array]::', this%name, '(', size(this%ptr), ' )'
       write(f_info, '(A,A,A,A,I6,A,I6)') this%name, tab, 'logical', tab, size(vptr), tab, idx
       write(f_data) vptr

    class default
       write(*, '(A,A)') '[Loki-debug] Error::Could not write ', this%name
    end select
  end subroutine write_array_1d

  subroutine write_array_2d(this, f_data, f_info, index)
    class(Array2D) :: this
    integer, intent(in) :: f_data, f_info
    integer, intent(in), optional :: index
    character, parameter :: tab=char(9)
    integer :: idx

    if (present(index)) then
       idx = index
    else
       idx = -1
    end if

    select type(vptr => this%ptr)
    type is (real(kind=JPRB))
       ! write(*, '(A,A,A,2I6,A)') '[Loki-debug] V[array]::', this%name, '(', shape(this%ptr), ' )'
       write(f_info, '(A,A,A,A,A,2I6,A,A,I6)') this%name, tab, 'float64', tab, '(', shape(vptr), ')', tab, idx
       write(f_data) vptr

    class default
       write(*, '(A,A)') '[Loki-debug] Error::Could not write ', this%name
    end select
  end subroutine write_array_2d

  subroutine write_array_3d(this, f_data, f_info, index)
    class(Array3D) :: this
    integer, intent(in) :: f_data, f_info
    integer, intent(in), optional :: index
    character, parameter :: tab=char(9)
    integer :: idx

    if (present(index)) then
       idx = index
    else
       idx = -1
    end if

    select type(vptr => this%ptr)
    type is (real(kind=JPRB))
       ! write(*, '(A,A,A,3I6,A)') '[Loki-debug] V[array]::', this%name, '(', shape(this%ptr), ' )'
       write(f_info, '(A,A,A,A,A,3I6,A,A,I6)') this%name, tab, 'float64', tab, '(', shape(vptr), ')', tab, idx
       write(f_data) vptr

    class default
       write(*, '(A,A)') '[Loki-debug] Error::Could not write ', this%name
    end select
  end subroutine write_array_3d

  subroutine write_derived_type(this, f_data, f_info, index)
    class(DerivedType) :: this
    integer, intent(in) :: f_data, f_info
    integer, intent(in), optional :: index
    integer :: i, idx

    if (present(index)) then
       idx = index
    else
       idx = -1
    end if

    write(*, '(A,A)') '[Loki-debug] V[state_type]::', this%name

    do i=1, size(this%items)
       call this%items(i)%v%write(f_data, f_info, index=index)
    end do

  end subroutine write_derived_type

  subroutine state_open(this, array_of_vars, filename)
    class(StateDump) :: this
    class(Item), target, intent(inout) :: array_of_vars(:)
    character(len=*), optional, intent(in) :: filename
    integer, parameter :: dummy_value = 1

    this%items => array_of_vars

    if (present(filename)) then
       this%fname_data = trim(filename)//'.data'
       this%fname_info = trim(filename)//'.info'
    else
       this%fname_data = 'loki_debug.data'
       this%fname_info = 'loki_debug.info'
    end if

    write(*, '(A,A)') '[Loki-debug] Initializing state: ', filename

    ! Open files for data and meta-data writes
    open(this%f_data, file=this%fname_data, status='replace', form='unformatted', access='stream')
    open(this%f_info, file=this%fname_info, status='replace', form='formatted')

    ! Write a dummy value for dynamic endian-ness checks
    ! I really hate myself for having to do this, argh..!
    write(this%f_data) dummy_value

  end subroutine state_open

  subroutine state_dump_indexed(this, index)
    class(StateDump) :: this
    integer, intent(in) :: index
    integer :: i

    ! Write variable info via type-specific routines
    do i=1, size(this%items)
       call this%items(i)%v%write(this%f_data, this%f_info, index=index)
    end do

  end subroutine state_dump_indexed

  subroutine state_close(this)
    class(StateDump) :: this

    ! Clean up
    deallocate(this%fname_data)
    deallocate(this%fname_info)
    close(this%f_data)
    close(this%f_info)
  end subroutine state_close

  subroutine dump_state(array_of_vars, filename)
    ! Write a list of variables to file with enough meta-information to
    ! recover the program state at that time.
    class(Item), intent(inout) :: array_of_vars(:)
    character(len=*), optional, intent(in) :: filename

    character(len=:), allocatable :: fname_data, fname_info, vname
    integer, parameter :: f_data=666, f_info=667
    character, parameter :: tab=char(9)
    integer :: i
    integer, parameter :: dummy_value = 1
    real(kind=JPRB), pointer :: arr1d(:), arr2d(:,:)

    if (present(filename)) then
       fname_data = trim(filename)//'.data'
       fname_info = trim(filename)//'.info'
    else
       fname_data = 'loki_debug.data'
       fname_info = 'loki_debug.info'
    end if

    write(*, '(A,I3)') '[Loki-debug] Serializing variable array of size ', size(array_of_vars)

    ! Open files for data and meta-data writes
    open(f_data, file=fname_data, status='replace', form='unformatted', access='stream')
    open(f_info, file=fname_info, status='replace', form='formatted')

    ! Write a dummy value for dynamic endian-ness checks
    ! I really hate myself for having to do this, argh..!
    write(f_data) dummy_value

    ! Write variable info via type-specific routines
    do i=1, size(array_of_vars)
       call array_of_vars(i)%v%write(f_data, f_info)
    end do

    ! Clean up
    deallocate(fname_data)
    deallocate(fname_info)
    close(f_data)
    close(f_info)

  end subroutine dump_state

end module loki_debug
