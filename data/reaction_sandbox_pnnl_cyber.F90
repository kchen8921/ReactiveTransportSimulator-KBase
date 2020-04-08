
module Reaction_Sandbox_Cyber_class

  use Reaction_Sandbox_Base_class

  use Global_Aux_module
  use Reactive_Transport_Aux_module

  use PFLOTRAN_Constants_module

  implicit none

  private

#include "petsc/finclude/petscsys.h"
  PetscInt, parameter :: HCO3_MASS_STORAGE_INDEX = 1
  PetscInt, parameter :: NH4_MASS_STORAGE_INDEX = 2
  PetscInt, parameter :: HS_MASS_STORAGE_INDEX = 3
  PetscInt, parameter :: H_MASS_STORAGE_INDEX = 4
  PetscInt, parameter :: BIOMASS_MASS_STORAGE_INDEX = 5
  PetscInt, parameter :: C5H2NOP_MASS_STORAGE_INDEX = 6
  PetscInt, parameter :: O2_MASS_STORAGE_INDEX = 7
  PetscInt, parameter :: C7H11NO_MASS_STORAGE_INDEX = 8

  type, public, &
    extends(reaction_sandbox_base_type) :: reaction_sandbox_cyber_type
    PetscInt :: hco3_id 
    PetscInt :: nh4_id 
    PetscInt :: hs_id 
    PetscInt :: h_id 
    PetscInt :: biomass_id 
    PetscInt :: c5h2nop_id 
    PetscInt :: o2_id 
    PetscInt :: c7h11no_id 
    PetscReal :: mu_max 
    PetscReal :: vh 
    PetscReal :: k_deg 
    PetscReal :: cc 
    PetscReal :: activation_energy 
    PetscReal :: reference_temperature 

    PetscReal :: nrxn
    PetscBool :: store_cumulative_mass
    PetscInt :: offset_auxiliary
  contains
    procedure, public :: ReadInput => CyberRead
    procedure, public :: Setup => CyberSetup
    procedure, public :: Evaluate => CyberReact
    procedure, public :: Destroy => CyberDestroy
  end type reaction_sandbox_cyber_type

  public :: CyberCreate

contains

! ************************************************************************** !

function CyberCreate()
#include "petsc/finclude/petscsys.h"
  use petscsys
  implicit none

  class(reaction_sandbox_cyber_type), pointer :: CyberCreate

  allocate(CyberCreate)
  CyberCreate%hco3_id = UNINITIALIZED_INTEGER 
  CyberCreate%nh4_id = UNINITIALIZED_INTEGER 
  CyberCreate%hs_id = UNINITIALIZED_INTEGER 
  CyberCreate%h_id = UNINITIALIZED_INTEGER 
  CyberCreate%biomass_id = UNINITIALIZED_INTEGER 
  CyberCreate%c5h2nop_id = UNINITIALIZED_INTEGER 
  CyberCreate%o2_id = UNINITIALIZED_INTEGER 
  CyberCreate%c7h11no_id = UNINITIALIZED_INTEGER 
  CyberCreate%mu_max = UNINITIALIZED_DOUBLE 
  CyberCreate%vh = UNINITIALIZED_DOUBLE 
  CyberCreate%k_deg = UNINITIALIZED_DOUBLE 
  CyberCreate%cc = UNINITIALIZED_DOUBLE 
  CyberCreate%activation_energy = UNINITIALIZED_DOUBLE 
  CyberCreate%reference_temperature = 298.15d0 ! 25 C

  CyberCreate%nrxn = UNINITIALIZED_INTEGER
  CyberCreate%store_cumulative_mass = PETSC_FALSE

  nullify(CyberCreate%next)
  print *, 'CyberCreat Done'
end function CyberCreate

! ************************************************************************** !

! ************************************************************************** !

subroutine CyberRead(this,input,option)

  use Option_module
  use String_module
  use Input_Aux_module

  implicit none

  class(reaction_sandbox_cyber_type) :: this
  type(input_type), pointer :: input
  type(option_type) :: option

  PetscInt :: i
  character(len=MAXWORDLENGTH) :: word, internal_units, units
  character(len=MAXSTRINGLENGTH) :: error_string

  error_string = 'CHEMISTRY,REACTION_SANDBOX,CYBER'
  call InputPushBlock(input,option)
  do
    call InputReadPflotranString(input,option)
    if (InputError(input)) exit
    if (InputCheckExit(input,option)) exit

    call InputReadCard(input,option,word)
    call InputErrorMsg(input,option,'keyword',error_string)
    call StringToUpper(word)

    select case(trim(word))

      case('MU_MAX')
        call InputReadDouble(input,option,this%mu_max)
        call InputErrorMsg(input,option,'mu_max',error_string)
        call InputReadAndConvertUnits(input,this%mu_max,'1/sec', &
                                      trim(error_string)//',mu_max',option)
        
      case('VH')
        call InputReadDouble(input,option,this%vh)
        call InputErrorMsg(input,option,'vh',error_string)
        call InputReadAndConvertUnits(input,this%vh,'m^3', &
                                      trim(error_string)//',vh',option)
        
      case('K_DEG')
        call InputReadDouble(input,option,this%k_deg)
        call InputErrorMsg(input,option,'k_deg',error_string)
        call InputReadAndConvertUnits(input,this%k_deg,'1/sec', &
                                      trim(error_string)//',k_deg',option)
        
      case('CC')
        call InputReadDouble(input,option,this%cc)
        call InputErrorMsg(input,option,'cc',error_string)
        call InputReadAndConvertUnits(input,this%cc,'M', &
                                      trim(error_string)//',cc',option)
        
      case('ACTIVATION_ENERGY')
        call InputReadDouble(input,option,this%activation_energy)
        call InputErrorMsg(input,option,'activation_energy',error_string)
        call InputReadAndConvertUnits(input,this%activation_energy,'J/mol', &
                                      trim(error_string)//',activation_energy',option)
        
      case('REFERENCE_TEMPERATURE')
        call InputReadDouble(input,option,this%reference_temperature)
        call InputErrorMsg(input,option,'reference temperature [C]', &
                           error_string)
        this%reference_temperature = this%reference_temperature + 273.15d0
        
      case default
        call InputKeywordUnrecognized(input,word,error_string,option)
    end select
  enddo
  call InputPopBlock(input,option)
end subroutine CyberRead

! ************************************************************************** !

subroutine CyberSetup(this,reaction,option)

  use Reaction_Aux_module, only : reaction_rt_type, GetPrimarySpeciesIDFromName
  use Reaction_Immobile_Aux_module, only : GetImmobileSpeciesIDFromName
  use Reaction_Mineral_Aux_module, only : GetKineticMineralIDFromName
  use Option_module

  implicit none

  class(reaction_sandbox_cyber_type) :: this
  class(reaction_rt_type) :: reaction
  type(option_type) :: option

  character(len=MAXWORDLENGTH) :: word
  PetscInt :: irxn

  PetscReal, parameter :: per_day_to_per_sec = 1.d0 / 24.d0 / 3600.d0

  word = 'HCO3-'
  this%hco3_id = &
    GetPrimarySpeciesIDFromName(word,reaction,option)
    
  word = 'NH4+'
  this%nh4_id = &
    GetPrimarySpeciesIDFromName(word,reaction,option)
    
  word = 'HS-'
  this%hs_id = &
    GetPrimarySpeciesIDFromName(word,reaction,option)
    
  word = 'H+'
  this%h_id = &
    GetPrimarySpeciesIDFromName(word,reaction,option)
    
  word = 'BIOMASS'
  this%biomass_id = &
    GetImmobileSpeciesIDFromName(word,reaction%immobile,option) + reaction%offset_immobile
        
  word = 'C5H2NOP'
  this%c5h2nop_id = &
    GetPrimarySpeciesIDFromName(word,reaction,option)
    
  word = 'O2'
  this%o2_id = &
    GetPrimarySpeciesIDFromName(word,reaction,option)
    
  word = 'C7H11NO'
  this%c7h11no_id = &
    GetPrimarySpeciesIDFromName(word,reaction,option)
    
  if (this%store_cumulative_mass) then
    this%offset_auxiliary = reaction%nauxiliary
    reaction%nauxiliary = reaction%nauxiliary + 16
  endif

end subroutine CyberSetup

! ************************************************************************** !

subroutine CyberAuxiliaryPlotVariables(this,list,reaction,option)

  use Option_module
  use Reaction_Aux_module
  use Output_Aux_module
  use Variables_module, only : REACTION_AUXILIARY

  implicit none

  class(reaction_sandbox_cyber_type) :: this
  type(output_variable_list_type), pointer :: list
  type(option_type) :: option
  class(reaction_rt_type) :: reaction

  character(len=MAXWORDLENGTH) :: names(8)
  character(len=MAXWORDLENGTH) :: word
  character(len=MAXWORDLENGTH) :: units
  PetscInt :: indices(8)
  PetscInt :: i

  names(1) = 'HCO3-'
  names(2) = 'NH4+'
  names(3) = 'HS-'
  names(4) = 'H+'
  names(5) = 'BIOMASS'
  names(6) = 'C5H2NOP'
  names(7) = 'O2'
  names(8) = 'C7H11NO'
  indices(1) = HCO3_MASS_STORAGE_INDEX
  indices(2) = NH4_MASS_STORAGE_INDEX
  indices(3) = HS_MASS_STORAGE_INDEX
  indices(4) = H_MASS_STORAGE_INDEX
  indices(5) = BIOMASS_MASS_STORAGE_INDEX
  indices(6) = C5H2NOP_MASS_STORAGE_INDEX
  indices(7) = O2_MASS_STORAGE_INDEX
  indices(8) = C7H11NO_MASS_STORAGE_INDEX

  if (this%store_cumulative_mass) then
    do i = 1, 8
      word = trim(names(i)) // ' Rate'
      units = 'mol/m^3-sec'
      call OutputVariableAddToList(list,word,OUTPUT_RATE,units, &
                                   REACTION_AUXILIARY, &
                                   this%offset_auxiliary+indices(i))
    enddo
    do i = 1, 8
      word = trim(names(i)) // ' Cum. Mass'
      units = 'mol/m^3'
      call OutputVariableAddToList(list,word,OUTPUT_GENERIC,units, &
                                   REACTION_AUXILIARY, &
                                   this%offset_auxiliary+8+indices(i))
    enddo
  endif

end subroutine CyberAuxiliaryPlotVariables

! ************************************************************************** !

subroutine CyberReact(this,Residual,Jacobian,compute_derivative, &
                         rt_auxvar,global_auxvar,material_auxvar,reaction, &
                         option)

  use Option_module
  use Reaction_Aux_module
  use Material_Aux_class

  implicit none

  class(reaction_sandbox_cyber_type) :: this
  type(option_type) :: option
  class(reaction_rt_type) :: reaction
  ! the following arrays must be declared after reaction
  PetscReal :: Residual(reaction%ncomp)
  PetscReal :: Jacobian(reaction%ncomp,reaction%ncomp)
  type(reactive_transport_auxvar_type) :: rt_auxvar
  type(global_auxvar_type) :: global_auxvar
  class(material_auxvar_type) :: material_auxvar

  PetscInt, parameter :: iphase = 1
  PetscReal :: L_water
  PetscReal :: kg_water

  PetscInt :: i, j, irxn
  PetscReal :: C_hco3,C_nh4,C_hs,C_h,C_biomass,C_c5h2nop,C_o2,C_c7h11no
  PetscReal :: r1doc,r1o2,r2doc,r2o2
  PetscReal :: r1kin,r2kin
  PetscReal :: sumkin
  PetscReal :: u1,u2
  PetscReal :: molality_to_molarity
  PetscReal :: temperature_scaling_factor
  PetscReal :: mu_max_scaled
  PetscReal :: k1_scaled,k2_scaled,k_deg_scaled
  PetscReal :: volume, rate_scale
  PetscBool :: compute_derivative

  PetscReal :: rate(2)

  volume = material_auxvar%volume
  L_water = material_auxvar%porosity*global_auxvar%sat(iphase)* &
            volume*1.d3 ! m^3 -> L
  kg_water = material_auxvar%porosity*global_auxvar%sat(iphase)* &
             global_auxvar%den_kg(iphase)*volume

  molality_to_molarity = global_auxvar%den_kg(iphase)*1.d-3

  if (reaction%act_coef_update_frequency /= ACT_COEF_FREQUENCY_OFF) then
    option%io_buffer = 'Activity coefficients not currently supported in &
      &CyberReact().'
    call printErrMsg(option)
  endif

  temperature_scaling_factor = 1.d0
  if (Initialized(this%activation_energy)) then
    temperature_scaling_factor = &
      exp(this%activation_energy/IDEAL_GAS_CONSTANT* &
          (1.d0/this%reference_temperature-1.d0/(global_auxvar%temp+273.15d0)))
  endif

  ! concentrations are molarities [M]
  C_hco3 = rt_auxvar%pri_molal(this%hco3_id)* &
        rt_auxvar%pri_act_coef(this%hco3_id)*molality_to_molarity
        
  C_nh4 = rt_auxvar%pri_molal(this%nh4_id)* &
        rt_auxvar%pri_act_coef(this%nh4_id)*molality_to_molarity
        
  C_hs = rt_auxvar%pri_molal(this%hs_id)* &
        rt_auxvar%pri_act_coef(this%hs_id)*molality_to_molarity
        
  C_h = rt_auxvar%pri_molal(this%h_id)* &
        rt_auxvar%pri_act_coef(this%h_id)*molality_to_molarity
        
  C_biomass = rt_auxvar%immobile(this%biomass_id-reaction%offset_immobile)
        
  C_c5h2nop = rt_auxvar%pri_molal(this%c5h2nop_id)* &
        rt_auxvar%pri_act_coef(this%c5h2nop_id)*molality_to_molarity
        
  C_o2 = rt_auxvar%pri_molal(this%o2_id)* &
        rt_auxvar%pri_act_coef(this%o2_id)*molality_to_molarity
        
  C_c7h11no = rt_auxvar%pri_molal(this%c7h11no_id)* &
        rt_auxvar%pri_act_coef(this%c7h11no_id)*molality_to_molarity
        
  mu_max_scaled = this%mu_max * temperature_scaling_factor
  k_deg_scaled = this%k_deg * temperature_scaling_factor

  r1doc = exp(-0.5076967667820371/(this%vh * C_c5h2nop))
  r1o2 = exp(-0.8538628754326395/(this%vh * C_o2))
  r1kin = mu_max_scaled * r1doc * r1o2
  r2doc = exp(-0.250127716266806/(this%vh * C_c7h11no))
  r2o2 = exp(-1.0760855882678508/(this%vh * C_o2))
  r2kin = mu_max_scaled * r2doc * r2o2

  sumkin = r1kin + r2kin

  u1 = 0.d0
  if (r1kin > 0.d0) u1 = r1kin/sumkin
  u2 = 0.d0
  if (r2kin > 0.d0) u2 = r2kin/sumkin

  rate(1) = u1*r1kin*(1-C_biomass/this%cc)
  rate(2) = u2*r2kin*(1-C_biomass/this%cc)

  Residual(this%hco3_id) = Residual(this%hco3_id)  &
                          - 1.5384838339101858 * rate(1) * C_biomass * L_water &
                          - 0.7508940138676419 * rate(2) * C_biomass * L_water
  Residual(this%nh4_id) = Residual(this%nh4_id)  &
                         - 0.3076967667820372 * rate(1) * C_biomass * L_water &
                         - 0.05012771626680597 * rate(2) * C_biomass * L_water
  Residual(this%hs_id) = Residual(this%hs_id)  &
                        - 0.5076967667820371 * rate(1) * C_biomass * L_water &
                        - 0.0 * rate(2) * C_biomass * L_water
  Residual(this%h_id) = Residual(this%h_id)  &
                       - 1.7384838339101851 * rate(1) * C_biomass * L_water &
                       - 0.7007662976008349 * rate(2) * C_biomass * L_water
  Residual(this%biomass_id) = Residual(this%biomass_id)  &
                             - 1.0 * rate(1) * C_biomass * L_water &
                             - 1.0 * rate(2) * C_biomass * L_water
  Residual(this%biomass_id) = Residual(this%biomass_id) + k_deg_scaled * C_biomass * L_water 

  Residual(this%c5h2nop_id) = Residual(this%c5h2nop_id)  &
                             + 0.5076967667820371 * rate(1) * C_biomass * L_water &
                             - 0.0 * rate(2) * C_biomass * L_water
  Residual(this%o2_id) = Residual(this%o2_id)  &
                        + 0.8538628754326395 * rate(1) * C_biomass * L_water &
                        + 1.0760855882678508 * rate(2) * C_biomass * L_water
  Residual(this%c7h11no_id) = Residual(this%c7h11no_id)  &
                             - 0.0 * rate(1) * C_biomass * L_water &
                             + 0.250127716266806 * rate(2) * C_biomass * L_water

  if (this%store_cumulative_mass) then
        rate_scale = C_biomass * L_water / volume
            i = this%offset_auxiliary + HCO3_MASS_STORAGE_INDEX
        rt_auxvar%auxiliary_data(i) = &
                        + 1.5384838339101858 * rate(1) * rate_scale &
                        + 0.7508940138676419 * rate(2) * rate_scale
        i = this%offset_auxiliary + NH4_MASS_STORAGE_INDEX
        rt_auxvar%auxiliary_data(i) = &
                        + 0.3076967667820372 * rate(1) * rate_scale &
                        + 0.05012771626680597 * rate(2) * rate_scale
        i = this%offset_auxiliary + HS_MASS_STORAGE_INDEX
        rt_auxvar%auxiliary_data(i) = &
                        + 0.5076967667820371 * rate(1) * rate_scale &
                        + 0.0 * rate(2) * rate_scale
        i = this%offset_auxiliary + H_MASS_STORAGE_INDEX
        rt_auxvar%auxiliary_data(i) = &
                        + 1.7384838339101851 * rate(1) * rate_scale &
                        + 0.7007662976008349 * rate(2) * rate_scale
        i = this%offset_auxiliary + BIOMASS_MASS_STORAGE_INDEX
        rt_auxvar%auxiliary_data(i) = &
                        + 1.0 * rate(1) * rate_scale &
                        + 1.0 * rate(2) * rate_scale
        i = this%offset_auxiliary + C5H2NOP_MASS_STORAGE_INDEX
        rt_auxvar%auxiliary_data(i) = &
                        + 0.5076967667820371 * rate(1) * rate_scale &
                        + 0.0 * rate(2) * rate_scale
        i = this%offset_auxiliary + O2_MASS_STORAGE_INDEX
        rt_auxvar%auxiliary_data(i) = &
                        + 0.8538628754326395 * rate(1) * rate_scale &
                        + 1.0760855882678508 * rate(2) * rate_scale
        i = this%offset_auxiliary + C7H11NO_MASS_STORAGE_INDEX
        rt_auxvar%auxiliary_data(i) = &
                        + 0.0 * rate(1) * rate_scale &
                        + 0.250127716266806 * rate(2) * rate_scale

  endif
    
end subroutine CyberReact


! ************************************************************************** !

subroutine CyberDestroy(this)
  use Utility_module

  implicit none

  class(reaction_sandbox_cyber_type) :: this

  print *, 'CyberDestroy Done'

end subroutine CyberDestroy

end module Reaction_Sandbox_Cyber_class
