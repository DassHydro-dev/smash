!%    This module `m_data` encapsulates all SMASH data
module m_input_data

    use m_common, only: sp, dp, lchar
    use m_setup, only: SetupDT
    use m_mesh, only: MeshDT
    
    implicit none
    
    !%      Input_DataDT type:
    !%
    !%      ====================    ==========================================================
    !%      `args`                  Description
    !%      ====================    ==========================================================

    !%      ====================    ==========================================================
    
    type Input_DataDT
    
        real(sp), dimension(:,:), allocatable :: qobs
        real(sp), dimension(:,:,:), allocatable :: prcp
        real(sp), dimension(:,:,:), allocatable :: pet
        
        real(sp), dimension(:,:), allocatable :: sparse_prcp
        real(sp), dimension(:,:), allocatable :: sparse_pet
    
    end type Input_DataDT
    
    contains
    
        subroutine Input_DataDT_initialise(input_data, setup, mesh)
        
            implicit none
            
            type(Input_DataDT), intent(inout) :: input_data
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh

            if (.not. setup%simulation_only) then
            
                allocate(input_data%qobs(mesh%ng, setup%ntime_step))
                input_data%qobs = -99._sp
                
            end if
            
            if (setup%sparse_storage) then
            
                allocate(input_data%sparse_prcp(mesh%nac, &
                & setup%ntime_step))
                input_data%sparse_prcp = -99._sp
                allocate(input_data%sparse_pet(mesh%nac, &
                & setup%ntime_step))
                input_data%sparse_pet = -99._sp
                
            else
            
                allocate(input_data%prcp(mesh%nrow, mesh%ncol, &
                & setup%ntime_step))
                input_data%prcp = -99._sp
                allocate(input_data%pet(mesh%nrow, mesh%ncol, &
                & setup%ntime_step))
                input_data%pet = -99._sp
            
            end if
            
        end subroutine Input_DataDT_initialise
    
    

end module m_input_data
