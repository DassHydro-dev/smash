!%      This module `mwd_parameters` encapsulates all SMASH parameters.
!%      This module is wrapped and differentiated.

!%      ParametersDT type:
!%      
!%      ======================== =======================================
!%      `Variables`              Description
!%      ======================== =======================================
!%      ``ci``                   Interception parameter          [mm]    (default: 1)     ]0, +Inf[
!%      ``cp``                   Production parameter            [mm]    (default: 200)   ]0, +Inf[
!%      ``beta``                 Percolation parameter           [-]     (default: 1000)  ]0, +Inf[
!%      ``cft``                  Fast transfer parameter         [mm]    (default: 500)   ]0, +Inf[
!%      ``cst``                  Slow transfer parameter         [mm]    (default: 500)   ]0, +Inf[
!%      ``alpha``                Transfer partitioning parameter [-]     (default: 0.9)   ]0, 1[
!%      ``exc``                  Exchange parameter              [mm/dt] (default: 0)     ]-Inf, +Inf[
!%      ``lr``                   Linear routing parameter        [min]   (default: 5)     ]0, +Inf[
!%      ======================== =======================================

module mwd_parameters

    use mwd_common !% only: sp, dp, lchar, np, ns
    use mwd_setup !% only: SetupDT
    use mwd_mesh  !% only: MeshDT
    
    implicit none
    
    type ParametersDT
        
        real(sp), dimension(:,:), allocatable :: ci
        real(sp), dimension(:,:), allocatable :: cp
        real(sp), dimension(:,:), allocatable :: beta
        real(sp), dimension(:,:), allocatable :: cft
        real(sp), dimension(:,:), allocatable :: cst
        real(sp), dimension(:,:), allocatable :: alpha
        real(sp), dimension(:,:), allocatable :: exc
        real(sp), dimension(:,:), allocatable :: lr
        
    end type ParametersDT
    
    contains
        
        subroutine ParametersDT_initialise(parameters, setup, mesh)
        
            implicit none
            
            type(SetupDT), intent(in) :: setup
            type(MeshDT), intent(in) :: mesh
            type(ParametersDT), intent(inout) :: parameters
            
            integer :: nrow, ncol
            
            nrow = mesh%nrow
            ncol = mesh%ncol
            
            allocate(parameters%ci(nrow, ncol))
            allocate(parameters%cp(nrow, ncol))
            allocate(parameters%beta(nrow, ncol))
            allocate(parameters%cft(nrow, ncol))
            allocate(parameters%cst(nrow, ncol))
            allocate(parameters%alpha(nrow, ncol))
            allocate(parameters%exc(nrow, ncol))
            allocate(parameters%lr(nrow, ncol))
            
            call vector_to_parameters(&
            & setup%default_parameters, parameters)

        end subroutine ParametersDT_initialise
        
        
!%      TODO comment  
        subroutine parameters_copy(parameters_in, &
        & parameters_out)
            
            implicit none
            
            type(ParametersDT), intent(in) :: parameters_in
            type(ParametersDT), intent(out) :: parameters_out
            
            parameters_out = parameters_in
        
        end subroutine parameters_copy
        
        
!%      TODO comment  
        subroutine parameters_to_matrix(parameters, matrix)
        
            implicit none
            
            type(ParametersDT), intent(in) :: parameters
            real(sp), dimension(size(parameters%cp, 1), &
            & size(parameters%cp, 2), np), intent(inout) :: matrix
            
            matrix(:,:,1) = parameters%ci(:,:)
            matrix(:,:,2) = parameters%cp(:,:)
            matrix(:,:,3) = parameters%beta(:,:)
            matrix(:,:,4) = parameters%cft(:,:)
            matrix(:,:,5) = parameters%cst(:,:)
            matrix(:,:,6) = parameters%alpha(:,:)
            matrix(:,:,7) = parameters%exc(:,:)
            matrix(:,:,8) = parameters%lr(:,:)
        
        end subroutine parameters_to_matrix
        
        
!%      TODO comment  
        subroutine matrix_to_parameters(matrix, parameters)
            
            implicit none
            
            type(ParametersDT), intent(inout) :: parameters
            real(sp), dimension(size(parameters%cp, 1), &
            & size(parameters%cp, 2), np), intent(in) :: matrix
            
            parameters%ci(:,:) = matrix(:,:,1)
            parameters%cp(:,:) = matrix(:,:,2)
            parameters%beta(:,:) = matrix(:,:,3)
            parameters%cft(:,:) = matrix(:,:,4)
            parameters%cst(:,:) = matrix(:,:,5)
            parameters%alpha(:,:) = matrix(:,:,6)
            parameters%exc(:,:) = matrix(:,:,7)
            parameters%lr(:,:) = matrix(:,:,8)
        
        end subroutine matrix_to_parameters
        
        
!%      TODO comment  
        subroutine vector_to_parameters(vector, parameters)
        
            implicit none
            
            type(ParametersDT), intent(inout) :: parameters
            real(sp), dimension(np), intent(in) :: vector
            
            parameters%ci = vector(1)
            parameters%cp = vector(2)
            parameters%beta = vector(3)
            parameters%cft = vector(4)
            parameters%cst = vector(5)
            parameters%alpha = vector(6)
            parameters%exc = vector(7)
            parameters%lr = vector(8)
        
        end subroutine vector_to_parameters

end module mwd_parameters