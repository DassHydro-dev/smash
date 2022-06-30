!%    This module (wrap) `mw_validate` encapsulates all SMASH validate (type, subroutines, functions)
module mw_validate
    
    use m_common !% only: sp, dp, lchar, np, np
    use mw_setup !% only: SetupDT
    use mw_mesh  !% only: MeshDT
    use mw_input_data !% only: Input_DataDT
    use mw_parameters !%  only: ParametersDT, parameters_derived_type_to_matrix
    use mw_states !% only: StatesDT
    use mw_output !% only: OutputDT
    
    implicit none
    
    contains
    
        !% Calling forward_d from forward/forward_d.f90
        !% Calling forward_b from forward/forward_b.f90
        subroutine scalar_product_test(setup, mesh, input_data, parameters, states, output, cost)
            
            implicit none
            
            type(SetupDT), intent(inout) :: setup
            type(MeshDT), intent(inout) :: mesh
            type(Input_DataDT), intent(inout) :: input_data
            type(ParametersDT), intent(inout) :: parameters
            type(StatesDT), intent(inout) :: states
            type(OutputDT), intent(inout) :: output
            real(sp), intent(inout) :: cost
            
            type(ParametersDT) :: parameters_d, parameters_b
            type(StatesDT) :: states_d, states_b, init_states
            type(OutputDT) ::  output_d, output_b
            real(sp) :: cost_d, cost_b
            real(sp), dimension(mesh%nrow, mesh%ncol, np) :: parameters_d_matrix, parameters_b_matrix 
            real(sp) :: sp1, sp2
            
            init_states = states
            
            call ParametersDT_initialise(parameters_d, setup, mesh)
            call ParametersDT_initialise(parameters_b, setup, mesh)
            
            call StatesDT_initialise(states_d, setup, mesh)
            call StatesDT_initialise(states_b, setup, mesh)
            
            call OutputDT_initialise(output_d, setup, mesh)
            call OutputDT_initialise(output_b, setup, mesh)
            
            call parameters_derived_type_to_matrix(parameters_d, parameters_d_matrix)
            
            parameters_d_matrix = 1._sp
            
            call matrix_to_parameters_derived_type(parameters_d_matrix, parameters_d)
            
            states = init_states
            cost = 0._sp
            cost_d = 0._sp
            
            call forward_d(setup, mesh, input_data, parameters, &
            & parameters_d, states, states_d, output, output_d, cost, cost_d)
            
            states = init_states
            cost = 0._sp
            cost_b = 1._sp
            
            call forward_b(setup, mesh, input_data, parameters, &
            & parameters_b, states, states_b, output, output_b, cost, cost_b)
            
            call parameters_derived_type_to_matrix(parameters_b, parameters_b_matrix)
            
            !% cost_b est reset à 0 dans l'adjoint ...
            sp1 = 1._sp * cost_d
            
            sp2 = sum(parameters_b_matrix * parameters_d_matrix)
            
            print*, 'sp1 ', sp1, 'sp2 ', sp2, 'err rel ', (sp1 - sp2) / sp1
        
        end subroutine scalar_product_test
        
        !% Calling forward from forward/forward.f90
        !% Calling forward_b from forward/forward_b.f90
        subroutine gradient_test_adj(setup, mesh, input_data, parameters, states, output, cost)
        
            implicit none
            
            type(SetupDT), intent(inout) :: setup
            type(MeshDT), intent(inout) :: mesh
            type(Input_DataDT), intent(inout) :: input_data
            type(ParametersDT), intent(inout) :: parameters
            type(StatesDT), intent(inout) :: states
            type(OutputDT), intent(inout) :: output
            real(sp), intent(inout) :: cost
            
            type(ParametersDT) :: parameters_b, init_parameters
            type(StatesDT) :: states_b, init_states
            type(OutputDT) :: output_b
            real(sp) :: cost_b
            real(sp), dimension(mesh%nrow, mesh%ncol, np) :: parameters_matrix, parameters_b_matrix
            real(sp) :: jk, jdk, dk, a, ia
            integer :: n
            
            init_states = states
            init_parameters = parameters
            dk = 10._sp
            
            call ParametersDT_initialise(parameters_b, setup, mesh)
            call StatesDT_initialise(states_b, setup, mesh)
            call OutputDT_initialise(output_b, setup, mesh)
            
            call forward(setup, mesh, input_data, parameters, states, output, cost)
            
            jk = cost
            
            states = init_states
            cost = 0._sp
            cost_b = 1._sp
            
            call forward_b(setup, mesh, input_data, parameters, &
            & parameters_b, states, states_b, output, output_b, cost, cost_b)
            
            call parameters_derived_type_to_matrix(parameters_b, parameters_b_matrix)

            do n=0, 15
            
                a = 2._sp ** (- n)
            
                call parameters_derived_type_to_matrix(init_parameters, parameters_matrix)
                
                parameters_matrix = parameters_matrix + a * dk
                
                call matrix_to_parameters_derived_type(parameters_matrix, parameters)
                
                states = init_states
                
                call forward(setup, mesh, input_data, parameters, states, output, cost)
                
                jdk = cost
                
                ia = (jdk - jk) / (a * sum(parameters_b_matrix * dk)) 
                
                print*, "n ", n, "a ", a,  "ia ", abs(ia - 1)
            
            end do
        
        end subroutine gradient_test_adj
        
        

end module mw_validate
