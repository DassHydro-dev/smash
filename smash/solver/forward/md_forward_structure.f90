!%      This module `md_forward_structure` encapsulates all SMASH forward_structure.
!%      This module is differentiated.
!%
!%      contains
!%
!%      [1] gr_a_forward
!%      [2] gr_b_forward
!%      [2] gr_c_forward
!%      [2] gr_d_forward
!%      [3] vic_a_forward

module md_forward_structure

    use md_constant !% only: sp
    use mwd_setup !% only: SetupDT
    use mwd_mesh !% only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_parameters !% only: ParametersDT
    use mwd_states !% only: StatesDT
    use mwd_output !% only: OutputDT
    use md_gr_operator !% only: gr_interception, gr_production, gr_exchange, &
    !% & gr_transfer
    use md_vic_operator !% only: vic_infiltration, vic_vertical_transfer, vic_interflow, vic_baseflow
    use md_routing_operator !% only: upstream_discharge, linear_routing

    implicit none

contains

    subroutine gr_a_forward(setup, mesh, input_data, parameters, states, output)

        implicit none

        !% =================================================================================================================== %!
        !%   Derived Type Variables (shared)
        !% =================================================================================================================== %!

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(in) :: parameters
        type(StatesDT), intent(inout) :: states
        type(OutputDT), intent(inout) :: output

        !% =================================================================================================================== %!
        !%   Local Variables (private)
        !% =================================================================================================================== %!
        real(sp), dimension(mesh%nrow, mesh%ncol) :: q
        real(sp) :: prcp, pet, ei, pn, en, pr, perc, l, prr, prd, &
        & qr, qd, qt, qup, qrout
        integer :: t, i, row, col, k, g

        !% =================================================================================================================== %!
        !%   Begin subroutine
        !% =================================================================================================================== %!

        do t = 1, setup%ntime_step !% [ DO TIME ]

            do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

                !% =============================================================================================================== %!
                !%   Local Variables Initialisation for time step (t) and cell (i)
                !% =============================================================================================================== %!

                ei = 0._sp
                pn = 0._sp
                en = 0._sp
                pr = 0._sp
                perc = 0._sp
                l = 0._sp
                prr = 0._sp
                prd = 0._sp
                qr = 0._sp
                qd = 0._sp
                qup = 0._sp
                qrout = 0._sp

                !% =========================================================================================================== %!
                !%   Cell indice (i) to Cell indices (row, col) following an increasing order of flow accumulation
                !% =========================================================================================================== %!

                if (mesh%path(1, i) .gt. 0 .and. mesh%path(2, i) .gt. 0) then !% [ IF PATH ]

                    row = mesh%path(1, i)
                    col = mesh%path(2, i)
                    if (setup%sparse_storage) k = mesh%rowcol_to_ind_sparse(row, col)

                    !% ======================================================================================================= %!
                    !%   Global/Local active cell
                    !% ======================================================================================================= %!

                    if (mesh%active_cell(row, col) .eq. 1 .and. mesh%local_active_cell(row, col) .eq. 1) then !% [ IF ACTIVE CELL ]

                        if (setup%sparse_storage) then

                            prcp = input_data%sparse_prcp(k, t)
                            pet = input_data%sparse_pet(k, t)

                        else

                            prcp = input_data%prcp(row, col, t)
                            pet = input_data%pet(row, col, t)

                        end if

                        if (prcp .ge. 0 .and. pet .ge. 0) then !% [ IF PRCP GAP ]

                            !% =============================================================================================== %!
                            !%   Interception module
                            !% =============================================================================================== %!

                            ei = min(pet, prcp)

                            pn = max(0._sp, prcp - ei)

                            en = pet - ei

                            !% =============================================================================================== %!
                            !%   Production module
                            !% =============================================================================================== %!

                            call gr_production(pn, en, parameters%cp(row, col), 1000._sp, &
                            & states%hp(row, col), pr, perc)

                            !% =============================================================================================== %!
                            !%   Exchange module
                            !% =============================================================================================== %!

                            call gr_exchange(parameters%exc(row, col), states%hft(row, col), l)

                        end if !% [ END IF PRCP GAP ]

                        !% =================================================================================================== %!
                        !%   Transfer module
                        !% =================================================================================================== %!

                        prr = 0.9_sp*(pr + perc) + l
                        prd = 0.1_sp*(pr + perc)

                        call gr_transfer(5._sp, prcp, prr, parameters%cft(row, col), states%hft(row, col), qr)

                        qd = max(0._sp, prd + l)

                        qt = (qr + qd)

                        !% =================================================================================================== %!
                        !%   Routing module
                        !% =================================================================================================== %!

                        call upstream_discharge(setup%dt, mesh%dx, mesh%nrow,&
                        &  mesh%ncol, mesh%flwdir, mesh%flwacc, row, col, q, qup)

                        call linear_routing(setup%dt, qup, parameters%lr(row, col), states%hlr(row, col), qrout)

                        q(row, col) = (qt + qrout*real(mesh%flwacc(row, col) - 1))&
                                     & *mesh%dx*mesh%dx*0.001_sp/setup%dt

                        !% =================================================================================================== %!
                        !%   Store simulated net rainfall on domain (optional)
                        !%   The net rainfall over a surface is a fictitious quantity that corresponds to
                        !%   the part of the rainfall water depth that actually causes runoff.
                        !% =================================================================================================== %!

                        if (setup%save_net_prcp_domain) then

                            if (setup%sparse_storage) then

                                output%sparse_net_prcp_domain(k, t) = qt

                            else

                                output%net_prcp_domain(row, col, t) = qt

                            end if

                        end if

                        !% =================================================================================================== %!
                        !%   Store simulated discharge on domain (optional)
                        !% =================================================================================================== %!

                        if (setup%save_qsim_domain) then

                            if (setup%sparse_storage) then

                                output%sparse_qsim_domain(k, t) = q(row, col)

                            else

                                output%qsim_domain(row, col, t) = q(row, col)

                            end if

                        end if

                    end if !% [ END IF ACTIVE CELL ]

                end if !% [ END IF PATH ]

            end do !% [ END DO SPACE ]

            !% =============================================================================================================== %!
            !%   Store simulated discharge at gauge
            !% =============================================================================================================== %!

            do g = 1, mesh%ng

                output%qsim(g, t) = q(mesh%gauge_pos(g, 1), mesh%gauge_pos(g, 2))

            end do

        end do !% [ END DO TIME ]

    end subroutine gr_a_forward

    subroutine gr_b_forward(setup, mesh, input_data, parameters, states, output)

        implicit none

        !% =================================================================================================================== %!
        !%   Derived Type Variables (shared)
        !% =================================================================================================================== %!

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(in) :: parameters
        type(StatesDT), intent(inout) :: states
        type(OutputDT), intent(inout) :: output

        !% =================================================================================================================== %!
        !%   Local Variables (private)
        !% =================================================================================================================== %!
        real(sp), dimension(mesh%nrow, mesh%ncol) :: q
        real(sp) :: prcp, pet, ei, pn, en, pr, perc, l, prr, prd, &
        & qr, qd, qt, qup, qrout
        integer :: t, i, row, col, k, g

        !% =================================================================================================================== %!
        !%   Begin subroutine
        !% =================================================================================================================== %!

        do t = 1, setup%ntime_step !% [ DO TIME ]

            do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

                !% =============================================================================================================== %!
                !%   Local Variables Initialisation for time step (t) and cell (i)
                !% =============================================================================================================== %!

                ei = 0._sp
                pn = 0._sp
                en = 0._sp
                pr = 0._sp
                perc = 0._sp
                l = 0._sp
                prr = 0._sp
                prd = 0._sp
                qr = 0._sp
                qd = 0._sp
                qup = 0._sp
                qrout = 0._sp

                !% =========================================================================================================== %!
                !%   Cell indice (i) to Cell indices (row, col) following an increasing order of flow accumulation
                !% =========================================================================================================== %!

                if (mesh%path(1, i) .gt. 0 .and. mesh%path(2, i) .gt. 0) then !% [ IF PATH ]

                    row = mesh%path(1, i)
                    col = mesh%path(2, i)
                    if (setup%sparse_storage) k = mesh%rowcol_to_ind_sparse(row, col)

                    !% ======================================================================================================= %!
                    !%   Global/Local active cell
                    !% ======================================================================================================= %!

                    if (mesh%active_cell(row, col) .eq. 1 .and. mesh%local_active_cell(row, col) .eq. 1) then !% [ IF ACTIVE CELL ]

                        if (setup%sparse_storage) then

                            prcp = input_data%sparse_prcp(k, t)
                            pet = input_data%sparse_pet(k, t)

                        else

                            prcp = input_data%prcp(row, col, t)
                            pet = input_data%pet(row, col, t)

                        end if

                        if (prcp .ge. 0 .and. pet .ge. 0) then !% [ IF PRCP GAP ]

                            !% =============================================================================================== %!
                            !%   Interception module
                            !% =============================================================================================== %!

                            call gr_interception(prcp, pet, parameters%ci(row, col), states%hi(row, col), pn, ei)

                            en = pet - ei

                            !% =============================================================================================== %!
                            !%   Production module
                            !% =============================================================================================== %!

                            call gr_production(pn, en, parameters%cp(row, col), 1000._sp, &
                            & states%hp(row, col), pr, perc)

                            !% =============================================================================================== %!
                            !%   Exchange module
                            !% =============================================================================================== %!

                            call gr_exchange(parameters%exc(row, col), states%hft(row, col), l)

                        end if !% [ END IF PRCP GAP ]

                        !% =================================================================================================== %!
                        !%   Transfer module
                        !% =================================================================================================== %!

                        prr = 0.9_sp*(pr + perc) + l
                        prd = 0.1_sp*(pr + perc)

                        call gr_transfer(5._sp, prcp, prr, parameters%cft(row, col), states%hft(row, col), qr)

                        qd = max(0._sp, prd + l)

                        qt = (qr + qd)

                        !% =================================================================================================== %!
                        !%   Routing module
                        !% =================================================================================================== %!

                        call upstream_discharge(setup%dt, mesh%dx, mesh%nrow,&
                        &  mesh%ncol, mesh%flwdir, mesh%flwacc, row, col, q, qup)

                        call linear_routing(setup%dt, qup, parameters%lr(row, col), states%hlr(row, col), qrout)

                        q(row, col) = (qt + qrout*real(mesh%flwacc(row, col) - 1))&
                                     & *mesh%dx*mesh%dx*0.001_sp/setup%dt

                        !% =================================================================================================== %!
                        !%   Store simulated net rainfall on domain (optional)
                        !%   The net rainfall over a surface is a fictitious quantity that corresponds to
                        !%   the part of the rainfall water depth that actually causes runoff.
                        !% =================================================================================================== %!

                        if (setup%save_net_prcp_domain) then

                            if (setup%sparse_storage) then

                                output%sparse_net_prcp_domain(k, t) = qt

                            else

                                output%net_prcp_domain(row, col, t) = qt

                            end if

                        end if

                        !% =================================================================================================== %!
                        !%   Store simulated discharge on domain (optional)
                        !% =================================================================================================== %!

                        if (setup%save_qsim_domain) then

                            if (setup%sparse_storage) then

                                output%sparse_qsim_domain(k, t) = q(row, col)

                            else

                                output%qsim_domain(row, col, t) = q(row, col)

                            end if

                        end if

                    end if !% [ END IF ACTIVE CELL ]

                end if !% [ END IF PATH ]

            end do !% [ END DO SPACE ]

            !% =============================================================================================================== %!
            !%   Store simulated discharge at gauge
            !% =============================================================================================================== %!

            do g = 1, mesh%ng

                output%qsim(g, t) = q(mesh%gauge_pos(g, 1), mesh%gauge_pos(g, 2))

            end do

        end do !% [ END DO TIME ]

    end subroutine gr_b_forward

    subroutine gr_c_forward(setup, mesh, input_data, parameters, states, output)

        implicit none

        !% =================================================================================================================== %!
        !%   Derived Type Variables (shared)
        !% =================================================================================================================== %!

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(in) :: parameters
        type(StatesDT), intent(inout) :: states
        type(OutputDT), intent(inout) :: output

        !% =================================================================================================================== %!
        !%   Local Variables (private)
        !% =================================================================================================================== %!
        real(sp), dimension(mesh%nrow, mesh%ncol) :: q
        real(sp) :: prcp, pet, ei, pn, en, pr, perc, l, prr, prl, prd, &
        & qr, ql, qd, qt, qup, qrout
        integer :: t, i, row, col, k, g

        !% =================================================================================================================== %!
        !%   Begin subroutine
        !% =================================================================================================================== %!

        do t = 1, setup%ntime_step !% [ DO TIME ]

            do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

                !% =============================================================================================================== %!
                !%   Local Variables Initialisation for time step (t) and cell (i)
                !% =============================================================================================================== %!

                ei = 0._sp
                pn = 0._sp
                en = 0._sp
                pr = 0._sp
                perc = 0._sp
                l = 0._sp
                prr = 0._sp
                prl = 0._sp
                prd = 0._sp
                qr = 0._sp
                ql = 0._sp
                qd = 0._sp
                qup = 0._sp
                qrout = 0._sp

                !% =========================================================================================================== %!
                !%   Cell indice (i) to Cell indices (row, col) following an increasing order of flow accumulation
                !% =========================================================================================================== %!

                if (mesh%path(1, i) .gt. 0 .and. mesh%path(2, i) .gt. 0) then !% [ IF PATH ]

                    row = mesh%path(1, i)
                    col = mesh%path(2, i)
                    if (setup%sparse_storage) k = mesh%rowcol_to_ind_sparse(row, col)

                    !% ======================================================================================================= %!
                    !%   Global/Local active cell
                    !% ======================================================================================================= %!

                    if (mesh%active_cell(row, col) .eq. 1 .and. mesh%local_active_cell(row, col) .eq. 1) then !% [ IF ACTIVE CELL ]

                        if (setup%sparse_storage) then

                            prcp = input_data%sparse_prcp(k, t)
                            pet = input_data%sparse_pet(k, t)

                        else

                            prcp = input_data%prcp(row, col, t)
                            pet = input_data%pet(row, col, t)

                        end if

                        if (prcp .ge. 0 .and. pet .ge. 0) then !% [ IF PRCP GAP ]

                            !% =============================================================================================== %!
                            !%   Interception module
                            !% =============================================================================================== %!

                            call gr_interception(prcp, pet, parameters%ci(row, col), states%hi(row, col), pn, ei)

                            en = pet - ei

                            !% =============================================================================================== %!
                            !%   Production module
                            !% =============================================================================================== %!

                            call gr_production(pn, en, parameters%cp(row, col), 1000._sp, &
                            & states%hp(row, col), pr, perc)

                            !% =============================================================================================== %!
                            !%   Exchange module
                            !% =============================================================================================== %!

                            call gr_exchange(parameters%exc(row, col), states%hft(row, col), l)

                        end if !% [ END IF PRCP GAP ]

                        !% =================================================================================================== %!
                        !%   Transfer module
                        !% =================================================================================================== %!

                        prr = 0.9_sp*0.6_sp*(pr + perc) + l
                        prl = 0.9_sp*0.4_sp*(pr + perc)
                        prd = 0.1_sp*(pr + perc)

                        call gr_transfer(5._sp, prcp, prr, parameters%cft(row, col), states%hft(row, col), qr)

                        call gr_transfer(5._sp, prcp, prl, parameters%cst(row, col), states%hst(row, col), ql)

                        qd = max(0._sp, prd + l)

                        qt = (qr + ql + qd)

                        !% =================================================================================================== %!
                        !%   Routing module
                        !% =================================================================================================== %!

                        call upstream_discharge(setup%dt, mesh%dx, mesh%nrow,&
                        &  mesh%ncol, mesh%flwdir, mesh%flwacc, row, col, q, qup)

                        call linear_routing(setup%dt, qup, parameters%lr(row, col), states%hlr(row, col), qrout)

                        q(row, col) = (qt + qrout*real(mesh%flwacc(row, col) - 1))&
                                     & *mesh%dx*mesh%dx*0.001_sp/setup%dt

                        !% =================================================================================================== %!
                        !%   Store simulated net rainfall on domain (optional)
                        !%   The net rainfall over a surface is a fictitious quantity that corresponds to
                        !%   the part of the rainfall water depth that actually causes runoff.
                        !% =================================================================================================== %!

                        if (setup%save_net_prcp_domain) then

                            if (setup%sparse_storage) then

                                output%sparse_net_prcp_domain(k, t) = qt

                            else

                                output%net_prcp_domain(row, col, t) = qt

                            end if

                        end if

                        !% =================================================================================================== %!
                        !%   Store simulated discharge on domain (optional)
                        !% =================================================================================================== %!

                        if (setup%save_qsim_domain) then

                            if (setup%sparse_storage) then

                                output%sparse_qsim_domain(k, t) = q(row, col)

                            else

                                output%qsim_domain(row, col, t) = q(row, col)

                            end if

                        end if

                    end if !% [ END IF ACTIVE CELL ]

                end if !% [ END IF PATH ]

            end do !% [ END DO SPACE ]

            !% =============================================================================================================== %!
            !%   Store simulated discharge at gauge
            !% =============================================================================================================== %!

            do g = 1, mesh%ng

                output%qsim(g, t) = q(mesh%gauge_pos(g, 1), mesh%gauge_pos(g, 2))

            end do

        end do !% [ END DO TIME ]

    end subroutine gr_c_forward

    subroutine gr_d_forward(setup, mesh, input_data, parameters, states, output)

        implicit none

        !% =================================================================================================================== %!
        !%   Derived Type Variables (shared)
        !% =================================================================================================================== %!

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(in) :: parameters
        type(StatesDT), intent(inout) :: states
        type(OutputDT), intent(inout) :: output

        !% =================================================================================================================== %!
        !%   Local Variables (private)
        !% =================================================================================================================== %!
        real(sp), dimension(mesh%nrow, mesh%ncol) :: q
        real(sp) :: prcp, pet, ei, pn, en, pr, perc, prr, qr, qt, qup, qrout
        integer :: t, i, row, col, k, g

        !% =================================================================================================================== %!
        !%   Begin subroutine
        !% =================================================================================================================== %!

        do t = 1, setup%ntime_step !% [ DO TIME ]

            do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

                !% =============================================================================================================== %!
                !%   Local Variables Initialisation for time step (t) and cell (i)
                !% =============================================================================================================== %!

                ei = 0._sp
                pn = 0._sp
                en = 0._sp
                pr = 0._sp
                perc = 0._sp
                prr = 0._sp
                qr = 0._sp
                qup = 0._sp
                qrout = 0._sp

                !% =========================================================================================================== %!
                !%   Cell indice (i) to Cell indices (row, col) following an increasing order of flow accumulation
                !% =========================================================================================================== %!

                if (mesh%path(1, i) .gt. 0 .and. mesh%path(2, i) .gt. 0) then !% [ IF PATH ]

                    row = mesh%path(1, i)
                    col = mesh%path(2, i)
                    if (setup%sparse_storage) k = mesh%rowcol_to_ind_sparse(row, col)

                    !% ======================================================================================================= %!
                    !%   Global/Local active cell
                    !% ======================================================================================================= %!

                    if (mesh%active_cell(row, col) .eq. 1 .and. mesh%local_active_cell(row, col) .eq. 1) then !% [ IF ACTIVE CELL ]

                        if (setup%sparse_storage) then

                            prcp = input_data%sparse_prcp(k, t)
                            pet = input_data%sparse_pet(k, t)

                        else

                            prcp = input_data%prcp(row, col, t)
                            pet = input_data%pet(row, col, t)

                        end if

                        if (prcp .ge. 0 .and. pet .ge. 0) then !% [ IF PRCP GAP ]

                            !% =============================================================================================== %!
                            !%   Interception module
                            !% =============================================================================================== %!

                            ei = min(pet, prcp)

                            pn = max(0._sp, prcp - ei)

                            en = pet - ei

                            !% =============================================================================================== %!
                            !%   Production module
                            !% =============================================================================================== %!

                            call gr_production(pn, en, parameters%cp(row, col), 1000._sp, &
                            & states%hp(row, col), pr, perc)

                        end if !% [ END IF PRCP GAP ]

                        !% =================================================================================================== %!
                        !%   Transfer module
                        !% =================================================================================================== %!

                        prr = pr + perc

                        call gr_transfer(5._sp, prcp, prr, parameters%cft(row, col), states%hft(row, col), qr)

                        qt = qr

                        !% =================================================================================================== %!
                        !%   Routing module
                        !% =================================================================================================== %!

                        call upstream_discharge(setup%dt, mesh%dx, mesh%nrow,&
                        &  mesh%ncol, mesh%flwdir, mesh%flwacc, row, col, q, qup)

                        call linear_routing(setup%dt, qup, parameters%lr(row, col), states%hlr(row, col), qrout)

                        q(row, col) = (qt + qrout*real(mesh%flwacc(row, col) - 1))&
                                     & *mesh%dx*mesh%dx*0.001_sp/setup%dt

                        !% =================================================================================================== %!
                        !%   Store simulated net rainfall on domain (optional)
                        !%   The net rainfall over a surface is a fictitious quantity that corresponds to
                        !%   the part of the rainfall water depth that actually causes runoff.
                        !% =================================================================================================== %!

                        if (setup%save_net_prcp_domain) then

                            if (setup%sparse_storage) then

                                output%sparse_net_prcp_domain(k, t) = qt

                            else

                                output%net_prcp_domain(row, col, t) = qt

                            end if

                        end if

                        !% =================================================================================================== %!
                        !%   Store simulated discharge on domain (optional)
                        !% =================================================================================================== %!

                        if (setup%save_qsim_domain) then

                            if (setup%sparse_storage) then

                                output%sparse_qsim_domain(k, t) = q(row, col)

                            else

                                output%qsim_domain(row, col, t) = q(row, col)

                            end if

                        end if

                    end if !% [ END IF ACTIVE CELL ]

                end if !% [ END IF PATH ]

            end do !% [ END DO SPACE ]

            !% =============================================================================================================== %!
            !%   Store simulated discharge at gauge
            !% =============================================================================================================== %!

            do g = 1, mesh%ng

                output%qsim(g, t) = q(mesh%gauge_pos(g, 1), mesh%gauge_pos(g, 2))

            end do

        end do !% [ END DO TIME ]

    end subroutine gr_d_forward

    subroutine vic_a_forward(setup, mesh, input_data, parameters, states, output)

        implicit none

        !% =================================================================================================================== %!
        !%   Derived Type Variables (shared)
        !% =================================================================================================================== %!

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(in) :: parameters
        type(StatesDT), intent(inout) :: states
        type(OutputDT), intent(inout) :: output

        !% =================================================================================================================== %!
        !%   Local Variables (private)
        !% =================================================================================================================== %!

        real(sp), dimension(mesh%nrow, mesh%ncol) :: q
        real(sp) :: prcp, pet, runoff, qi, qb, qt, qup, qrout
        integer :: t, i, row, col, k, g

        !% =================================================================================================================== %!
        !%   Begin subroutine
        !% =================================================================================================================== %!

        do t = 1, setup%ntime_step !% [ DO TIME ]

            do i = 1, mesh%nrow*mesh%ncol !% [ DO SPACE ]

                !% =============================================================================================================== %!
                !%   Local Variables Initialisation for time step (t) and cell (i)
                !% =============================================================================================================== %!

                runoff = 0._sp
                qi = 0._sp
                qb = 0._sp
                qt = 0._sp
                qup = 0._sp
                qrout = 0._sp

                !% =========================================================================================================== %!
                !%   Cell indice (i) to Cell indices (row, col) following an increasing order of flow accumulation
                !% =========================================================================================================== %!

                if (mesh%path(1, i) .gt. 0 .and. mesh%path(2, i) .gt. 0) then !% [ IF PATH ]

                    row = mesh%path(1, i)
                    col = mesh%path(2, i)
                    if (setup%sparse_storage) k = mesh%rowcol_to_ind_sparse(row, col)

                    !% ======================================================================================================= %!
                    !%   Global/Local active cell
                    !% ======================================================================================================= %!

                    if (mesh%active_cell(row, col) .eq. 1 .and. mesh%local_active_cell(row, col) .eq. 1) then !% [ IF ACTIVE CELL ]

                        if (setup%sparse_storage) then

                            prcp = input_data%sparse_prcp(k, t)
                            pet = input_data%sparse_pet(k, t)

                        else

                            prcp = input_data%prcp(row, col, t)
                            pet = input_data%pet(row, col, t)

                        end if

                        if (prcp .ge. 0 .and. pet .ge. 0) then !% [ IF PRCP GAP ]

                            !% =============================================================================================== %!
                            !%   Infiltration module
                            !% =============================================================================================== %!

                            call vic_infiltration(prcp, parameters%cusl1(row, col), parameters%cusl2(row, col), &
                            & parameters%b(row, col), states%husl1(row, col), states%husl2(row, col), &
                            & runoff)

                            !% =============================================================================================== %!
                            !%   Vertical transfer module
                            !% =============================================================================================== %!

                            call vic_vertical_transfer(pet, parameters%cusl1(row, col), parameters%cusl2(row, col), &
                            & parameters%clsl(row, col), parameters%ks(row, col), states%husl1(row, col), &
                            & states%husl2(row, col), states%hlsl(row, col))

                        end if !% [ END IF PRCP GAP ]

                        !% =================================================================================================== %!
                        !%   Horizontal transfer module
                        !% =================================================================================================== %!

                        call vic_interflow(5._sp, parameters%cusl2(row, col), states%husl2(row, col), qi)

                        call vic_baseflow(parameters%clsl(row, col), parameters%ds(row, col), &
                        & parameters%dsm(row, col), parameters%ws(row, col), states%hlsl(row, col), qb)

                        qt = (runoff + qi + qb)

                        !% =================================================================================================== %!
                        !%   Routing module
                        !% =================================================================================================== %!

                        call upstream_discharge(setup%dt, mesh%dx, mesh%nrow,&
                        &  mesh%ncol, mesh%flwdir, mesh%flwacc, row, col, q, qup)

                        call linear_routing(setup%dt, qup, parameters%lr(row, col), states%hlr(row, col), qrout)

                        q(row, col) = (qt + qrout*real(mesh%flwacc(row, col) - 1))&
                                     & *mesh%dx*mesh%dx*0.001_sp/setup%dt

                        !% =================================================================================================== %!
                        !%   Store simulated net rainfall on domain (optional)
                        !%   The net rainfall over a surface is a fictitious quantity that corresponds to
                        !%   the part of the rainfall water depth that actually causes runoff.
                        !% =================================================================================================== %!

                        if (setup%save_net_prcp_domain) then

                            if (setup%sparse_storage) then

                                output%sparse_net_prcp_domain(k, t) = qt

                            else

                                output%net_prcp_domain(row, col, t) = qt

                            end if

                        end if

                        !% =================================================================================================== %!
                        !%   Store simulated discharge on domain (optional)
                        !% =================================================================================================== %!

                        if (setup%save_qsim_domain) then

                            if (setup%sparse_storage) then

                                output%sparse_qsim_domain(k, t) = q(row, col)

                            else

                                output%qsim_domain(row, col, t) = q(row, col)

                            end if

                        end if

                    end if !% [ END IF ACTIVE CELL ]

                end if !% [ END IF PATH ]

            end do !% [ END DO SPACE ]

            !% =============================================================================================================== %!
            !%   Store simulated discharge at gauge
            !% =============================================================================================================== %!

            do g = 1, mesh%ng

                output%qsim(g, t) = q(mesh%gauge_pos(g, 1), mesh%gauge_pos(g, 2))

            end do

        end do !% [ END DO TIME ]

    end subroutine vic_a_forward

end module md_forward_structure
