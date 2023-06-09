!%    This module `mwd_cost` encapsulates all SMASH cost (type, subroutines, functions)
!%    This module is wrapped and differentiated.
!%
!%      contains
!%
!%      [1]  compute_jobs
!%      [2]  compute_jreg
!%      [3]  compute_cost
!%      [4]  nse
!%      [5]  kge_components
!%      [6]  kge
!%      [7]  se
!%      [8]  rmse
!%      [9]  logarithmic
!%      [10] reg_prior

module mwd_cost

    use md_constant !% only: sp, dp, lchar, GNP, GNS
    use mwd_setup  !% only: SetupDT
    use mwd_mesh   !%only: MeshDT
    use mwd_input_data !% only: Input_DataDT
    use mwd_parameters !% only: ParametersDT, Hyper_ParametersDT
    use mwd_states !% only: StatesDT, Hyper_StatesDT
    use mwd_output !% only: OutputDT
    use mwd_parameters_manipulation !% only: get_parameters
    use mwd_states_manipulation !%only: get_states

    implicit none

    public :: compute_jobs, compute_jreg, compute_cost

contains

    !% Way to improve: try do one single for loop to compute all cost function
    !% ATM, each cost function are computed separately with n for loop
    subroutine compute_jobs(setup, mesh, input_data, output, jobs)

        !% Notes
        !% -----
        !%
        !% Jobs computation subroutine
        !%
        !% Given SetupDT, MeshDT, Input_DataDT, OutputDT,
        !% it returns the result of Jobs computation
        !%
        !% Jobs = f(Q*,Q)
        !%
        !% See Also
        !% --------
        !% nse
        !% kge
        !% se
        !% rmse
        !% logarithmic

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(OutputDT), intent(inout) :: output
        real(sp), intent(out) :: jobs

        real(sp), dimension(setup%ntime_step - setup%optimize%optimize_start_step + 1) :: po, qo, qs
        real(sp), dimension(mesh%ng) :: arr_gauge_jobs
        real(sp) :: imd, j_imd, gauge_jobs
        integer :: g, row, col, j, arr_size

        jobs = 0._sp

        arr_gauge_jobs = 0._sp

        arr_size = 0

        do g = 1, mesh%ng

            gauge_jobs = 0._sp

            if ((setup%optimize%wgauge(g) .gt. 0._sp) .or. (setup%optimize%wgauge(g) .lt. 0._sp)) then

                po = input_data%mean_prcp(g, setup%optimize%optimize_start_step:setup%ntime_step)

                qs = output%qsim(g, setup%optimize%optimize_start_step:setup%ntime_step) &
                    & *setup%dt/mesh%area(g)*1e3_sp

                row = mesh%gauge_pos(g, 1)
                col = mesh%gauge_pos(g, 2)

                qo = input_data%qobs(g, setup%optimize%optimize_start_step:setup%ntime_step) &
                    & *setup%dt/(real(mesh%flwacc(row, col))*mesh%dx*mesh%dx) &
                    & *1e3_sp

                do j = 1, setup%optimize%njf

                    if (any(qo .ge. 0._sp)) then

                        select case (setup%optimize%jobs_fun(j))

                        case ("nse")

                            j_imd = nse(qo, qs)

                        case ("kge")

                            j_imd = kge(qo, qs)

                        case ("kge2")

                            imd = kge(qo, qs)
                            j_imd = imd*imd

                        case ("se")

                            j_imd = se(qo, qs)

                        case ("rmse")

                            j_imd = rmse(qo, qs)

                        case ("logarithmic")

                            j_imd = logarithmic(qo, qs)

                        case ("Crc", "Cfp2", "Cfp10", "Cfp50", "Cfp90", "Erc", "Elt", "Epf") ! CASE OF SIGNATURES

                            j_imd = signature(po, qo, qs, &
                            & setup%optimize%mask_event(g, setup%optimize%optimize_start_step:setup%ntime_step), &
                            & setup%optimize%jobs_fun(j))

                        end select

                    end if

                    gauge_jobs = gauge_jobs + setup%optimize%wjobs_fun(j)*j_imd

                end do

                if (setup%optimize%wgauge(g) .gt. 0._sp) then

                    jobs = jobs + setup%optimize%wgauge(g)*gauge_jobs

                else
                    arr_size = arr_size + 1

                    arr_gauge_jobs(arr_size) = gauge_jobs

                end if

            end if

        end do

        if (arr_size .gt. 0) jobs = quantile(arr_gauge_jobs(1:arr_size), 0.5)

    end subroutine compute_jobs

    !% WIP
    subroutine compute_jreg(setup, mesh, input_data, parameters, parameters_bgd, states, states_bgd, jreg)

        !% Notes
        !% -----
        !%
        !% Jreg computation subroutine
        !%
        !% Given SetupDT, MeshDT, ParametersDT, ParametersDT_bgd, StatesDT, STatesDT_bgd,
        !% it returns the result of Jreg computation
        !%
        !% Jreg = f(theta_bgd,theta)
        !%
        !% See Also
        !% --------
        !% reg_prior

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(in) :: parameters, parameters_bgd
        type(StatesDT), intent(in) :: states, states_bgd
        real(sp), intent(inout) :: jreg

        real(sp) :: parameters_jreg, states_jreg
        real(sp), dimension(mesh%nrow, mesh%ncol, GNP) :: parameters_matrix, parameters_bgd_matrix
        real(sp), dimension(mesh%nrow, mesh%ncol, GNS) :: states_matrix, states_bgd_matrix

        integer :: i

        call get_parameters(mesh, parameters, parameters_matrix)
        call get_parameters(mesh, parameters_bgd, parameters_bgd_matrix)

        call get_states(mesh, states, states_matrix)
        call get_states(mesh, states_bgd, states_bgd_matrix)

        jreg = 0._sp
        parameters_jreg = 0._sp
        states_jreg = 0._sp

        do i = 1, setup%optimize%njr

            select case (setup%optimize%jreg_fun(i))

            case ("prior")

                parameters_jreg = parameters_jreg + setup%optimize%wjreg_fun(i)* &
                & reg_prior(setup, setup%optimize%optim_parameters, parameters_matrix, parameters_bgd_matrix)

                states_jreg = states_jreg + setup%optimize%wjreg_fun(i)*&
                & reg_prior(setup, setup%optimize%optim_states, states_matrix, states_bgd_matrix)

            case ("smoothing")

                parameters_jreg = parameters_jreg + setup%optimize%wjreg_fun(i)**2._sp* &
                & reg_smoothing(setup, mesh, setup%optimize%optim_parameters, parameters_matrix, parameters_bgd_matrix,.true.)

                states_jreg = states_jreg + setup%optimize%wjreg_fun(i)**2._sp* &
                & reg_smoothing(setup, mesh, setup%optimize%optim_states, states_matrix, states_bgd_matrix,.true.)

            case ("hard_smoothing")

                parameters_jreg = parameters_jreg + setup%optimize%wjreg_fun(i)**2._sp* &
                & reg_smoothing(setup, mesh, setup%optimize%optim_parameters, parameters_matrix, parameters_bgd_matrix,.false.)

                states_jreg = states_jreg + setup%optimize%wjreg_fun(i)**2._sp* &
                & reg_smoothing(setup, mesh, setup%optimize%optim_states, states_matrix, states_bgd_matrix,.false.)


            case ("distance_correlation")

                parameters_jreg = parameters_jreg + setup%optimize%wjreg_fun(i)&
                &*distance_correlation_descriptors(&
                &setup, mesh, input_data, "params", GNP, parameters_matrix)

                states_jreg = states_jreg + setup%optimize%wjreg_fun(i)&
                &*distance_correlation_descriptors(&
                &setup, mesh, input_data, "states", GNS, states_matrix)

            end select

        end do

        jreg = parameters_jreg + states_jreg

    end subroutine compute_jreg

    subroutine compute_cost(setup, mesh, input_data, parameters, parameters_bgd, states, states_bgd, output, cost)

        !% Notes
        !% -----
        !%
        !% cost computation subroutine
        !%
        !% Given SetupDT, MeshDT, Input_DataDT, ParametersDT, ParametersDT_bgd, StatesDT, STatesDT_bgd, OutputDT
        !% it returns the result of cost computation
        !%
        !% cost = Jobs + wJreg * Jreg
        !%
        !% See Also
        !% --------
        !% compute_jobs
        !% compute_jreg

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(ParametersDT), intent(inout) :: parameters
        type(ParametersDT), intent(in) :: parameters_bgd
        type(StatesDT), intent(inout) :: states
        type(StatesDT), intent(in) :: states_bgd
        type(OutputDT), intent(inout) :: output
        real(sp), intent(inout) :: cost

        real(sp) :: jobs, jreg

        jobs = 0._sp

        call compute_jobs(setup, mesh, input_data, output, jobs)

        jreg = 0._sp

        if (setup%optimize%denormalize_forward) then

            call normalize_parameters(setup, mesh, parameters)
            call normalize_states(setup, mesh, states)

        end if

        call compute_jreg(setup, mesh, input_data, parameters, parameters_bgd, states, states_bgd, jreg)

        if (setup%optimize%denormalize_forward) then

            call denormalize_parameters(setup, mesh, parameters)
            call denormalize_states(setup, mesh, states)

        end if

        cost = jobs + setup%optimize%wjreg*jreg

        output%cost = cost
        output%cost_jobs = jobs
        output%cost_jreg = jreg

    end subroutine compute_cost

    !% TODO comment and refactorize
    subroutine hyper_compute_cost(setup, mesh, input_data, &
    & hyper_parameters, hyper_parameters_bgd, hyper_states, &
    & hyper_states_bgd, output, cost)

        !% Notes
        !% -----
        !%
        !% cost computation subroutine
        !%
        !% Given SetupDT, MeshDT, Input_DataDT, ParametersDT, ParametersDT_bgd, StatesDT, STatesDT_bgd, OutputDT
        !% it returns the result of cost computation
        !%
        !% cost = Jobs + wJreg * Jreg
        !%
        !% See Also
        !% --------
        !% compute_jobs
        !% compute_jreg

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        type(Hyper_ParametersDT), intent(in) :: hyper_parameters, hyper_parameters_bgd
        type(Hyper_StatesDT), intent(in) :: hyper_states, hyper_states_bgd
        type(OutputDT), intent(inout) :: output
        real(sp), intent(inout) :: cost

        real(sp) :: jobs, jreg

        call compute_jobs(setup, mesh, input_data, output, jobs)

        jreg = 0._sp

        cost = jobs + setup%optimize%wjreg*jreg
        output%cost = cost
        output%cost_jobs = jobs

    end subroutine hyper_compute_cost

    function nse(x, y) result(res)

        !% Notes
        !% -----
        !%
        !% NSE computation function
        !%
        !% Given two single precision array (x, y) of dim(1) and size(n),
        !% it returns the result of NSE computation
        !% num = sum(x**2) - 2 * sum(x*y) + sum(y**2)
        !% den = sum(x**2) - n * mean(x) ** 2
        !% NSE = num / den

        implicit none

        real(sp), dimension(:), intent(in) :: x, y
        real(sp) :: res

        real(sp) :: sum_x, sum_xx, sum_yy, sum_xy, mean_x, num, den
        integer :: i, n

        !% Metric computation
        n = 0
        sum_x = 0._sp
        sum_xx = 0._sp
        sum_yy = 0._sp
        sum_xy = 0._sp

        do i = 1, size(x)

            if (x(i) .ge. 0._sp) then

                n = n + 1
                sum_x = sum_x + x(i)
                sum_xx = sum_xx + (x(i)*x(i))
                sum_yy = sum_yy + (y(i)*y(i))
                sum_xy = sum_xy + (x(i)*y(i))

            end if

        end do

        mean_x = sum_x/n

        !% NSE numerator / denominator
        num = sum_xx - 2*sum_xy + sum_yy
        den = sum_xx - n*mean_x*mean_x

        !% NSE criterion
        res = num/den

    end function nse

    subroutine kge_components(x, y, r, a, b)

        !% Notes
        !% -----
        !%
        !% KGE components computation subroutine
        !%
        !% Given two single precision array (x, y) of dim(1) and size(n),
        !% it returns KGE components r, a, b
        !% r = cov(x,y) / std(y) / std(x)
        !% a = mean(y) / mean(x)
        !% b = std(y) / std(x)

        implicit none

        real(sp), dimension(:), intent(in) :: x, y
        real(sp), intent(inout) :: r, a, b

        real(sp) :: sum_x, sum_y, sum_xx, sum_yy, sum_xy, mean_x, mean_y, &
        & var_x, var_y, cov
        integer :: n, i

        ! Metric computation
        n = 0
        sum_x = 0._sp
        sum_y = 0._sp
        sum_xx = 0._sp
        sum_yy = 0._sp
        sum_xy = 0._sp

        do i = 1, size(x)

            if (x(i) .ge. 0._sp) then

                n = n + 1
                sum_x = sum_x + x(i)
                sum_y = sum_y + y(i)
                sum_xx = sum_xx + (x(i)*x(i))
                sum_yy = sum_yy + (y(i)*y(i))
                sum_xy = sum_xy + (x(i)*y(i))

            end if

        end do

        mean_x = sum_x/n
        mean_y = sum_y/n
        var_x = (sum_xx/n) - (mean_x*mean_x)
        var_y = (sum_yy/n) - (mean_y*mean_y)
        cov = (sum_xy/n) - (mean_x*mean_y)

        ! KGE components (r, alpha, beta)
        r = (cov/sqrt(var_x))/sqrt(var_y)
        a = sqrt(var_y)/sqrt(var_x)
        b = mean_y/mean_x

    end subroutine kge_components

    function kge(x, y) result(res)

        !% Notes
        !% -----
        !%
        !% KGE computation function
        !%
        !% Given two single precision array (x, y) of dim(1) and size(n),
        !% it returns the result of KGE computation
        !% KGE = sqrt((1 - r) ** 2 + (1 - a) ** 2 + (1 - b) ** 2)
        !%
        !% See Also
        !% --------
        !% kge_components

        implicit none

        real(sp), dimension(:), intent(in) :: x, y
        real(sp) :: res

        real(sp) :: r, a, b

        call kge_components(x, y, r, a, b)

        ! KGE criterion
        res = sqrt(&
        & (r - 1)*(r - 1) + (b - 1)*(b - 1) + (a - 1)*(a - 1) &
        & )

    end function kge

    function se(x, y) result(res)

        !% Notes
        !% -----
        !%
        !% Square Error (SE) computation function
        !%
        !% Given two single precision array (x, y) of dim(1) and size(n),
        !% it returns the result of SE computation
        !% SE = sum((x - y) ** 2)

        implicit none

        real(sp), dimension(:), intent(in) :: x, y
        real(sp) :: res

        integer :: i

        res = 0._sp

        do i = 1, size(x)

            if (x(i) .ge. 0._sp) then

                res = res + (x(i) - y(i))*(x(i) - y(i))

            end if

        end do

    end function se

    function rmse(x, y) result(res)

        !% Notes
        !% -----
        !%
        !% Root Mean Square Error (RMSE) computation function
        !%
        !% Given two single precision array (x, y) of dim(1) and size(n),
        !% it returns the result of SE computation
        !% RMSE = sqrt(SE / n)
        !%
        !% See Also
        !% --------
        !% se

        implicit none

        real(sp), dimension(:), intent(in) :: x, y
        real(sp) :: res

        integer :: i, n

        n = 0

        do i = 1, size(x)

            if (x(i) .ge. 0._sp) then

                n = n + 1

            end if

        end do

        res = sqrt(se(x, y)/n)

    end function rmse

    function logarithmic(x, y) result(res)

        !% Notes
        !% -----
        !%
        !% Logarithmic (LGRM) computation function
        !%
        !% Given two single precision array (x, y) of dim(1) and size(n),
        !% it returns the result of LGRM computation
        !% LGRM = sum(x * log(y/x) ** 2)

        implicit none

        real(sp), dimension(:), intent(in) :: x, y
        real(sp) :: res

        integer :: i

        res = 0._sp

        do i = 1, size(x)

            if (x(i) .gt. 0._sp .and. y(i) .gt. 0._sp) then

                res = res + x(i)*log(y(i)/x(i))*log(y(i)/x(i))

            end if

        end do

    end function logarithmic

    subroutine heap_sort(n, arr)

        !% Notes
        !% -----
        !%
        !% Implement heap sort algorithm
        !%
        !% Computational complexity is O(n log n)

        implicit none

        integer, intent(in) :: n
        real(sp), dimension(n), intent(inout) :: arr

        integer :: l, ir, i, j
        real(sp) :: arr_l

        l = n/2 + 1

        ir = n

10      continue

        if (l .gt. 1) then

            l = l - 1

            arr_l = arr(l)

        else

            arr_l = arr(ir)

            arr(ir) = arr(1)

            ir = ir - 1

            if (ir .eq. 1) then

                arr(1) = arr_l

                return

            end if

        end if

        i = l

        j = l + l

20      if (j .le. ir) then

            if (j .lt. ir) then

                if (arr(j) .lt. arr(j + 1)) j = j + 1

            end if

            if (arr_l .lt. arr(j)) then

                arr(i) = arr(j)

                i = j; j = j + j

            else

                j = ir + 1

            end if

            goto 20

        end if

        arr(i) = arr_l

        goto 10

    end subroutine heap_sort

    function quantile(dat, p) result(res)

        !% Notes
        !% -----
        !%
        !% Quantile function using linear interpolation
        !%
        !% Similar to numpy.quantile

        implicit none

        real(sp), intent(in) :: p
        real(sp), dimension(:), intent(in) :: dat
        real(sp), dimension(size(dat)) :: sorted_dat
        integer :: n
        real(sp) :: res, q1, q2, frac

        res = dat(1)

        n = size(dat)

        if (n .gt. 1) then

            sorted_dat = dat

            call heap_sort(n, sorted_dat)

            frac = (n - 1)*p + 1

            if (frac .le. 1) then

                res = sorted_dat(1)

            else if (frac .ge. n) then

                res = sorted_dat(n)

            else
                q1 = sorted_dat(int(frac))

                q2 = sorted_dat(int(frac) + 1)

                res = q1 + (q2 - q1)*(frac - int(frac)) ! linear interpolation

            end if

        end if

    end function quantile

    subroutine flow_percentile(qo, qs, p, num, den)

        !% Notes
        !% -----
        !%
        !% Compute flow percentiles
        !%
        !% From observed (qo) and simulated signature (qs)
        !% Apply quantile function by ignoring negative values

        implicit none

        real(sp), dimension(:), intent(in) :: qo, qs
        real(sp), intent(in) :: p

        real(sp), intent(inout) :: num, den

        real(sp), dimension(size(qo)) :: pos_qo, pos_qs

        integer :: i, j, n

        n = size(qo)

        pos_qo = 0._sp
        pos_qs = 0._sp

        j = 0

        do i = 1, n

            if (qo(i) .ge. 0._sp .and. qs(i) .ge. 0._sp) then

                j = j + 1

                pos_qo(j) = qo(i)
                pos_qs(j) = qs(i)

            end if

        end do

        num = quantile(pos_qs(1:j), p)

        den = quantile(pos_qo(1:j), p)

    end subroutine flow_percentile

    function signature(po, qo, qs, mask_event, stype) result(res)

        !% Notes
        !% -----
        !%
        !% Signatures-based cost computation (SBC)
        !%
        !% Given two single precision array (x, y) of dim(1) and size(n),
        !% it returns the result of SBC computation
        !% SBC_i = (s_i(y)/s_i(x) - 1) ** 2
        !% where i is a signature i and s_i is its associated signature computation function

        implicit none

        real(sp), dimension(:), intent(in) :: po, qo, qs
        integer, dimension(:), intent(in) :: mask_event
        character(len=*), intent(in) :: stype

        real(sp) :: res

        logical, dimension(size(mask_event)) :: lgc_mask_event
        integer :: n_event, i, j, start_event, ntime_step_event
        real(sp) :: sum_qo, sum_qs, sum_po, &
        & max_qo, max_qs, max_po, num, den
        integer :: imax_qo, imax_qs, imax_po

        res = 0._sp

        n_event = 0

        if (stype(:1) .eq. "E") then

            ! Reverse loop on mask_event to find number of event (array sorted filled with 0)
            do i = size(mask_event), 1, -1

                if (mask_event(i) .gt. 0) then

                    n_event = mask_event(i)
                    exit

                end if

            end do

            do i = 1, n_event

                lgc_mask_event = (mask_event .eq. i)

                do j = 1, size(mask_event)

                    if (lgc_mask_event(j)) then

                        start_event = j
                        exit

                    end if

                end do

                ntime_step_event = count(lgc_mask_event)

                sum_qo = 0._sp
                sum_qs = 0._sp
                sum_po = 0._sp

                max_qo = 0._sp
                max_qs = 0._sp
                max_po = 0._sp

                imax_qo = 0
                imax_qs = 0
                imax_po = 0

                do j = start_event, start_event + ntime_step_event - 1

                    if (qo(j) .ge. 0._sp .and. po(j) .ge. 0._sp) then

                        sum_qo = sum_qo + qo(j)
                        sum_qs = sum_qs + qs(j)
                        sum_po = sum_po + po(j)

                        if (qo(j) .gt. max_qo) then

                            max_qo = qo(j)
                            imax_qo = j

                        end if

                        if (qs(j) .gt. max_qs) then

                            max_qs = qs(j)
                            imax_qs = j

                        end if

                        if (po(j) .gt. max_po) then

                            max_po = po(j)
                            imax_po = j

                        end if

                    end if

                end do

                select case (stype)

                case ("Epf")

                    num = max_qs
                    den = max_qo

                case ("Elt")

                    num = imax_qs - imax_po
                    den = imax_qo - imax_po

                case ("Erc")

                    if (sum_po .gt. 0._sp) then

                        num = sum_qs/sum_po
                        den = sum_qo/sum_po

                    end if

                end select

                if (den .gt. 0._sp) then

                    res = res + abs(num/den - 1._sp)

                end if

            end do

            if (n_event .gt. 0) then

                res = res/n_event

            end if

        else

            select case (stype)

            case ("Crc")

                sum_qo = 0._sp
                sum_qs = 0._sp
                sum_po = 0._sp

                do i = 1, size(qo)

                    if (qo(i) .ge. 0._sp .and. po(i) .ge. 0._sp) then

                        sum_qo = sum_qo + qo(i)
                        sum_qs = sum_qs + qs(i)
                        sum_po = sum_po + po(i)

                    end if

                end do

                if (sum_po .gt. 0._sp) then

                    num = sum_qs/sum_po
                    den = sum_qo/sum_po

                end if

            case ("Cfp2")

                call flow_percentile(qo, qs, 0.02_sp, num, den)

            case ("Cfp10")

                call flow_percentile(qo, qs, 0.1_sp, num, den)

            case ("Cfp50")

                call flow_percentile(qo, qs, 0.5_sp, num, den)

            case ("Cfp90")

                call flow_percentile(qo, qs, 0.9_sp, num, den)

            end select

            if (den .gt. 0._sp) then

                res = abs(num/den - 1._sp)

            end if

        end if

    end function signature

    !%TODO: Add "distance_correlation" once clearly verified
    function distance_correlation_descriptors(&
    &setup, mesh, input_data, target_control, nbz, parameters_matrix) &
    &result(penalty_total)

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        type(Input_DataDT), intent(in) :: input_data
        character(6), intent(in) :: target_control
        integer, intent(in) :: nbz
        real(sp), dimension(mesh%nrow, mesh%ncol, nbz), intent(in) :: parameters_matrix
        real(sp) :: penalty_total

        real :: penalty, penalty_class, distance
        integer :: i, j, ii, jj, p, label, minmask, maxmask, indice, nbpixbyclass
        integer :: jj_start, ii_start

        integer, dimension(setup%nd) :: descriptor_indexes
        integer :: desc
        integer, dimension(nbz) :: optim

        if (target_control == "params") then
            optim = setup%optimize%optim_parameters
        end if

        if (target_control == "states") then
            optim = setup%optimize%optim_states
        end if

        ! penality term
        penalty_total = 0.0

        do p = 1, nbz ! loop on all parameters

            if (optim(p) > 0) then

                penalty = 0.0

                !#TODO seklect approriate descriptor for parameter
                if (target_control == "params") then
                    descriptor_indexes = setup%optimize%reg_descriptors_for_params(p, :)
                end if
                if (target_control == "states") then
                    descriptor_indexes = setup%optimize%reg_descriptors_for_states(p, :)
                end if

                do desc = 1, setup%nd

                    if (descriptor_indexes(desc) > 0) then

                        minmask = 0
                        maxmask = int(maxval(input_data%descriptor(:, :, desc)))

                        !Boucle sur les différents indices du masque
                        do indice = minmask, maxmask

                            label = indice
                            nbpixbyclass = 0
                            penalty_class = 0.

                            do i = 1, mesh%nrow
                                do j = 1, mesh%ncol

                                    if (int(input_data%descriptor(i, j, p)) == label) then

                                        nbpixbyclass = nbpixbyclass + 1

                                        jj_start = j + 1
                                        ii_start = i
                                        if ((j == mesh%ncol) .and. (i < mesh%nrow)) then
                                            jj_start = 1
                                            ii_start = i + 1
                                        end if

                                        if ((i == mesh%nrow) .and. (j == mesh%ncol)) then
                                            jj_start = j
                                            ii_start = i
                                        end if

                                        do ii = ii_start, mesh%nrow

                                            do jj = jj_start, mesh%ncol

                                                if (int(input_data%descriptor(ii, jj, desc)) == label) then

                                                    distance = sqrt(((ii - i))**2.+((jj - j))**2.)
                                                    if (distance < 1.) then
                                                        distance = 1.
                                                    end if

                                                    penalty_class = penalty_class + (1./(distance**2.))*&
                                                    &(parameters_matrix(i, j, p) - parameters_matrix(ii, jj, p))**2.

                                                end if

                                            end do

                                            jj_start = 1

                                        end do

                                    end if

                                end do
                            end do

                            if (nbpixbyclass >= 1) then
                                penalty = penalty + penalty_class/(real(nbpixbyclass))
                            else
                                penalty = penalty + penalty_class
                            end if

                        end do

                    end if

                end do

                penalty_total = penalty_total + penalty

            end if

        end do

    end function distance_correlation_descriptors

    function reg_smoothing(setup, mesh, optim_arr, matrix, matrix_bgd, rel_to_bgd) result(res)

        !% Notes
        !% -----
        !%
        !% Smoothing regularization computation function
        !%
        !% Given one matrix of dim(3) and size(mesh%nrow, mesh%ncol, size_mat3),
        !% it returns the spatial second derivative multiplicated by pond_smoothing**4

        implicit none

        type(SetupDT), intent(in) :: setup
        type(MeshDT), intent(in) :: mesh
        integer, dimension(:), intent(in) :: optim_arr
        real(sp), dimension(:, :, :), intent(in) :: matrix, matrix_bgd
        logical, intent(in):: rel_to_bgd
        real(sp) :: res

        real(sp), dimension(size(matrix, 1), size(matrix, 2), size(matrix, 3)) :: mat
        integer :: i, col, row, min_col, max_col, min_row, max_row

        res = 0._sp

        ! matrix relative to the bgd. We don't want to penalize initial spatial variation.
        if (rel_to_bgd) then
            mat = matrix - matrix_bgd
        else
            mat = matrix
        end if

        do i = 1, size(matrix, 3)

            if (optim_arr(i) .gt. 0) then

                do col = 1, size(matrix, 2)

                    do row = 1, size(matrix, 1)

                        if (mesh%active_cell(row, col) .eq. 1) then

                            ! do not point out of the domain
                            min_col = max(1, col - 1)
                            max_col = min(size(matrix, 2), col + 1)
                            min_row = max(1, row - 1)
                            max_row = min(size(matrix, 1), row + 1)

                            ! if active_cell, do not take into account the cells outside the catchment
                            ! since only cells in active_cell are included in the control vector
                            if (mesh%active_cell(row, min_col) .eq. 0) then
                                min_col = col
                            end if

                            if (mesh%active_cell(row, max_col) .eq. 0) then
                                max_col = col
                            end if

                            if (mesh%active_cell(min_row, col) .eq. 0) then
                                min_row = row
                            end if

                            if (mesh%active_cell(max_row, col) .eq. 0) then
                                max_row = row
                            end if

                            res = res + ((mat(max_row, col, i) - 2._sp*mat(row, col, i) + mat(min_row, col, i))**2._sp &
                            & + (mat(row, max_col, i) - 2._sp*mat(row, col, i) + mat(row, min_col, i))**2._sp)

                        end if

                    end do

                end do

            end if

        end do

    end function reg_smoothing

    function reg_prior(setup, optim_arr, matrix, matrix_bgd) result(res)

        !% Notes
        !% -----
        !%
        !% Prior regularization (PR) computation function
        !%
        !% Given two matrix of dim(3) and size(mesh%nrow, mesh%ncol, size_mat3),
        !% it returns the result of PR computation. (Square Error between matrix)
        !%
        !% PR = sum((mat1 - mat2) ** 2)

        implicit none

        type(SetupDT), intent(in) :: setup
        integer, dimension(:), intent(in) :: optim_arr
        real(sp), dimension(:, :, :), intent(in) :: matrix, matrix_bgd
        real(sp) :: res

        integer :: i, col, row

        res = 0._sp

        do i = 1, size(matrix, 3)

            if (optim_arr(i) .gt. 0) then

                do col = 1, size(matrix, 2)

                    do row = 1, size(matrix, 1)

                        res = res + (matrix(row, col, i) - matrix_bgd(row, col, i))**2._sp
                
                    end do
                
                end do

            end if

        end do

    end function reg_prior

end module mwd_cost
