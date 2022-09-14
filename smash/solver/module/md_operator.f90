!%      This module `md_operator` encapsulates all SMASH operator.
!%      This module is differentiated.
!%
!%      contains
!%
!%      [1] GR_interception
!%      [2] GR_production
!%      [3] GR_exchange
!%      [4] GR_transferN
!%      [5] upstream_discharge
!%      [6] sparse_upstream_discharge
!%      [7] GR_transfer1

module md_operator
    
    use mwd_common !% only : sp

    implicit none
    
    contains
        
        !% TODO Renommer argument pour etre global
        subroutine GR_interception(prcp, pet, ci, hi, pn, ei)
        
            implicit none
            
            real(sp), intent(in) :: prcp, pet, ci
            real(sp), intent(inout) :: hi
            real(sp), intent(out) :: pn, ei
            
            ei = min(pet, prcp + hi * ci)
            
            pn = max(0._sp, prcp - ci * (1._sp - hi) - ei)
            
            hi = hi + (prcp - ei - pn) / ci
            
        end subroutine GR_interception
    
    
        subroutine GR_production(pn, en, cp, beta, hp, pr, perc)
        
            implicit none
            
            real(sp), intent(in) :: pn, en, cp, beta
            real(sp), intent(inout) :: hp
            real(sp), intent(out) :: pr, perc
            
            real(sp) :: inv_cp, ps, es, hp_imd
            
            inv_cp = 1._sp / cp
            pr = 0._sp
            
            ps = cp * (1._sp - hp * hp) * tanh(pn * inv_cp) / &
            & (1._sp + hp * tanh(pn * inv_cp))
            
            es = (hp * cp) * (2._sp - hp) * tanh(en * inv_cp) / &
            & (1._sp + (1._sp - hp) * tanh(en * inv_cp))
            
            hp_imd = hp + (ps - es) * inv_cp
            
            if (pn .gt. 0) then
            
                pr = pn - (hp_imd - hp) * cp
            
            end if
            
            perc = (hp_imd * cp) * (1._sp - (1._sp + (hp_imd / beta) ** 4) ** (- 0.25_sp))
            
            hp = hp_imd - perc * inv_cp

        end subroutine GR_production
        
        
        subroutine GR_exchange(exc, hft, l)
        
            implicit none
            
            real(sp), intent(in) :: exc
            real(sp), intent(inout) :: hft
            real(sp), intent(out) :: l
            
            l = exc * hft ** 3.5_sp
        
        end subroutine GR_exchange

        
        subroutine GR_transferN(n, prcp, pr, ct, ht, q)
        
            implicit none
            
            real(sp), intent(in) :: n, prcp, pr, ct
            real(sp), intent(inout) :: ht
            real(sp), intent(out) :: q
            
            real(sp) :: pr_imd, ht_imd, nm1, d1pnm1
            
            nm1 = n - 1._sp
            d1pnm1 = 1._sp / nm1
            
            if (prcp .lt. 0._sp) then
            
                pr_imd = ((ht * ct) ** (- nm1) - ct ** (- nm1)) ** (- d1pnm1) - (ht * ct)
                
            else
            
                pr_imd = pr
                
            end if
            
            ht_imd = max(1.e-6_sp, ht + pr_imd / ct)
            
            ht = (((ht_imd * ct) ** (- nm1) + ct ** (- nm1)) ** (- d1pnm1)) / ct
            
            q = (ht_imd - ht) * ct
        
        end subroutine GR_transferN
        
        
        subroutine upstream_discharge(dt, dx, nrow, ncol, &
        & flwdir, drained_area, row, col, q, qup)
        
            implicit none

            real(sp), intent(in) :: dt, dx
            integer, intent(in) :: nrow, ncol, row, col
            integer, dimension(nrow, ncol), intent(in) :: flwdir, drained_area
            real(sp), dimension(nrow, ncol), intent(in) :: q
            real(sp), intent(out) :: qup
            
            integer :: i, row_imd, col_imd
            integer, dimension(8) :: dcol = [0, -1, -1, -1, 0, 1, 1, 1]
            integer, dimension(8) :: drow = [1, 1, 0, -1, -1, -1, 0, 1]
            integer, dimension(8) :: dkind = [1, 2, 3, 4, 5, 6, 7, 8]
            
            qup = 0._sp
            
            if (drained_area(row, col) .gt. 1) then
            
                do i=1, 8
                    
                    col_imd = col + dcol(i)
                    row_imd = row + drow(i)
                    
                    if (col_imd .gt. 0 .and. col_imd .le. ncol .and. &
                    &   row_imd .gt. 0 .and. row_imd .le. nrow) then
                    
                        if (flwdir(row_imd, col_imd) .eq. dkind(i)) then
                        
                            qup = qup + q(row_imd, col_imd)
                            
                        end if
                        
                    end if
                
                end do
                
                qup = (qup * dt) / &
                & (0.001_sp * dx * dx * real(drained_area(row, col) - 1))
            
            end if
        
        end subroutine upstream_discharge
        
        
        subroutine sparse_upstream_discharge(dt, dx, nrow, ncol, nac, &
        & flwdir, drained_area, ind_sparse, row, col, q, qup)
            
            implicit none

            real(sp), intent(in) :: dt, dx
            integer, intent(in) :: nrow, ncol, nac, row, col
            integer, dimension(nrow, ncol), intent(in) :: flwdir, drained_area, ind_sparse
            real(sp), dimension(nac), intent(in) :: q
            real(sp), intent(out) :: qup
            
            integer :: i, row_imd, col_imd, k
            integer, dimension(8) :: dcol = [0, -1, -1, -1, 0, 1, 1, 1]
            integer, dimension(8) :: drow = [1, 1, 0, -1, -1, -1, 0, 1]
            integer, dimension(8) :: dkind = [1, 2, 3, 4, 5, 6, 7, 8]
            
            qup = 0._sp
            
            if (drained_area(row, col) .gt. 1) then
            
                do i=1, 8
                    
                    col_imd = col + dcol(i)
                    row_imd = row + drow(i)
                    
                    if (col_imd .gt. 0 .and. col_imd .le. ncol .and. &
                    &   row_imd .gt. 0 .and. row_imd .le. nrow) then
                    
                        if (flwdir(row_imd, col_imd) .eq. dkind(i)) then
                            
                            k = ind_sparse(row_imd, col_imd)
                            qup = qup + q(k)
                            
                        end if
                        
                    end if
                
                end do
                
                qup = (qup * dt) / &
                & (0.001_sp * dx * dx * real(drained_area(row, col) - 1))
            
            end if
        
        end subroutine sparse_upstream_discharge


        subroutine GR_transfer1(dt, qup, lr, hr, qrout)
            
            implicit none
            
            real(sp), intent(in) :: dt
            real(sp), intent(in) :: qup, lr
            real(sp), intent(inout) :: hr
            real(sp), intent(out) :: qrout
            
            real(sp) :: hr_imd
            
            hr_imd = hr + qup
            
            hr = hr_imd * exp(- dt / (lr * 60._sp))
            
            qrout = hr_imd - hr
        
        end subroutine GR_transfer1
        
        
end module md_operator
