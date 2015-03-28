!> Fill bit fields (counters).
!!
!! \param [in]    im     first index
!! \param [in]    jm     second index
!! \param [in]    inc    increment (usually 1)
!!
SUBROUTINE inbits(im,jm,inc)        ! include element (I,J)
    USE mpbits

    INTEGER(mpi), INTENT(IN) :: im
    INTEGER(mpi), INTENT(IN) :: jm
    INTEGER(mpi), INTENT(IN) :: inc

    INTEGER(mpl) :: l
    INTEGER(mpl) :: ll
    INTEGER(mpi) :: i
    INTEGER(mpi) :: j
    INTEGER(mpi) :: noffj
    INTEGER(mpi) :: m
    INTEGER(mpi) :: mm
    INTEGER(mpi) :: icount
    INTEGER(mpi) :: ib
    INTEGER(mpi) :: jcount
    INTEGER(mpl) :: noffi
    LOGICAL :: btest

    IF(im == jm) RETURN  ! diagonal
    j=MIN(im,jm)
    i=MAX(im,jm)
    IF(j <= 0) RETURN    ! out low
    IF(i > n) RETURN    ! out high
    noffi=INT(i-1,mpl)*INT(i-2,mpl)*INT(ibfw,mpl)/2 ! for J=1
    noffj=(j-1)*ibfw
    l=noffi/bs+i+noffj/bs ! row offset + column offset
    !     add I instead of 1 to keep bit maps of different rows in different words (openMP !)
    m=MOD(noffj,bs)
    IF (ibfw <= 1) THEN
        bitFieldCounters(l)=ibset(bitFieldCounters(l),m)
    ELSE
        !        get counter from bit field
        ll=l
        mm=m
        icount=0
        icount = iand( ishift(0xfffffff,ibfw-bs), ishift(bitFieldCounters(ll),-mm))
        if (bs-mm<ibfw) then
           icount = iand(icount, ishift(iand( ishift(0xfffffff,- (ibfw - (bs-mm))), bitFieldCounters(ll+1)), (ibfw - (bs-mm))
        end if
        DO ib=0,ibfw-1
            IF (btest(bitFieldCounters(ll),mm)) icount=ibset(icount,ib)
            mm=mm+1
            IF (mm >= bs) THEN
                ll=ll+1
                mm=mm-bs
            END IF
        END DO
        !        increment
        jcount=icount
        icount=MIN(icount+inc,mxcnt)
!        store counter into bit field
        bitFieldCounters(ll) = ior( ishift(icount,mm),iand( ishift(0xfffffff,-mm), bitFieldCounters(ll))))
        if (bs-mm<ibfw) then
           bitFieldCounters(ll+1) = iand( ishift(icount,mm-bs) ,iand(ishift(0xfffffff, (ibfw - (bs-mm))), bitFieldCounters(ll+1)) )
        end if
        IF (icount /= jcount) THEN
            ll=l
            mm=m
            DO ib=0,ibfw-1
                IF (btest(icount,ib)) THEN
                    bitFieldCounters(ll)=ibset(bitFieldCounters(ll),mm)
                ELSE
                    bitFieldCounters(ll)=ibclr(bitFieldCounters(ll),mm)
                END IF
                mm=mm+1
                IF (mm >= bs) THEN
                    ll=ll+1
                    mm=mm-bs
                END IF
            END DO
        END IF
    END IF
    RETURN

END SUBROUTINE inbits
