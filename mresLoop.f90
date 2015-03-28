SUBROUTINE avprod(n,x,b)
    IMPLICIT NONE
    ! precision constants
    INTRINSIC :: selected_real_kind
    INTRINSIC :: selected_int_kind
    INTEGER, PARAMETER :: mpi4  = selected_int_kind(9)         !>  4 byte integer
    INTEGER, PARAMETER :: mpi8  = selected_int_kind(18)        !>  8 byte integer
    INTEGER, PARAMETER :: mpr4  = selected_real_kind(6, 37)    !>  4 byte float
    INTEGER, PARAMETER :: mpr8  = selected_real_kind(15, 307)  !>  8 byte float
    INTEGER, PARAMETER :: mpr16 = selected_real_kind(33, 4931) !> 16 byte float, gcc needs libquadmath    INTEGER, PARAMETER :: mpi = selected_int_kind(9)         !>  4 byte integer
    INTEGER, PARAMETER :: mpi  = mpi4                          !>  integer
    INTEGER, PARAMETER :: mpl  = mpi8                          !>  long integer
    INTEGER, PARAMETER :: mps  = mpr4                          !>  single precision
    INTEGER, PARAMETER :: mpd  = mpr8                          !>  double precision
    !> list items from steering file
    TYPE listItem
        INTEGER(mpi) :: label
        REAL(mpd) :: value
    END TYPE listItem
     
    INTEGER(mpi), DIMENSION(:), ALLOCATABLE :: sparseMatrixColumns     !< (compressed) list of columns for sparse matrix
    REAL(mpd), DIMENSION(:), ALLOCATABLE :: globalMatD !< global matrix 'A' (double, full or sparse)
    INTEGER(mpl), DIMENSION(:,:), ALLOCATABLE :: sparseMatrixOffsets !< row offsets for column list, sparse matrix elements


    INTEGER(mpi) :: i
    INTEGER(mpi) :: iencdb
    INTEGER(mpi) :: iencdm
    INTEGER(mpi) :: ir
    INTEGER(mpi) :: j
    INTEGER(mpi) :: jc
    INTEGER(mpi) :: jj
    INTEGER(mpi) :: jn

    INTEGER(mpi), INTENT(IN)                      :: n
    REAL(mpd), INTENT(IN)             :: x(n)
    REAL(mpd), INTENT(OUT)            :: b(n)
    INTEGER(mpl) :: kk
    INTEGER(mpl) :: kl
    INTEGER(mpl) :: ku
    INTEGER(mpl) :: ll
    INTEGER(mpl) :: lj
    INTEGER(mpl) :: indij
    INTEGER(mpl) :: indid
    SAVE

    i=n-4
            b(i)=globalMatD(i)*x(i)    ! diagonal elements
            !                                ! off-diagonals double precision
            ir=i
            kk=sparseMatrixOffsets(1,ir) ! offset in 'd' (column lists)
            ll=sparseMatrixOffsets(2,ir) ! offset in 'j' (matrix)
            kl=0
            ku=sparseMatrixOffsets(1,ir+1)-1-kk
            indid=kk
            indij=ll


                lj=0
                ku=((ku+1)*8)/9-1         ! number of regions (-1)
                indid=indid+ku/8+1        ! skip group offsets
                DO kl=0,ku
                    jc=sparseMatrixColumns(indid+kl)
                    j=ishft(jc,-iencdb)
                    jn=IAND(jc, iencdm)
                    DO CONCURRENT (jj=1:jn)
                        b(j)=b(j)+globalMatD(indij+lj)*x(i)
                        b(i)=b(i)+globalMatD(indij+lj)*x(j)
                        j=j+1
                        lj=lj+1
                    END DO
                END DO
     RETURN
END SUBROUTINE avprod
