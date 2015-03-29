program atomic
    use iso_fortran_env
    implicit none
    integer (atomic_logical_kind) :: atom[*],old
    integer n, tmp, tot,i
    integer OMP_GET_THREAD_NUM, iproc
    real b
    REAL, ALLOCATABLE :: a(:)
    allocate(a(1:100))
    a = 0.0    
    b = 0
    n = 50
    atom[1]=1
    tot = 0
    !$OMP PARALLEL private(iproc,tmp,old) &
    !$OMP  REDUCTION(+:B) REDUCTION(+:A)
    iproc=0
    IPROC=OMP_GET_THREAD_NUM()         ! thread number
    Do
     tmp = atom[1]
     do while( tmp<=n)
       call atomic_cas(atom[1],old,tmp,tmp+1)
       if (tmp .eq. old) exit
       tmp = atom[1]
     end do
     if (tmp > n) EXIT
     b = b+1
     a(tmp) = 1
     a(91) = a(91)+1
     !$OMP atomic
     tot = tot + 1
     write(*,*) iproc, tot, tmp, old
    end do
    !$OMP END PARALLEL

    write (*,*) b
    write (*,*) a
    do i=1,2
      write(*,*) "this is fortran " ,i
    enddo


end program atomic
