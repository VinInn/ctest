subroutine loop(n,b,x,globalMatD,indij,i,j,jn,lj)
   real*8 b(n),x(n),globalMatD(n), bb
   !DIMENSION(:) b,x,globalMatD
   integer n,indij 
   integer jj,jn,j,lj,i

   bb = b(i)
   DO CONCURRENT (jj=1:jn)
     b(j)=b(j)+globalMatD(indij+lj)*x(i)
     bb=bb+globalMatD(indij+lj)*x(j)
     j=j+1
     lj=lj+1
   END DO
   b(i)=bb
end subroutine
