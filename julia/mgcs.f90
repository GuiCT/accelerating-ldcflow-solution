module multigrid

contains
subroutine mgcs(f, tf, nx, ny, dx, dy)
  
  implicit none
  integer nx, ny, iteracao
  real*8 dx, dy, dx2, dy2, erro, w
  real*8 f(nx, ny), tf(nx, ny), res(nx, ny)
  real*8 f2(nx, ny), tf2(nx, ny), corr(nx, ny)
  
  w = 0.8d0
  dx2 = dx * dx
  dy2 = dy * dy

  do iteracao = 1, 5000
    call jacobi(f,tf,nx,ny,dx2,dy2,w,2)
    call residuo(res,f,tf,nx,ny,dx2,dy2)
    call verifica(res,erro,nx,ny)

    if (erro.lt.1d-6) then
      return
    endif

    call restrict(f2,tf2,res,nx,ny,dx2,dy2)
    call sormg(f2,tf2,nx,ny,dx2,dy2,1.7d0,100)

    call intpol(corr,f2,nx,ny,dx2,dy2)
    call correction(f,corr,nx,ny)
  end do

  return
end

subroutine jacobi(f, tf, nx, ny, dx2, dy2, w, itera)

  implicit none
  integer i, j, nx, ny, it, itera
  real*8 f(nx,ny), fold(nx,ny), tf(nx,ny), dx2, dy2, beta, w
  
  beta = dx2/dy2
  do it = 1, itera
     fold = f
     do j = 2, ny-1
        do i = 2, nx-1
           f(i,j) = (1.d0-w)*f(i,j)+w*(-dx2*tf(i,j)+fold(i-1,j)+fold(i+1,j) &
                  + beta*(fold(i,j-1)+fold(i,j+1)))/(2.d0*(beta+1.d0))
        end do
     end do
  end do
 
  return
end

subroutine residuo(res, f, tf, nx, ny, dx2, dy2)

  implicit none
  integer i, j, nx, ny
  real*8 f(nx,ny), res(nx,ny), tf(nx,ny), dx2, dy2
 
  do j = 2, ny-1
     do i = 2, nx-1
        res(i,j) = (tf(i,j)-(f(i-1,j)-2.d0*f(i,j)+f(i+1,j))/dx2 &
                 - (f(i,j-1)-2.d0*f(i,j)+f(i,j+1))/dy2 )
     end do
  end do
 
  return
end

subroutine verifica(res, errol, nx, ny)

  implicit none
  integer i, j, nx, ny
  real*8 res(nx, ny), errol
 
  errol = 0.d0
  do j = 2, ny-1
     do i = 2, nx-1
        errol = max(dabs(res(i,j)),errol)
     end do
  end do
  
  return
end

subroutine restrict(fc, sc, res, nx, ny, dx2, dy2)

  implicit none
  integer i, j, nx, ny, fi, fj
  real*8 dx2, dy2, fc(nx,ny), sc(nx,ny), res(nx,ny)
 
  ! new values for dx2, dy2, ptsx e ptsy
  nx = (nx + 1)/2
  ny = (ny + 1)/2
  dx2   = 4.d0 * dx2
  dy2   = 4.d0 * dy2
  fc    = 0.d0
  
  ! restriction in the middle
  fj = 3
  do j = 2, nx - 1
     fi = 3
     do i = 2, ny - 1
        sc(i,j) = 6.25d-2*(2.d0*(res(fi-1,fj)+res(fi+1,fj) &
                + res(fi,fj-1)+res(fi,fj+1) )+4.d0*res(fi,fj) &
                + res(fi+1,fj+1)+res(fi+1,fj-1)+res(fi-1,fj+1)+res(fi-1,fj-1))
        fi = fi + 2
     end do
     fj = fj + 2
  end do
 
  return
end

subroutine intpol(temp, fc, nx, ny, dx2, dy2)
  
  implicit none
  integer i, j, nx, ny, fi, fj
  real*8 dx2, dy2, fc(nx,ny), temp(nx,ny)
 
  ! new values for dx2 e dy2
  dx2 = 0.25d0 * dx2
  dy2 = 0.25d0 * dy2
 
  ! copying fcc to temp
  fj = 1
  do j = 1, ny
     fi = 1
     do i = 1, nx
        temp(fi,fj) = fc(i,j)  !(círculo preto)
        fi = fi + 2
     end do
     fj = fj + 2
  end do
 
  ! new values for nptsx and nptsy
  nx = 2 * nx - 1
  ny = 2 * ny - 1
 
  ! Interpolation in x
  do j = 1, ny, 2
     do i = 2, nx - 1, 2
        temp(i,j) = 0.5d0 * ( temp(i-1,j) + temp(i+1,j) )
     end do
  end do
  
    ! Interpolation in y 
  do j = 2, ny - 1, 2
     do i = 1, nx
        temp(i,j) = 0.5d0 * ( temp(i,j+1) + temp(i,j-1) )
     end do
  end do
 
  return
end

subroutine correction(f, corr, nx, ny)

  implicit none
  integer i, j, nx, ny
  real*8 f(nx,ny), corr(nx,ny)
 
  ! Sum to new function
  do j = 2, ny - 1
     do i = 2, nx - 1
        f(i,j) = f(i,j) + corr(i,j)
     end do
  end do
 
  return
end

subroutine sormg(f, tf, nx, ny, dx2, dy2, w, itera)
                         
  implicit none    
  integer i, j, nx, ny, it, itera
  real*8 f(nx,ny), tf(nx,ny), dx2, dy2, beta, w
 
  beta = dx2/dy2
  do it = 1, itera        
     do j = 2, ny-1      
        do i = 2, nx-1
           f(i,j) = (1.d0-w)*f(i,j)+w*(-dx2*tf(i,j)+f(i-1,j) &
                  + f(i+1,j)+beta*(f(i,j-1)+f(i,j+1)))/(2.d0*(beta+1.d0))
        end do
     end do
  end do
  
  return
end

end module multigrid