! Export matched, kappa-resolved scattering partial waves from ONCVPSP.
! This is a generation tool linked into an isolated copy of ONCVPSP, never
! a production GW kernel. ONCV's native up arrays are d(u)/d(log-grid index).
subroutine export_atomic_reconstruction(lmax,lloc,nproj,rr,vfull,vp,vkb,evkb, &
                                      ep,rc,zz,mmax,nc,na,la,ea,fa)
  implicit none
  integer, parameter :: dp=kind(1.0d0)
  integer :: ntrain,nhold,ne
  integer :: lmax,lloc,mmax,nc,na(30),la(30),nproj(6)
  real(dp) :: rr(mmax),vfull(mmax),vp(mmax,5,2),vkb(mmax,2,4,2)
  real(dp) :: evkb(2,4,2),ep(6,2),rc(6),zz,ea(30,2),fa(30)
  real(dp), allocatable :: ps(:),psp(:),ae(:,:),aep(:,:),left(:,:),leftp(:,:)
  real(dp) :: ra,al,e0,e1,eps,eta,etest,f,amplitude,du,dv,eupper
  real(dp) :: elo,ehi,flo,fhi,target
  integer :: mch,ll,l1,ikap,kap,mkap,ivkb,ie,ierr,it,ir,train,nn,nch,ich
  integer :: node_reference,node_matched,jj,skip,ios
  character(len=256) :: controls
  allocate(ps(mmax),psp(mmax),ae(mmax,2),aep(mmax,2))
  allocate(left(mmax,2),leftp(mmax,2))
  al=0.01d0*dlog(rr(101)/rr(1))
  eupper=5.d0
  ntrain=128
  nhold=32
  open(73,file='atomic_export.in',status='old',action='read',iostat=ios)
  if(ios==0) then
    read(73,'(a)',iostat=ios) controls
    close(73)
    if(ios/=0) stop 'ERROR invalid atomic reconstruction controls'
    read(controls,*,iostat=ios) eupper,ntrain,nhold
    if(ios>0 .or. eupper<=0.d0 .or. ntrain<8 .or. nhold<1) then
      stop 'ERROR invalid atomic reconstruction energy window or sample counts'
    end if
  end if
  ne=ntrain+nhold
  ra=1.2d0*maxval(rc(1:lmax+1))+0.15d0
  mch=minloc(abs(rr-ra),dim=1)
  ra=rr(mch)
  nch=2*(lmax+2)-1
  open(71,file='atomic_bank.dat',status='new',action='write')
  write(71,'(a)') '# lorrax.atomic_scattering_bank.v1; u=rR; derivative=du/dr; bohr,Ha'
  write(71,'(3i8,2es25.16)') nch,ne,mch,ra,zz
  ich=0
  do ll=0,lmax+1
    l1=ll+1
    mkap=2
    if(ll==0) mkap=1
    ivkb=0
    if(ll<=lmax) ivkb=nproj(l1)
    do ikap=1,mkap
      ich=ich+1
      kap=-(ll+1)
      if(ikap==2) kap=ll
      e0=-0.3d0
      if(ll<=lmax) e0=min(0.d0,ep(l1,ikap))-0.3d0
      e1=eupper
      write(71,'(4i8,2es25.16)') ich,ll,kap,ivkb,e0,e1
      skip=count(la(1:nc)==ll)
      do ie=1,ne
        train=1
        if(ie<=ntrain) then
          eps=e0+(e1-e0)*dble(ie-1)/dble(ntrain-1)
        else
          train=0
          eps=e0+(e1-e0)*(dble(ie-ntrain)-0.5d0)/dble(nhold)
        end if
        call lschvkbs(ll,ivkb,eps,rr,vp(1,lloc+1,1), &
                      vkb(1,1,l1,ikap),evkb(1,l1,ikap),ps,psp,mmax,mch)
        eta=eps
        call ldiracfs(ll,kap,ierr,eta,rr,zz,vfull,ae,aep,mmax,mch)
        if(ierr/=0) stop 'ERROR atomic AE scattering reference failed'
        node_reference=count(ae(1:mch-1,1)*ae(2:mch,1)<0.d0)
        ! OCEAN normangnodes/baregrip use an unwrapped phase. A determinant
        ! alone also vanishes on every wrong radial-node branch.
        target=atomic_phase(ps,psp(mch)/al,skip,.true.)
        elo=e0-1.d0
        ehi=e1+1.d0
        do it=1,20
          etest=elo
          call ldiracfs(ll,kap,ierr,etest,rr,zz,vfull,left,leftp,mmax,mch)
          flo=atomic_phase(left(:,1),leftp(mch,1)/al,0,.false.)-target
          if(flo>=0.d0) exit
          elo=elo-2.d0**it
        end do
        if(flo<0.d0) stop 'ERROR atomic AE phase lower bracket failed'
        do it=1,20
          etest=ehi
          call ldiracfs(ll,kap,ierr,etest,rr,zz,vfull,left,leftp,mmax,mch)
          fhi=atomic_phase(left(:,1),leftp(mch,1)/al,0,.false.)-target
          if(fhi<=0.d0) exit
          ehi=ehi+2.d0**it
        end do
        if(fhi>0.d0) stop 'ERROR atomic AE phase upper bracket failed'
        do it=1,120
          eta=(elo*fhi-ehi*flo)/(fhi-flo)
          eta=max(elo+0.1d0*(ehi-elo),min(ehi-0.1d0*(ehi-elo),eta))
          call ldiracfs(ll,kap,ierr,eta,rr,zz,vfull,ae,aep,mmax,mch)
          if(ierr/=0) stop 'ERROR atomic AE scattering iteration failed'
          f=atomic_phase(ae(:,1),aep(mch,1)/al,0,.false.)-target
          if(abs(f)<2.d-13) exit
          if(f>0.d0) then
            elo=eta; flo=f
          else
            ehi=eta; fhi=f
          end if
        end do
        if(it>120) then
          write(6,*) 'phase convergence failure: channel,energy,Eps,Eae,F',ich,ie,eps,eta,f
          stop 'ERROR atomic phase matching did not converge'
        end if
        node_matched=count(ae(1:mch-1,1)*ae(2:mch,1)<0.d0)
        du=aep(mch,1)/al
        dv=psp(mch)/al
        amplitude=(ae(mch,1)*ps(mch)+du*dv)/(ae(mch,1)**2+du**2)
        ae=amplitude*ae
        aep=amplitude*aep
        write(71,'(4i8,4es25.16)') ie,train,node_reference,node_matched,eps,eta,f,amplitude
        do ir=1,mch
          write(71,'(7es25.16)') rr(ir),ps(ir),psp(ir)/(al*rr(ir)), &
               ae(ir,1),aep(ir,1)/(al*rr(ir)),ae(ir,2),aep(ir,2)/(al*rr(ir))
        end do
      end do
    end do
  end do
  close(71)
  ! True frozen-core orbitals are diagnostics, never an NLCC substitution.
  open(72,file='atomic_core.dat',status='new',action='write')
  write(72,'(a)') '# lorrax.atomic_frozen_core.v1; Dirac large/small u=rR; bohr,Ha'
  write(72,'(2i8)') nc,mch
  do jj=1,nc
    ll=la(jj)
    mkap=2
    if(ll==0) mkap=1
    do ikap=1,mkap
      kap=-(ll+1)
      if(ikap==2) kap=ll
      etest=ea(jj,ikap)
      call ldiracfb(na(jj),ll,kap,ierr,etest,rr,zz,vfull,ae,aep,mmax,nn)
      if(ierr/=0) stop 'ERROR atomic frozen-core bound state failed'
      write(72,'(4i8,2es25.16)') jj,na(jj),ll,kap,etest,fa(jj)
      do ir=1,mch
        write(72,'(3es25.16)') rr(ir),ae(ir,1),ae(ir,2)
      end do
    end do
  end do
  close(72)
  deallocate(ps,psp,ae,aep,left,leftp)
contains
  real(dp) function atomic_phase(u,rdudr,node_skip,pseudo)
    real(dp), intent(in) :: u(mmax),rdudr
    integer, intent(in) :: node_skip
    logical, intent(in) :: pseudo
    integer :: i,nodes
    real(dp) :: orientation
    nodes=0
    do i=1,mch-1
      ! ONCV's nonlocal outward solver has roundoff nodes deep inside the
      ! pseudized core. AE node counting retains every true core node.
      if(pseudo .and. rr(i)<0.1d0) cycle
      if(u(i)*u(i+1)<0.d0) nodes=nodes+1
    end do
    orientation=sign(1.d0,u(mch))
    atomic_phase=atan2(orientation*rdudr,abs(u(mch)))- &
                 4.d0*atan(1.d0)*dble(nodes+node_skip)
  end function atomic_phase
end subroutine export_atomic_reconstruction
