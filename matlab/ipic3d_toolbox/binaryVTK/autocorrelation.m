dx=Lx/Nx; dy=Ly/Ny; dz=Lz/Nz;
rhoi=vthi/mean(B(:));
rhoe=rhoi/sqrt(abs(qom));
rhoLH=sqrt(rhoe*rhoi);

xflow='outflow'
close all
soglia=AAz(:,:,:)./max(AAz(:));
s_0=.5;c=.1;
% -() for inflow +() for outflow
switch xflow
    case 'outflow'
        st=.5*(1+tanh(+(soglia-s_0)./c^2));
    otherwise
        st=.5*(1+tanh(-(soglia-s_0)./c^2));  
end

wci=(mean(B(:).*st(:))/mean(st(:)));
wce=wci*abs(qom);
rhoi=vthi/wci;
rhoe=vthe/wce;
rhoLH=sqrt(rhoe*rhoi);

Qe=Qex.^2+Qey.^2+Qez.^2; Qe=sqrt(Qe);
%bbb=Qe.*st;
%figure(1)
%imagesc(bbb(:,:,1))

[autX1,autY1,autZ1]=compute_autocorrelation_directional(Qe,st);

%bbb=S.*st;
[autX2,autY2,autZ2]=compute_autocorrelation_directional(S,st);

bbb=abs(Jex.*Ex+Jey.*Ey+Jez.*Ez);%.*st;
[autX3,autY3,autZ3]=compute_autocorrelation_directional(bbb,st);

h=figure(2)
set(h,'Position', [103 3 827 665])
hax=subplot(1,3,1)
loglog((0:Nx/2-1)*dx,autX1(1:Nx/2),(0:Ny/2-1)*dy,autY1(1:Ny/2),(0:Nz/2-1)*dz,autZ1(1:Nz/2))
hold on 
p=polyfit(log((1:3)*dx), log(autX1(2:4)'),1)
n=p(1);
loglog((1:20)*dx,((1:20)/dx).^n.*dx^n*autX1(2))
xlabel('$d_i/\ell$','interpreter','latex','fontsize',12)
ylabel('S_Q^2')
%line([1 1]/rhoi,get(hax,'YLim'),'Color','k')
%line([1 1]/rhoLH,get(hax,'YLim'),'Color','y')
legend('Q_e:l_x','l_y','l_z',['n=' num2str(n)],'location','southoutside')
grid on
hax=subplot(1,3,2)
loglog(1./(0:Nx/2-1)/dx,autX2(1:Nx/2),1./(0:Ny/2-1)/dy,autY2(1:Ny/2),1./(0:Nz/2-1)/dz,autZ2(1:Nz/2))
hold on 
p=polyfit(log(1./(1:3)/dx), log(autX2(2:4)'),1)
n=p(1);
loglog(1./(1:20)/dx,(1./(1:20)/dx).^n.*dx^n*autX2(2))
xlabel('$d_i/\ell$','interpreter','latex','fontsize',12)
ylabel('S_S^2')
grid on
%line([1 1]/rhoi,get(hax,'YLim'),'Color','k')
%line([1 1]/rhoLH,get(hax,'YLim'),'Color','y')
legend('S:l_x','l_y','l_z',['n=' num2str(n)],'location','southoutside')
hax=subplot(1,3,3)
loglog(1./(0:Nx/2-1)/dx,autX3(1:Nx/2),1./(0:Ny/2-1)/dy,autY3(1:Ny/2),1./(0:Nz/2-1)/dz,autZ3(1:Nz/2))
hold on 
p=polyfit(log(1./(1:3)/dx), log(autX3(2:4)'),1)
n=p(1);
loglog(1./(1:20)/dx,(1./(1:20)/dx).^n.*dx^n*autX3(2))
xlabel('$d_i/\ell$','interpreter','latex','fontsize',12)
ylabel('S_{JeE}^2')
grid on
%line([1 1]/rhoi,get(hax,'YLim'),'Color','k')
%line([1 1]/rhoLH,get(hax,'YLim'),'Color','y')
legend('JeE:l_x','l_y','l_z',['n=' num2str(n)],'location','southoutside')
print('-dpng', '-r300', ['structure_ele' xflow])
return
close all

Qi=Qix.^2+Qiy.^2+Qiz.^2; Qi=sqrt(Qi);
%bbb=Qi.*st;
%figure(1)
%imagesc(bbb(:,:,1))

[autX1,autY1,autZ1]=compute_strct_funct_directional(Qi,st);

%bbb=S.*st;
[autX2,autY2,autZ2]=compute_strct_funct_directional(S,st);

bbb=abs(Jix.*Ex+Jiy.*Ey+Jiz.*Ez);%.*st;
[autX3,autY3,autZ3]=compute_strct_funct_directional(bbb,st);

h=figure(2)
set(h,'Position', [103 3 827 665])
hax=subplot(1,3,1)
loglog(1./(0:Nx/2-1)/dx,autX1(1:Nx/2),1./(0:Ny/2-1)/dy,autY1(1:Ny/2),1./(0:Nz/2-1)/dz,autZ1(1:Nz/2))
hold on 
p=polyfit(log(1./(1:3)/dx), log(autX1(2:4)'),1)
n=p(1);
loglog(1./(1:20)/dx,(1./(1:20)/dx).^n.*dx^n*autX1(2))
xlabel('$d_i/\ell$','interpreter','latex','fontsize',12)
ylabel('S_Q^2')
%line([1 1]/rhoi,get(hax,'YLim'),'Color','k')
%line([1 1]/rhoLH,get(hax,'YLim'),'Color','y')
legend('Q_i:l_x','l_y','l_z',['n=' num2str(n)],'location','southoutside')
grid on
hax=subplot(1,3,2)
loglog(1./(0:Nx/2-1)/dx,autX2(1:Nx/2),1./(0:Ny/2-1)/dy,autY2(1:Ny/2),1./(0:Nz/2-1)/dz,autZ2(1:Nz/2))
hold on 
p=polyfit(log(1./(1:3)/dx), log(autX2(2:4)'),1)
n=p(1);
loglog(1./(1:20)/dx,(1./(1:20)/dx).^n.*dx^n*autX2(2))
xlabel('$d_i/\ell$','interpreter','latex','fontsize',12)
ylabel('S_S^2')
grid on
%line([1 1]/rhoi,get(hax,'YLim'),'Color','k')
%line([1 1]/rhoLH,get(hax,'YLim'),'Color','y')
legend('S:l_x','l_y','l_z',['n=' num2str(n)],'location','southoutside')
hax=subplot(1,3,3)
loglog(1./(0:Nx/2-1)/dx,autX3(1:Nx/2),1./(0:Ny/2-1)/dy,autY3(1:Ny/2),1./(0:Nz/2-1)/dz,autZ3(1:Nz/2))
hold on 
p=polyfit(log(1./(1:3)/dx), log(autX3(2:4)'),1)
n=p(1);
loglog(1./(1:20)/dx,(1./(1:20)/dx).^n.*dx^n*autX3(2))
xlabel('$d_i/\ell$','interpreter','latex','fontsize',12)
ylabel('S_{JiE}^2')
grid on
%line([1 1]/rhoi,get(hax,'YLim'),'Color','k')
%line([1 1]/rhoLH,get(hax,'YLim'),'Color','y')
%legend('JiE:l_x','l_y','l_z',['n=' num2str(n)],'1/\rho_i','1/\rho_{LH}','location','southoutside')
legend('JiE:l_x','l_y','l_z',['n=' num2str(n)],'location','southoutside')
print('-dpng', '-r300', ['structure_ion' xflow])