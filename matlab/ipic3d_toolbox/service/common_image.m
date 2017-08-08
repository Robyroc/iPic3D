function [J]=common_image(x,y,J1,Az,nlabel,name,clmt, Nsm, nfig)

global color_choice symmetric_color labelx labely labelc reversex Ncycle


J=smoothbc(J1,Nsm);

if(nargin>8)
h=figure(nfig)
else
h=figure(1)
end

%set(h, 'Position', [167 26 515 776])
set(h, 'Position',[204 376 764 429])

if(clmt(1) == clmt(2))
clmt(1)=min(J(:));
clmt(2)=max(J(:));
if(symmetric_color==1)
clmt(1)=-max(abs(clmt))
clmt(2)=-clmt(1)
elseif (symmetric_color==-1)
clmt(1)=-min(abs(clmt))
clmt(2)=-clmt(1)
end
end

Ncut=max(Nsm*3,1)

imagesc(x(1,:),y(:,1),J(Ncut:end-Ncut,Ncut:end-Ncut)',clmt)
hold on
contour(x,y,Az',30,'k')
caxis(clmt)

if(color_choice==0)
        colormap jet
elseif (color_choice==1)
        load cm_new
        colormap(cm_kbwrk)
elseif (color_choice==2)
        colormap hot
        elseif (color_choice==3)
        load cm_multi4
        colormap(cm_cool_hot_2)
elseif (color_choice==-1)
        colormap parula
end

ch=colorbar
set(get(ch,'title'),'string',labelc);

set(gca,'fontsize',[14])


xlabel(labelx,'fontsize',[14])
ylabel(labely,'fontsize',[14])


title([ nlabel ],'fontsize',[14])


axis image
axis xy
if(reversex==1) 
    set(gca,'xdir','reverse')
end

%set(gca,'xdir','reverse','TickDir','out')
%print('-depsc','-r300',[name '.eps'])
print('-dpng','-r300',[name Ncycle '.png'])
%saveas(gcf,[name '.fig'])
