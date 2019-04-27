function islr=compute_islr(response,interp_sz)
    %width and height of the respone map
    w=interp_sz(2);
    h=interp_sz(1);
    %compute the peak value
    max_score=max(response(:));
    stand=max_score/sqrt(2);
   
    response=fftshift(response);
    [row, col] = find(response == max(response(:)), 1);
    %define the diameters
    dh1=0;
    dh2=0;
    dw1=0;
    dw2=0;
    
    for i=row:1:h
        if(response(i,col)<stand)
            dh1=i-row;
            break;
        end
        if i==h
            dh1=i-row;
        end
    end
    for i=row:-1:1
        if(response(i,col)<stand)
            dh2=row-i;
            break;
        end
        if i==1
            dh2=row-i;
        end
    end
    dh=dh1+dh2;
    rh=dh/2;
    for j=col:1:w
        if(response(row,j)<stand)
            dw1=j-col;
            break;
        end
        if j==w
            dw1=j-col;
        end
    end
    for j=col:-1:1
        if(response(row,j)<stand)
            dw2=col-j;
            break;
        end
        if j==1
            dw2=col-j;
        end
    end
    dw=dw1+dw2;
    rw=dw/2;
    %compute the r
    r=ceil((rh+rw)/2);
    %compute the energy 
    energy_h=0;
    energy_w=0;
    energy_h_10=0;
    energy_w_10=0;
    y_min=max(1,row-r);
    y_max=min(row+r,h);
    x_min=max(1,col-r);
    x_max=min(col+r,w);
    for i=y_min:1:y_max
       energy_h=energy_h+(response(i,col))^2;
    end
    for i=x_min:1:x_max
       energy_w=energy_w+(response(row,i))^2;
    end
    
    y_min=max(1,row-10*r);
    y_max=min(row+10*r,h);
    x_min=max(1,col-10*r);
    x_max=min(col+10*r,w);    
    for i=y_min:1:y_max
       energy_h_10=energy_h_10+(response(i,col))^2;
    end
    for i=x_min:1:x_max
        energy_w_10=energy_w_10+(response(row,i))^2;
    end
    %compute the islr
    islr=-10*log10((energy_h_10/energy_h)*(energy_w_10/energy_w)-1);
end