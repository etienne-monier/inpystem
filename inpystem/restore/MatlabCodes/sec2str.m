function res = sec2str(t)
    
    h = floor(t/3600);
    m = floor( (t-3600*h)/60 );
    s = t - 3600 * h - 60 * m;

    if h == 0 && m == 0
        res = [ num2str(s, '%05.2f') 's' ];
    elseif h == 0 && m ~= 0
        res = [ num2str(m, '%02d') 'm ' num2str(s, '%05.2f') 's' ];
    else
        res = [ num2str(m, '%d') 'h ' num2str(m, '%02d') 'm ' num2str(s, '%05.2f') 's' ];
    end

end