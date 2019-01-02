function surface_contour_plot()
   while 1 ==1

    %u = udp('10.0.0.2', 'LocalHost', '', 'LocalPort', 8037, 'RemotePort', 8023);
    u = udp('', 'LocalHost', '', 'LocalPort', 8037, 'RemotePort', 8023);
    u.ByteOrder = 'littleEndian';
    set(u,'DatagramTerminateMode','off', ...
          'InputBufferSize', 2048, ...
          'Timeout', 5); % I think only one call of set is needed here

    fopen(u);

    meta = fread(u, 2, 'int')

    if size(meta, 1) == 2
        meta
        fid = meta(1);
        l = meta(2);

        data = zeros(l, 1, 'double');

        packetSize = 10;

        c = 0;
        while c < l
            if l - c >= packetSize
                [fdata, count] = fread(u, packetSize, 'double');
                data(c+1:c+count) = fdata;
                c = c + count;
            else
                [fdata, count] = fread(u, 1, 'double');
                data(c+1:c+count) = fdata;
                c = c + count;
            end

            if count < packetSize
                packetSize = 1;
            end
        end

        data = reshape(data,[4, size(data, 1) / 4])'
        %data = reshape(data,[4, size(data, 1) / 5])'
        
        figure('rend','painters','pos',[10 10 500 500], 'color', [247 / 255 247 / 255 247 / 255])
        set(gca,'LooseInset',get(gca,'TightInset'))
        
        [X,Y,Z, Level] = fs(fid);
        
        hold on;
        s = surf(X,Y,Z); 
        
        if fid == 0
            view(63,30);
        elseif fid == 1
            view(70,50);
        elseif fid == 2
            view(80,36);
        elseif fid == 3
            view(40,30);
        elseif fid == 4
            view(-54,32);
        elseif fid == 5
            view(-40,30);
        elseif fid == 6
            view(-20,45);
        elseif fid == 7
            view(-40,25);
        end

        s.EdgeColor = 'none';
        s.FaceColor = 'none';

        [j,h] = contourf(X, Y, Z);
        h.LineColor = 'none';
        h.LineStyle = 'none';
        h.ContourZLevel = Level;
        h.ZData = zeros(size(h.ZData))

        set(gca,'xtick',[])
        set(gca,'ytick',[])
        set(gca,'Visible','off')
        set(gca,'color',[247 / 255 247 / 255 247 / 255])
    
        fig = gcf;
        fig.Color = [247 / 255 247 / 255 247 / 255];
        fig.InvertHardcopy = 'off';

        window = 254;        
        indices = [];
        for p=1:window:size(data(:, 2), 1)
            
            pw = p + window;
            if pw >= size(data(:, 2), 1)
                pw = size(data(:, 2), 1);
            end

            objarray = [];
            
            pc = 1;
            for ps = p:pw
                %objarray = [objarray; plot3(data(ps, 2), data(ps, 3), data(ps, 4), 'Marker', '.', 'LineStyle', 'none', 'Color', [0, 0, pc * (1 / 255)], 'MarkerSize', 0.0000000000001, 'LineWidth', 0.0000000000001)];
                objarray = [objarray; plot3(data(ps, 3), data(ps, 4), data(ps, 1), 'Marker', '.', 'LineStyle', 'none', 'Color', [0, 0, pc * (1 / 255)], 'MarkerSize', 0.0000000000001, 'LineWidth', 0.0000000000001)];
                disp('plot')
                [data(ps, 3), data(ps, 4), data(ps, 1)]
                pc = pc + 1;
            end
            
            F = getframe(gcf);
            [X, Map] = frame2im(F);
            idx = find(X(:,:,2) == 0);  % ~=
            
            
            
            csvwrite('output2.csv', X(:,:,2))
            csvwrite('output3.csv', X(:,:,3))
            %asoqweoed()
            
            idx
            HL = X(:,:,2);
            unique(HL(:))
            
            B = X(:,:,3);
            [~,sidx] = sort(B(idx));
            idx = idx(sidx);
            
            indices = [indices; idx];
            
            if pw ~= size(data(:, 2), 1)
                disp('delete')
                for ps = 1:size(objarray, 1)
                    delete(objarray(ps));
                end
            end         
        end
        
        size(indices)
        indices
        

        packetSize = 20;
        fwrite(u, size(indices, 1), 'int32');
        for i=1:packetSize:size(indices, 1)
            if (i + packetSize) < size(indices, 1)
                fwrite(u, indices(i:i+packetSize-1), 'int32');
            else
                fwrite(u, indices(i:size(indices, 1)), 'int32');
            end
        end
        
        close(gcf)
    end
    
    % Clean up
    fclose(u);
    delete(u);
    clear u

    end
end

function [X,Y,Z, Level] = fs(id)
    if id == 0
        [X,Y,Z, Level] = booth_function();
    elseif id == 1
        [X,Y,Z, Level] = styblinski_tang_function();
    elseif id == 2
        [X,Y,Z, Level] = bukin_function();
    elseif id == 3
        [X,Y,Z, Level] = dropwave_function();
    elseif id == 4
        [X,Y,Z, Level] = mc_cormick_function();
    elseif id == 5
        [X,Y,Z, Level] = rastrigin_function();
    elseif id == 6
        [X,Y,Z, Level] = sphere_function();
    elseif id == 7
        [X,Y,Z, Level] = easom_function();
    end
end

function [X,Y,Z, Level] = booth_function()
    [X,Y] = meshgrid(-11:0.3:11);
    Z = (X + 2 .* Y - 7).^2 + (2 .* X + Y - 5).^2;
    Level = -2500;
end

function [X,Y,Z, Level] = styblinski_tang_function()
    [X,Y] = meshgrid(-5:0.05:5);
    Z = 0.5 * ((X.^4 - 16 .* X.^2 + 5 .* X) + (Y.^4 - 16 .* Y.^2 + 5 .* Y));
    Z = Z .* 4;
    Level = -1000;
end

function [X,Y,Z, Level] = bukin_function()
    [X,Y] = meshgrid(-10:0.1:10);
    Z1 = 100 .* sqrt(abs(Y - 0.01 .* Y.^2));
    Z2 = 0.01 * abs(X + 10);
    Z = Z1 + Z2;
    Level = -300;
end

function [X,Y,Z, Level] = dropwave_function()
    [X,Y] = meshgrid(-6:0.05:6);
    Z1 = 1 + cos(12*sqrt(X.^2+Y.^2));
    Z2 = 0.5 .* (X.^2+Y.^2) + 2;
    Z = -Z1 ./ Z2;
    Level = -1;
end

function [X,Y,Z, Level] = mc_cormick_function()
    [X,Y] = meshgrid(-4:0.05:4);
    Z1 = sin(X + Y);
    Z2 = (X - Y).^2;
    Z3 = -1.5 .* X;
    Z4 = 2.5 .* Y;
    Z = Z1 + Z2 + Z3 + Z4 + 1;
    Level = -150;
end

function [X,Y,Z, Level] = rastrigin_function()
    [X,Y] = meshgrid(-5:0.05:5);
    Z = 10 .* 2.0 + ((X.^2 - 10 .* cos(2 .* pi .* X)) + (Y.^2 - 10 .* cos(2 .* pi .* Y)));    
    Level = -50;
end

function [X,Y,Z, Level] = sphere_function()
    [X,Y] = meshgrid(-6:0.1:6);
    Z = X .^2 + Y.^2;
    Level = -50;
end

function [X,Y,Z, Level] = easom_function()
    [X,Y] = meshgrid(-20:0.08:20);
    Z1 = -cos(X) .* cos(Y);
    Z2 = exp(-(X - pi).^2 - (Y - pi).^2);
    Z = Z1 .* Z2;
    Level = -1;
end
