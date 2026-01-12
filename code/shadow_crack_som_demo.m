clc;
clear all;
close all;
tic;

%% Example: SOM-based crack segmentation on a shadowed concrete image
% Input image should be an RGB photo named 'shadow.png' in the same folder.
I = imread('shadow.png');
Igray = int16(rgb2gray(I));
[m, n] = size(Igray);

%% Feature 1: Grayscale (cropped to avoid boundary)
Igraycut = Igray(3:m-2, 3:n-2);
Igraysom = reshape(Igraycut, 1, (m-4)*(n-4));

%% Feature 2: Local contrast (max diff to 3x3 mean in 5x5 neighborhood)
for i = 3:m-2
   for j = 3:n-2
        ma(i,j) = ( ...
            Igray(i-1,j-1)+Igray(i-1,j)+Igray(i-1,j+1)+ ...
            Igray(i  ,j-1)+Igray(i  ,j)+Igray(i  ,j+1)+ ...
            Igray(i+1,j-1)+Igray(i+1,j)+Igray(i+1,j+1) );

        dddd1 (i,j) = abs(ma(i,j) - Igray(i-2,j-2));
        dddd2 (i,j) = abs(ma(i,j) - Igray(i-2,j-1));
        dddd3 (i,j) = abs(ma(i,j) - Igray(i-2,j  ));
        dddd4 (i,j) = abs(ma(i,j) - Igray(i-2,j+1));
        dddd5 (i,j) = abs(ma(i,j) - Igray(i-2,j+2));
        dddd6 (i,j) = abs(ma(i,j) - Igray(i-1,j-2));
        dddd7 (i,j) = abs(ma(i,j) - Igray(i-1,j+2));
        dddd8 (i,j) = abs(ma(i,j) - Igray(i  ,j-2));
        dddd9 (i,j) = abs(ma(i,j) - Igray(i  ,j+2));
        dddd10(i,j) = abs(ma(i,j) - Igray(i+1,j-2));
        dddd11(i,j) = abs(ma(i,j) - Igray(i+1,j+2));
        dddd12(i,j) = abs(ma(i,j) - Igray(i+2,j-2));
        dddd13(i,j) = abs(ma(i,j) - Igray(i+2,j-1));
        dddd14(i,j) = abs(ma(i,j) - Igray(i+2,j  ));
        dddd15(i,j) = abs(ma(i,j) - Igray(i+2,j+1));
        dddd16(i,j) = abs(ma(i,j) - Igray(i+2,j+2));

        ddddp1(i,j) = max(max(max(max(max(max(max( ...
            dddd1(i,j), dddd2(i,j)), dddd3(i,j)), dddd4(i,j)), ...
            dddd5(i,j)), dddd6(i,j)), dddd7(i,j)), dddd8(i,j));

        ddddp2(i,j) = max(max(max(max(max(max(max( ...
            dddd9(i,j), dddd10(i,j)), dddd11(i,j)), dddd12(i,j)), ...
            dddd13(i,j)), dddd14(i,j)), dddd15(i,j)), dddd16(i,j));

        dddmax(i,j) = max(ddddp1(i,j), ddddp2(i,j));
   end
end
Cnew           = dddmax(3:m-2, 3:n-2);
Contrastnewsom = reshape(Cnew, 1, (m-4)*(n-4));

%% Feature 3: Edge sharpness (combined directional responses)
for i = 3:m-2
   for j = 3:n-2
       A(i,j) = abs(Igray(i  ,j-1)+Igray(i+1,j-1)+Igray(i-1,j-1) ...
                  - Igray(i  ,j+1)-Igray(i-1,j+1)-Igray(i+1,j+1));
       B(i,j) = abs(Igray(i-1,j-1)+Igray(i-1,j)+Igray(i-1,j+1) ...
                  - Igray(i+1,j-1)-Igray(i+1,j)-Igray(i+1,j+1));
       C(i,j) = abs(Igray(i  ,j-1)+Igray(i+1,j-1)+Igray(i+1,j) ...
                  - Igray(i-1,j  )-Igray(i-1,j+1)-Igray(i  ,j+1));
       D(i,j) = abs(Igray(i-1,j-1)+Igray(i-1,j)+Igray(i  ,j-1) ...
                  - Igray(i  ,j+1)-Igray(i+1,j  )-Igray(i+1,j+1));
       Sc3(i,j) = max(max(max(A(i,j),B(i,j)),C(i,j)),D(i,j));

       AA(i,j) = abs(2*(Igray(i-1,j-1)+Igray(i  ,j-1)) ...
                  - (Igray(i-1,j+1)+Igray(i  ,j+1)+Igray(i+1,j)+Igray(i+1,j+1)));
       BB(i,j) = abs(2*(Igray(i-1,j+1)+Igray(i  ,j+1)) ...
                  - (Igray(i-1,j-1)+Igray(i  ,j-1)+Igray(i+1,j-1)+Igray(i+1,j)));
       CC(i,j) = abs(2*(Igray(i  ,j-1)+Igray(i+1,j-1)) ...
                  - (Igray(i+1,j+1)+Igray(i  ,j+1)+Igray(i-1,j+1)+Igray(i-1,j)));
       DD(i,j) = abs(2*(Igray(i  ,j+1)+Igray(i+1,j+1)) ...
                  - (Igray(i-1,j-1)+Igray(i-1,j)+Igray(i  ,j-1)+Igray(i+1,j-1)));
       EE(i,j) = abs(2*(Igray(i-1,j-1)+Igray(i-1,j)) ...
                  - (Igray(i+1,j-1)+Igray(i+1,j)+Igray(i+1,j+1)+Igray(i  ,j+1)));
       FF(i,j) = abs(2*(Igray(i-1,j)+Igray(i-1,j+1)) ...
                  - (Igray(i  ,j-1)+Igray(i+1,j-1)+Igray(i+1,j)+Igray(i+1,j+1)));
       GG(i,j) = abs(2*(Igray(i+1,j-1)+Igray(i+1,j)) ...
                  - (Igray(i-1,j-1)+Igray(i-1,j)+Igray(i-1,j+1)+Igray(i  ,j+1)));
       HH(i,j) = abs(2*(Igray(i+1,j)+Igray(i+1,j+1)) ...
                  - (Igray(i  ,j-1)+Igray(i-1,j-1)+Igray(i-1,j)+Igray(i-1,j+1)));

       Sc4(i,j) = max(max(max(max(max(max(max( ...
           AA(i,j),BB(i,j)),CC(i,j)),DD(i,j)),EE(i,j)),FF(i,j)),GG(i,j)),HH(i,j));
       Contrast3(i,j) = max(Sc3(i,j),Sc4(i,j));
   end
end
C3            = Contrast3(3:m-2, 3:n-2);
Normaledgesom = reshape(C3, 1, (m-4)*(n-4));

%% Feature 4: Thinness (directional intensity ratio)
for i = 3:m-2
    for j = 3:n-2
        ao(i,j) = (Igray(i  ,j  )+Igray(i+1,j-1)+Igray(i-1,j+1))/3;
        bo(i,j) = (Igray(i-1,j-1)+Igray(i-1,j)+Igray(i  ,j-1)+ ...
                   Igray(i  ,j+1)+Igray(i+1,j)+Igray(i+1,j+1))/6;
        ratioo(i,j) = ao(i,j)/bo(i,j);

        ap(i,j) = (Igray(i-1,j  )+Igray(i  ,j)+Igray(i+1,j))/3;
        bp(i,j) = (Igray(i-1,j-1)+Igray(i  ,j-1)+Igray(i+1,j-1)+ ...
                   Igray(i-1,j+1)+Igray(i  ,j+1)+Igray(i+1,j+1))/6;
        ratiop(i,j) = ap(i,j)/bp(i,j);

        aq(i,j) = (Igray(i-1,j-1)+Igray(i  ,j)+Igray(i+1,j+1))/3;
        bq(i,j) = (Igray(i-1,j  )+Igray(i  ,j+1)+Igray(i  ,j-1)+ ...
                   Igray(i  ,j+1)+Igray(i+1,j-1)+Igray(i+1,j  ))/6;
        ratioq(i,j) = aq(i,j)/bq(i,j);

        ar(i,j) = (Igray(i  ,j-1)+Igray(i  ,j)+Igray(i  ,j+1))/3;
        br(i,j) = (Igray(i-1,j-1)+Igray(i-1,j)+Igray(i-1,j+1)+ ...
                   Igray(i+1,j-1)+Igray(i+1,j)+Igray(i+1,j+1))/6;
        ratior(i,j) = ar(i,j)/br(i,j);

        as(i,j) = (Igray(i-1,j  )+Igray(i  ,j)+Igray(i+1,j-1))/3;
        bs(i,j) = (Igray(i-1,j-1)+Igray(i-1,j+1)+Igray(i  ,j-1)+ ...
                   Igray(i  ,j+1)+Igray(i+1,j)+Igray(i+1,j+1))/6;
        ratios(i,j) = as(i,j)/bs(i,j);

        at(i,j) = (Igray(i  ,j-1)+Igray(i  ,j)+Igray(i+1,j+1))/3;
        bt(i,j) = (Igray(i-1,j-1)+Igray(i-1,j)+Igray(i-1,j+1)+ ...
                   Igray(i  ,j+1)+Igray(i+1,j-1)+Igray(i+1,j+1))/6;
        ratiot(i,j) = at(i,j)/bt(i,j);

        au(i,j) = (Igray(i-1,j+1)+Igray(i  ,j)+Igray(i+1,j))/3;
        bu(i,j) = (Igray(i-1,j-1)+Igray(i-1,j)+Igray(i  ,j-1)+ ...
                   Igray(i  ,j+1)+Igray(i-1,j+1)+Igray(i+1,j+1))/6;
        ratiou(i,j) = au(i,j)/bu(i,j);

        av(i,j) = (Igray(i-1,j-1)+Igray(i  ,j)+Igray(i  ,j+1))/3;
        bv(i,j) = (Igray(i-1,j  )+Igray(i-1,j+1)+Igray(i  ,j-1)+ ...
                   Igray(i+1,j-1)+Igray(i+1,j)+Igray(i+1,j+1))/6;
        ratiov(i,j) = av(i,j)/bv(i,j);

        aw(i,j) = (Igray(i-1,j  )+Igray(i  ,j)+Igray(i+1,j+1))/3;
        bw(i,j) = (Igray(i-1,j-1)+Igray(i-1,j+1)+Igray(i  ,j-1)+ ...
                   Igray(i  ,j+1)+Igray(i+1,j-1)+Igray(i+1,j  ))/6;
        ratiow(i,j) = aw(i,j)/bw(i,j);

        ax(i,j) = (Igray(i  ,j-1)+Igray(i  ,j)+Igray(i-1,j+1))/3;
        bx(i,j) = (Igray(i-1,j-1)+Igray(i-1,j)+Igray(i  ,j+1)+ ...
                   Igray(i+1,j-1)+Igray(i+1,j)+Igray(i+1,j+1))/6;
        ratiox(i,j) = ax(i,j)/bx(i,j);

        ay(i,j) = (Igray(i-1,j-1)+Igray(i  ,j)+Igray(i+1,j))/3;
        by(i,j) = (Igray(i-1,j  )+Igray(i-1,j+1)+Igray(i  ,j-1)+ ...
                   Igray(i  ,j+1)+Igray(i+1,j+1)+Igray(i+1,j-1))/6;
        ratioy(i,j) = ay(i,j)/by(i,j);

        az(i,j) = (Igray(i+1,j-1)+Igray(i  ,j)+Igray(i  ,j+1))/3;
        bz(i,j) = (Igray(i-1,j-1)+Igray(i-1,j)+Igray(i-1,j+1)+ ...
                   Igray(i  ,j-1)+Igray(i+1,j)+Igray(i+1,j+1))/6;
        ratioz(i,j) = az(i,j)/bz(i,j);

        thin(i,j) = min(min(min(min(min(min(min(min(min(min(min( ...
            ratioo(i,j), ratiop(i,j)), ratioq(i,j)), ratior(i,j)), ...
            ratios(i,j)), ratiot(i,j)), ratiou(i,j)), ratiov(i,j)), ...
            ratiow(i,j)), ratiox(i,j)), ratioy(i,j)), ratioz(i,j));
    end
end
Th      = thin(3:m-2, 3:n-2) * 100;
Thinsom = reshape(Th, 1, (m-4)*(n-4));

%% Feature 5–6: Local mean over 3x3 and 5x5 neighborhoods
for i = 3:m-2
   for j = 3:n-2
       sum8neighbors(i,j) = ( ...
           Igray(i-1,j-1)+Igray(i-1,j)+Igray(i-1,j+1)+ ...
           Igray(i  ,j-1)+Igray(i  ,j)+Igray(i  ,j+1)+ ...
           Igray(i+1,j-1)+Igray(i+1,j)+Igray(i+1,j+1) )/9;
   end
end
Sum1    = sum8neighbors(3:m-2, 3:n-2);
Sum1som = reshape(Sum1, 1, (m-4)*(n-4));

for i = 3:m-2
   for j = 3:n-2
       sum24neighbors(i,j) = ( ...
           Igray(i-2,j-2)+Igray(i-2,j-1)+Igray(i-2,j)+Igray(i-2,j+1)+Igray(i-2,j+2)+ ...
           Igray(i-1,j-2)+Igray(i-1,j-1)+Igray(i-1,j)+Igray(i-1,j+1)+Igray(i-1,j+2)+ ...
           Igray(i  ,j-2)+Igray(i  ,j-1)+Igray(i  ,j)+Igray(i  ,j+1)+Igray(i  ,j+2)+ ...
           Igray(i+1,j-2)+Igray(i+1,j-1)+Igray(i+1,j)+Igray(i+1,j+1)+Igray(i+1,j+2)+ ...
           Igray(i+2,j-2)+Igray(i+2,j-1)+Igray(i+2,j)+Igray(i+2,j+1)+Igray(i+2,j+2) )/25;
   end
end
Sum2    = sum24neighbors(3:m-2, 3:n-2);
Sum2som = reshape(Sum2, 1, (m-4)*(n-4));

%% Feature 7–8: Row/column normalized grayscale
row    = median(Igray, 2);
column = median(Igray, 1);
for i = 3:m-2
    for j = 3:n-2
        p(i,j) = Igray(i,j) * 100 ./ row(i,1);     % vs row median
        q(i,j) = Igray(i,j) * 100 ./ column(1,j);  % vs column median
    end
end
p    = p(3:m-2, 3:n-2);
q    = q(3:m-2, 3:n-2);
psom = reshape(p, 1, (m-4)*(n-4));
qsom = reshape(q, 1, (m-4)*(n-4));

%% Feature 9: Hue (from RGB)
r = I(:,:,1);
g = I(:,:,2);
b = I(:,:,3);
for i = 3:m-2
   for j = 3:n-2
        if  r(i,j) >  g(i,j) && g(i,j) >  b(i,j)
            Hue(i,j) = 60 * (g(i,j)-b(i,j)) / (r(i,j)-b(i,j));
        elseif r(i,j) >= g(i,j) && g(i,j) <= b(i,j)
            Hue(i,j) = 60 * (g(i,j)-b(i,j)) / (r(i,j)-g(i,j));
        elseif g(i,j) >  r(i,j) && r(i,j) >  b(i,j)
            Hue(i,j) = 120 + 60 * (b(i,j)-r(i,j)) / (g(i,j)-b(i,j));
        elseif g(i,j) >= r(i,j) && r(i,j) <= b(i,j)
            Hue(i,j) = 120 + 60 * (b(i,j)-r(i,j)) / (g(i,j)-r(i,j));
        elseif  b(i,j) >  r(i,j) && r(i,j) >  g(i,j)
            Hue(i,j) = 240 + 60 * (r(i,j)-g(i,j)) / (b(i,j)-g(i,j));
        elseif  b(i,j) >= r(i,j) && r(i,j) <= g(i,j)
            Hue(i,j) = 240 + 60 * (r(i,j)-g(i,j)) / (b(i,j)-r(i,j));
        end
   end
end
Hue    = Hue(3:m-2, 3:n-2);
Huesom = reshape(Hue, 1, (m-4)*(n-4));

%% Assemble 9-D SOM feature vector
input = double([ ...
    Igraysom; Contrastnewsom; Normaledgesom; Thinsom; ...
    Sum1som;  Sum2som;      psom;           qsom;   Huesom ]);

%% Train SOM (5 units for shadowed cracks + backgrounds)
nClasses = 5;
net      = selforgmap(nClasses);
[net, tr] = train(net, input);
op        = vec2ind(net(input));
weight    = getwb(net);
[b, Iw, Lw] = separatewb(net, weight); 
z         = dist(Iw{1,1}, input);
euclidean = min(z);                     
[ro, col] = find(z == euclidean);       
op_som    = reshape(op, m-4, n-4);

%% Visualize SOM classes (optional)
figure;
for k = 1:nClasses
    classMask = uint8(255 * ones(size(op_som)));
    classMask(op_som == k) = 0;
    subplot(2,3,k);
    imshow(classMask);
    title(sprintf('%d-th class', k));
end

%% Automatically select the crack class (darkest mean intensity)
Igraycut = Igray(3:m-2, 3:n-2);   % same crop as features
mean_intensity = zeros(1, nClasses);
for k = 1:nClasses
    mean_intensity(k) = mean(Igraycut(op_som == k), 'all');
end
[~, crack_class] = min(mean_intensity);
fprintf('Detected crack class: %d\n', crack_class);

%% Binary crack mask and area-based cleaning
I_bin = (op_som == crack_class);       % logical: 1 = crack, 0 = background

[L, num] = bwlabel(I_bin);
stats    = regionprops(L, 'Area');
Area     = [stats.Area];

total_crack_area = sum(Area);
thresh           = 0.01 * total_crack_area;

I_clean = false(size(I_bin));
for i = 1:num
    if Area(i) >= thresh
        I_clean(L == i) = true;
    end
end

figure;
imshowpair(I_bin, I_clean, 'montage');
title('Left: Original Binary | Right: After Cleaning');

% Invert for visualization: cracks as black
I_clean = ~I_clean;
figure;
imshow(I_clean);

toc;
 