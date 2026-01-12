clc;
clear all;
tic;

% ================================================================
% DEMO: SOM-based crack segmentation on a single concrete image
%
% - Extract 9-D pixel features from an RGB concrete image
% - Train a 3-node Self-Organizing Map (SOM) on these features
% - Visualize the 3 SOM classes and a cleaned crack mask
%
% Usage:
%   1) Put 'concrete.png' in the same folder as this script.
%   2) Run this script.
%   3) The script will show:
%        - Three SOM classes
%        - Binary crack mask before/after small-component removal
%
% Requirements:
%   - MATLAB
%   - Deep Learning Toolbox (selforgmap, train, vec2ind)
%
% This demo focuses on segmentation and post-processing. If you need
% a crack-area ratio, you can compute it directly from I_clean.
% ================================================================


% Read image and convert to grayscale
I      = imread('concrete.png');
Igray  = int16(rgb2gray(I));
[m, n] = size(Igray);

%% 1) Crop 2-pixel border (to avoid boundary issues in neighborhoods)
Igraycut = Igray(3:m-2, 3:n-2);
Igraysom = reshape(Igraycut, 1, (m-4)*(n-4));   % feature 1: grayscale

%% 2) Local contrast feature (max deviation from 3x3 mean in a 5x5 neighborhood)
for i = 3:m-2
   for j = 3:n-2
        % 3x3 local mean around (i,j)
        ma(i,j) = ( ...
            Igray(i-1,j-1)+Igray(i-1,j)+Igray(i-1,j+1)+ ...
            Igray(i  ,j-1)+Igray(i  ,j)+Igray(i  ,j+1)+ ...
            Igray(i+1,j-1)+Igray(i+1,j)+Igray(i+1,j+1) );

        % Differences between local mean and all 16 positions on the
        % outer ring of a 5x5 neighborhood
        d1  = abs(ma(i,j) - Igray(i-2,j-2));
        d2  = abs(ma(i,j) - Igray(i-2,j-1));
        d3  = abs(ma(i,j) - Igray(i-2,j  ));
        d4  = abs(ma(i,j) - Igray(i-2,j+1));
        d5  = abs(ma(i,j) - Igray(i-2,j+2));
        d6  = abs(ma(i,j) - Igray(i-1,j-2));
        d7  = abs(ma(i,j) - Igray(i-1,j+2));
        d8  = abs(ma(i,j) - Igray(i  ,j-2));
        d9  = abs(ma(i,j) - Igray(i  ,j+2));
        d10 = abs(ma(i,j) - Igray(i+1,j-2));
        d11 = abs(ma(i,j) - Igray(i+1,j+2));
        d12 = abs(ma(i,j) - Igray(i+2,j-2));
        d13 = abs(ma(i,j) - Igray(i+2,j-1));
        d14 = abs(ma(i,j) - Igray(i+2,j  ));
        d15 = abs(ma(i,j) - Igray(i+2,j+1));
        d16 = abs(ma(i,j) - Igray(i+2,j+2));

        % Split into two 8-neighbor groups and take max of each group
        d_group1 = max([d1 d2 d3 d4 d5 d6 d7 d8]);
        d_group2 = max([d9 d10 d11 d12 d13 d14 d15 d16]);

        % Overall local contrast at (i,j)
        dddmax(i,j) = max(d_group1, d_group2);
   end
end
Cnew          = dddmax(3:m-2, 3:n-2);
Contrastnewsom = reshape(Cnew, 1, (m-4)*(n-4));   % feature 2: local contrast

%% 3) Edge / line-strength feature (multiple oriented configurations)
for i = 3:m-2
   for j = 3:n-2
       % 4 basic oriented responses (Sc3)
       A(i,j) = abs(Igray(i  ,j-1)+Igray(i+1,j-1)+Igray(i-1,j-1) ...
                   - Igray(i  ,j+1)-Igray(i-1,j+1)-Igray(i+1,j+1));
       B(i,j) = abs(Igray(i-1,j-1)+Igray(i-1,j)+Igray(i-1,j+1) ...
                   - Igray(i+1,j-1)-Igray(i+1,j)-Igray(i+1,j+1));
       C(i,j) = abs(Igray(i  ,j-1)+Igray(i+1,j-1)+Igray(i+1,j) ...
                   - Igray(i-1,j  )-Igray(i-1,j+1)-Igray(i  ,j+1));
       D(i,j) = abs(Igray(i-1,j-1)+Igray(i-1,j)+Igray(i  ,j-1) ...
                   - Igray(i  ,j+1)-Igray(i+1,j  )-Igray(i+1,j+1));

       Sc3(i,j) = max([A(i,j) B(i,j) C(i,j) D(i,j)]);

       % 8 additional oriented responses (Sc4)
       AA(i,j) = abs(2*(Igray(i-1,j-1)+Igray(i  ,j-1)) ...
                    - (Igray(i-1,j+1)+Igray(i  ,j+1)+Igray(i+1,j  )+Igray(i+1,j+1)));
       BB(i,j) = abs(2*(Igray(i-1,j+1)+Igray(i  ,j+1)) ...
                    - (Igray(i-1,j-1)+Igray(i  ,j-1)+Igray(i+1,j-1)+Igray(i+1,j  )));
       CC(i,j) = abs(2*(Igray(i  ,j-1)+Igray(i+1,j-1)) ...
                    - (Igray(i+1,j+1)+Igray(i  ,j+1)+Igray(i-1,j+1)+Igray(i-1,j  )));
       DD(i,j) = abs(2*(Igray(i  ,j+1)+Igray(i+1,j+1)) ...
                    - (Igray(i-1,j-1)+Igray(i-1,j  )+Igray(i  ,j-1)+Igray(i+1,j-1)));
       EE(i,j) = abs(2*(Igray(i-1,j-1)+Igray(i-1,j  )) ...
                    - (Igray(i+1,j-1)+Igray(i+1,j  )+Igray(i+1,j+1)+Igray(i  ,j+1)));
       FF(i,j) = abs(2*(Igray(i-1,j  )+Igray(i-1,j+1)) ...
                    - (Igray(i  ,j-1)+Igray(i+1,j-1)+Igray(i+1,j  )+Igray(i+1,j+1)));
       GG(i,j) = abs(2*(Igray(i+1,j-1)+Igray(i+1,j  )) ...
                    - (Igray(i-1,j-1)+Igray(i-1,j  )+Igray(i-1,j+1)+Igray(i  ,j+1)));
       HH(i,j) = abs(2*(Igray(i+1,j  )+Igray(i+1,j+1)) ...
                    - (Igray(i  ,j-1)+Igray(i-1,j-1)+Igray(i-1,j  )+Igray(i-1,j+1)));

       Sc4(i,j) = max([AA(i,j) BB(i,j) CC(i,j) DD(i,j) ...
                       EE(i,j) FF(i,j) GG(i,j) HH(i,j)]);

       % Final edge/line-strength feature
       Contrast3(i,j) = max(Sc3(i,j), Sc4(i,j));
   end
end
C3            = Contrast3(3:m-2, 3:n-2);
Normaledgesom = reshape(C3, 1, (m-4)*(n-4));      % feature 3: edge strength

%% 4) Thinness feature (ratio-based across many directions)
for i = 3:m-2
    for j = 3:n-2
        % Each pair (a?, b?) defines a directional contrast ratio
        ao(i,j) = (Igray(i  ,j  )+Igray(i+1,j-1)+Igray(i-1,j+1))/3;
        bo(i,j) = (Igray(i-1,j-1)+Igray(i-1,j  )+Igray(i  ,j-1)+ ...
                   Igray(i  ,j+1)+Igray(i+1,j  )+Igray(i+1,j+1))/6;
        ratioo(i,j) = ao(i,j)/bo(i,j);

        ap(i,j) = (Igray(i-1,j  )+Igray(i  ,j  )+Igray(i+1,j  ))/3;
        bp(i,j) = (Igray(i-1,j-1)+Igray(i  ,j-1)+Igray(i+1,j-1)+ ...
                   Igray(i-1,j+1)+Igray(i  ,j+1)+Igray(i+1,j+1))/6;
        ratiop(i,j) = ap(i,j)/bp(i,j);

        aq(i,j) = (Igray(i-1,j-1)+Igray(i  ,j  )+Igray(i+1,j+1))/3;
        bq(i,j) = (Igray(i-1,j  )+Igray(i  ,j+1)+Igray(i  ,j-1)+ ...
                   Igray(i  ,j+1)+Igray(i+1,j-1)+Igray(i+1,j  ))/6;
        ratioq(i,j) = aq(i,j)/bq(i,j);

        ar(i,j) = (Igray(i  ,j-1)+Igray(i  ,j  )+Igray(i  ,j+1))/3;
        br(i,j) = (Igray(i-1,j-1)+Igray(i-1,j  )+Igray(i-1,j+1)+ ...
                   Igray(i+1,j-1)+Igray(i+1,j  )+Igray(i+1,j+1))/6;
        ratior(i,j) = ar(i,j)/br(i,j);

        as(i,j) = (Igray(i-1,j  )+Igray(i  ,j  )+Igray(i+1,j-1))/3;
        bs(i,j) = (Igray(i-1,j-1)+Igray(i-1,j+1)+Igray(i  ,j-1)+ ...
                   Igray(i  ,j+1)+Igray(i+1,j  )+Igray(i+1,j+1))/6;
        ratios(i,j) = as(i,j)/bs(i,j);

        at(i,j) = (Igray(i  ,j-1)+Igray(i  ,j  )+Igray(i+1,j+1))/3;
        bt(i,j) = (Igray(i-1,j-1)+Igray(i-1,j  )+Igray(i-1,j+1)+ ...
                   Igray(i  ,j+1)+Igray(i+1,j-1)+Igray(i+1,j+1))/6;
        ratiot(i,j) = at(i,j)/bt(i,j);

        au(i,j) = (Igray(i-1,j+1)+Igray(i  ,j  )+Igray(i+1,j  ))/3;
        bu(i,j) = (Igray(i-1,j-1)+Igray(i-1,j  )+Igray(i  ,j-1)+ ...
                   Igray(i  ,j+1)+Igray(i-1,j+1)+Igray(i+1,j+1))/6;
        ratiou(i,j) = au(i,j)/bu(i,j);

        av(i,j) = (Igray(i-1,j-1)+Igray(i  ,j  )+Igray(i  ,j+1))/3;
        bv(i,j) = (Igray(i-1,j  )+Igray(i-1,j+1)+Igray(i  ,j-1)+ ...
                   Igray(i+1,j-1)+Igray(i+1,j  )+Igray(i+1,j+1))/6;
        ratiov(i,j) = av(i,j)/bv(i,j);

        aw(i,j) = (Igray(i-1,j  )+Igray(i  ,j  )+Igray(i+1,j+1))/3;
        bw(i,j) = (Igray(i-1,j-1)+Igray(i-1,j+1)+Igray(i  ,j-1)+ ...
                   Igray(i  ,j+1)+Igray(i+1,j-1)+Igray(i+1,j  ))/6;
        ratiow(i,j) = aw(i,j)/bw(i,j);

        ax(i,j) = (Igray(i  ,j-1)+Igray(i  ,j  )+Igray(i-1,j+1))/3;
        bx(i,j) = (Igray(i-1,j-1)+Igray(i-1,j  )+Igray(i  ,j+1)+ ...
                   Igray(i+1,j-1)+Igray(i+1,j  )+Igray(i+1,j+1))/6;
        ratiox(i,j) = ax(i,j)/bx(i,j);

        ay(i,j) = (Igray(i-1,j-1)+Igray(i  ,j  )+Igray(i+1,j  ))/3;
        by(i,j) = (Igray(i-1,j  )+Igray(i-1,j+1)+Igray(i  ,j-1)+ ...
                   Igray(i  ,j+1)+Igray(i+1,j+1)+Igray(i+1,j-1))/6;
        ratioy(i,j) = ay(i,j)/by(i,j);

        az(i,j) = (Igray(i+1,j-1)+Igray(i  ,j  )+Igray(i  ,j+1))/3;
        bz(i,j) = (Igray(i-1,j-1)+Igray(i-1,j  )+Igray(i-1,j+1)+ ...
                   Igray(i  ,j-1)+Igray(i+1,j  )+Igray(i+1,j+1))/6;
        ratioz(i,j) = az(i,j)/bz(i,j);

        % Final thinness feature = minimum ratio across all directions
        thin(i,j) = min([ ...
            ratioo(i,j), ratiop(i,j), ratioq(i,j), ratior(i,j), ...
            ratios(i,j), ratiot(i,j), ratiou(i,j), ratiov(i,j), ...
            ratiow(i,j), ratiox(i,j), ratioy(i,j), ratioz(i,j) ]);
    end
end
Th      = thin(3:m-2, 3:n-2) * 100;
Thinsom = reshape(Th, 1, (m-4)*(n-4));           % feature 4: thinness

%% 5) Local mean features (3x3 and 5x5)
for i = 3:m-2
   for j = 3:n-2
       sum8neighbors(i,j) = ( ...
            Igray(i-1,j-1)+Igray(i-1,j)+Igray(i-1,j+1)+ ...
            Igray(i  ,j-1)+Igray(i  ,j)+Igray(i  ,j+1)+ ...
            Igray(i+1,j-1)+Igray(i+1,j)+Igray(i+1,j+1) ) / 9;
   end
end
Sum1    = sum8neighbors(3:m-2, 3:n-2);
Sum1som = reshape(Sum1, 1, (m-4)*(n-4));         % feature 5: 3x3 mean

for i = 3:m-2
   for j = 3:n-2
       sum24neighbors(i,j) = ( ...
            Igray(i-2,j-2)+Igray(i-2,j-1)+Igray(i-2,j  )+Igray(i-2,j+1)+Igray(i-2,j+2)+ ...
            Igray(i-1,j-2)+Igray(i-1,j-1)+Igray(i-1,j  )+Igray(i-1,j+1)+Igray(i-1,j+2)+ ...
            Igray(i  ,j-2)+Igray(i  ,j-1)+Igray(i  ,j  )+Igray(i  ,j+1)+Igray(i  ,j+2)+ ...
            Igray(i+1,j-2)+Igray(i+1,j-1)+Igray(i+1,j  )+Igray(i+1,j+1)+Igray(i+1,j+2)+ ...
            Igray(i+2,j-2)+Igray(i+2,j-1)+Igray(i+2,j  )+Igray(i+2,j+1)+Igray(i+2,j+2) ) / 25;
   end
end
Sum2    = sum24neighbors(3:m-2, 3:n-2);
Sum2som = reshape(Sum2, 1, (m-4)*(n-4));         % feature 6: 5x5 mean

%% 6) Row/column-relative grayscale features (p, q)
row    = median(Igray,  2);
column = median(Igray,  1);
for i = 3:m-2
    for j = 3:n-2
        p(i,j) = Igray(i,j) * 100 / row(i,1);     % vs row median
        q(i,j) = Igray(i,j) * 100 / column(1,j);  % vs column median
    end
end
p    = p(3:m-2, 3:n-2);
q    = q(3:m-2, 3:n-2);
psom = reshape(p, 1, (m-4)*(n-4));               % feature 7: row-relative
qsom = reshape(q, 1, (m-4)*(n-4));               % feature 8: column-relative

%% 7) Hue feature (from RGB)
r = I(:,:,1);
g = I(:,:,2);
b = I(:,:,3);
for i = 3:m-2
   for j = 3:n-2
        if  r(i,j) > g(i,j) && g(i,j) > b(i,j)
            Hue(i,j) = 60*(g(i,j)-b(i,j))/(r(i,j)-b(i,j));
        elseif r(i,j) >= g(i,j) && g(i,j) <= b(i,j)
            Hue(i,j) = 60*(g(i,j)-b(i,j))/(r(i,j)-g(i,j));
        elseif g(i,j) > r(i,j) && r(i,j) > b(i,j)
            Hue(i,j) = 120 + 60*(b(i,j)-r(i,j))/(g(i,j)-b(i,j));
        elseif g(i,j) >= r(i,j) && r(i,j) <= b(i,j)
            Hue(i,j) = 120 + 60*(b(i,j)-r(i,j))/(g(i,j)-r(i,j));
        elseif  b(i,j) > r(i,j) && r(i,j) > g(i,j)
            Hue(i,j) = 240 + 60*(r(i,j)-g(i,j))/(b(i,j)-g(i,j));
        elseif  b(i,j) >= r(i,j) && r(i,j) <= g(i,j)
            Hue(i,j) = 240 + 60*(r(i,j)-g(i,j))/(b(i,j)-r(i,j));
        end
   end
end
Hue    = Hue(3:m-2, 3:n-2);
Huesom = reshape(Hue, 1, (m-4)*(n-4));           % feature 9: hue

%% 8) Assemble 9-D feature vector for SOM
input = double([ ...
    Igraysom; ...
    Contrastnewsom; ...
    Normaledgesom; ...
    Thinsom; ...
    Sum1som; ...
    Sum2som; ...
    psom; ...
    qsom; ...
    Huesom ]);

%% 9) Train a 3-node SOM and get class labels
% Fix random seed for reproducibility
rng(0);

net       = selforgmap(3);
[net, tr] = train(net, input);
op        = vec2ind(net(input));          % 1D label vector (1..3)

% Reshape labels back to image size
op_som = reshape(op, m-4, n-4);

% -------------------------------------------------
% Automatically choose which SOM class is "crack"
% Heuristic: crack regions are the darkest on average
% -------------------------------------------------
Igraycut = Igray(3:m-2, 3:n-2);  % same region as features
meanGray = zeros(1,3);

for k = 1:3
    mask_k = (op_som == k);
    % Avoid empty class
    if any(mask_k(:))
        meanGray(k) = mean(Igraycut(mask_k));
    else
        meanGray(k) = inf;
    end
end

[~, CRACK_CLASS] = min(meanGray);   % darkest class as crack

%% 10) Visualize three SOM classes
figure;
for k = 1:3
    C = uint8(255 * ones(size(op_som)));  % white background
    C(op_som == k) = 0;                   % class k as black
    subplot(2,3,k);
    imshow(C);
    title(sprintf('%d^{th} class', k));
end

% Mark which class was automatically selected as crack
subplot(2,3,4);
C_crack = uint8(255 * ones(size(op_som)));
C_crack(op_som == CRACK_CLASS) = 0;
imshow(C_crack);
title(sprintf('Selected crack class = %d', CRACK_CLASS));

toc;

%% 11) Post-processing: keep only large crack components
% Binary mask: crack = 1, background = 0
I_bin = (op_som == CRACK_CLASS);

% Connected components
[L, num] = bwlabel(I_bin);
stats    = regionprops(L, 'Area');
Area     = [stats.Area];

% Total crack area (in pixels)
total_crack_area = sum(Area);

% Remove tiny components (< 1% of total crack area)
thresh  = 0.01 * total_crack_area;
I_clean = zeros(size(I_bin));

for k = 1:num
    if Area(k) >= thresh
        I_clean(L == k) = 1;
    end
end

% Show before/after cleaning
figure;
imshowpair(I_bin, I_clean, 'montage');
title('Left: Raw SOM crack mask | Right: After removing small components');

% Optional: invert if you prefer white cracks on black background
I_clean = 1 - I_clean;
figure;
imshow(I_clean);
title('Final crack mask');