function [P, lam_vol] = Q_singleVex(true_Q, N, Q, Q_num, lam_domain, lam_dim, ...
    lam_samples, bin_ratio, nbins, lam_vol)

if nargin < 9
    nbins = 20;  % number of bins in each direction
else
    if isempty(nbins)
        nbins = 20;  % number of bins in each direction
    end
end

calc_lam_vol = false;
if nargin < 10
    calc_lam_vol = true;
else
    if isempty(lam_vol)
        calc_lam_vol = 2;
    end
end

% Determine the approriate bin_size for this QoI
bin_size = zeros(size(Q_num));

for i=1:numel(Q_num)
    bin_size(i) = max(Q(:,Q_num(i))) - min(Q(:,Q_num(i)));
end

bin_size = bin_size / bin_ratio;
b375 = bin_size*.375;
b50 = bin_size*.50;
b75 = bin_size*.75;


% This is a script for parameter identification, i.e. an "error" problem

M = 50; % Defines number M samples in D used to define rho_{D,M}
% The choice of M is something of an "art" - play around with it
% and you can get reasonable results with a relatively small
% number here like 50.

% Create M samples defining M bins in D used to define rho_{D,M}
% This choice of rho_D was based on looking at Q(Lambda) and getting a
% sense of what a reasonable amount of error was for this problem. Notice I
% use uniform distributions of various lengths depending on which
% measurement I chose from the 4 I made above. Also, this does not have to
% be random. I can choose these bins deterministically but that doesn't
% scale well typically. These are also just chosen to determine bins and do
% not necessarily have anything to do with probabilities. I use "smaller"
% uniform densities below. This was just to setup a discretization of D in
% some random way so that I put bins near where the output probability is
% (why would I care about binning zero probability events?).
q_distr_samples = zeros(M, numel(Q_num));
for i=1:numel(Q_num)
    q_distr_samples(:,i) = true_Q(Q_num(i))-b50(i)+...
        bin_size(i)*rand(M,1);
end

% Now compute probabilities for rho_{D,M} by sampling from rho_D
% First generate samples of rho_D - I sometimes call this emulation
num_q_emulate = 1E6;
q_distr_emulate = zeros(num_q_emulate, numel(Q_num));
for i=1:numel(Q_num)
    q_distr_emulate(:,i) = true_Q(Q_num(i))-b375(i)+b75(i)*rand(num_q_emulate,1);
end

% Now bin samples of rho_D in the M bins of D to compute rho_{D,M}
tic
count_neighbors = zeros(M,1);
T = delaunayn(q_distr_samples);
k = dsearchn(q_distr_samples, T, q_distr_emulate);
for i = 1:M
    count_neighbors(i) = sum(k==i);
end
toc

% Now define probability of the q_distr_samples
q_distr_prob = count_neighbors / (num_q_emulate); %This together with q_distr_samples defines rho_{D,M}
% NOTE: The computation of q_distr_prob, q_distr_emulate, q_distr_samples
% above, while informed by the sampling of the map Q, do not require
% solving the model EVER! This can be done "offline" so to speak.

% Determine which inputs go to which M bins using the QoI
io_ptr = dsearchn(q_distr_samples, T, Q(:,Q_num));

P = zeros(N,1);

% No longer applying standard MC assumption/approximation
% Estimate the volume of voronoi cells
tic
disp('Estimating the volume of voronoi cells')
lam_width = lam_domain(:,2)-lam_domain(:,1);

if calc_lam_vol
    % Estimate using voronoi cells
    % Add fake samples outside of lam_domain to close Voronoi tesselations at
    lam_bound = lam_samples;
    pts_per_edge = nbins;
    sides = zeros(2, pts_per_edge);
    for i =1:lam_dim
        sides(i,:) = linspace(lam_domain(i,1), lam_domain(i,2), pts_per_edge);
    end
    % add midpoints
    for i=1:lam_dim
        new_pt = sides;
        new_pt(i,:) = repmat(lam_domain(i,1)-lam_width(i)/pts_per_edge, 1, pts_per_edge);
        lam_bound = [lam_bound; new_pt'];
        new_pt = sides;
        new_pt(i,:) = repmat(lam_domain(i,2)+lam_width(i)/pts_per_edge, 1, pts_per_edge);
        lam_bound = [lam_bound; new_pt'];
    end
    % add corners
    corners = zeros(2^lam_dim, lam_dim)';
    for i=1:lam_dim
        corners(i,:) = lam_domain(i,repmat([ones(1,2^(i-1)) 2*ones(1,2^(i-1))],1,2^(lam_dim-i)));
        corners(i,:) = corners(i,:) + ...
            lam_width(i)*repmat([ones(1,2^(i-1)) -1*ones(1,2^(i-1))],1,2^(lam_dim-i))/pts_per_edge;
    end
    lam_bound = [lam_bound; corners'];
    
    [v, c] = voronoin(lam_bound);
    
    lam_vol = zeros(N,1);
    for i = 1:N
        [~, lam_vol(i)] = convhulln(v(c{i},:));
    end
end
toc

tic
for i=1:M
    Itemp = find(io_ptr == i);
    if ~isempty(Itemp)
        P(Itemp) = lam_vol(Itemp)/(sum(lam_vol(Itemp))) * q_distr_prob(i);
    end
end
toc
P = P/sum(P);

disp('Plotting now')

plot_voronoi_probs(P,lam_samples,lam_dim,lam_domain, 1, nbins)
%plot_voronoi_regions(P,lam_samples,lam_dim,lam_domain)
drawnow
