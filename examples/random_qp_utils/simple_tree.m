
% Generate a tree with arbitrary structure, solve it with yalmip and code
% generate data to be solved in treeQP

clear all; clc

CLIPPING = false;

%% define dimensions

nk = [2 2 1 0 0 0];  % number of successors (children) of each node
nx = [2 3 2 1 1 1];  % number of states of each node
nu = [1 2 1 0 0 0];  % number of controls of each node


if sum(nk) ~= length(nk) - 1
   error('wrong data in nk')
end
if length(nk) ~= length(nx)
   error('wrong dimension of nx')
end
if length(nk) ~= length(nu)
   error('wrong dimension of nu')
end

%% generate random tree

agents = generate_random_tree(nk, nx, nu, CLIPPING);

for ii = 1:length(agents)
    disp(['Node ' num2str(ii)])
    agents(ii)
end

%% solve with yalmip

for ii = 1:length(agents)
    x{ii} = sdpvar(nx(ii),1);
    u{ii} = sdpvar(nu(ii),1);
end

con = [];
for ii = 2:length(agents)
    dad = agents(ii).dad;
    con = [con; x{ii} == agents(ii).A*x{dad} + agents(ii).B*u{dad} + agents(ii).b];
end

obj = 0;
for ii = 1:length(agents)
    obj = obj + 0.5*x{ii}'*agents(ii).Q*x{ii} + 0.5*u{ii}'*agents(ii).R*u{ii} + ...
        u{ii}'*agents(ii).S*x{ii} + x{ii}'*agents(ii).q + u{ii}'*agents(ii).r;
end

opts  = sdpsettings('solver','quadprog');

diagn = optimize(con, obj, opts);

disp(' ')
disp(diagn.info)
disp(' ')

for ii = 1:length(agents)
    agents(ii).xopt = double(x{ii});
    agents(ii).uopt = double(u{ii});
end

% double(obj)

%% code generate data for c

code_generate_tree(agents, 'data.c', CLIPPING)
code_generate_json(agents, 'data.json')