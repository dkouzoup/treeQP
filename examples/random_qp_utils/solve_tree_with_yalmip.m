
function agents = solve_tree_with_yalmip(agents, TOL, MAXITER)

if nargin < 2
    TOL = 1e-20;
end

if nargin < 3
    MAXITER = 1000;
end

con = [];
obj = 0;

Nn = length(agents);

x = cell(Nn, 1);
u = cell(Nn, 1);

for ii = 1:Nn
    
    % dimensions
    nx = length(agents(ii).q);
    nu = length(agents(ii).r);
    
    % optimization variables
    x{ii} = sdpvar(nx, 1, 'full');
    u{ii} = sdpvar(nu, 1, 'full');
    
    % objective
    if nx > 0
        obj = obj + 0.5*x{ii}'*agents(ii).Q*x{ii} + agents(ii).q'*x{ii};
    end
    if nu > 0
        obj = obj + 0.5*u{ii}'*agents(ii).R*u{ii} + agents(ii).r'*u{ii};
    end
    if nx > 0 && nu > 0
        obj = obj + u{ii}'*agents(ii).S*x{ii};
    end
    
    % equality constraints
    if ii > 1
        
        dad   = agents(ii).dad;
        nxdad = length(agents(dad).q);
        
        if nxdad > 0
            con = [con; x{ii} == agents(ii).A*x{dad} + agents(ii).B*u{dad} + agents(ii).b];
        else
            con = [con; x{ii} == agents(ii).B*u{dad} + agents(ii).b];
        end
        
    end
    
    % bounds
    
    if isfield(agents(ii), 'xmin') && ~isempty(agents(ii).xmin)
        con = [con; agents(ii).xmin <= x{ii} <= agents(ii).xmax];
    end
    if isfield(agents(ii), 'umin') && ~isempty(agents(ii).umin)
        con = [con; agents(ii).umin <= u{ii} <= agents(ii).umax];
    end
end

% setup options
opts = sdpsettings('solver','quadprog');
opts.quadprog.MaxIter = MAXITER;
opts.quadprog.Display = 'off';

opts.TolCon = TOL;
opts.TolFun = TOL;
opts.TolX   = TOL;

% solve
diagn = optimize(con,obj,opts);

if diagn.problem ~= 0
    warning('yalmip failed')
    keyboard
end


for ii = 1:Nn
    agents(ii).xopt = value(x{ii});
    agents(ii).uopt = value(u{ii});
end

end
