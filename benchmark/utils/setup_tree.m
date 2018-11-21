function [ agents ] = setup_tree(dynamics, weights, constraints, Nr, Nh, md )

% SETUP_TREE Create an array of structures with one element per tree
%   node.

if ~isfield(weights,'q')
    weights.q = [];
end
if ~isfield(weights,'r')
    weights.r = [];
end
if ~isfield(weights,'p')
    weights.p = [];
end

[nx, nu]   = size(dynamics(1).B);
Nscenarios = md^Nr;
Nnodes     = get_number_of_nodes(md, Nr, Nh );

zmin = [constraints.xmin; constraints.umin];
zmax = [constraints.xmax; constraints.umax];

% Build tree
agents(Nnodes).stage = Nh; % preallocate tree structure
count = 1; % counter for realizations

for ii = 1:Nnodes
    
    if ii == 1 % root
        
        agents(ii).stage  = 0;          % time stage
        agents(ii).child  = 2:md+1;     % children nodes
        agents(ii).parent = [];         % parent node
        agents(ii).real   = [];         % realization of dynamics      
        agents(ii).dyn    = [];         % struct with linear dynamics
        agents(ii).ind    = 1:nx+nu;    % index of variables in optimal solution
        agents(ii).zmin   = zmin;       % lower bounds
        agents(ii).zmax   = zmax;       % upper bounds
        agents(ii).H      = [];         % weighted quadratic cost
        agents(ii).Hv     = [];         % weighted diagonal quadratic cost in vector
        agents(ii).Hvinv  = [];         % inverse of diagonal quadratic cost in vector
        agents(ii).h      = [];         % weighted linear cost    

        % initial value constraint
        agents(ii).zmin(1:nx) = constraints.x0;
        agents(ii).zmax(1:nx) = constraints.x0;
        
        % add scaled weights 
        agents(ii).H     = Nscenarios*diag([weights.Q; weights.R]);
        agents(ii).Hv    = Nscenarios*[weights.Q; weights.R];
        agents(ii).Hvinv = 1./agents(ii).Hv;
        agents(ii).h     = Nscenarios*[weights.q; weights.r]; 
        
    else % children
        
        % set stage
        agents(ii).stage = agents(agents(ii).parent).stage+1;
        
        % set realization number
        real = mod(count,md);
        if real == 0
            real = md;
        end
        agents(ii).real = real;
                
        % set dynamics
        agents(ii).dyn = dynamics(agents(ii).real);
        
        % find children
        if ii <= Nnodes-md^Nr % not leafs (NOTE: can't use the util functions before the tree is built!)
            if agents(ii).stage < Nr % branch
                agents(ii).child = agents(ii-1).child(end)+1:agents(ii-1).child(end)+md;
            else % do not branch
                agents(ii).child = agents(ii-1).child(end)+1;
            end
        end
        
        % set index in solution vector, upper/lower bounds and average stage cost
        if ii <= Nnodes-md^Nr % not leafs
            agents(ii).ind  = (ii-1)*(nx+nu)+1:ii*(nx+nu);
            agents(ii).zmin = zmin;
            agents(ii).zmax = zmax;
            
            if agents(ii).stage <= Nr
                w = md^(Nr-agents(ii).stage);
            else
                w = 1;
            end
            agents(ii).H = w*diag([weights.Q;weights.R]);
            agents(ii).h = w*[weights.q;weights.r];
            agents(ii).Hv= w*[weights.Q;weights.R];
        else % leafs
            agents(ii).ind  = (Nnodes-Nscenarios)*(nx+nu)+(count-1)*nx+1:(Nnodes-Nscenarios)*(nx+nu)+count*nx;   
            agents(ii).zmin = constraints.xNmin;
            agents(ii).zmax = constraints.xNmax;
            agents(ii).H    = diag(weights.P);
            agents(ii).Hv   = weights.P;
            agents(ii).h    = weights.p;
        end
        agents(ii).Hvinv = 1./agents(ii).Hv;
        
        % set counter for next agent
        if ( count == md^agents(ii).stage && agents(ii).stage <= Nr ) ...
                || ( count == Nscenarios && agents(ii).stage > Nr ) 
            count = 1;
        else
            count = count+1;
        end
        
    end
    
    % Set parent node of all children
    nc  = length(agents(ii).child); % number of children
    par = num2cell(ii*ones(nc,1));
    [agents(agents(ii).child).parent] = par{:};
    
    % Check which stage QP solution method should be used
    % TODO also check that there are no polytopic constraints
    if max(abs(agents(ii).Hv - diag(agents(ii).H))) < 1e-15
        agents(ii).clipping = 1;
    else
        agents(ii).clipping = 0;
    end
    
end

end

