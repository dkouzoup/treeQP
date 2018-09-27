function agents = code_generate_tree_from_json(qp, fname)

Nn = length(qp.nodes);

agents(1).dad = -1;

nx = zeros(Nn,1);
nu = zeros(Nn,1);

CLIPPING = 1;
CALCULATE_OPTIMAL_SOLUTION = 0;

% remove cells 
struct_fields = fields(qp.nodes(1)); 
for ii = 1:length(struct_fields)
    if eval(['iscell(qp.nodes(1).' struct_fields{ii} ')'])
        eval(['qp.nodes(1).' struct_fields{ii} '= [];'])
    end
end
struct_fields = fields(qp.edges(1)); 
for ii = 1:length(struct_fields)
    if eval(['iscell(qp.edges(1).' struct_fields{ii} ')'])
        eval(['qp.edges(1).' struct_fields{ii} '= [];'])
    end
end

for ii = 1:Nn
 
    agents(ii).idx = ii;
    
    nx(ii) = size(qp.nodes(ii).Q, 1);
    nu(ii) = size(qp.nodes(ii).R, 1);
    
    agents(ii).Q = qp.nodes(ii).Q;
    agents(ii).R = qp.nodes(ii).R;
    agents(ii).S = qp.nodes(ii).S;
    agents(ii).q = qp.nodes(ii).q;
    agents(ii).r = qp.nodes(ii).r;
   
    if isfield(qp.nodes(ii), 'xopt')
        agents(ii).xopt = qp.nodes(ii).xopt;
        agents(ii).uopt = qp.nodes(ii).uopt;
        
        assert(size(agents(ii).xopt, 1) == nx(ii));
        assert(size(agents(ii).uopt, 1) == nu(ii));
    else
        CALCULATE_OPTIMAL_SOLUTION = 1;
    end
    
    assert(size(agents(ii).Q, 1) == nx(ii));
    assert(size(agents(ii).Q, 2) == nx(ii));
    assert(size(agents(ii).q, 1) == nx(ii));
    
    assert(size(agents(ii).R, 1) == nu(ii));
    assert(size(agents(ii).R, 2) == nu(ii));
    assert(size(agents(ii).r, 1) == nu(ii));
        
    if (ii < Nn)
        
        from = qp.edges(ii).from+1;
        to = qp.edges(ii).to+1;

        agents(to).A = qp.edges(ii).A;
        agents(to).B = qp.edges(ii).B;
        agents(to).b = qp.edges(ii).b;
                
        agents(to).dad = from;
    end
    disp(['node ' num2str(ii) ' processed']) 
    
    agents(ii).nkids = 0; % derived later
    
    % constraints
    if isfield(qp.nodes, 'lx')
        agents(ii).xmin = qp.nodes(ii).lx;
        agents(ii).umin = qp.nodes(ii).lu;
        agents(ii).xmax = qp.nodes(ii).ux;
        agents(ii).umax = qp.nodes(ii).uu;
    end
    
    % check clipping conditions
    if ~isdiag(agents(ii).Q) || ~isdiag(agents(ii).R) || ...
            (~iscell(agents(ii).S) && sum(sum(abs(agents(ii).S))) ~= 0)
        CLIPPING = 0;
    end
end

% TODO: FIX
% for ii = 1:Nn-1
%     
%     from = qp.edges(ii).from+1;
%     to = qp.edges(ii).to+1;
%     
%     assert(size(agents(to).A, 1) == nx(to));
%     assert(size(agents(to).A, 2) == nx(from));
%     
%     assert(size(agents(ii+1).B, 1) == nx(to));
%     assert(size(agents(ii+1).B, 2) == nu(from));
%     
%     assert(size(agents(ii+1).b, 1) == nx(to));
%     disp(['edge from agent ' num2str(qp.edges(ii).from) ' to agent ' num2str(qp.edges(ii).to)]) 
% end

if CALCULATE_OPTIMAL_SOLUTION
    agents = solve_tree_with_yalmip(agents);
end

for ii = 1:Nn


    dad = agents(ii).dad;
    if dad > 0
        agents(dad).nkids = agents(dad).nkids + 1;
    end    
end

code_generate_tree(agents, fname, CLIPPING)


end

