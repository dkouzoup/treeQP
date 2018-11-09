function code_generate_json(fname, agents, opts)

if ~strcmp(fname(end-4:end), '.json')
    fname = [fname '.json'];
end

N = length(agents); % number of nodes

treeqp = struct;

for ii = 1:N
    treeqp.nodes{ii}.Q = agents(ii).Q;
    treeqp.nodes{ii}.R = agents(ii).R;
    treeqp.nodes{ii}.S = agents(ii).S;
    treeqp.nodes{ii}.q = agents(ii).q;
    treeqp.nodes{ii}.r = agents(ii).r;
    
    if isfield(agents, 'xopt')
        treeqp.nodes{ii}.xopt = agents(ii).xopt;
        treeqp.nodes{ii}.uopt = agents(ii).uopt;
    end
    
    if isfield(agents, 'xmin')
        treeqp.nodes{ii}.lx = agents(ii).xmin;
        treeqp.nodes{ii}.ux = agents(ii).xmax;
        treeqp.nodes{ii}.lu = agents(ii).umin;
        treeqp.nodes{ii}.uu = agents(ii).umax;   
    end
end


for ii = 2:N
    
    treeqp.edges{ii-1}.A = agents(ii).A;
    treeqp.edges{ii-1}.B = agents(ii).B;
    treeqp.edges{ii-1}.b = agents(ii).b;
    
    treeqp.edges{ii-1}.from = agents(ii).dad-1;
    treeqp.edges{ii-1}.to = ii-1;
    
    if nargin > 2 && strcmp(opts.solver, 'tdunes') && opts.warmstart
        treeqp.edges{ii-1}.lam0 = agents(ii).lambda;
    end
end

if nargin > 2
    treeqp.options = opts;
end

content = jsonencode(treeqp);

jsonfile = fopen(fname, 'w');
fprintf(jsonfile, content);
fclose(jsonfile);

end