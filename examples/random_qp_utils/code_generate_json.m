

N = length(agents); % number of nodes

treeqp = struct('edges', [], 'nodes', []);

agents

for ii = 1:N
    treeqp.nodes{ii}.Q = agents(ii).Q;
    treeqp.nodes{ii}.R = agents(ii).R;
    treeqp.nodes{ii}.S = agents(ii).S;
    treeqp.nodes{ii}.q = agents(ii).q;
    treeqp.nodes{ii}.r = agents(ii).r;
    
    treeqp.nodes{ii}.xopt = agents(ii).xopt;
    treeqp.nodes{ii}.uopt = agents(ii).uopt;
end


for ii = 2:N
   
    treeqp.edges{ii-1}.A = agents(ii).A;
    treeqp.edges{ii-1}.B = agents(ii).B;
    treeqp.edges{ii-1}.b = agents(ii).b;
    
    treeqp.edges{ii-1}.from = agents(ii).dad-1;
    treeqp.edges{ii-1}.to = ii-1;
    
end

content = jsonencode(treeqp);

jsonfile = fopen('treeqp.json', 'w');
fprintf(jsonfile, content);
fclose(jsonfile);