function [H, h, Aeq, beq, lb, ub] = setup_quadprog(agents, Nr, Nh, md)

% SETUP_QUADPROG Transform QP in quadprog format


[nx, nu]   = size(agents(2).dyn.B);
Nscenarios = md^Nr;
Nnodes     = length(agents);
nvar       = (Nnodes-Nscenarios)*(nx+nu) + Nscenarios*nx;
neq        = (Nnodes-1)*nx;

% Build Aeq, beq
%Aeq  = zeros(neq,nvar);
Aeq  = sparse(neq,nvar);
beq  = zeros(neq,1);

for ii = 2:Nnodes
   
    % row index
    rind = (ii-2)*nx+1:(ii-1)*nx;
    
    % column indicies
    cind_xp = (agents(ii).parent-1)*(nx+nu)+1:(agents(ii).parent-1)*(nx+nu)+nx;
    cind_up = (agents(ii).parent-1)*(nx+nu)+nx+1:agents(ii).parent*(nx+nu);
    cind_x  = agents(ii).ind(1:nx);
    
    Aeq(rind,cind_xp) = agents(ii).dyn.A;
    Aeq(rind,cind_up) = agents(ii).dyn.B;
    Aeq(rind,cind_x)  = -eye(nx);

    beq(rind) = -agents(ii).dyn.c;
    
end

% Build bounds and cost
ub  =  inf(nvar,1);
lb  = -inf(nvar,1);
H   = [];
h   = [];

for ii = 1:Nnodes
    lb(agents(ii).ind) = agents(ii).zmin;
    ub(agents(ii).ind) = agents(ii).zmax;
    H = blkdiag(H,agents(ii).H);
    h = [h; agents(ii).h];
end

H = sparse(H);

end

