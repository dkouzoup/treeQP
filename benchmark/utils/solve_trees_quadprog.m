function [ trajectories, lam_quadprog, diagn, t_quadprog ] = solve_trees_quadprog( agents, Nh, Nr, opts)

% SOLVE_TREES_QUADPROG Solve the tree-structured problem with quadprog

md     = length(agents(1).child); % TODO Check that this is correct also in marginal choices of md, Nr etc
Nnodes = length(agents);
nx     = size(agents(2).dyn.A,1);

% Setup quadprog tree data
[H, h, Aeq, beq, lb, ub] = setup_quadprog(agents, Nr, Nh, md);

% Solve with quadprog
tic
[sol, fval, flag, output, lam_quadprog] = quadprog(H,h,[],[],Aeq,beq,lb,ub,[],opts);
t_quadprog = toc;

if flag == -2
    error('Problem is infeasible')
end

% Populate tree with optimal solution
for ii = 1:Nnodes
    xind = agents(ii).ind(1:nx);
    uind = agents(ii).ind(nx+1:end);
    agents(ii).xit = sol(xind);
    agents(ii).uit = sol(uind);
end

trajectories = build_trajectories_from_trees(agents, Nh, Nr);

diagn = []; % TODO quadprog message

% Check consistency of dynamics
for ii = 1:Nnodes
    if is_not_leaf(agents,ii)
        for jj = 1:length(agents(ii).child)
            child = agents(ii).child(jj);
            xnext = agents(child).xit;
            xsim  = agents(child).dyn.A*agents(ii).xit + agents(child).dyn.B*agents(ii).uit + agents(child).dyn.c;
            if max(abs(xnext-xsim)) > 1e-10
                disp('VIOLATION OF DYNAMICS')
                keyboard
            end
        end
    end
end

end

