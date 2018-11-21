function err = check_error_in_trajectories(trajectories_1, trajectories_2)

% CHECK_ERROR_IN_TRAJECTORIES Check maximum error between two solutions

% some sanity checks
if iscell(trajectories_1) && ~iscell(trajectories_2) || ~iscell(trajectories_1) && iscell(trajectories_2) 
    error('One input is a cell of problems while the other is not');
end
if iscell(trajectories_1) && iscell(trajectories_2) && length(trajectories_1) ~= length(trajectories_2)
    error('One input contains more problems than the other');
end
if iscell(trajectories_1)
    Nproblems = length(trajectories_1);
    Nscenarios = length(trajectories_1{1});
else
    Nproblems  = 1;
    Nscenarios = length(trajectories_1);
    tmp1 = trajectories_1;
    tmp2 = trajectories_2;
    clear trajectories_1;
    clear trajectories_2;
    trajectories_1{1} = tmp1;
    trajectories_2{1} = tmp2;
end
err = zeros(Nproblems, Nscenarios);

for ii = 1:Nproblems
    trajectory_1 = trajectories_1{ii};
    trajectory_2 = trajectories_2{ii};
    for jj = 1:Nscenarios
        err_u = max(max(abs(trajectory_1(jj).u-trajectory_2(jj).u)));
        err_x = max(max(abs(trajectory_1(jj).x-trajectory_2(jj).x)));
        err(ii, jj) = max(err_x, err_u);
    end
end

end

