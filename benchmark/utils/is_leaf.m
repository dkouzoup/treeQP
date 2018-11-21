function resp = is_leaf(agents, ii)

% Check if node is a leaf
if isempty(agents(ii).child)
    resp = 1;
else
    resp = 0;
end

end

