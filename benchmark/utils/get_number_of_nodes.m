function [ Nnodes ] = get_number_of_nodes(md, Nr, Nh )

% GET_NUMBER_OF_NODES Calculate number of nodes in tree

if md == 1 % i.e., standard block-banded structure
    Nnodes = Nh+1;
else
    Nnodes = (Nh-Nr)*md^Nr + (md^(Nr+1)-1)/(md-1);
end

end

