function [ answer ] = benchmark_exists(benchmark)

% BENCHMARK_EXISTS Check if specified benchmark problem exists.

answer = false;

if contains(benchmark, 'linear_chain')
    answer = true;
end

if strcmp(benchmark, 'quadcopter')
    answer = true;
end    

if strcmp(benchmark, 'crane')
    answer = true;
end    

end

