function code_generate_tree( agents, fname, CLIPPING )

% CODE_GENERATE_TREE Write tree data in a c file to be processed by treeQP

%% initialization

datafile = fopen(fname, 'w');
Nnodes   = length(agents);

dimA = 0;
dimB = 0;
dimb = 0;
dimQ = 0;
dimR = 0;
dimS = 0;
dimq = 0;
dimr = 0;

for ii = 1:Nnodes
    if ii > 1
        dimA = dimA + size(agents(ii).A,1)*size(agents(ii).A,2);
        dimB = dimB + size(agents(ii).B,1)*size(agents(ii).B,2);
        dimb = dimb + size(agents(ii).b,1);
    end
    if CLIPPING
        dimQ = dimQ + size(agents(ii).Q,1);
        dimR = dimR + size(agents(ii).R,1);
    else
        dimQ = dimQ + size(agents(ii).Q,1)*size(agents(ii).Q,2);
        dimR = dimR + size(agents(ii).R,1)*size(agents(ii).R,2);
        dimS = dimS + size(agents(ii).S,1)*size(agents(ii).S,2);
    end
    dimq = dimq + size(agents(ii).q,1);
    dimr = dimr + size(agents(ii).r,1);
end

%% dimensions
if CLIPPING
    fprintf(datafile, '\n#define CLIPPING\n');
end

fprintf(datafile, '\n/* Dimensions */\n\n');

fprintf(datafile,'int Nn = %d;\n', Nnodes);

% print nc
fprintf(datafile,'int nc[%d] = { ', Nnodes);
for ii = 1:Nnodes
	fprintf(datafile,'%d, ', agents(ii).nkids);
end
fprintf(datafile,'};\n');

% print nx
fprintf(datafile,'int nx[%d] = { ', Nnodes);
for ii = 1:Nnodes
	fprintf(datafile,'%d, ', size(agents(ii).Q,1));
end
fprintf(datafile,'};\n');

% print nu
fprintf(datafile,'int nu[%d] = { ', Nnodes);
for ii = 1:Nnodes
	fprintf(datafile,'%d, ', size(agents(ii).R,1));
end
fprintf(datafile,'};\n');

%% data 

fprintf(datafile, '\n/* Data */\n\n');

% print A
fprintf(datafile,'double A[%d] = { ', dimA);
for kk = 2:Nnodes
    for jj = 1:size(agents(kk).A, 2)
        for ii = 1:size(agents(kk).A,1)
            fprintf(datafile,'%1.15e, ', agents(kk).A(ii,jj));
        end
    end
end
fprintf(datafile,'};\n');

% print B
fprintf(datafile,'double B[%d] = { ', dimB);
for kk = 2:Nnodes
    for jj = 1:size(agents(kk).B, 2)
        for ii = 1:size(agents(kk).B,1)
            fprintf(datafile,'%1.15e, ', agents(kk).B(ii,jj));
        end
    end
end
fprintf(datafile,'};\n');

% print b
fprintf(datafile,'double b[%d] = { ', dimb);
for kk = 2:Nnodes
    for ii = 1:size(agents(kk).b,1)
        fprintf(datafile,'%1.15e, ', agents(kk).b(ii));
    end
end
fprintf(datafile,'};\n');


if CLIPPING

    % print Q
    fprintf(datafile,'double Qd[%d] = { ', dimQ);
    for kk = 1:Nnodes
        for ii = 1:size(agents(kk).Q,1)
            fprintf(datafile,'%1.15e, ', agents(kk).Q(ii, ii));
        end
    end
    fprintf(datafile,'};\n');

    % print R
    fprintf(datafile,'double Rd[%d] = { ', dimR);
    for kk = 1:Nnodes
        for ii = 1:size(agents(kk).R,1)
            fprintf(datafile,'%1.15e, ', agents(kk).R(ii, ii));
        end
    end
    fprintf(datafile,'};\n');
    
else
    
    % print Q
    fprintf(datafile,'double Q[%d] = { ', dimQ);
    for kk = 1:Nnodes
        for jj = 1:size(agents(kk).Q, 2)
            for ii = 1:size(agents(kk).Q,1)
                fprintf(datafile,'%1.15e, ', agents(kk).Q(ii,jj));
            end
        end
    end
    fprintf(datafile,'};\n');

    % print R
    fprintf(datafile,'double R[%d] = { ', dimR);
    for kk = 1:Nnodes
        for jj = 1:size(agents(kk).R, 2)
            for ii = 1:size(agents(kk).R,1)
                fprintf(datafile,'%1.15e, ', agents(kk).R(ii,jj));
            end
        end
    end
    fprintf(datafile,'};\n');

    % print S
    fprintf(datafile,'double S[%d] = { ', dimS);
    for kk = 1:Nnodes
        for jj = 1:size(agents(kk).S, 2)
            for ii = 1:size(agents(kk).S,1)
                fprintf(datafile,'%1.15e, ', agents(kk).S(ii,jj));
            end
        end
    end
    fprintf(datafile,'};\n');

end

% print q
fprintf(datafile,'double q[%d] = { ', dimq);
for kk = 1:Nnodes
    for ii = 1:size(agents(kk).q,1)
        fprintf(datafile,'%1.15e, ', agents(kk).q(ii));
    end
end
fprintf(datafile,'};\n');

% print r
fprintf(datafile,'double r[%d] = { ', dimr);
for kk = 1:Nnodes
    for ii = 1:size(agents(kk).r,1)
        fprintf(datafile,'%1.15e, ', agents(kk).r(ii));
    end
end
fprintf(datafile,'};\n');

%% optimal solution

fprintf(datafile, '\n/* Optimal solution */\n\n');

% print xopt
fprintf(datafile,'double xopt[%d] = { ', dimq);
for kk = 1:Nnodes
    for ii = 1:size(agents(kk).xopt,1)
        fprintf(datafile,'%1.15e, ', agents(kk).xopt(ii));
    end
end
fprintf(datafile,'};\n');

% print uopt
fprintf(datafile,'double uopt[%d] = { ', dimr);
for kk = 1:Nnodes
    for ii = 1:size(agents(kk).uopt,1)
        fprintf(datafile,'%1.15e, ', agents(kk).uopt(ii));
    end
end
fprintf(datafile,'};\n');

fclose(datafile);

end

