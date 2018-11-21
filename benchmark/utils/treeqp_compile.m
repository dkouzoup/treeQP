function treeqp_compile(solver_opts, NRUNS, TREEQP_ROOT)

% TREEQP_COMPILE Compile treeQP and solve_qp_json executable if any options
%                are updated (assumes treeQP has been compiled with current
%                Makefile.rule parameters).


COMPILE = 0;

%% Edit Makefile.rule

COMPILE = makefile_rule_set_option_if_changed('DEBUG', 'OFF', TREEQP_ROOT, COMPILE);  % not to have -g -O0 flags

COMPILE = makefile_rule_set_option_if_changed('PRINT_LEVEL', 0, TREEQP_ROOT, COMPILE);

% TODO: pass as option instead
COMPILE = makefile_rule_set_option_if_changed('NREP', NRUNS, TREEQP_ROOT, COMPILE);

if isfield(solver_opts, 'openmp')
    if solver_opts.openmp.ON == 0
        COMPILE = makefile_rule_set_option_if_changed('OPENMP', 'OFF', TREEQP_ROOT, COMPILE);
    else
        COMPILE = makefile_rule_set_option_if_changed('OPENMP', 'ON', TREEQP_ROOT, COMPILE);
    end
end

COMPILE = makefile_rule_set_option_if_changed('SAVE_DETAILED_RESULTS', 'OFF', TREEQP_ROOT, COMPILE);

if isfield(solver_opts, 'DETAILED_TIMINGS') && solver_opts.DETAILED_TIMINGS == 1
    PROFILE = 3; % log timer per key operation per iteration for dual Newton algorithms (no total timings)
else
    % NOTE: needs to be 2 if we want to log ls_iters
    PROFILE = 0;
end

COMPILE = makefile_rule_set_option_if_changed('PROFILING_LEVEL', PROFILE, TREEQP_ROOT, COMPILE);

%% Set up BLASFEO

DEEP_CLEAN = 0;

DEEP_CLEAN = makefile_rule_set_option_if_changed('BLASFEO_VERSION', solver_opts.BLASFEO_LA, TREEQP_ROOT, DEEP_CLEAN);

DEEP_CLEAN = makefile_rule_set_option_if_changed('BLASFEO_TARGET', solver_opts.BLASFEO_TARGET, TREEQP_ROOT, DEEP_CLEAN);

%% compile treeQP

here        = pwd;
source_file = ['examples' filesep 'solve_qp_json.out'];
target_file = [here filesep 'solve_qp_json.out'];

if COMPILE || DEEP_CLEAN || ~exist(target_file, 'file')
    
    cd(TREEQP_ROOT)
    
    if DEEP_CLEAN
        system('make deep_clean')
    else
        system('make clean')
    end
    
    system('make solve_qp_json -j 4')
    
    if exist(target_file, 'file')
        delete(target_file)
    end
    copyfile(source_file, here)
    delete(source_file)
    
    cd(here)
    
end

end


function COMPILE = makefile_rule_set_option_if_changed(OPTION, VALUE, FILEPATH, COMPILE)

% MAKEFILE_RULE_SET_OPTION_IF_CHANGED if given value is different than the
%                                     existing one, change value and
%                                     indicate that software needs
%                                     recompilation.

if ~ischar(VALUE)
    VALUE = num2str(VALUE);
end

VALUE_PREV = makefile_rule_get_option(OPTION, FILEPATH);

if ~contains(VALUE_PREV, VALUE)
    COMPILE = 1;
    makefile_rule_set_option(OPTION, VALUE, FILEPATH);
end

end


function VALUE = makefile_rule_get_option(OPTION, FILEPATH)

% MAKEFILE_RULE_GET_OPTION Find line 'OPTION = ' in Makefile.rule located
%                          in FILEPATH and return its VALUE. 

% open the two files
fid_in  = fopen([FILEPATH 'Makefile.rule'], 'r');

if fid_in < 0
    error('Did not find Makefile.rule');
end

str   = [OPTION ' ='];
found = 0;

VALUE = NaN;

% scan Makefile
while ~feof(fid_in)

    % read line
    fline = fgetl(fid_in);

    % check for flag in this line (but without # before, which is a comment)    
    if contains(fline, str) && fline(1) ~= '#'
        found = 1;
        
        % sanity check
        if ~strcmp(fline(1:length(OPTION)), OPTION) 
            error('line found, but does not start with option name')
        end
        VALUE = fline(strfind(fline,'=')+1:end);
        break
    end
end

% close files
fclose(fid_in);

% copy and delete temp

if found == 0
   error('Option not found in Makefile.rule') 
end

end


function found = makefile_rule_set_option(OPTION, VALUE, FILEPATH)

% MAKEFILE_RULE_SET_OPTION Find line 'OPTION = ' in Makefile.rule located
%                          in FILEPATH and set its value to VALUE. 

% open the two files
fid_in  = fopen([FILEPATH 'Makefile.rule'], 'r');

if fid_in < 0
    error('Did not find Makefile.rule');
end

fid_tmp = fopen('tmp','wt');

if ~ischar(VALUE)
    VALUE = num2str(VALUE);
end

str   = [OPTION ' ='];
found = 0;

% scan Makefile
while ~feof(fid_in)

    % read line
    fline = fgetl(fid_in);

    % check for flag in this line (but without # before, which is a comment)    
    if contains(fline, str) && fline(1) ~= '#'
        found = 1;
        
        % sanity check
        if ~strcmp(fline(1:length(OPTION)), OPTION) 
            error('line found, but does not start with option name')
        end  
        fline = [OPTION ' = ' VALUE];
    end

    % replace escape characters for printf
    fline = strrep(fline, '\', '\\');
    
    % add end of line character
    fnew  = [fline '\n'];
    
    % print in temp file
    fprintf(fid_tmp, fnew);    
end

% close files
fclose(fid_in);
fclose(fid_tmp);

% copy and delete temp
copyfile('tmp', [FILEPATH 'Makefile.rule']);
delete('tmp');

if found == 0
   error('Option not found in Makefile.rule') 
end

end
