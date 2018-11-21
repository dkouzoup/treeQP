function [ val ] = extract_param_value(model_str, param_str)

% EXTRACT_PARAM_VALUE extract arithmetic value of param_str from model_str,
%                     with model_str following convenion 'YYY_param_str_X'.

idx_start = strfind(model_str, param_str);

if isempty(idx_start)
    val = nan;
else
    idx_start = idx_start + length(param_str)+1;
    idx_end   = idx_start;
    
    for ii = idx_start+1:length(model_str)
        if isnan(str2double(model_str(ii)))
            break
        else
            idx_end = idx_end + 1;
        end
    end
    
    val = str2double(model_str(idx_start:idx_end));
    
end

end

