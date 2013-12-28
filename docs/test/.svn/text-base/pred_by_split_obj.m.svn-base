function [splitobj_mean_p_per_split] = pred_by_split_obj(result_fileprefix)

r = load(strcat(result_fileprefix, '_NPAIRSJresult.mat'));
p = r.npairs_result.prediction.priors.pp_true_class;
split_obj_labels = r.npairs_result.split_obj_labels;           
testvols = r.npairs_result.split_test_vols;         
uniq_split_obj = unique(split_obj_labels);

num_split = size(testvols,1); % actually num split halves...
num_split_obj = size(uniq_split_obj,2);
splitobj_mean_p_per_split = zeros(num_split, num_split_obj);

for s = 1:num_split_obj,
    strcat('Split Object no. ',num2str(s));
    % scan nos (1-rel) of current split object:
    curr_splitobj_vols = find(~(split_obj_labels - uniq_split_obj(s)));
          
    for r = 1:num_split,
	strcat('Split half no. ', num2str(r));
	curr_split_vols = testvols(r,:);
        curr_split_p = p(r,:);
        % loc contains the (highest) index in curr_split_vols
        % for each element in curr_splitobj_vols that is a member of
        % curr_split_vols.  For elements of curr_splitobj_vols that 
        % do not occur in curr_split_vols, ismember returns 0.
	  [tf, loc] = ismember(curr_splitobj_vols, curr_split_vols);

        if (sum(tf) == 0)
            strcat('Split Object ',num2str(uniq_split_obj(s)),...
		   ' is not in split half no. ',num2str(r));
           
        elseif (sum(tf) < size(curr_splitobj_vols,1)) 
            strcat('Warning: some vols belonging to split object ',...
		   num2str(uniq_split_obj(s)),' are not contained',...
                   ' in split half no.', num2str(r))
        else 
	    curr_p = curr_split_p(loc);
	    splitobj_mean_p_per_split(r, s) = mean(curr_split_p(loc));
        end 
    end
end    
end
            
            
           

