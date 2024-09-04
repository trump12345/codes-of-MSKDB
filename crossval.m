function  cv_ren =  crossval_ren(goal, classify_algorithm)

sample = size(goal,1);
switch classify_algorithm
    
    case 'CV-5'
        cv = cvpartition(goal, 'kfold', 5);
        cv_ren=cv;
    case 'CV-5-stable'
        fold = 5;
        indices = ceil(fold*(1:sample)/sample);
        cv_ren.NumTestSets = fold;
        for i = 1:fold %循环3次，分别取出第i部分作为测试样本，其余两部分作为训练样本
            cv_ren.test{i} = logical(indices == i);
            cv_ren.training{i}  = ~cv_ren.test{i} ;
        end
    case 'CV-10-stable'
        fold = 10;
        indices = ceil(fold*(1:sample)/sample);
        cv_ren.NumTestSets = fold;
        for i = 1:fold %循环3次，分别取出第i部分作为测试样本，其余两部分作为训练样本
            cv_ren.test{i} = logical(indices == i);
            cv_ren.training{i}  = ~cv_ren.test{i} ;
        end
    otherwise
        % error classify algorithm
        fprintf('%s classifier algorithm is not support',classify_algorithm);
        return;
end

end
