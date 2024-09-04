function [zol,accuracy ,learn_classify_time,rmse_matrix] ...
    = learn_struct_and_classified(data,classify_node_num, struct_algorithm, classify_algorithm,varargin)


% set default parameters  设置默认参数
node_flag = 'A';
label = {};
score = 'bic';
% get variable parameters
if ~isempty(varargin)
    args = varargin;
    nargs = length(args);
    
    if length(args) > 0
        if ischar(args{1})
            for i=1:2:nargs
                switch args{i}
                    case 'node_flag'
                        node_flag = args{i+1};
                    case 'label'
                        label = args{i+1};
                    case 'score'
                        score = args{i+1};
                    case 'run_times'
                        run_times = args{i+1};
                    case 'fan'
                        fan=args{i+1};
                    case 'params'
                        if isempty(args{i+1})
                            params = cell(1,n);
                        else
                            params = args{i+1};
                        end
                end
            end
        end
    end
end


for run_time_i = 1:run_times    
    disp(['第' num2str(run_time_i) '次独立运行的计算']);

    if contains(classify_algorithm, 'stable') % 固定分组        
         rng(run_time_i); % 设置种子为run_time_i
         p = randperm(size(data,2));  % 生成一个随机排列 
         data= data(:,p); % 固定打乱顺序
    end
            
    [r,c]=size(data);
    dNodes =1:r;
    % date_preprocess
    [~, node_sizes, node_type,node_names] = data_process(r,data,node_flag);

    % 1 对数据进行划分
    goal = data(classify_node_num,:)';
    Acutural_Y = data(classify_node_num,:);
    % 将y值变为矩阵形式
    acutural_Y_matrix  = full(sparse(1:numel(Acutural_Y),Acutural_Y,1));

    

    switch classify_algorithm        
        case 'HOLD_OUT'
            cv = cvpartition(goal, 'HoldOut',1/10);
        case 'CV-5'
            cv = cvpartition(goal, 'kfold',5);
        case 'CV-5-stable'
            cv =  crossval(goal, classify_algorithm);
        case 'CV-10'
            cv = cvpartition(goal, 'kfold',10);
        case 'CV-10-stable'
            cv =  crossval(goal, classify_algorithm);
        otherwise
            % error classify algorithm
            fprintf('%s classifier algorithm is not support',classify_algorithm);
            confusion_matrix = [];
            correct_rate = [];
            return;
    end   
    
    data_temp = num2cell(data);
%     maxNumCompThreads(10);
    for cv_i = 1:cv.NumTestSets
        if isa(cv, 'cvpartition') 
            trainData= data(:,cv.training(cv_i)); % 训练模型数据
            testData =data_temp(:,cv.test(cv_i));
            actual_Y_test = acutural_Y_matrix(cv.test(cv_i),:);
            actual_Y_train = acutural_Y_matrix(cv.training(cv_i),:);
        else
            trainData= data(:,cv.training{cv_i}); % 训练模型数据
            testData =data(:,cv.test{cv_i});
            testData = num2cell(testData);
            
            actual_Y_test = acutural_Y_matrix(cv.test{cv_i},:);
            
        end

        testData(classify_node_num,:) = {[]}; %# remove class     
        
        start_time = cputime; % 开始时间
        %  2 结构学习 调用结构学习函数
        dag = learn_struct(trainData,struct_algorithm, node_sizes, node_type,classify_node_num,varargin);        
        
        
        % 2.1 参数学习
        bnet = mk_bnet(dag, node_sizes, 'discrete',dNodes, 'names',node_names);
        for node_i=1:numel(dNodes)
            name = node_names{dNodes(node_i)};
            bnet.CPD{node_i} = tabular_CPD(bnet, node_i, ...
                'prior_type','dirichlet',"dirichlet_type","unif"); %BDeu  unif
        end
        %# 用训练集数据进行参数学习
        Bnet = learn_params(bnet, trainData);
        
        learn_struct_time(cv_i,run_time_i) = cputime - start_time;  % 第cv_i折运行时间

        % 3  testing分类  调用分类接口
        start_classify_time = cputime;
       
        [~,correct_rate,rmse] = classified_test(Bnet,classify_node_num,testData,actual_Y_test ,node_sizes);
        end_classified_time(cv_i,run_time_i) = cputime - start_classify_time;        
        correct_rate_matrix(cv_i,run_time_i) = correct_rate;
        rmse_matrix(cv_i,run_time_i) = rmse;
%          
    end
    
    disp(['01 = =' num2str(1-(mean(correct_rate_matrix(:,run_time_i))'/100))]);
end

[zol,rmse_matrix, accuracy, learn_classify_time] = result_process(rmse_matrix,correct_rate_matrix,learn_struct_time,end_classified_time,run_times);

end
function [zol,rmse_matrix, accuracy, learn_classify_time] = result_process(rmse,correct_rate_mat,learn_t,classified_t,run_t)
% 
if run_t >1
    %  计算每轮结果的平均值，
    rmse_matrix = mean(rmse)';
    accuracy = mean(correct_rate_mat)/100;
    accuracy  = accuracy';
    zol = 1 - accuracy;
    learn_classify_time = [mean(learn_t)' mean(classified_t)'];
else
    %  只有一轮，直接保存每折交叉验证结果
    rmse_matrix = rmse;
    accuracy = (correct_rate_mat)/100;    
    zol = 1 -accuracy;
    learn_classify_time = [learn_t classified_t];
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% check data
function [r,c] = check_data_integer(data)

[r,c] = size(data);
if (r == 1)
    disp('The row of data can not equal 1 \n');
    return
end

for i=1:r
    for j=1:c
        if(data(i,j) ~= fix(data(i,j)))%Y = fix(X) 将 X 的每个元素朝零方向四舍五入为最近的整数。对于正方向的 X，fix 的行为与 floor 相同。对于负方向的 X，fix 的行为与 ceil 相同。
            fprintf('[%d,%d] is not integer \n',i,j)
            break
        end
    end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% data process
function [bnet ,node_sizes, node_type,node_name] = data_process(r,...
    data, node_flag)

dag = zeros(r,r);
[node_sizes,node_name,node_type] = get_node(data,node_flag);
bnet = mk_bnet(dag, node_sizes, 'names', node_name, 'discrete', 1:r);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% get node sizes
function [node_sizes,node_name,node_type] = get_node(data,node_flag)

[r,~] = size(data);

node_class_num = cell(1,r);
node_sizes = cell(1,r);
node_name = cell(1,r);
node_type = cell(1,r);
for i = 1:r
    eval([node_flag, num2str(i) '=num2str(i);']);
    node_name{i} = [node_flag, num2str(i)];
    node_class_num{i} = unique(data(i,:));
    node_sizes{i} = length(node_class_num{i});
    node_type{i} = 'tabular';
end

node_sizes = cell2mat(node_sizes);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% learn struct
function dag = learn_struct(data, struct_algorithm, node_sizes,...
    node_type,classify_node_num, varargin)

[r,~] = size(data);

switch struct_algorithm
   
    case 'MSKDB'
        % KDB_H algorithm;
         dag = learn_struct_mskdb(data, classify_node_num, node_sizes);
 
    otherwise
        fprintf('%s struct algorithm is not support \n',struct_algorithm);
        dag = zeros(r,r);
        return
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% classified test
function [confusion_matrix,correct_rate,RMSE_RESULT] = classified_test(bnet, classify_node_num,...
    test_data, true_Y_test, node_sizes)

node_flag = 'A';
data = test_data;

prob = zeros(node_sizes(classify_node_num), size(test_data,2));
engine = jtree_inf_engine(bnet);         %# Inference engine
for j = 1:size(test_data,2)
    [engine,loglik] = enter_evidence(engine, test_data(:,j));
    marg = marginal_nodes(engine, classify_node_num);
    prob(:,j) = marg.T;
end

% 选出概率最大的值的状态设置为1，然后其他的类节点状态设置为0
predInd = prob;
predInd(max(predInd)==predInd)=1; % 将最大值位置设置为1 其他位置设置为0
predInd(max(predInd)~=predInd)=0;
predInd =  predInd';


[C, RATE] = confmat(predInd, true_Y_test);
correct_rate = RATE(:,1);
confusion_matrix =C;
%%%%%%%%%%%%%% 2 计算均方误差 %%%%%%%%%%
rmse_temp = prob';
rmse_temp(true_Y_test==0)=0;
rmse =rmse_temp-true_Y_test;
rmse2 = rmse.^2;

RMSE_RESULT = sqrt(sum(sum(rmse2))/size(true_Y_test,1));


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%