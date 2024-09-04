 clc;
clear all;
close all;

% 1读取数据
data_path = "your_File_path"; %读入目录下文件信息存储为结构体形式
a = dir(fullfile(data_path, '*.xlsx'));
      
b=struct2cell(a);  %将格式转为cell形式
c=b(1,:);        % 取出其中文件名单元
[h,col]=size(c);   %计算文件个数


total_result{1,1}="序号";
total_result{1,2}="数据集";
total_result{1,3}="01损失";
total_result{1,4}="均方误差";
total_result{1,5}="学习时间";
total_result{1,6}="分类时间";
for ii=3:col
    if strfind(c{ii},'.xlsx')    %如果是xls文件格式 注意括号要使用cell的括号
        str=[num2str(ii) '当前数据集=' c{ii}];         %c{ii} 为数据集名字
        disp(str);
         
        data = xlsread(data_path+c{ii});
        data =data';   
        classify_node_num = size(data,1);

        kfold ="CV-5-stable";
        % kfold ="CV-10-stable";
        run_times = 5;

        [zol,accuracy ,learn_classify_time,rmse] = learn_struct_and_classified(data,classify_node_num, ...
            "MSKDB", kfold,'run_times',run_times);
        disp(mean(zol));
        result = [zol  rmse learn_classify_time accuracy];
        re_name =["01损失" "均方误差"  "学习时间" "分类时间" "准确度"];
        result_1 = [re_name;result];

        save_data_path  = fullfile(pwd, 'result\'); %  实验结果路径
        fileaddress1=strcat( save_data_path,kfold,"_T_",num2str(run_times),"_MSKDB__", c{ii});

        xlswrite(fileaddress1,result_1); % 写入数据
        %%%%%%%%%%% 2 %%%%%%%%%%%%%%%%%
        % aa_time = mean(learn_classify_time);
        aa_time = [mean(learn_classify_time(:,1))  mean(learn_classify_time(:,2))];   
        total_result{ii+1,1}=ii;
        total_result{ii+1,2}=c{ii};
        total_result{ii+1,3}= mean(zol);
        total_result{ii+1,4}= mean(rmse);
        total_result{ii+1,5}= aa_time(1);
        total_result{ii+1,6}=aa_time(2);

        fileaddress2=strcat( save_data_path,kfold,"_TT_",num2str(run_times),'_1__MSKDB__实验结果.xlsx');

        xlswrite(fileaddress2,total_result);

    end
    disp("=============================");
end



