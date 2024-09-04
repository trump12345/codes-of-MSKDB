 clc;
clear all;
close all;

% 1��ȡ����
data_path = "your_File_path"; %����Ŀ¼���ļ���Ϣ�洢Ϊ�ṹ����ʽ
a = dir(fullfile(data_path, '*.xlsx'));
      
b=struct2cell(a);  %����ʽתΪcell��ʽ
c=b(1,:);        % ȡ�������ļ�����Ԫ
[h,col]=size(c);   %�����ļ�����


total_result{1,1}="���";
total_result{1,2}="���ݼ�";
total_result{1,3}="01��ʧ";
total_result{1,4}="�������";
total_result{1,5}="ѧϰʱ��";
total_result{1,6}="����ʱ��";
for ii=3:col
    if strfind(c{ii},'.xlsx')    %�����xls�ļ���ʽ ע������Ҫʹ��cell������
        str=[num2str(ii) '��ǰ���ݼ�=' c{ii}];         %c{ii} Ϊ���ݼ�����
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
        re_name =["01��ʧ" "�������"  "ѧϰʱ��" "����ʱ��" "׼ȷ��"];
        result_1 = [re_name;result];

        save_data_path  = fullfile(pwd, 'result\'); %  ʵ����·��
        fileaddress1=strcat( save_data_path,kfold,"_T_",num2str(run_times),"_MSKDB__", c{ii});

        xlswrite(fileaddress1,result_1); % д������
        %%%%%%%%%%% 2 %%%%%%%%%%%%%%%%%
        % aa_time = mean(learn_classify_time);
        aa_time = [mean(learn_classify_time(:,1))  mean(learn_classify_time(:,2))];   
        total_result{ii+1,1}=ii;
        total_result{ii+1,2}=c{ii};
        total_result{ii+1,3}= mean(zol);
        total_result{ii+1,4}= mean(rmse);
        total_result{ii+1,5}= aa_time(1);
        total_result{ii+1,6}=aa_time(2);

        fileaddress2=strcat( save_data_path,kfold,"_TT_",num2str(run_times),'_1__MSKDB__ʵ����.xlsx');

        xlswrite(fileaddress2,total_result);

    end
    disp("=============================");
end



