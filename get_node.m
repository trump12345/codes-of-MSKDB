    function [node_sizes,node_name,node_type] = get_node(data,node_flag)
   
    [r,~] = size(data);  %忽略列输出
    node_class_num = cell(1,r);%初始化元胞数组%1x32
    node_sizes = cell(1,r);
    node_name = cell(1,r);
    node_type = cell(1,r);
    for i = 1:r
        eval([node_flag, num2str(i) '=num2str(i);']);%  A=num2str(i)，node_flag=A1,A2...
        node_name{i} = [node_flag, num2str(i)];%节点名字，同上
        node_class_num{i} = unique(data(i,:));%第i行的唯一值 分几类？类节点数量？
        node_sizes{i} = length(node_class_num{i});%第i个节点的特征取值个数
        node_type{i} = 'tabular';
    end
    node_sizes = cell2mat(node_sizes);%元胞数组转成普通数组
    end
