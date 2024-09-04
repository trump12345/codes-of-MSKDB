function DAG = learn_struct_mskdb(data, class_node, node_sizes,k)
% learn_struct_kdb_h: �����������  I(C;xi)/H(xi) �Ӵ�С���򣬸��ڵ��ѡ��ΪCMI(xi;xj|c)
% (with discrete nodes)
% dag = learn_struct_kdb_h(data, class_node, node_sizes,k,scoring_fn)
% Input :
% 	data(i,m) is the value of node i in case m���ڵ� �� ����
% 	class_node is the class node
%   node_sizes ÿ���ڵ���ɢ״̬��
%
% Output :
%	dag = adjacency matrix of the dag

if nargin == 3
    k=2;
end;

N=size(data,1);
dag=zeros(N);
notClass = setdiff(1:N,class_node);
[sort_attribute,CMI,MI_xc] = attrbute_sort(data,node_sizes,class_node);

% disp(['��������=' num2str(sort_attribute)]);
DAG= learn_struct(dag,sort_attribute,CMI,class_node,k);
% plot(digraph(DAG));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sort_attribute,CMI,MI_xc] =   attrbute_sort(data,node_sizes,class_node)

Nnode = size(data,1);
notClass = setdiff(1:Nnode,class_node);

CMI = zeros(Nnode,Nnode); % CMI(x,y|c)
MI_xc = zeros(1,length(notClass)); % MI(x,c)
H = zeros(1,Nnode); % ������ H(x)
MI_Ratio = zeros(1,length(notClass)); % MI_xc/H(x)

for entropy_i = 1: Nnode % size(notclass,2)   % ȡ��notclass�е�ÿ��Ԫ�ؼ�������
    % ����ÿ���ڵ����  H(X)
    H(entropy_i) = entropy_ren(entropy_i ,data,node_sizes(entropy_i)); % �ڵ����
    if entropy_i~=class_node
        laplas=0;
        MI_xc(entropy_i) = mutual_info_score_ren(data,entropy_i ,class_node ,laplas ,node_sizes(entropy_i) ,node_sizes(class_node)); % ��������ڵ�֮��Ļ���Ϣ
        MI_Ratio(entropy_i) =  MI_xc(entropy_i)/H(entropy_i);
    end
    for entropy_j = 1: Nnode % size(notclass,2)
        if entropy_i~=entropy_j && entropy_j~=class_node && entropy_i~=class_node
            CMI(entropy_i,entropy_j) = cond_mutual_info_score_ren(data,entropy_i,entropy_j,class_node,node_sizes(entropy_i),node_sizes(entropy_j),node_sizes(class_node));
        end
    end
end
[~, MI_Ratio_I] = sort(MI_Ratio,'descend');
sort_attribute =MI_Ratio_I;
end

function [dag]= learn_struct(dag,sort_att,CMI,class_node,K)
% Ϊÿ��sort�еĽڵ��ҵ����ڵ�
for  i =1:length(sort_att)    
    if i ==1 % ���Ϊ��1���ڵ㣬ֱ�ӽ�����뵽������
        dag(class_node,sort_att(i))=1;
        % disp(['��ǰ����=' num2str(sort_att(i)) "���ڵ㣺=" num2str(class_node)]);
    else % ������ǵ�һ���ڵ㣬�ҵ���ǰ�ڵ�ǰ��i-1���ڵ��У�ʹ�õ�ǰ�ڵ�����CMI
        if i<= (K+1)  % ���Ϊ�ڶ����͵������ڵ㣬ֱ�ӽ�����뵽������
            dag(class_node,sort_att(i))=1;
            for  j =1:i-1 % ǰ���i-1���ڵ㶼Ϊ�ڵ�i�ĸ��ڵ�
                dag(sort_att(j),sort_att(i))=1;
            end
        else  % ˵����ǰ�����Ľڵ㳬����3������ôѡ��i-1������k��Ϊ�ڵ�i�ĸ��ڵ�
%             temp =zeros(1,i-1);
            for  j =1:i-1
                temp(j,1) = sort_att(j); %  ����ڵ�
                temp(j,2) = CMI(sort_att(i),sort_att(j));  %  ���滥��Ϣֵ
            end            
            % ǰ��K��Ϊi�ĸ��ڵ�            
            [~,I]=sort(temp(:,2),'descend');
            A=temp(I,:);
            parent_set = A(1:K,1);
            dag(class_node,sort_att(i))=1;
            dag(parent_set,sort_att(i))=1;
            % disp(['��ǰ����=' num2str(sort_att(i)) "���ڵ㣺=" num2str(parent_set')]);                        
            clear temp;
        end
    end
end
end




