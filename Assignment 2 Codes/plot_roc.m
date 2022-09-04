function  auc = plot_roc(predict, ground_truth)  
% plot the ROC curve
   
x = 1.0;  
y = 1.0;  

%calculate numbers of positive and negative samples
pos_num = sum(ground_truth==1);  
neg_num = sum(ground_truth==0);  

%calculate step size for plot
x_step = 1.0/neg_num;  
y_step = 1.0/pos_num;  

%sort the output value 
[~,index] = sort(predict);  
ground_truth = ground_truth(index);  

%plot the ROC curve
for i=1:length(ground_truth)  
    if ground_truth(i) == 1  
        y = y - y_step;  
    else  
        x = x - x_step;  
    end  
    X(i)=x;  
    Y(i)=y;  
end  

hold on;
grid on;
set(gcf,'Position',[50/0.277 50/0.277 100/0.277 100/0.277]);
plot(X,Y,'LineWidth',2,'MarkerSize',3);  
plot(X,X,'--k','LineWidth',0.5)
xlim([0 1]); ylim([0 1]);

xlabel('False Positive Rate', 'Interpreter', 'latex');  
ylabel('True Positive Rate', 'Interpreter', 'latex');  
title('ROC Curve', 'Interpreter', 'latex');  
%calculate the area under ROC curve, i.e. auc 
auc = -trapz(X,Y); 
end 