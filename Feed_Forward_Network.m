%% Introduction - creating the X shape
clc
clear

% creating X shape
xv = [0 0.4 0 0.1 0.5 0.9 1 0.6 1 0.9 0.5 0.1 0]; % x_values 
yv = [0.1 0.5 0.9 1 0.6 1 0.9 0.5 0.1 0 0.4 0 0.1]; % y_values

% randomizing 1000 x,y coordinates
dots(:,1) = rand(1000,1);
dots(:,2) = rand(1000,1);

% classifying the coordinates to 'inside' and 'outside'
truth = inpolygon(dots(:,1),dots(:,2),xv,yv); 

% plotting the X shape with the dots
figure()
plot(dots(truth,1),dots(truth,2),'r+') % points inside
hold on
plot(dots(~truth,1),dots(~truth,2),'bo') % points outside
plot(xv, yv, 'LineWidth',1)
hold off
title('Example of X shape with 1000 random points','FontSize', 13)
legend({'inside points','outside points'})

%% one layer
clc
clear

% X shape
xw = [0 0.4 0 0.1 0.5 0.9 1 0.6 1 0.9 0.5 0.1 0];
yw = [0.1 0.5 0.9 1 0.6 1 0.9 0.5 0.1 0 0.4 0 0.1];
 
number_of_dots = 10000; % number of random coordinates of each iteration

% first train-test iteration
dots(:,1) = rand(number_of_dots,1);
dots(:,2) = rand(number_of_dots,1);
dots(:,3) = ones(number_of_dots,1); % bias
w = rand(1,3); % initial weights
truth = inpolygon(dots(:,1),dots(:,2),xw,yw); % ground truth
W(1,:) = one_layer(w, dots, truth, 0.1, number_of_dots, 'sigmoid'); % one-layer network function (see function below)
success(1) = test(W,dots(number_of_dots*0.9+1:number_of_dots,:),truth(0.9*number_of_dots+1:number_of_dots)); % success calculation function (documantation below)

% running the network through 1000 train-test iterations
iteration = 1000;
    for i=2:iteration
        dots(:,1) = rand(number_of_dots,1);
        dots(:,2) = rand(number_of_dots,1);
        dots(:,3) = ones(number_of_dots,1); % bias
        w = W(i-1,:);
        truth = inpolygon(dots(:,1),dots(:,2),xw,yw); % correct answers
        W(i,:) = one_layer(w, dots, truth, 0.1, number_of_dots, 'sigmoid');
        success(1,i) = test(W(i,:),dots(0.9*number_of_dots+1:number_of_dots,:),truth(0.9*number_of_dots+1:number_of_dots));
    end


ave_suc = mean(success(1,200:end)); % calculating average network success

% plotting the success over the iterations
figure()
plot([1:iteration],success);
title('Perceptron Perrformance on the "X" Problem')
ylim([0 100])
xlabel('# Training')
ylabel('Success of the Network (%)')
yline(ave_suc,'--');
hold on
text(50,73,['Average success rate: ',num2str(round(ave_suc)),'%'])

%% two layers
clc
clear

% X shape
xw = [0 0.4 0 0.1 0.5 0.9 1 0.6 1 0.9 0.5 0.1 0];
yw = [0.1 0.5 0.9 1 0.6 1 0.9 0.5 0.1 0 0.4 0 0.1];

% first train-test iteration
dots(:,1) = rand(10000,1);
dots(:,2) = rand(10000,1);
dots(:,3) = ones(10000,1); % bias
w_1 = rand(3,14); % initial weights of first layer
w_2 = rand(15,1); % initial weights of second layer
truth = inpolygon(dots(:,1),dots(:,2),xw,yw); % ground truth
[w1(:,:,1),w2(:,1)] = two_layers(w_1, w_2, dots, truth, 0.1, 10000,"sigmoid","sigmoid"); % two-layer network function (see function below)
[success(1),output(1,:)] = test_two(w1(:,:,1),w2(:,1),dots(9001:10000,:),truth(9001:10000,:),"sigmoid","sigmoid"); % success calculation function (see function below)

% running the network through 1000 train-test iterations
iteration = 500;
    for i=2:iteration
        dots(:,1) = rand(10000,1);
        dots(:,2) = rand(10000,1);
        dots(:,3) = ones(10000,1); % bias
        w_1 = w1(:,:,i-1);
        w_2 = w2(:,i-1);
        truth = inpolygon(dots(:,1),dots(:,2),xw,yw);
        [w1(:,:,i),w2(:,i)] = two_layers(w_1, w_2, dots, truth, 0.1, 10000,"sigmoid","sigmoid");
        [success(i),output(i,:)] = test_two(w1(:,:,i),w2(:,i),dots(9001:10000,:),truth(9001:10000),"sigmoid","sigmoid");
    end

% creating plots of network performance

error = 100-success; % calculating error rates
ave_err_max = mean(error(1:100)); % average initial error rates
ave_err_min = mean(error(500:end)); % average final error rates
ave_suc = mean(success(1,500:end)); % average final success rates

% plotting error rates over the iterations
figure()
plot([1:iteration],error);
title('Network Perrformance on the "X" Problem')
ylim([0 100])
xlabel('# Training')
ylabel('Error rates (%)')
yline(ave_err_max,'--',['Initial average error rate: ',num2str(round(ave_err_max)),'%']);
yline(ave_err_min,'--',['Final average error rate: ',num2str(round(ave_err_min)),'%']);

% plotting success rates over the iterations
figure()
plot([1:iteration],success);
title('Network Perrformance on the "X" Problem')
ylim([0 100])
xlabel('# Training')
ylabel('Success rate (%)')
yline(ave_suc,'--');
hold on
text(270,92,['Average success rate: ',num2str(round(ave_suc)),'%'])

% plotting final result of the X problem

final_test_dots = dots(9001:10000,:); % last 1000 test coordinates
final_true_dots = truth(9001:10000,:); % last ground truth
test_output = output';
test_output = test_output(:,iteration);

% plotting over X shape
figure()
plot(final_test_dots(final_true_dots,1),final_test_dots(final_true_dots,2),'r+') % points inside
hold on
plot(final_test_dots(~final_true_dots,1),final_test_dots(~final_true_dots,2),'bo') % points outside
plot(final_test_dots(test_output~=final_true_dots,1),final_test_dots(test_output~=final_true_dots,2),'k*') % points outside
plot(xw, yw, 'LineWidth',1)
hold off
title('Network performance after trainings','FontSize', 15)
legend({'inside points','outside points','wrong points'})

%% Neural inspection - looking into 5 middle-layer's neurons

% getting 500 random input x,y coordinates
dot_number = 500;

% running the network while focusing on the middle layer's neurons
neurons = [1:14]; % number of middle-layer's neurons
w1_neurons = w1(:,neurons,iteration); % final w1 weights
w2_neurons = w2(neurons,iteration); % final w2 weights

% running the network
input_neurons = [];
for neuron = 1:length(neurons)
    for dot = 1:dot_number
        input_neurons(neuron,dot) = sigmoid(dots(dot,:)*w1_neurons(:,neuron));
    end
end

output_neurons = [];
for neuron = 1:length(neurons)
    output_neurons(neuron,:) = sigmoid(w2_neurons(neuron)*input_neurons(neuron,:));
end

% ploting the unique neural activity of 5 middle layer's neurons

figure()
neu_num = [9,10,11,3,4]; % specific neurons
for i=1:length(neu_num)
    subplot(2,3,i)
    plot(xw,yw)
    hold on
    scatter(dots(1:dot_number,1),dots(1:dot_number,2),[],output_neurons(neu_num(i),:))
    title(['Neuron no.', num2str(neu_num(i))])
    xlabel("input's x value")
    ylabel("input's y value")
    colorbar
end
sgtitle("Unique Activity of Middle Layer's Neurons")

%% functions

% various activation functions

% ReLu
function [rslt,rslt_tag] = relu(x)
    for j = 1:size(x,2)
        for i=1:size(x,1)
            if x(i,j) > 0
                rslt(i,j) = x(i,j);
                rslt_tag(i,j) = 1;
            else
                rslt(i,j) = 0;
                rslt_tag(i,j) = 0;
            end
        end
    end
end

% sigmoid
function [y,y_tag] = sigmoid(x)
    y = 1./(1+exp(-x));
    y_tag = y.*(1-y);
end

% hyperbolic tangent
function [y,y_tag] = tang(x)
    y = tanh(x);
    y_tag = 1-(tanh(x)).^2;
end

% function of all activation functions
function [y,y_tag] = activation(x,type)
    if type == "sigmoid"
        [y,y_tag] = sigmoid(x);
    elseif type == "arctan"
        y = atan(x);
        y_tag = 1./(1+((atan(x)).^2));
    elseif type == "tang"
        y = tanh(x);
        y_tag = 1-(tanh(x)).^2;
    elseif type == "relu"
        [y,y_tag] = relu(x);
    end
end


% one-layer network function
% the function recieves the various inputs requiered for running the
% network, and returns the final weights

function W = one_layer(w, input, truth, eta, repeat, active)
    for i=1:0.9*repeat % training set - 90% of the input
        input_layer_1 = input(i,:); % getting the dots from each line

        % feed forward
        [output_layer_1,output_layer_1_tag] = activation(input_layer_1*w',active);
        
        % back propagating
        error = 0.5*(output_layer_1-truth(i,:))^2;
        error_tag = output_layer_1-truth(i,:);        
        dW = -eta.*(error_tag*output_layer_1_tag*input_layer_1);
        
        % updating weights
        W(1)=w(1)+dW(1);
        W(2)=w(2)+dW(2);
        W(3)=w(3)+dW(3);

        end
end

% success of one-layer network function
% the function recieves the output weights after training the network and
% returns the success rates and the final output

function [success,output] = test(W,input,truth)
    for i=1:length(input)
        input_layer_1 = input(i,:);
        output_layer_1 = input_layer_1*W';
        output(i,1) = round(sigmoid(output_layer_1));
    end
    total = (output-truth).^2;
    success = 100*(length(input)-sum(total))/length(input);
end

% two-layers network function
% the function recieves the various inputs requiered for running the
% network, and returns the final weights

function [w1,w2] = two_layers(w_1, w_2, input, truth, eta, repeat,active_layer_1,active_layer_2) % add activation
    for i=1:0.9*repeat % training set - 90% of the input

        % feed forward
        input_layer_1 = input(i,:);
        [output_layer_1,output_layer_1_tag] = activation(input_layer_1*w_1,active_layer_1);
        
        input_layer_2 = [output_layer_1,1];
        [output_layer_2,output_layer_2_tag] = activation(input_layer_2*w_2,active_layer_2);
        
        % back propagating
        error = 0.5*(output_layer_2-truth(i,:))^2;
        error_tag = output_layer_2-truth(i,:);
        
        dW2 = -eta.*(error_tag*output_layer_2_tag*input_layer_2);
        for j=1:size(w_1,2)
            dW1(:,j) = -eta.*(error_tag*output_layer_2_tag*w_2(j)*output_layer_1_tag(j)*input_layer_1);
        end

        % updating weights
        w_2 = w_2+dW2';
        w_1 = w_1+dW1;
    end

    % final weights
    w1 = w_1;
    w2 = w_2;
end

% success of two-layers network function
% the function recieves the output weights after training the network and
% returns the success rates and the final output

function [success,output] = test_two(w1, w2, input, truth, active1,active2)
    for i=1:length(input)
        input_layer_1 = input(i,:);
        output_layer_1 = activation(input_layer_1*w1,active1);
        
        input_layer_2 = [output_layer_1,1];
        output_layer_2(i,:) = round(activation(input_layer_2*w2,active2));
    end

    total = (output_layer_2-truth).^2;
    output = output_layer_2;
    success = 100*(length(input)-sum(total))/length(input);
end

