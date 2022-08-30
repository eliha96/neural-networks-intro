%% Q1: A 3-Memory Hopfield Network

clc
clear

% creating 3 random memories (see function's details below)

memory_1 = random_memory(100);
memory_2 = random_memory(100);
memory_3 = random_memory(100);
memories = [memory_1;memory_2;memory_3];

% creating weights' table
W = 1/numel(memory_1).*(memory_1'*memory_1 + memory_2'*memory_2 + memory_3'*memory_3);
W = zero_diagonal(W);

%creating a noisy memory (se function' details below)
noisy_memory_1 = random_noise(memory_1,0.1);

% Creating a figure of the memories for demonstration
noisy_for_show = reshape(noisy_memory_1,[10,10]);
noise_place = find(noisy_memory_1~=memory_1);
noisy_for_show(noise_place) = 0;

figure('Position', [150 150 1000 500])
subplot(1,2,1)
imagesc(reshape(memory_1,[10,10]))
title('Original Memory','FontSize', 20)
colormap gray
axis off
subplot(1,2,2)
imagesc(reshape(noisy_memory_1,[10,10]))
title('Memory with 10% Noise','FontSize', 20)
subtitle('changed tiles appear in grey','FontSize', 20)
colormap gray
axis off
hold on
imagesc(noisy_for_show)
colormap gray
axis off

% running the network with the noisy memory and the weights table

is_similar = 0; % for breaking out of the loop
S = noisy_memory_1;
t = [1,10000]; % number of steps available
iter = randi(numel(memory_1),t); % randomizing for neuron updating
for i=1:length(iter)
    sum_S = W(iter(i),:)*S';
    S(1,iter(i)) = sign(sum_S);
    S(S==0)=1;
    S_log(i,:) = S;
    if S == memory_1
        disp(['The system is now memory 1 after ', num2str(i), ' steps'])
        is_similar = 1;
    elseif S == memory_2
        disp(['The system is now memory 2 after ', num2str(i), ' steps'])
        is_similar = 1;
    elseif S == memory_3
        disp(['The system is now memory 3 after ', num2str(i), ' steps'])
        is_similar = 1;
    elseif i > numel(memory_1)*50
        if S == S_log(i-1,:)
            if S == S_log(i-2,:)
                disp('The system is stuck')
                is_similar = 1;
            end
        end
    end
    if(is_similar == 1)
        break
    end
end

%% running the system 100 times
num_runs = 100;
for i=1:num_runs
    [success(i),steps(i)] = hopfield(noisy_memory_1,W,memories,5000);
end

% calculating convergance for one run
similarity(1) = sum(noisy_memory_1==memory_1)/numel(memory_1);
for i=2:length(S_log)+1
    similarity(i) = sum(S_log(i-1,:) == memory_1)/numel(memory_1);
end
similarity = similarity.*100;

% plotting convergance & steps variability of the system
figure('Position', [150 150 1000 500])
subplot(1,2,1)
plot([1:length(S_log)+1],similarity)
title('Convergence of the system for one run','FontSize', 12)
ylabel('Similarity to Original Memory (%)')
ylim([90, 100])
xlim([0,length(S_log)+5])
xlabel('Steps')
subplot(1,2,2)
plot(1:num_runs,steps)
title('Number of steps until reaching original memory for 100 runs','FontSize', 12)
ylabel('Steps until reaching Original Memory')
xlabel('# run')


%% Q2: Comperhesive Performance Analysis of the Network
% WARNING: this code runs for up to 2 minute

clc
clear

% running the network 50 times over 2-22 memories, each run with 10%-70% noise
%see the function details below
tic
[scoreboard,results] = network_performance([2:2:20],[0.1:0.1:0.7],50);
toc

%% showing success rates and average steps number of the different combinations

% heat-map of success rates
figure()
noise = {'10','20','30','40','50','60','70'};
memories = {'2','4','6','8','10','12','14','16','18','20'};
h = heatmap(memories,noise,scoreboard(:,:,1)');
h.Title = 'Success of the Network';
h.XLabel = 'Number of Memories';
h.YLabel = 'Noise (%)';

% steps until reaching convergance
noise = [10:10:70];
figure()
plot(noise,scoreboard(1,:,2),noise,scoreboard(3,:,2),noise,scoreboard(5,:,2),noise,scoreboard(7,:,2),noise,scoreboard(10,:,2))
title('Average steps until convergance in different memories and noises','FontSize', 11)
xlim([10 70])
xlabel('Noise %')
ylabel('Average steps until convergance')
xticks([10 20 30 40 50 60 70 80 90])
xticklabels({'10','20','30','40','50','60','70'})
yticklabels({'200','300','400','500','600','700','800','900','>1000'})
legend({'2 memories','6 memories','10 memories','14 memories','20 memories'},'Position',[0.77 0.25 0 0])


%% Q3: Random vs. Deterministic Noise
clc
clear

% creating 5 random memories
for i=1:5
    memories(i,:) = random_memory(100);
    W(:,:,i) = 1/numel(memories(i,:)).*(memories(i,:)'*memories(i,:));
end

W = sum(W,3);
W = zero_diagonal(W);

% creting noise matrix - for random and deterministic noise
% see function details below

noises = [0.1:0.1:0.9]; % precent of noises
for i=1:length(noises)
    noise_mat(i,:,1) = random_noise(memories(1,:),noises(i));
    noise_mat(i,:,2) = determinist(memories(1,:),noises(i));
end
%% WARNING: this code runs for up to 40 seconds

% running the network 100 times over the different noise levels, for random and
% deterministic noises

iteration = 100;
for type = [1,2] % type of noise - random and deterministic
    for noise=1:size(noise_mat,1)
        for i=1:iteration      
            [rslt(i,noise,type),steps(i,noise,type)] = hopfield(noise_mat(noise,:,type),W,memories,1000);
        end
        success = round(100*sum(rslt(:,noise,type)==1)/iteration);
        mean_steps = round(mean(steps(:,noise,type)));
        score_board(noise,1,type) = success;
        score_board(noise,2,type) = mean_steps;
    end
end


% calculating statistics for success and average steps until convergance

[h_suc,p_suc] = ttest2(score_board(:,1,1),score_board(:,1,2)); % t-test for success rates
err_suc = [(std(score_board(:,1,1)))/sqrt(length(score_board(:,1,1))),(std(score_board(:,1,2)))/sqrt(length(score_board(:,1,2)))]; % STE for success rates
[h_step,p_step] = ttest2(score_board(:,2,1),score_board(:,2,2)); % t-test for number of steps
err_step = [(std(score_board(:,2,1)))/sqrt(length(score_board(:,2,1))),(std(score_board(:,2,2)))/sqrt(length(score_board(:,2,2)))]; % STE for number of steps


%% Plotting the comparisons between random & deterministic noises

% Creating a figure of the memories for demonstration
noisy_random_for_show = reshape(noise_mat(3,:,1),[10,10]);
noise_place = find(noise_mat(3,:,1)~=memories(1,:));
noisy_random_for_show(noise_place) = 0;

noisy_deter_for_show = reshape(noise_mat(3,:,2),[10,10]);
noise_place = find(noise_mat(3,:,2)~=memories(1,:));
noisy_deter_for_show(noise_place) = 0;

figure('Position', [150 150 1000 500])
subplot(1,2,1)
imagesc(reshape(noise_mat(3,:,1),[10,10]))
title('Random noise of 30%','FontSize', 20)
subtitle('changed tiles appear in grey','FontSize', 15)
colormap gray
hold on
imagesc(noisy_random_for_show)
colormap gray
axis off
subplot(1,2,2)
imagesc(reshape(noise_mat(3,:,2),[10,10]))
title('Deterministic noise of 30%','FontSize', 20)
subtitle('changed tiles appear in grey','FontSize', 20)
colormap gray
axis off
hold on
imagesc(noisy_deter_for_show)
colormap gray
axis off

% success rates over the noise kinds and levels
figure()
bar([score_board(:,1,1),score_board(:,1,2)])
title('Success of the system in Random vs. Deterministic noise')
xlabel('Noise (%)')
ylabel('Success (%)')
xticklabels({'10','20','30','40','50','60','70','80','90'})
legend({'Random noise','Deterministic noise'})

% Average success over the noises & statistical analysis
figure()
bar([mean(score_board(:,1,1)),mean(score_board(:,1,2))])
hold on
errorbar([mean(score_board(:,1,1)),mean(score_board(:,1,2))],err_suc,'LineStyle','None')
title('Average success of the system over the noise levels')
text(2.3,50,['p-value = ',num2str(round(p_suc,2))])
xticklabels({'Random noise','Deterministic noise'})
ylabel('Success (%)')

% Average steps until convergance over the noise kinds and levels
figure()
plot([10:10:90],score_board(:,2,1),[10:10:90],score_board(:,2,2))
title('Steps untill convergance in Random vs. Deterministic noise')
xlabel('Noise (%)')
ylabel('Average steps no.')
yticklabels({'200','300','400','500','600','700','800','900','>1000'})
legend({'Random noise','Deterministic noise'})

% Steps statistical analysis
figure()
bar([mean(score_board(:,2,1)),mean(score_board(:,2,2))])
hold on
errorbar([mean(score_board(:,2,1)),mean(score_board(:,2,2))],err_step,'LineStyle','None')
title('Average steps until convergance of the system over the noise levels')
text(2.3,800,['p-value = ',num2str(round(p_step,2))])
xticklabels({'Random noise','Deterministic noise'})
ylabel('steps until convergance')



%% Functions

% 1. SAMPLE: the function generates a random-with-replacement vector of selected
% values

% vector - the values to be randomly sampled
% number - number of values to be sampled
function rslt = sample(vector,number)
    for i=1:number
        rslt(i) = vector(randi(length(vector)));
    end
end

% 2. RANDOM_MEMORY: the function generates a vector of -1 and 1 values
% number: the length of the vector
function rslt = random_memory(number)
    pool = [-1 1];
    rslt = sample(pool,number);
end

% 3. ZERO_DIAGONAL: the function turns the diagonal values of the matrix into
% zeros

function rslt = zero_diagonal(matrix)
    for i=1:size(matrix,1)
        matrix(i,i) = 0;
    end
    rslt = matrix;
end

% 4. RANDOM_NOISE: the function reverses the sign of x percent of the
% matrix

% memory: the matrix to be reversed
% noise: the percent of cells to be reversed in the matrix
function rslt = random_noise(memory,noise)
  noise_place = randperm(numel(memory),round(numel(memory)*noise));
  noisy_memory = memory;
  noisy_memory(noise_place) = noisy_memory(noise_place)*-1;
  rslt = noisy_memory;
end

% 5. HOPFIELD: the function runs a hopfield-network dynamic and returns the
% memory/error it converged to, and the steps it took to convergance

% noisy memory: the noisy memory presented to the system
% W: weights table
% memories: matrix of original memory embeded in the system
% iterations: number of runs
function [rslt,steps] = hopfield(noisy_memory,W,memories,iterations)
    is_similar = 0;
    output = 0;
    S = noisy_memory;
    t = [1,iterations];
    iter = randi(numel(noisy_memory),t);
    for i=1:length(iter)
        sum_S = W(iter(i),:)*S';
        S(1,iter(i)) = sign(sum_S);
        S(S==0)=1;
        S_log(i,:) = S;
        for line=1:size(memories,1)
            if S == memories(line,:)
                output = line;
                is_similar = 1;
            end
        end
        if i > numel(noisy_memory)*10
            if S == S_log(i-1,:)
                if S == S_log(i-2,:)
                    is_similar = 1;
                end
            end
        end
        if(is_similar == 1)
            break
        end
    end
    steps = i;
    rslt = output;

end

% 6. NETWORK_PERFORMANCE: comprehesive analysis of the system. the function
% runs a hopfield-network dynamics over number of memories and noises and
% returns success rates and steps

% num_memo: a vector contains number of memories to be embeded
% num_noise: a vector of noises percentage
% iterations: number of runs

% score_board: average success rates and steps over the iterations
% all_results: results of every run of the network
function [score_board,all_results] = network_performance(num_memo,num_noise,iteration)
    for memory = 1:length(num_memo)
        for noise = 1:length(num_noise)
            for i=1:iteration
                memories = [];
                for num_memory = 1:num_memo(memory)
                    memories(num_memory,:) = random_memory(100);
                    W(:,:,num_memory) = 1/numel(memories(num_memory,:)).*(memories(num_memory,:)'*memories(num_memory,:));
                end

                W = sum(W,3);
                W = zero_diagonal(W);
             
                noisy_memory_1 = random_noise(memories(1,:),num_noise(noise));
                [rslt,step] = hopfield(noisy_memory_1,W,memories,1000);
                results(i) = rslt;
                steps(i) = step;
                all_results(i,noise,memory) = step;
            end
            success = round(100*sum(results==1)/iteration);
            mean_steps = round(mean(steps));
            score_board(memory,noise,1) = success;
            score_board(memory,noise,2) = mean_steps;
        end
    end
end

% 7. DETERMINIST: the function creates a deterministic noise to be put into a memory

% memory: original memory matrix
% percentage of deterministic noise (0.1-0.9)
function rslt = determinist(memory,noise)
    memo_10 = ones(10);
    memo_10(5:6,3:7) = -1;
    memories(1,:) = reshape(memo_10,[1,100]);
    
    memo_20 = ones(10);
    memo_20(4:7,3:7) = -1;
    memories(2,:) = reshape(memo_20,[1,100]);
    
    memo_30 = ones(10);
    memo_30(3:7,3:8) = -1;
    memories(3,:) = reshape(memo_30,[1,100]);
    
    memo_40 = ones(10);
    memo_40(4:8,2:9) = -1;
    memories(4,:) = reshape(memo_40,[1,100]);
    
    memo_50 = ones(10);
    memo_50(4:8,1:10) = -1;
    memories(5,:) = reshape(memo_50,[1,100]);
    
    memo_60 = ones(10);
    memo_60(3:8,1:10) = -1;
    memories(6,:) = reshape(memo_60,[1,100]);
    
    memo_70 = ones(10);
    memo_70(3:9,1:10) = -1;
    memories(7,:) = reshape(memo_70,[1,100]);
    
    memo_80 = ones(10);
    memo_80(2:9,1:10) = -1;
    memories(8,:) = reshape(memo_80,[1,100]);
    
    memo_90 = ones(10);
    memo_90(1:9,1:10) = -1;
    memories(9,:) = reshape(memo_90,[1,100]);
    
    memo_100 = ones(10).*-1;
    memories(10,:) = reshape(memo_100,[1,100]);
    noise = round(noise.*10);
    rslt = memory.*memories(noise,:);
end 