clc, close all, clear all;
withBias = 1;

nMeshx = 5; nMeshy = 5;
nTilex = 1; nTiley = 1;

functionApproximator = 'kwtaNN';
shunt = 1.0;

% control task could be 'grid_world' or 'puddle_world'
task = 'puddle_world';
% function approximator can be either 'kwtaNN' or 'regularBPNN'


% goal in continouos state
% g is goal and it is dynamic this time

% Input of function approximator
xgridInput = 1.0 / nMeshx;
ygridInput = 1.0 / nMeshy;
xInputInterval = 0 : xgridInput : 1.0;
yInputInterval = 0 : ygridInput : 1.0;

% the number of states -- This is the gross mesh states ; the 1st tiling 
nStates = ( length(xInputInterval) * length(yInputInterval) ); 

% on each grid we can choose from among this many actions 
% [ up , down, right, left ]
% (except on edges where this action is reduced): 
nActions = 4; 

%% kwta and regular BP Neural Network
% Weights from input (x,y,x_goal,y_goal) to hidden layer
InputSize = length(xInputInterval) + length(yInputInterval );
goal_size = nStates;
nCellHidden = 2*round( nStates );
mu = 0.1;

W_ih_big = mu * (rand(InputSize,nCellHidden,goal_size) - 0.5);
b_ih_big = mu * ( rand(1,nCellHidden,goal_size) - 0.5 );

W_ho_big = mu * (rand(nCellHidden,nActions,goal_size) - 0.5);
% b_ho_big = mu * ( rand(1,nActions,goal_size) - 0.5 );
b_ho_big = zeros(1,nActions,goal_size);
% Wih = mu * (rand(InputSize,nCellHidden) - 0.5);
% biasih = mu * ( rand(1,nCellHidden) - 0.5 );
% % Weights from hidden layer to output
% Who = mu * (rand(nCellHidden,nActions) - 0.5);
% biasho = mu * ( rand(1,nActions) - 0.5 );

% on each grid we can choose from among this many actions 
% [ up , down, right, left ]
% (except on edges where this action is reduced): 
nActions = 4; 

gamma = 0.99;    % discounted task 
epsilon_max = 0.1;
epsilon_min = 0.001;
epsilon = epsilon_max;  % epsilon greedy parameter


alpha_min = 0.001;
alpha_max = 0.005;
alpha = alpha_max;
% Max number of iteration in ach episde to break the loop if AGENT
% can't reach the GOAL 

             
%% Input of function approximator
xgridInput = 1.0 / nMeshx;
ygridInput = 1.0 / nMeshy;
xInputInterval = 0 : xgridInput : 1.0;
yInputInterval = 0 : ygridInput : 1.0;
% smoother state space with tiling
xgrid = 1 / (nMeshx * nTilex);
ygrid = 1 / (nMeshy * nTiley);
% parameter of Gaussian Distribution
sigmax = 1.0 / nMeshx; 
sigmay = 1.0 / nMeshy;

xVector = 0:xgrid:1;
yVector = 0:ygrid:1;

nStates = length(xInputInterval) * length(yInputInterval);

%% Different Max number of episodes
maxNumEpisodes = 2000000;
close_to_convergence = false;
nGoodEpisodes = 0; % a variable for checking the convergence
convergence = false;
agentReached2Goal = false;
agentBumped2wall = false;
%% Episode Loops
ei = 0;
delta_sum = [];
total_num_steps = 0;
radius = 0.11;
results_basic_score = [];
results_basic_success = [];
results_hard_score = [];
results_medium_score = [];
results_hard_success = [];
results_medium_success = [];
results_easy_success = [];
results_easy_score = [];
results_hard_score_best = [];
results_medium_score_best = [];
results_basic_score_best = [];

while (ei < maxNumEpisodes && ~convergence ), % ei<maxNumEpisodes && % ei is counter for episodes
    
    if mod(ei,1000)==0
        [successful_key_door_episodes, successful_key_episodes, successful_easy_episodes, successful_basic_key_door, scores_vec,scores_vec_basic,score_key,score_easy, total_episodes] = test_func_s_g_no_puddle(W_ih_big, b_ih_big, W_ho_big, b_ho_big)
        results_basic_score = [results_basic_score,mean(scores_vec_basic)];
        results_basic_success = [results_basic_success,length(successful_basic_key_door)/total_episodes];
        results_hard_score = [results_hard_score, mean(scores_vec)];
        results_hard_success = [results_hard_success,length(successful_key_door_episodes)/total_episodes];
        results_medium_success = [results_medium_success, length(successful_key_episodes)/total_episodes];
        results_medium_score = [results_medium_score,mean(score_key)];
        results_easy_success = [results_easy_success, length(successful_easy_episodes)/total_episodes];
        results_hard_score_best = [results_hard_score_best,max(scores_vec)];
        results_medium_score_best = [results_medium_score_best,max(score_key)];
        results_basic_score_best = [results_basic_score_best,max(scores_vec_basic)];
        results_easy_score = [results_easy_success,mean(score_easy)];
        
        fprintf('average success hard     : %.4f \n',length(successful_key_door_episodes)/total_episodes);
        fprintf('average success basic    : %.4f \n',length(successful_basic_key_door)/total_episodes);
        fprintf('average success medium   : %.4f \n',length(successful_key_episodes)/total_episodes);
        fprintf('average success easy     : %.4f \n',length(successful_easy_episodes)/total_episodes);
        fprintf('average score   hard     : %.4f \n',mean(scores_vec));
        fprintf('average score   basic    : %.4f \n',mean(scores_vec_basic));
        fprintf('average score   medium   : %.4f \n',mean(score_key));
        fprintf('average score   easy     : %.4f \n',mean(score_easy));
        fprintf('max score       hard     : %.4f \n',max(scores_vec));
        fprintf('max score       basic    : %.4f \n',max(scores_vec_basic));
        fprintf('max score       medium   : %.4f \n',max(score_key));

        pause(5)
    end
      
     s0 = initializeState(xVector,yVector);
     s = s0;
     g = initializeState(xVector,yVector);
     %g = neighbor_state_no_puddle(s0,xVector,yVector,radius);
     
     % Gaussian Distribution on continuous state
     sx = sigmax * sqrt(2*pi) * normpdf(xInputInterval,s(1),sigmax);
     sy = sigmay * sqrt(2*pi) * normpdf(yInputInterval,s(2),sigmay);
     % Using st as distributed input for function approximator
    
     st = [sx,sy];     
          
     % initializing time
     ts = 1;
     %[Q,h,id] = kwta_NN_forward_new(st,Wih,biasih,Who,biasho);
     [Q,h_vec,h_id_vec,gid_vec] = kwta_NN_forward_s_g(st,g,nMeshx,nMeshy,W_ih_big,b_ih_big, W_ho_big,b_ho_big);
     
     act = e_greedy_policy(Q,nActions,epsilon);
     ei = ei + 1;
     deltaForStepsOfEpisode = [];
     maxIteratonEpisode = 4*nMeshx;
     agentReached2Goal = success(s,g);
    %% Episode While Loop
    while( ~agentReached2Goal && ts < maxIteratonEpisode ),
        % update state to state+1
        sp1 = UPDATE_STATE(s,act,xgrid,xVector,ygrid,yVector);
        xp1 = sp1(1); yp1 = sp1(2);
        % PDF of stp1
        sxp1 = sigmax * sqrt(2*pi) * normpdf(xInputInterval,xp1,sigmax);
        syp1 = sigmay * sqrt(2*pi) * normpdf(yInputInterval,yp1,sigmay);
        
        stp1=[sxp1,syp1];
        if ( success(sp1,g) ),
            agentReached2Goal = true;
            agentBumped2wall = false;
            rew = 1;
        elseif ( success(s,sp1) ),
            agentBumped2wall = true;
            agentReached2Goal = false;
            rew = -2;
        else
            agentBumped2wall = false;
            agentReached2Goal = false;
           rew = -1;
        end
        
        % reward/punishment from Environment
        %rew = ENV_REWARD(sp1,agentReached2Goal,agentBumped2wall);
        % [Qp1,hp1,idp1] = kwta_NN_forward_new(stp1,Wih,biasih,Who,biasho);
        [Qp1,h_vec_p1,h_id_vec_p1,gid_vec] = kwta_NN_forward_s_g(stp1,g,nMeshx,nMeshy,W_ih_big,b_ih_big, W_ho_big,b_ho_big);        
        % make the greedy action selection in st+1: 
        actp1 = e_greedy_policy(Qp1,nActions,epsilon);
    
        if( ~agentReached2Goal ) 
            % stp1 is not the terminal state
            delta = rew + gamma * Qp1(actp1) - Q(act);
            deltaForStepsOfEpisode = [deltaForStepsOfEpisode,delta];           
            % Update Neural Net
            [W_ih_big,b_ih_big,W_ho_big,b_ho_big] = Update_kwtaNN_s_g(st,act,h_vec,h_id_vec,gid_vec,delta,alpha, W_ih_big,b_ih_big, W_ho_big,b_ho_big);        
        else
            delta = rew - Q(act);
            deltaForStepsOfEpisode = [deltaForStepsOfEpisode,delta];
            [W_ih_big,b_ih_big,W_ho_big,b_ho_big] = Update_kwtaNN_s_g(st,act,h_vec,h_id_vec,gid_vec,delta,alpha, W_ih_big,b_ih_big, W_ho_big,b_ho_big);        
            % stp1 is the terminal state ... no Q(s';a') term in the sarsa update
            fprintf('Success: episode = %d, s0 = (%g , %g), goal: (%g , %g), step = %d, mean(delta) = %f \n',ei,s0,g,ts,mean(deltaForStepsOfEpisode));
            break;
        end
        % update (st,at) pair:
        st = stp1;  s = sp1; act = actp1; h_id_vec = h_id_vec_p1; h_vec = h_vec_p1; 
        Q = Qp1; ts = ts + 1;
    end % while loop
    %epsilon = epsilon_max/ei;
    total_num_steps = total_num_steps + ts;
    meanDeltaForEpisode(ei) = mean(deltaForStepsOfEpisode);
    delta_sum(ei) = sum(deltaForStepsOfEpisode);
    varianceDeltaForEpisode(ei) =var(deltaForStepsOfEpisode);
    stdDeltaForEpisode(ei) = std(deltaForStepsOfEpisode);

    %% Exploration vs. Exploitation    

    if length(successful_key_door_episodes)/total_episodes > 0.98
       close_to_convergence =  close_to_convergence + 1;
    else
        close_to_convergence = 0;
    end
    
    if close_to_convergence > 10
        convergence = true;
    end
       
    if mod(ei,10000) == 0
        file_name = 'all_variables_April_16th.mat';
        save(file_name) 
    end
end  % end episode loop