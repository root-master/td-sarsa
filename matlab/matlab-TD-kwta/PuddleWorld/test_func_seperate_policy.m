function [successful_key_door_episodes, successful_key_episodes, successful_easy_episodes, successful_basic_key_door, scores_vec,scores_vec_basic,score_key,score_easy, total_episodes] = test_func_seperate_policy(W_ih_big, b_ih_big, W_ho_big, b_ho_big)

nMeshx = 10; nMeshy = 10;

successful_key_door_episodes = [];
successful_key_episodes = [];
scores_vec_basic = [];
successful_basic_key_door = [];
scores_vec = [];
score_key = [];
score_easy = [];
% Input of function approximator
xgridInput = 1.0 / nMeshx;
ygridInput = 1.0 / nMeshy;
xInputInterval = 0 : xgridInput : 1.0;
yInputInterval = 0 : ygridInput : 1.0;
xVector = xInputInterval;
yVector = yInputInterval;
xgrid = 1 / (nMeshx);
ygrid = 1 / (nMeshy);
% parameter of Gaussian Distribution
sigmax = 1.0 / nMeshx; 
sigmay = 1.0 / nMeshy;

ep_id = 1;
max_iter = 500;
total_episodes = 0;


%fprintf('door = %g %g \n',door);


for x=xInputInterval,
    for y=yInputInterval,
        keyinPuddle = true;
        while keyinPuddle
            key = initializeState(xInputInterval,yInputInterval);
            [keyinPuddle,~] = CreatePuddle(key);
        end
        %fprintf('key = %g %g \n',key);

        doorinPuddle = true;
        while doorinPuddle
            door = initializeState(xInputInterval,yInputInterval);
            [doorinPuddle,~] = CreatePuddle(door);
        end

        t = 1;
        scores = 0;
        agenthaskey = false;
        g = key;
        s=[x,y];
        [agentinPuddle,~] = CreatePuddle(s);
        if agentinPuddle
            continue
        end
        %fprintf('s0 = %g %g and goal = %g %g \n',s,g);
        while(t<=max_iter)
            %fprintf('s = %g %g \n',s);
            if success(s,key) && ~agenthaskey
                scores = scores + 10;
                g = door;
                %fprintf('goal changed to %g %g \n',g);
                successful_key_episodes = [successful_key_episodes, ep_id];
                score_key = [score_key,scores];
                agenthaskey = true;
            end
            
            if agenthaskey
               if success(s,door)
                   agentReached2Door = true;
                   scores = scores + 100;
                   successful_key_door_episodes = [successful_key_door_episodes, ep_id];
                   scores_vec = [scores_vec, scores];
                   %fprintf('goal acheived \n');
                   break
               end
            end
            
             [~,gidx] = min(dist(g(1),xInputInterval));
             [~,gidy] = min(dist(g(2),yInputInterval));
             gid = sub2ind([length(xVector),length(yVector)],gidx,gidy);
             Wih = W_ih_big(:,:,gid);
             biasih = b_ih_big(:,:,gid);
             Who = W_ho_big(:,:,gid);
             biasho = b_ho_big(:,:,gid);
            
             sx = xInputInterval == s(1);
             sy = yInputInterval == s(2);

            % Using st as distributed input for function approximator
             st = [sx,sy];                
             Q = kwta_NN_forward_new(st, Wih, biasih, Who, biasho);
             [~,a] = max(Q);
             sp1 = UPDATE_STATE(s,a,xgrid,xInputInterval,ygrid,yInputInterval);
             [agent_in_puddle,dist_2_edge] = CreatePuddle(sp1);
             rew = 0;
             if agent_in_puddle
                 rew = -400*dist_2_edge;
             end
             scores = scores + rew;
             
             s = sp1;
             t = t+1;
            if t == max_iter
                scores_vec = [scores_vec, scores];
                if ~agenthaskey
                    score_key = [score_key,scores];
                end
                    
            end
        end
                
        ep_id = ep_id + 1;
        total_episodes = total_episodes + 1;
    end
end




radius = 0.11;
successful_easy_episodes = [];
ep_id = 1;
for x=xInputInterval,
    for y=yInputInterval,
        t = 1;
        scores = 0;
        s0=[x,y];
        [agentinPuddle,~] = CreatePuddle(s0);
        if agentinPuddle
            continue
        end
        s = s0;
        g = neighbor_state(s0,xVector,yVector,radius);
        while(t<=max_iter)
            %fprintf('s = %g %g \n',s);
            if success(s,g)
                successful_easy_episodes = [successful_easy_episodes, ep_id];
                score_easy = [score_easy,scores];
                break
            end
             [~,gidx] = min(dist(g(1),xInputInterval));
             [~,gidy] = min(dist(g(2),yInputInterval));
             gid = sub2ind([length(xVector),length(yVector)],gidx,gidy);
             Wih = W_ih_big(:,:,gid);
             biasih = b_ih_big(:,:,gid);
             Who = W_ho_big(:,:,gid);
             biasho = b_ho_big(:,:,gid);

             sx = sigmax * sqrt(2*pi) * normpdf(xInputInterval,s(1),sigmax);
             sy = sigmay * sqrt(2*pi) * normpdf(yInputInterval,s(2),sigmay);

            % Using st as distributed input for function approximator
             st = [sx,sy];                
             Q = kwta_NN_forward_new(st, Wih, biasih, Who, biasho);
             [~,a] = max(Q);
             sp1 = UPDATE_STATE(s,a,xgrid,xInputInterval,ygrid,yInputInterval); 
             rew = 0;
             [agent_in_puddle,dist_2_edge] = CreatePuddle(sp1);
             if agent_in_puddle
                 rew = -400*dist_2_edge;
             end
             scores = scores + rew;

             s = sp1;
             t = t+1; 
             if t == max_iter
                score_easy = [score_easy, scores];
            end

        end                
        ep_id = ep_id + 1;
    end
end



% keyinPuddle = true;
% while keyinPuddle
%     key = initializeState(xInputInterval,yInputInterval);
%     [keyinPuddle,~] = CreatePuddle(key);
% end
% %fprintf('key = %g %g \n',key);
% 
% doorinPuddle = true;
% while doorinPuddle
%     door = initializeState(xInputInterval,yInputInterval);
%     [doorinPuddle,~] = CreatePuddle(door);
% end

key = [0.0,0.0];
door = [1,1];
ep_id = 1;

for x=xInputInterval,
    for y=yInputInterval,
        t = 1;
        scores = 0;
        agenthaskey = false;
        g = key;
        s=[x,y];
        [agentinPuddle,~] = CreatePuddle(s);
        if agentinPuddle
            continue
        end
        %fprintf('s0 = %g %g and goal = %g %g \n',s,g);
        while(t<=max_iter)
            %fprintf('s = %g %g \n',s);
            if success(s,key) && ~agenthaskey
                scores = scores + 10;
                g = door;
                %fprintf('goal changed to %g %g \n',g);
                agenthaskey = true;
            end
            
            if agenthaskey
               if success(s,door)
                   agentReached2Door = true;
                   scores = scores + 100;
                   successful_basic_key_door = [successful_basic_key_door, ep_id];
                   scores_vec_basic = [scores_vec_basic, scores];
                   %fprintf('goal acheived \n');
                   break
               end
            end
            
             [~,gidx] = min(dist(g(1),xInputInterval));
             [~,gidy] = min(dist(g(2),yInputInterval));
             gid = sub2ind([length(xVector),length(yVector)],gidx,gidy);
             Wih = W_ih_big(:,:,gid);
             biasih = b_ih_big(:,:,gid);
             Who = W_ho_big(:,:,gid);
             biasho = b_ho_big(:,:,gid);
            
             sx = xInputInterval == s(1);
             sy = yInputInterval == s(2);

            % Using st as distributed input for function approximator
             st = [sx,sy];                
             Q = kwta_NN_forward_new(st, Wih, biasih, Who, biasho);
             [~,a] = max(Q);
             sp1 = UPDATE_STATE(s,a,xgrid,xInputInterval,ygrid,yInputInterval);
             [agent_in_puddle,dist_2_edge] = CreatePuddle(sp1);
             rew = 0;
             if agent_in_puddle
                 rew = -400*dist_2_edge;
             end
             scores = scores + rew;
             
             s = sp1;
             t = t+1;
            if t == max_iter
                scores_vec_basic = [scores_vec_basic, scores];
            end
        end
                
        ep_id = ep_id + 1;
    end
end





