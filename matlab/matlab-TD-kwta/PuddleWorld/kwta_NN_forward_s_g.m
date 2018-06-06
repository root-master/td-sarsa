function [o,h_vec,h_id_vec,gid_vec]  = kwta_NN_forward_s_g(st,g,nMeshx,nMeshy, W_ih_big,b_ih_big, W_ho_big,b_ho_big) 

shunt = 1;

gid_vec = neighbors_of_goal(g,nMeshx,nMeshy);
gate_weights = 0.01 * ones(1,length(gid_vec));
gate_weights(1) = 1;
gate_weights = gate_weights ./ sum(gate_weights);

o = zeros(1,4);
k_rate = 0.1;

h_vec = [];
h_id_vec = [];
for i=1:length( gid_vec )
     gid = gid_vec(i);
 
     Wih = W_ih_big(:,:,gid);
     biasih = b_ih_big(:,:,gid);
     Who = W_ho_big(:,:,gid);
     biasho = b_ho_big(:,:,gid);

     nCellHidden = length(Wih);


    k = round(k_rate* nCellHidden); % number of winners

    net = st * Wih + biasih;

    [netSorted,idsort] = sort(net,'descend');
    q = 0.25; % constant 0 < q < 1 determines where exactly
              % to place the inhibition between the k and k + 1th units

    biaskwta = netSorted(k+1) + q * ( netSorted(k) - netSorted(k+1) );
    h_id = idsort(1:k);

    eta = net - biaskwta - shunt; % shunt is a positive number which is the shift to left in activation-eta

    % hidden activation
    h = zeros(size(eta));
    h(h_id) = 1./(1 + exp(-eta(h_id)) );
    %h = 1./(1 + exp(-eta) );
    
    h_vec = [h_vec;h];
    h_id_vec = [h_id_vec;h_id];

    Q = h * Who + biasho; % Output
    
        o = o + Q* gate_weights(i);
    end
end
