 function [W_ih_big,b_ih_big,W_ho_big,b_ho_big] = Update_kwtaNN_s_g(st,act,h_vec,h_id_vec,gid_vec,delta,alpha, W_ih_big,b_ih_big, W_ho_big,b_ho_big)

gate_weights = 0.001 * ones(1,length(gid_vec));
gate_weights(1) = 1;
gate_weights = gate_weights ./ sum(gate_weights);
  

for i=1:length(gid_vec)
    gid = gid_vec(i);
    Wih = W_ih_big(:,:,gid);
    biasih = b_ih_big(:,:,gid);
    Who = W_ho_big(:,:,gid);
    biasho = b_ho_big(:,:,gid);
    
    alpha_eff = alpha * gate_weights(i);

    h = h_vec(i,:);
    h_id = h_id_vec(i,:);
    deltaj = zeros(1,length(h));
    deltaj(h_id) = (- delta * Who(h_id,act))' .* (1-h(h_id)) .* h((h_id)); 
    Who(h_id,act) = Who(h_id,act) + alpha_eff * delta * h(h_id)';
    Wih(:,h_id) = Wih(:,h_id) - alpha_eff * st' * deltaj(h_id);
    biasih(h_id) = biasih(h_id) - alpha_eff * deltaj(h_id);

    W_ih_big(:,:,gid) = Wih ;
    b_ih_big(:,:,gid) = biasih;
    W_ho_big(:,:,gid) = Who;
    b_ho_big(:,:,gid) = biasho;

end
