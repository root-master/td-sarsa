function g = neighbor_state_no_puddle(s0,xVector,yVector,radius)

x0 = s0(1);
y0 = s0(2);
x_close_vec = xVector( abs(xVector - x0) < radius );
y_close_vec = yVector( abs(yVector - y0) < radius );

gi = randi(length(x_close_vec));
gj = randi(length(y_close_vec));
g =  [x_close_vec(gi),y_close_vec(gj)];

