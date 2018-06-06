function [id_vec] = neighbors_of_goal(g,nMeshx,nMeshy)
neighbors = [];
id_vec = [];

dx = 1.0 / nMeshx;
dy = 1.0 / nMeshy;
xVector = 0 : dx : 1.0;
yVector = 0 : dy : 1.0;

if success([0,0],g)
    neighbors = [neighbors;g];
    neighbors = [neighbors;[0,dy]];
    neighbors = [neighbors;[dx,0]];
end

if success([1,0],g)
    neighbors = [neighbors;g];
    neighbors = [neighbors;[1-dx,0]];
    neighbors = [neighbors;[1,dy]];
end

if success([1,1],g)
    neighbors = [neighbors;g];
    neighbors = [neighbors;[1-dx,1]];
    neighbors = [neighbors;[1,1-dy]];
end

if success([0,1],g)
    neighbors = [neighbors;g];
    neighbors = [neighbors;[dx,1]];
    neighbors = [neighbors;[0,1-dy]];
end


if ~( success([0,0],g) || success([0,1],g) || success([1,1],g) || success([1,0],g) )
    x = g(1);
    y = g(2);
    neighbors = [neighbors;g];
    neighbors = [neighbors;[x+dx,y]];
    neighbors = [neighbors;[x,y+dy]];
    neighbors = [neighbors;[x-dx,y]];
    neighbors = [neighbors;[x,y-dy]];
end

for i=1:length(neighbors)
    g_tmp = neighbors(i,:);
    [~,gidx] = min(dist(g_tmp(1),xVector));
    [~,gidy] = min(dist(g_tmp(2),yVector));
    gid = sub2ind([length(xVector),length(yVector)],gidx,gidy);
    id_vec = [id_vec,gid];
end



