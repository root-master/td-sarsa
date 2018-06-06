function done = success(state,goal)
 
done = all( abs(state-goal)<0.01 );

