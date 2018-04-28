distXY = function(X1,x2){
  # X1 is a n*m matrix of training data
  # x2 is a new row with m columns
  # D is a vector of length n, where D[i] is the distance between x2 and X1[i,]
  D = rep(0,nrow(X1))
  for(i in 1:nrow(X1)){
    D[i] = sum( (X1[i,]-x2)^2 )
  }
  return(D)}



knn_predict = function(k, train_X, train_Y, test_X){
  # k the number of neighbors to select
  # train_X the n*m training covariate data matrix
  # train_Y outcomes for the n training observations
  # test_X a 1*m row for the new observation's covariates
  
  scale_function = 
  train_X_scaled = 
  test_X_scaled = 
  distances = 
  distance_rank = 
  nearest_neighbors = 
  neighbor_outcomes = 
  prediction = 
  
  return(prediction)
}


knn_by_Treatment = function(k, train_Treatment, train_X, train_Y, test_X){
  # k the number of neighbors to select
  # train_Treatment the treatment assigned to each training row
  # train_X the n*m training covariate data matrix
  # train_Y outcomes for the n training observations
  # test_X a 1*m row for the new observation's covariates
  
  # find out how many rows are in our test set
  n_test = nrow(test_X)
  
  # make a list of the treatments to consider
  treatments = sort(unique(train_Treatment))
  
  # make a matrix to store predicted outcomes 
  # (1 row per observation, 1 column per treatment)
  predicted_outcomes = matrix(0, nrow=n_test, ncol=length(treatments) )
  # rename the columns by treatment
  colnames(predicted_outcomes) = paste("T_", treatments, sep="")
  
  # now, loop over every observation and every  treatment to make predictions
  for(t in treatments){
    # extract train data for patients who had treatment t
    t_rows = which(train_Treatment==t)
    T_X = train_X[t_rows,] 
    T_Y = train_Y[t_rows]
    
    # make a prediction for each observation:
    for(i in 1:n_test){
    predicted_outcomes[i,t] = knn_predict(k, T_X,T_Y, test_X[i,])
  }}
  
  return(predicted_outcomes)
}



knn_LOOCV = function(k_range, train_Treatment, train_X, train_Y){
  # k_range the number of neighbors to select
  # train_Treatment the treatment assigned to each training row
  # train_X the n*m training covariate data matrix
  # train_Y outcomes for the n training observations
  
  # make a list of the treatments to consider
  treatments = sort(unique(train_Treatment))
  
  # make a matrix to store predicted outcomes 
  # (1 row per observation, 1 column per k value)
  # NB: we are only comparing predictions to data that we actually observe
  # So we only predict one outcome pre patient (for the treatment they received)

  predicted_outcomes = matrix(0, nrow=nrow(train_X), ncol=length(k_range) )
  prediction_errors = matrix(0, nrow=nrow(train_X), ncol=length(k_range) )
  # rename the columns by treatment
  colnames(predicted_outcomes) = paste("k=", k_range, sep="")
  
  # now, loop over every observation and every  treatment to make predictions
  for(t in treatments){
    # extract train data for patients who had treatment t
    t_rows = which(train_Treatment==t)
    T_X = train_X[t_rows,] 
    T_Y = train_Y[t_rows]
    train_X_scaled = predict(preProcess(T_X), T_X)
    DM = as.matrix(dist(train_X_scaled, method = "euclidean"))
    
    # make predictions for each observation
      for(i in 1:nrow(T_X) ){
        # list outcomes in order of neighbor closeness
        ordered_outcomes = T_Y[order(DM[i,])]
        # remove individual's outcome
        ordered_outcomes = ordered_outcomes[-1]
        K_means = cumsum(ordered_outcomes)/(1:length(ordered_outcomes))
        predicted_outcomes[t_rows[i], ] = K_means[k_range]
        prediction_errors[t_rows[i], ] = (train_Y[i] - predicted_outcomes[t_rows[i], ])
      } }
  # calculate total error for each column (k)
  SSE = colSums(prediction_errors^2)
  
  return(SSE)
}
