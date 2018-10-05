
#data is a data table, attribute is the attribute you want to know
knn <- function (data, attribute, unknown, k){
  distances <- c()
  indexes <- c()
  attributes <- c()
  j=0
  for(i in 1:length(data)){
    difference <- abs(data[i]-unknown)
    distances <- c(distances, difference)
  }
  while(j < k){
    for(i in 1:length(data)){
      if(data[i] == max(data)){
        indexes <- c(indexes, i)
        data[i] <- NULL
      }
    }
    j <- j + 1
  }
  for(i in 1:length(indexes)){
    attributes <- c(attributes, data$attribute[indexes[i]])
  }
  return (which.max(attributes))
  
}
