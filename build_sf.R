


BuildSF = function(fname, objno)
{
  
  # Read in the file provided by fname
  
  fullfile = read.table(fname, header=T, sep=",")
  
  
  # Find the unique object IDs
  
  uniIDs = unique(fullfile$ID)
  
  
  # Pull out the information associated with object objno
  
  mjds = fullfile$MJD[fullfile$ID == uniIDs[objno]]
  mags = fullfile$Mag[fullfile$ID == uniIDs[objno]]
  
  
  # Find all pairs of time and magnitude differences
  
  nr = length(mags)
  
  allpairs = matrix(0,ncol=2,nrow=nr*(nr-1)/2)
  
  pos = 1
  for(i in 1:(nr-1))
  {
    for(j in (i+1):nr)
    {
      allpairs[pos,1] = mjds[i]-mjds[j]
      allpairs[pos,2] = mags[i]-mags[j]
      pos = pos + 1
    }
  }
  
  
  # Find the absolute time difference
  
  timediff = abs(allpairs[,1])
  
  
  # Consider magnitude differences on the log scale
  
  magdiff = log10(abs(allpairs[,2]))
  
  
  # Remove any infinite values
  
  timediff = timediff[is.finite(magdiff)]
  magdiff = magdiff[is.finite(magdiff)]
  
  timediff = timediff[is.finite(timediff)]
  magdiff = magdiff[is.finite(timediff)]
  
  return(cbind(timediff,magdiff))
}
