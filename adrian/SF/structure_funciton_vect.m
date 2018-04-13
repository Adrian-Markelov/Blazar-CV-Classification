function [ SF ] = structure_function_vect( fname, objno ) 

%Read in the file provided by fname
  
fullfile = readtable(fname);
  
  
%Find the unique object IDs
  
uniIDs = unique(fullfile.ID, 'stable');
  
  
% Pull out the information associated with object objno
  
mjds = fullfile.MJD(fullfile.ID == uniIDs(objno));
mags = fullfile.Mag(fullfile.ID == uniIDs(objno));
  
% Find all pairs of time and magnitude differences
  
nr = length(mags);
  
allpairs = zeros(2,nr*(nr-1)/2);
  
pos = 1;
for i = 1:(nr-1)
    for j = (i+1):nr
        allpairs(pos,1) = mjds(i)-mjds(j);
        allpairs(pos,2) = mags(i)-mags(j);
        pos = pos + 1;
        %fprintf('index: %d, %d\n', i, j);
    end
    fprintf('index: %d\n', i);
end
  
  
% Find the absolute time difference
  
timediff = abs(allpairs(:,1));
  
  
% Consider magnitude differences on the log scale
  
magdiff = log10(abs(allpairs(:,2)));
  
  
% Remove any infinite values

timediff = timediff(isfinite(magdiff));
magdiff = magdiff(isfinite(magdiff));
  
timediff = timediff(isfinite(timediff));
magdiff = magdiff(isfinite(timediff));

SF = [timediff,magdiff];

end

