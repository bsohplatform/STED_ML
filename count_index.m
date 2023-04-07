A=readmatrix('BAS_DB.csv');

i = 0;
for j = 1:length(A)-1
    if A(j,16) == A(j+1,16)
        A(j,1) = i;
    else
        A(j,1) = i;
        i = i+1;
    end
end