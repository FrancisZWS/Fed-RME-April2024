% Specify parameters
numTrain = 1000;
numTest = 100;
matrixSize = [32, 32, 1];

% Preallocate cell array to store data matrices
Traindata = cell(1, numTrain);
Testdata = cell(1, numTest);

% Generate and store data matrices
for i = 1:numTrain
    Traindata{i} = randn(matrixSize);
end

for i = 1:numTest
    Testdata{i} = randn(matrixSize);
end

% Save the list of data matrices in a MATLAB file
save('map_list.mat', 'Traindata','Testdata');
