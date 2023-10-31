% 
% (c) 2023 Naoki Masuyama
% 
% CIM-based ART with Edge and Ages (CAEA) Classifier Clustering (CAEAC-C) is proposed in:
% 
% N. Masuyama, Y. Nojima, F. Dawood, and Z. Liu, "Class-wise classifier design capable of continual learning using adaptive resonance theory-based topological clustering," 
% Applied Sciences, 2023.
% 
% Run "mainCAMD.m"
% 
% Please contact masuyama@omu.ac.jp if you have any problems.
% 


% clc
clear

% load data
% datalist = {'Iris', 'OptDigits'};
tmpD = load('dataset/OptDigits');

DATA = tmpD.data;
LABEL = tmpD.target;


% Randamize data 
rng(1)
ran = randperm(size(DATA,1));
DATA = DATA(ran,:);
LABEL = LABEL(ran,:);


classNums = unique(LABEL);

% Parameters of CAEAC =====================================================
CAEACnet = cell(1, max(classNums));
ATTnet = cell(1, max(classNums));
for k = 1:max(classNums)
    CAEACnet{k}.numNodes    = 0;   % the number of nodes
    CAEACnet{k}.weight      = [];  % node position
    CAEACnet{k}.CountNode = [];    % winner counter for each node
    CAEACnet{k}.adaptiveSig = [];  % kernel bandwidth for CIM in each node
    CAEACnet{k}.edge = zeros(2,2); % Initial connections (edges) matrix
    CAEACnet{k}.LabelCluster = [];
    CAEACnet{k}.edgeAge = zeros(2,2);  % Age of edge
    CAEACnet{k}.CountEdge = [];
    CAEACnet{k}.CIMthreshold = [];
    CAEACnet{k}.CountLabel = [];
    
    CAEACnet{k}.Lambda = 90;       % an interval for calculating a kernel bandwidth for CIM
    CAEACnet{k}.edgeAgeMax = 16;   % Maximum node age
    
    % Parameters for attritute clustering
    ATTnet{k}.numNodes    = 0;   % the number of nodes
    ATTnet{k}.weight      = [];  % node position
    ATTnet{k}.CountNode = [];    % winner counter for each node
    ATTnet{k}.adaptiveSig = [];  % kernel bandwidth for CIM in each node
    ATTnet{k}.Lambda = CAEACnet{k}.Lambda;   % an interval for calculating a kernel bandwidth for CIM
end
% =========================================================================


time_caeac = 0;

% Training
tic
parfor k = 1:length(classNums)
% for k = 1:length(classNums)
    % Divide data by class labels
    trainData_div = DATA(LABEL == classNums(k),:);
    trainLabels_div = LABEL(LABEL == classNums(k),:);

    % Training CAEA for each class
    CAEACnet{k} = CAEA_Clustering_Train(trainData_div, trainLabels_div, 1, CAEACnet{k}, ATTnet{k});

end
time_caeac = time_caeac + toc;


% Combining Centroids
centroids=[];
label_centroids=[];
for k = 1:length(classNums)
    centroids = [centroids; CAEACnet{k}.weight];
    tmplabel = zeros(size(CAEACnet{k}.weight,1),1) + k;
    label_centroids = [label_centroids; tmplabel];
end


% Test
mdl_CAEAC = fitcknn(centroids, label_centroids, 'NumNeighbors', 1);
estimated_labels = predict(mdl_CAEAC, DATA);


% Evaluation
output = CAEAC_Test(LABEL, label_centroids, estimated_labels);


% Results
disp('Results (CAEAC-C):');
disp(['ACC: ', num2str(output.ACC)]);
disp(['Macro Fscore: ',num2str(output.macroF)]);
disp(['NMI: ',num2str(output.NMI)]);
disp(['ARI: ',num2str(output.ARI)]);
disp(['Training Time: ',num2str(time_caeac)]);
disp(' ');







