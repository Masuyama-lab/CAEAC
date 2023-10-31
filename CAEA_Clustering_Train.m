% 
% (c) 2023 Naoki Masuyama
% 
% CIM-based ART with Edge and Ages (CAEA) Classifier (CAEAC) is proposed in:
% 
% N. Masuyama, Y. Nojima, F. Dawood, and Z. Liu, "Class-wise classifier design capable of continual learning using adaptive resonance theory-based topological clustering," 
% Applied Sciences, 2023.
% 
% Please contact masuyama@omu.ac.jp if you have any problems.
% 
function [net, ATTnet] = CAEA_Clustering_Train(DATA, LABELS, maxLABEL, net, ATTnet)

numNodes = net.numNodes;         % the number of nodes
weight = net.weight;             % node position
CountNode = net.CountNode;       % winner counter for each node
adaptiveSig = net.adaptiveSig;   % kernel bandwidth for CIM in each node
Lambda = net.Lambda;             % an interval for calculating a kernel bandwidth for CIM

edge = net.edge;
edgeAge = net.edgeAge;
edgeAgeMax = net.edgeAgeMax;

CountEdge = net.CountEdge;
CIMthreshold = net.CIMthreshold;

CountLabel = net.CountLabel;

% Set a size of CountLabel
if size(weight) == 0
    CountLabel = zeros(1, maxLABEL);
end


for sampleNum = 1:size(DATA,1)
    
    % Current data sample.
    input = DATA(sampleNum,:);
    label = LABELS(sampleNum, 1);
    
    % The number of inputs that directly become nodes.
    bufferInput = Lambda/2;
    
    if size(weight,1) < bufferInput % In the case of the number of nodes in the entire space is small.
        % Add Node
        numNodes = numNodes + 1;
        weight(numNodes,:) = input;
        CountNode(numNodes) = 1;
        a = SigmaEstimation(DATA, sampleNum, Lambda);
        adaptiveSig(numNodes,:) = SigmaEstimation(DATA, sampleNum, Lambda);
        edge(numNodes, :) = 0;
        edge(:, numNodes) = 0;
        edgeAge(numNodes, :) = 0;
        edgeAge(:, numNodes) = 0;
        CountEdge(numNodes, :) = 0;
        CountEdge(:, numNodes) = 0;
        
        CountLabel(numNodes,label) = 1;
        
        % Assign similarlity threshold to the initial nodes.
        if numNodes == bufferInput
            
            % Clustering attributes of data.
            ATTnet = ClusteringAtribute(DATA, sampleNum, Lambda, CIMthreshold, ATTnet);
            
            
            % Find the most similar attribute to ATTnet.weight.
            [~, idx] = min( pdist2(ATTnet.weight, input', 'euclidean') );
            
            tmpTh = zeros(1,bufferInput);
            for k = 1:bufferInput
                tmpCIM = CIM(weight(k,:), weight, mean(mean(adaptiveSig)));
                [~, s1] = min(tmpCIM);
                tmpCIM(s1) = inf; % Remove CIM between weight(k,:) and weight(k,:).
                tmpTh(k) = min(tmpCIM);
                
            end
            meanTmpTh = mean(tmpTh);
            
            CIMthreshold = zeros(1,bufferInput);
            for k = 1:bufferInput
                CIMthreshold(k) = meanTmpTh;
            end
            
        else
            CIMthreshold(1:numNodes) = mean(CIMthreshold);
        end
        
    else
        
        
        % Clustering attributes of data.
        if mod(sampleNum, Lambda) == 0
            ATTnet = ClusteringAtribute(DATA, sampleNum, Lambda, CIMthreshold, ATTnet);
        end
        
        % Find the most similar attribute to ATTnet.weight.
        [~, idx] = min( pdist2(ATTnet.weight, input', 'euclidean') );
        
        % Compute CIM for grouped attributes.
        totalCIM = 0;
        for h = unique(idx)
           groupAtt = find(idx == h);
           exS = mean(mean(adaptiveSig(:,groupAtt)));
           totalCIM = totalCIM + CIM(input(:,groupAtt)', weight(:,groupAtt), exS);
        end
        
        globalCIM = totalCIM/size(unique(idx),2);
        gCIM = globalCIM;
        
        
        % Set CIM state between the local winner nodes and the input for Vigilance Test.
        [Vs1, s1] = min(gCIM);
        gCIM(s1) = inf;
        [Vs2, s2] = min(gCIM);
        
        if CIMthreshold(s1) < Vs1 % Case 1 i.e., V < CIM_k1
            % Add Node
            numNodes = numNodes + 1;
            weight(numNodes,:) = input;
            CountNode(numNodes) = 1;
            adaptiveSig(numNodes,:) = SigmaEstimation(DATA, sampleNum, Lambda);
            edge(numNodes, :) = 0;
            edge(:, numNodes) = 0;
            edgeAge(numNodes, :) = 0;
            edgeAge(:, numNodes) = 0;
            CountLabel(numNodes,label) = 1;
            CountEdge(numNodes, :) = 0;
            CountEdge(:, numNodes) = 0;
            
            % Assigne similarlity threshold
            CIMthreshold(numNodes) = CIMthreshold(s1);
            
            
        else % Case 2 i.e., V >= CIM_k1
            weight(s1,:) = weight(s1,:) + (1/CountNode(s1)) * (input - weight(s1,:));
            CountNode(s1) = CountNode(s1) + 1;
            CountEdge(s1,s2) = CountEdge(s1,s2) + 1;
            
            CountLabel(s1,label) = CountLabel(s1, label) + 1;
            
            if CIMthreshold(s2) >= Vs2 % Case 3 i.e., V >= CIM_k2
                % Update weight of s2 node.
%                 weight(s2,:) = weight(s2,:) + (1/(10*CountNode(s2))) * (input - weight(s2,:));
                
                s1Neighbors = find( edge(s1,:) );
                for k = s1Neighbors
                    weight(k,:) = weight(k,:) + ( 1/(10*CountNode(k) )) * (input - weight(k,:));
                end
                
                % Create an edge between s1 and s2 nodes.
                edge(s1,s2) = 1;
                edge(s2,s1) = 1;
                edgeAge(s1,s2) = 0;
                edgeAge(s2,s1) = 0;
            end
            
            % Increment age
            edgeAge(s1,:) = edgeAge(s1,:) + 1;
            edgeAge(:,s1) = edgeAge(:,s1) + 1;
            
            % If age > ageMAX, detele edge
            deleteAge = find( edgeAge(s1,:) > edgeAgeMax );
            edge(s1, deleteAge) = 0;
            edge(deleteAge, s1) = 0;
            edgeAge(s1, deleteAge) = 0;
            edgeAge(deleteAge, s1) = 0;
            CountEdge(s1, deleteAge) = 0;
            CountEdge(deleteAge, s1) = 0;
            
        end % if minCIM < Lcim_s1 % Case 1 i.e., V < CIM_k1
    end % if size(weight,1) < 2
    
    
    % Topology Adjustment
    % If activate the following function, CAEAF shows a noise reduction ability.
    if mod(sampleNum, Lambda) == 0 && size(weight,1) > 1
        % -----------------------------------------------------------------
        % Delete Node based on number of neighbors
        nNeighbor = sum(edge);
        deleteNodeEdge = (nNeighbor == 0);
        
        % Delete process
        numNodes = numNodes - sum(deleteNodeEdge);
        weight(deleteNodeEdge, :) = [];
        CountNode(deleteNodeEdge) = [];
        edge(deleteNodeEdge, :) = [];
        edge(:, deleteNodeEdge) = [];
        edgeAge(deleteNodeEdge, :) = [];
        edgeAge(:, deleteNodeEdge) = [];
        adaptiveSig(deleteNodeEdge,:) = [];
        CIMthreshold(deleteNodeEdge) = [];
        
        CountLabel(deleteNodeEdge, :) = [];
        
    end % if mod(sampleNum, Lambda) == 0
    
end % for sampleNum = 1:size(DATA,1)

% Cluster Labeling based on edge (Functions are available above R2015b.)
connection = graph(edge ~= 0);
LabelCluster = conncomp(connection);

net.numNodes = numNodes;      % Number of nodes
net.weight = weight;          % Mean of nodes
net.CountNode = CountNode;    % Counter for each node
net.adaptiveSig = adaptiveSig;

net.LabelCluster = LabelCluster;
net.edge = edge;
net.edgeAge = edgeAge;
net.CountEdge = CountEdge;
net.CIMthreshold = CIMthreshold;

net.CountLabel = CountLabel;

end


% Compute an initial kernel bandwidth for CIM based on data points.
function estSig = SigmaEstimation(DATA, sampleNum, Lambda)

if size(DATA,1) < Lambda
    exNodes = DATA;
elseif (sampleNum - Lambda) <= 0
    exNodes = DATA(1:sampleNum,:);
elseif (sampleNum - Lambda) > 0
    exNodes = DATA( (sampleNum+1)-Lambda:sampleNum, :);
end

% Add a small value for handling categorical data.
if size(exNodes,1) == 1
    qStd = zeros(size(exNodes));
else
    qStd = std(exNodes);
end
qStd(qStd==0) = 1.0E-6;

% normal reference rule-of-thumb
% https://www.sciencedirect.com/science/article/abs/pii/S0167715212002921
[n,d] = size(exNodes);
estSig = ((4/(2+d))^(1/(4+d))) * qStd * n^(-1/(4+d));

end


% Correntropy induced Metric (Gaussian Kernel based)
function cim = CIM(X,Y,sig)
% X : 1 x n
% Y : m x n
[n, att] = size(Y);
g_Kernel = zeros(n, att);

for i = 1:att
    g_Kernel(:,i) = GaussKernel(X(i)-Y(:,i), sig);
end

% ret0 = GaussKernel(0, sig);
ret0 = 1;
ret1 = mean(g_Kernel, 2);

cim = sqrt(ret0 - ret1)';
end

function g_kernel = GaussKernel(sub, sig)
g_kernel = exp(-sub.^2/(2*sig^2));
% g_kernel = 1/(sqrt(2*pi)*sig) * exp(-sub.^2/(2*sig^2));
end

% Clustering attributes of data.
function ATTnet = ClusteringAtribute(DATA, sampleNum, Lambda, CIMthreshold, ATTnet)

if size(DATA,1) < Lambda
    D = DATA;
elseif (sampleNum - Lambda) <= 0
    D = DATA(1:sampleNum,:);
elseif (sampleNum - Lambda) > 0
    D = DATA( (sampleNum+1)-Lambda:sampleNum, :);
end

D = mean(D);
rD = reshape(D,[numel(D),1]); % reshape n*m matrix to 1*(n*m)

% Randamize data
ran = randperm(size(rD,1));
rD = rD(ran,:);

% Attribution Clustering
ATTnet.minCIM = CIMthreshold;
ATTnet = CA_forAttClustering(rD, ATTnet);

end

