%
% CIM-based Adaptive Resonance Theory with Edge and Age (CAEA)
%
function net = CAEA_Train(DATA, net)

numNodes = net.numNodes;         % the number of nodes
weight = net.weight;             % node position
CountNode = net.CountNode;       % winner counter for each node
adaptiveSig = net.adaptiveSig;   % kernel bandwidth for CIM in each node
Lambda = net.Lambda;             % an interval for calculating a kernel bandwidth for CIM

edge = net.edge;
edgeAge = net.edgeAge;
edgeAgeMax = net.edgeAgeMax;

CIMthreshold = net.CIMthreshold;


for sampleNum = 1:size(DATA,1)
    
    % Current data sample.
    input = DATA(sampleNum,:);
    
    % The number of inputs that directly becomes nodes.
    bufferInput = Lambda/2;
    
    if size(weight,1) < bufferInput % In the case of the number of nodes in the entire space is small.
        % Add Node
        numNodes = numNodes + 1;
        weight(numNodes,:) = input;
        CountNode(numNodes) = 1;
        adaptiveSig(numNodes) = SigmaEstimation(DATA, sampleNum, bufferInput);
        edge(numNodes, :) = 0;
        edge(:, numNodes) = 0;
        edgeAge(numNodes, :) = 0;
        edgeAge(:, numNodes) = 0;
        
        % Assign similarlity threshold to the initial nodes.
        if numNodes == bufferInput
            tmpTh = zeros(1,bufferInput);
            for k = 1:bufferInput
                tmpCIMs1 = CIM(weight(k,:), weight, mean(adaptiveSig));
                [~, s1] = min(tmpCIMs1);
                tmpCIMs1(s1) = inf; % Remove CIM between weight(k,:) and weight(k,:).
                tmpTh(k) = min(tmpCIMs1);
            end
            CIMthreshold = repmat(mean(tmpTh), bufferInput, 1);
        else
            CIMthreshold(1:numNodes) = mean(CIMthreshold);
        end
        
    else
        
        % Calculate CIM based on global mean adaptiveSig.
        gCIM = CIM(input, weight, mean(adaptiveSig));
        
        % Set CIM state between the local winner nodes and the input for Vigilance Test.
        [Vs1, s1] = min(gCIM);
        gCIM(s1) = inf;
        [Vs2, s2] = min(gCIM);
        
        if CIMthreshold(s1) < Vs1 % Case 1 i.e., V < CIM_k1
            % Add Node
            numNodes = numNodes + 1;
            weight(numNodes,:) = input;
            CountNode(numNodes) = 1;
            adaptiveSig(numNodes) = SigmaEstimation(DATA, sampleNum, bufferInput);
            edge(numNodes, :) = 0;
            edge(:, numNodes) = 0;
            edgeAge(numNodes, :) = 0;
            edgeAge(:, numNodes) = 0;
            
            % Assigne similarlity threshold
            CIMthreshold(numNodes) = CIMthreshold(s1);
            
        else % Case 2 i.e., V >= CIM_k1
            
            % Increment age
            edgeAge(s1,:) = edgeAge(s1,:) + 1;
            edgeAge(:,s1) = edgeAge(:,s1) + 1;
            
            % If age > ageMAX, detele edge
            deleteAge = find( edgeAge(s1,:) > edgeAgeMax );
            edge(s1, deleteAge) = 0;
            edge(deleteAge, s1) = 0;
            edgeAge(s1, deleteAge) = 0;
            edgeAge(deleteAge, s1) = 0;
            
            % Update s1 weight
            weight(s1,:) = weight(s1,:) + (1/CountNode(s1)) * (input - weight(s1,:));
            CountNode(s1) = CountNode(s1) + 1;
            
            % Update s1 neighbor
            if CIMthreshold(s2) >= Vs2 % Case 3 i.e., V >= CIM_k2
                
                % Update weight of s2 node.
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
        adaptiveSig(deleteNodeEdge) = [];
        CIMthreshold(deleteNodeEdge) = [];
        
    end % if mod(sampleNum, Lambda) == 0
    
    
    % Drawing
%     if mod(sampleNum, 100) == 0
%         net.weight = weight;
%         net.edge = edge;
%         connection = graph(edge ~= 0);
%         net.LabelCluster = conncomp(connection);
%         try
%             set(0,'CurrentFigure',2)
%         catch
%             figure(2);
%         end
%         myPlot(DATA, net, ' ');
%         drawnow
%     end
    
    
    
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
net.CIMthreshold = CIMthreshold;

end


% Compute an initial kernel bandwidth for CIM based on data points.
function estSig = SigmaEstimation(DATA, sampleNum, bufferInput)

if (sampleNum - bufferInput) <= 0
    exNodes = DATA(1:sampleNum,:);
elseif (sampleNum - bufferInput) > 0
    exNodes = DATA( (sampleNum+1)-bufferInput:sampleNum, :);
end

% Add a small value for handling categorical data.
qStd = std(exNodes);
qStd(qStd==0) = 1.0E-6;

% normal reference rule-of-thumb
% https://www.sciencedirect.com/science/article/abs/pii/S0167715212002921
[n,d] = size(exNodes);
estSig = median( ((4/(2+d))^(1/(4+d))) * qStd * n^(-1/(4+d)) );

end


% Correntropy induced Metric
function cim = CIM(X,Y,sig)
% X : 1 x n
% Y : m x n
[~, att] = size(Y);
cim = sqrt(1 - mean(exp(-(X(1:att)-Y(:,1:att)).^2/(2*sig^2)), 2))';
end



