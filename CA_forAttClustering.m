%
% CIM-based Adaptive Resonance Theory (CA)
%
function net = CA_forAttClustering(DATA, net)

numNodes = net.numNodes;         % the number of nodes
weight = net.weight;             % node position
CountNode = net.CountNode;       % winner counter for each node
adaptiveSig = net.adaptiveSig;   % kernel bandwidth for CIM in each node

Lambda = net.Lambda;             % an interval for calculating a kernel bandwidth for CIM
minCIM = net.minCIM;             % similarity threshold


for sampleNum = 1:size(DATA,1)
    
    % Compute a kernel bandwidth for CIM based on data points.
    if isempty(weight) == 1 || mod(sampleNum, Lambda) == 0
        estSigCA = SigmaEstimation(DATA, sampleNum, Lambda);
    end
    
    % Current data sample.
    input = DATA(sampleNum,:);
    
    if size(weight,1) < 1 % In the case of the number of nodes in the entire space is small.
        % Add Node
        numNodes = numNodes + 1;
        weight(numNodes,:) = input;
        CountNode(numNodes) = 1;
        adaptiveSig(numNodes) = estSigCA;
        
    else
        
        % Calculate CIM based on global mean adaptiveSig.
        globalCIM = CIM(input, weight, mean(adaptiveSig));
        gCIM = globalCIM;
        
        % Set CIM state between the local winner nodes and the input for Vigilance Test.
        [Lcim_s1, s1] = min(gCIM);
        
        if minCIM < Lcim_s1 % Case 1 i.e., V < CIM_k1
            % Add Node
            numNodes = numNodes + 1;
            weight(numNodes,:) = input;
            CountNode(numNodes) = 1;
            adaptiveSig(numNodes) = SigmaEstimation(DATA, sampleNum, Lambda);
            
        else % Case 2 i.e., V >= CIM_k1
            weight(s1,:) = weight(s1,:) + (1/CountNode(s1)) * (input - weight(s1,:));
            CountNode(s1) = CountNode(s1) + 1;
            
        end % if minCIM < Lcim_s1 % Case 1 i.e., V < CIM_k1
    end % if size(weight,1) < 2
    
    
end % for sampleNum = 1:size(DATA,1)

net.numNodes = numNodes;      % Number of nodes
net.weight = weight;          % Mean of nodes
net.CountNode = CountNode;    % Counter for each node
net.adaptiveSig = adaptiveSig;
net.Lambda = Lambda;

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

% Scaling [0,1]
% normalized = (exNodes-min(exNodes))./(max(exNodes)-min(exNodes));
% qStd = std(normalized);
% qStd(isnan(qStd))=0;

% Add a small value for handling categorical data.
qStd = std(exNodes);
qStd(qStd==0) = 1.0E-6;

% normal reference rule-of-thumb
% https://www.sciencedirect.com/science/article/abs/pii/S0167715212002921
[n,d] = size(exNodes);
estSig = median( ((4/(2+d))^(1/(4+d))) * qStd * n^(-1/(4+d)) );

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



