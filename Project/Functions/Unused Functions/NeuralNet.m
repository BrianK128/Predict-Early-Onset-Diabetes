classdef NeuralNet
    %MYNEURALNET Neural Net Class
    %   Implements a binary classifier using a feedforward and
    %   backpropagation neural network with a single hidden layer.
    
    properties
        w1 %weights for input -> hidden layer
        w2 %weights for hidden layer -> input
        b1 %biases for hidden layer neurons
        b2 %bias for output layer
    end
    
    methods
        function nn = NeuralNet(i)
            %MYNEURALNET Construct an instance of this class
            %   Initializing a NN obj for the first time will create
            %   a neural net object with the following weights and biases. Given input
            %   is the number of features in the dataset.
            %   Number of hidden layers is equal to the mean of the sum of
            %   the number of input and output layers.
            
            h = floor((i + 1)/2);       %Binary classifier so only 1 output layer.
            nn.w1 = 4*rand(h,i)-2;          %Weight matrix set to 1s bias.
            nn.w2 = 4*rand(1,h)-2;    
           % nn.w1 = .5*ones(h,i);      %Initial weight matrix 
           % nn.w2 = .5*ones(1,h);
            %zero bias initialization.
            nn.b1 = zeros(h,1);
            nn.b2 = zeros(1);  
        end
        
        function nn = train(nn,D,L,r)
            %train. Train neural net with test data set.
            %   Inputs:
            %   nn: NeuralNet object
            %   D:  Dataset with rows representing individual samples and
            %   columns representing unique features.
            %   L:  Class labels for dataset samples.
            %   r:  rate constant (step size).
            
            ns = size(D,1);
            %for every training sample,
            for i = 1:ns            
               %% feedforward                
               x = D(i,:).'; 
               %Outputs for hidden layer (H) and outer layer (Y).
                   H = sig(nn.w1*x+nn.b1); 
                   y = sig(nn.w2*H+nn.b2);                
               %% backpropagation
               %desired output
               yd = L(i);
               
               a = size(nn.w1,1);
               b = size(nn.w1,2);
               
               %Modify weights from input layer to hidden layer.
               for p = 1:a               
                   for q = 1:b
                   do = y*(1-y)*(yd-y);                                              
                   dno = H(p)*(1-H(p))*nn.w2(p)*do;
                   dwn = r*x(q)*dno;
                   nn.w1(p,q) = nn.w1(p,q) + dwn;
                   end
               end
               
               %Modify weights from hidden layer to output layer.
               %weight matrix for H->O is a 1xc matrix since there is only
               %a single neuron in the output layer. (Binary classifier)
               c = length(nn.w2);
               %Modify bias for hidden layer (same length as weight matrix 
               %for hidden to outer so it is done in the same loop.
               for j = 1:c               
                   nn.w2(j) = r*H(j)*y*(1-y)*(yd - y);
                   nn.b1(j) = nn.b1(j) + (yd - y);
               end
               %Modify bias at output neuron.
               nn.b2 = nn.b2 + (yd - y);
            end                                                
        end
        function [L,p ]= classify(nn,D)
           %classify. Classify training samples. Return a column vector
           %with the labels for the given samples.nn.w2
           %Inputs:
           %nn: NeuralNet object
           %D: Matrix with training sample data. rows represent individual
           %samples and columns represent unique features.
           %func: 1 for rectified linear unit, else use logistic sigmoid.
           ns = size(D,1);
           L = zeros(ns,1);
           p = zeros(ns,1);
           for i = 1:ns               
               % Feedforward
               x = D(i,:).'; 
               H = sig(nn.w1*x+nn.b1);               
               temp = sig(nn.w2*H+nn.b2);
                 
               if(temp > 0.5)               
                   L(i) = 1;                   
               else
                   L(i) = 0;
               end
               p(i) = temp;
           end           
        end
    end
end

