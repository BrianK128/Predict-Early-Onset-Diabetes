classdef NN
    %NN Neural Network class. Binary classifier neural net.
    %   NN object is a neural network object that can be trained using
    %   a feedforward operation and backpropogation algorithm. 
    
    properties
        w1 %weights for input -> hidden layer
        w2 %weights for hidden layer -> input
        b1 %biases for hidden layer neurons
        b2 %bias for output layer
        %Jw %current J(w), training error.
        %thrReached %1 if Threshold theta was reached.
    end
    
    methods
        function nn = NN(i,j)
            % NN Construct an instance of a neural net.
            %   input:
            %   i: number of features in data set.
            %   j: number of hidden layers.
            %   Initializing a NN obj for the first time will create
            %   a neural net object with random weights and biases between 
            %   -2 to 4. 
            %   Number of hidden layers is equal to the mean of the sum of
            %   the number of input and output layers.
            %
            %% *Developers note
            %   Jw and threshold theta is currently unsupported due to
            %   problems with determining the optimal threshold. 
            %% Set Initial Properties. Random Weights and zero bias.
            nn.w1 = 4*rand(j,i)-2;          
            nn.w2 = 4*rand(1,j)-2;      %Single output neuron.
            %bias initialization.
            nn.b1 = zeros(j,1);
            nn.b2 = 0;  
            %nn.Jw = 0;
            %nn.thrReached = 0;
        end
        
        function nn = train(nn,D,L,r)
           %Trains the Neural Net object nn. 
           %Inputs:
           %nn: NN object
           %D: Training Data in the format: rows:samples cols: features
           %L: Training Label. Corresponding tag for class for given sample
           %r: Training rate.
           
           ns = size(D,1);
           
           for samp = 1:ns
               %% Feed Forward
               x = D(samp,:).';
               Y = sig(nn.w1*x+nn.b1);
               Z = sig(nn.w2*Y+nn.b2);
               
               %% Backpropogation and Calculate Error 
               t = L(samp);  
               
               %JwC = (.5)*(t - Z)^2;
               
               %if(abs(nn.Jw - JwC) < 0.01)
               %   nn.thrReached = 1;
                   
               %else
               
                   [w1r,w1c] = size(nn.w1);
                   [w2r,w2c] = size(nn.w2);
                   
                   dk = (t-Z)*dsig(Z);
                   %calc delta wkj, change in weights for hidden->outer
                   %layer.
                   dw2 = zeros(w2r,w2c);
                   for k = 1:w2r
                       for j = 1:w2c
                           dw2(k,j) = r*dk*Y(j);                                              
                           nn.b1(j) = nn.b1(j) + (t-Z)*r;
                       end
                   end
                   %calc delta wji, change in weights for input->hidden
                   dj = nn.w2 * dk .* dsig(Y).';
                   dw1 = zeros(w1r,w1c);
                   for j1 = 1:w1r
                       for i = 1:w1c
                           dw1(j1,i) = r*dj(j1)*x(i);                                                   
                       end
                   end
                   
                   nn.b2 = nn.b2 + (t-Z)*r;
                   nn.w1 = nn.w1 + dw1;
                   nn.w2 = nn.w2 + dw2;   
                   %nn.Jw = JwC;
               %end
           end          
        end
        
        function L = classify(nn,D)
           ns = size(D,1);
           L = zeros(ns,1);
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
           end
        end
        
    end
end

