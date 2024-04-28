% a) Learning POD Summary
% POD members: Raymond, Meghan
% Lab struggle: I did not know how to eliminate question to one expression.
% I originally did not think to use what expression to link the two parts.
% discussion with POD help: After discussed with my podmates, I learned
% that I could use "," to connect the two random portion.

%% 1.5.2
%% 1. 
% What is M(2,3)? Ans: M(2,3) = 6
M = [1, 3, 5, 7, 9; 2, 4, 6, 8, 10]
M(2,3)
%% 2. 
% What will happen if you type MP(2,3)
% Ans: there is an error message
% Index in position 2 exceeds array bounds. Index must not exceed 2.
MP = M'
% MP(2, 3) // need to comm
%% 3. 
% What will happen if type MP(4,2)? Ans: 8
MP(4,2)

%% 1.5.4
%% 4. 
% Create a row vector fo the numbers 1-100 in reverse order
A = flip(1:100)

%% 5. 
% 10x10 matrix with 5 for all diagonal entries and -3 for off diagonal
% entries
B = eye(10,10) * 8 + (-3) * ones(10,10)

%% 6. 
% Create a 5x3 matrix comprising random numbers selected uniformly from
% the range 10 to 20
C = 10*rand(5,3) + 10

%% 7. 
% Create a vector of length 20, whose first 15 numbers are randomly 
% chosen between 2 and 3, and whose last 5 numbers are randomly chosen 
% between -1 and -2
vector = [2 + (3-2).*rand(1,15), -2 + (-1+2).*rand(1,5)]

%% 8. 
% plot y = 3x for a range of x from -5 to +5 with steps of 0.1.
% On the same graph, in a different color, plot y = 4x. Label the axes
% and indicate with a legend which color corresponds to which line
x = -5:0.1:5;
y1 = 3 * x;
plot(x,y1)

hold on
y2 = 4 * x;
plot(x,y2)
legend("y1=3x", "y2=4x")
xlabel("x")
ylabel("y")

hold off

%% 9. 
% Write a code that will plot y = x^2 - 3 for a range of x from -3 to +3
% with steps of 0.1. Save the code as an .m file and run that file from
% the command window
x = -3:0.1:3;
y = x .^ 2 - 3;
plot(x, y)
xlabel("x")
ylabel("y")

%% 1.5.8
%% 10. 
% Plot a rectified sine wave (negative values are fixed at zero) over
% the range of values -pi <= x <= pi
x = linspace(-pi, pi, 1000);
y = sin(x);
y_rectified = max(y, 0); % % The max() function compares each element of 
%                           y with 0 and keeps the greater value
plot(x, y_rectified)
xlabel("x")
ylabel("y")

%% 11. 
% Write a code to indicate whether the cube root of 6 is greater than
% the square root of 3.
cubeRoot = nthroot(6, 3);
squareRoot = sqrt(3);
if cubeRoot > squareRoot
    disp('The cube root of 6 is greater than the square root of 3.');
else
    disp('The cube root of 6 is not greater than the square root of 3.');
end

%% 1.5.9
%% 12. 
% Sum all the cubed positive integers up to 15^3
sum = 0;
for i = 1:1:15
    sum = sum + i^3;
end
disp(sum)

%% 13. 
% find the positive integer n such that n + n^2 + n^3 + n^4 = 88740
n = 1;
while (n + n^2 + n^3 + n^4 ~= 88740)
    n = n+1;
end
disp(n)

%% 1.5.10
%% 14
% 14a. Write a function that takes as input a single vector and returns the
% sum of the squares of its elements as its single output.
function y = square_element_sum(x)
    y = sum(x .^ 2);
end

% 14b. Use that function to sum the square of the numbers from 27 to 37
% inclusive.
sum_vector = square_element_sum(27:37);
disp(sum_vector)

%% 15. 
% Write a function that takes as input a single vector of numbers and
% returns the mean, the mode, and the median as three separate variables
% for its output.
function [meanVal, modeVal, medianVal] = stats(vector)
    % Calculate mean
    sumVal = 0;
    for i = 1:length(vector)
        sumVal = sumVal + vector(i);
    end
    meanVal = sumVal / length(vector);

    % Calculate mode
    % Using hist to count occurrences and finding the value with the 
    % maximum count
    [counts, values] = hist(vector, unique(vector));
    [~, maxIdx] = max(counts);
    modeVal = values(maxIdx);

    % Calculate median
    sortedVector = sort(vector);
    n = length(sortedVector);
    if mod(n, 2) == 0
        medianVal = (sortedVector(n/2) + sortedVector(n/2 + 1)) / 2;
    else
        medianVal = sortedVector((n + 1) / 2);
    end
end

% Test Case 1: Odd number of elements
vector1 = [10, 2, 8, 6, 3];
[meanVal1, modeVal1, medianVal1] = stats(vector1);
disp(['Test Case 1 - Mean: ', num2str(meanVal1), ', Mode: ', ...
    num2str(modeVal1), ', Median: ', num2str(medianVal1)]);

% Test Case 2: Even number of elements
vector2 = [1, 2, 3, 4, 5, 6];
[meanVal2, modeVal2, medianVal2] = stats(vector2);
disp(['Test Case 2 - Mean: ', num2str(meanVal2), ', Mode: ', ...
    num2str(modeVal2), ', Median: ', num2str(medianVal2)]);

% Test Case 3: Repeated elements
vector3 = [4, 2, 4, 2, 4, 3];
[meanVal3, modeVal3, medianVal3] = stats(vector3);
disp(['Test Case 3 - Mean: ', num2str(meanVal3), ', Mode: ', ...
    num2str(modeVal3), ', Median: ', num2str(medianVal3)]);

% Test Case 4: A single element
vector4 = [7];
[meanVal4, modeVal4, medianVal4] = stats(vector4);
disp(['Test Case 4 - Mean: ', num2str(meanVal4), ', Mode: ', ...
    num2str(modeVal4), ', Median: ', num2str(medianVal4)]);

% Test Case 5: Negative numbers
vector5 = [-3, -1, -4, -2, -5];
[meanVal5, modeVal5, medianVal5] = stats(vector5);
disp(['Test Case 5 - Mean: ', num2str(meanVal5), ', Mode: ', ...
    num2str(modeVal5), ', Median: ', num2str(medianVal5)]);


%% 1.5.11
% Suppose you want to write a code that indicates whenever the tangent
% function has a value greater than a threshold, here set as 2.

% Writing and verifying that the two code options yield the same results
% both aim to find the points at which the tangent of a time vector, 
% scaled by 2pi, exceeds a certain threshold. They then plot the tangent 
% function and mark the points where it surpasses this threshold.

% The first script threshold_find.m uses a for-loop to iterate over each 
% element of the time vector tvector, compute the tangent value, and 
% check if it's above the threshold.

% The second script threshold_find2.m uses vectorized operations to 
% compute the tangent values for all time points at once and create a 
% logical vector findhigh indicating where the tangent exceeds the 
% threshold. This approach is more efficient because it leverages 
% MATLAB's ability to operate on entire arrays at once, instead of 
% looping through individual elements.

% threshold_find.m
%clear
thresh = 2;
tmax = 10;
tvector = 0:0.001:tmax;
Nt = length(tvector);
tanval = zeros(size(tvector));          % to store tan of tvector
findhigh = zeros(size(tvector));        % stores when tan > thresh
for i=1:Nt                              % for all values of tvector
    tanval(i) = tan(2*pi*tvector(i));   % set tangent of t
    if (tanval(i) > thresh)                 % if tan is high
        findhigh(i) = 1;                % store value of t end
    end
end

% Now plot the results
figure(1)
subplot(2,1,1)
plot(tvector,tanval)                    % plot tan(2.pi.t) versus t
axis([0 tmax -5 5])
subplot(2,1,2)
plot(tvector,findhigh)                  %plot t where tan(2.pi.t)>2
axis([0 tmax -0.5 1.5])
% We can color the portions corresponding to findhigh=1 using
% MATLAB’s find command, which extracts the indices of the
% non-zero entries of a matrix:
highindices = find(findhigh);           % indices above threshold
subplot(2,1,1);
hold on
plot(tvector(highindices),tanval(highindices),'r.');


% threshold_find2.m
%clear
thresh = 2;
tmax = 10;
tvector = 0:0.001:tmax;
Nt = length(tvector);
tanval = tan(tvector);                  % operates on all values at once
findhigh = tanval>thresh;               % gives 1 or 0 for all entreis

% Now plot the results
figure(2)
%clf                                     % clears figure for a new plot
subplot(2,1,1)
plot(tvector,tanval)
axis([0 tmax -5 5])
subplot(2,1,2)
plot(tvector,findhigh)
axis([0 tmax -0.1 1.1])
