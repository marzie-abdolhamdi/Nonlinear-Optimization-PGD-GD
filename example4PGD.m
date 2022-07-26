%Example 4
clc
clear
format short
% Function Definition:
syms x1 x2;
%Objective function
f = (x1-1)^2 + (x2-2)^2 -4;
%Constrains :
g(1) = x1+2*x2-5;       % x1+2*x2<=5
g(2) = 4*x1 +3*x2-6;    % 4*x1 +3*x2<=6   
g(3) = 6*x1+x2-7;       % 6*x1+x2-7
g(4) = -x1;             % x1>=0
g(5) = -x2;             % x2>=0
N = 5; %number of constraints
%tolerances
eps1 = 0.001;
conv = 1; %initialize the convergance criteria
%Step 1: Initial Guess (Choose Initial Guesses):
i = 1;
x_1(i) = 1;
x_2(i) = 1;
%Save the gradient of given objective function
Search_dir = -gradient(f);
%Objective function value at initial guess
fun_value = (subs(f,[x1,x2], [x_1(i),x_2(i)]));
while conv > eps1
    %matrix for constratint at initial guess
    const(1) = (subs(g(1),[x1,x2], [x_1(i),x_2(i)]));
    const(2) = (subs(g(2),[x1,x2], [x_1(i),x_2(i)]));
    const(3) = vpa(subs(g(3),[x1,x2], [x_1(i),x_2(i)]),4);
    const(4) = (subs(g(4),[x1,x2], [x_1(i),x_2(i)]));
    const(5) = (subs(g(5),[x1,x2], [x_1(i),x_2(i)]));
    
    %Checking the optimum point [x_1,x_2] will violate the constraint or not
    if max(const) < 0 %not violated
        %calculate the search diection since point satisfies constraint
        S = subs(Search_dir,[x1,x2], [x_1(1),x_2(1)]);
    else %if constraint violate
        for j = 1:N
            if const(j)==0
                %calculate the projection matrix of violated constaraint
                N = (subs(gradient(g(j)),[x1,x2], [x_1(i),x_2(i)]));
                P = eye(2)-(N*inv(N'*N)*N');
                %Since point violate the constraint so calculate again
                %serach direction
                S = -((P*double(subs(gradient(f),[x1,x2], [x_1(i),x_2(i)]))));
                break;
            end
        end
    end
    %calculating the step length
    if norm(S)== 0
        %If search direction is zero
        lambda = -inv(N'*N)*N'*double(subs(gradient(f),[x1,x2], [x_1(i),x_2(i)]));
        if lambda>0
            %lambda more than zero then will get our optimum point
            optima = [x_1(i),x_2(i)];
            optimum = double(subs(f,[x1,x2], [x_1(i),x_2(i)]));
            break;
        end
    else %if search direction isn't zero
        S = S/norm(S);
        syms lambda
        const(1) = vpa((subs(g(1),[x1,x2], [x_1(i)+ lambda*S(1), x_2(i)+lambda*S(2)])));
        const(2) = vpa((subs(g(2),[x1,x2], [x_1(i)+ lambda*S(1), x_2(i)+lambda*S(2)])));
        lam2 = vpa(solve(const(1)==0,lambda),5);
        const(3) = vpa((subs(g(3),[x1,x2], [x_1(i)+ lambda*S(1), x_2(i)+lambda*S(2)])));
        lam3 = vpa(solve(const(3)==0,lambda),5);
        const(4) = vpa((subs(g(4),[x1,x2], [x_1(i)+ lambda*S(1), x_2(i)+lambda*S(2)])));
        lam4 = vpa(solve(const(4)==0,lambda),5);
        const(5) = vpa((subs(g(5),[x1,x2], [x_1(i)+ lambda*S(1), x_2(i)+lambda*S(2)])));
        lam5 = vpa(solve(const(5)==0,lambda),5);
        lam = max(lam2,max(lam3,max(lam4,lam5)));
        lambd = max(lam);
        %put the maximum value of point in objective function 
        func_lambda = vpa(subs(f,[x1,x2], [x_1(i)+ lambda*S(1),x_2(i)+ lambda*S(2)]));
        %differentiate the objective function respect to step length
        dfunc_lambda = diff(func_lambda,lambda);
        dfunc_lambda_lambd = vpa(subs(dfunc_lambda,[lambda],[lambd]));
        if dfunc_lambda_lambd>0
            lambda = vpa(solve(dfunc_lambda==0),5);
        else
            lambda = lambd;
        end
    end
    %new point
    x_1(i+1) = x_1(i)+lambda*S(1);
    x_2(i+1) = x_2(i)+lambda*S(2);
    fun_value1 = vpa(subs(f,[x1,x2], [x_1(i),x_2(i)]),6);
    fun_value2 = vpa(subs(f,[x1,x2], [x_1(i+1),x_2(i+1)]),6);
    %convergance criteria
    conv = (abs(fun_value1)-abs(fun_value2))/(abs(fun_value1));
    %increase the iteration number
    i = i+1
end
%% draw table in column view
Iter = 1:i;
X_coordinate = x_1';
Y_coordinate = x_2';
Iterations = Iter';
for i=1:length(X_coordinate)
    Objective_value(i) = double(subs(f,[x1,x2], [x_1(i),x_2(i)]));
end
Objective_value = Objective_value';
%table view
T = table(Iterations,X_coordinate,Y_coordinate, Objective_value);
%Output
fprintf('Initial Objective Function Value: %d\n\n',double(subs(f,[x1,x2], [x_1(1),x_2(1)])));
if (conv < eps1)
    fprintf('Minimum succesfully obtained...\n\n');
end
fprintf('Number of Iterations for Convergence: %d\n\n', i);
fprintf('Point of Minima: [%d,%d]\n\n', x_1(i), x_2(i));
fprintf('Objective Function Minimum Value: %d\n\n', double(subs(f,[x1,x2], [x_1(i),x_2(i)])));
%Plots
hold on;
%Contour Drawing
fcontour(f,[-5,5,-5,5],'MeshDensity',500, 'Fill', 'On');
[x1,x2] = meshgrid(-5:0.05:5);
Z1 =(x1>=0 & x2>=0 );
Z2 = (x1 +2*x2 <= 5) ;
Z3= (4*x1 +3*x2 <= 10);
Z4= (6*x1 +x2 <= 7);
contour(x1,x2,Z1,'-y','LineWidth',0.5)
contour(x1,x2,Z2,'-g','LineWidth',0.5)
contour(x1,x2,Z3,'-c','LineWidth',0.5)
contour(x1,x2,Z4,'-m','LineWidth',0.5)
for n = 1:i-1
    %for tracing the points in contour plot
    plot(x_1(n),x_2(n),'Og','LineWidth',2);
    plot(x_1(n+1),x_2(n+1),'Or','LineWidth',2);
    plot(x_1(n:n+1),x_2(n:n+1),'*-r')
    xlabel('x1')
    ylabel('x2')
    %title of the contour plot
    title('Rosens Gradient Projection method');
    %These annotation used for arrow with text, You can edit the arrow position from
    %contour plot where you will get the position [x1,y1][x2,y2]
    annotation('textarrow',[0.724 0.691],[0.802 0.633],'Color','g','String', 'x1 + 2*x2 <=5');
    annotation('textarrow',[0.430 0.575],[0.697 0.738],'Color','m', 'String', '6x1 + x2 <=7');
    annotation('textarrow',[0.764 0.792],[0.238 0.404],'Color','c','String', '4x1 + 3x2 <=10');
    annotation('textarrow',[0.548 0.580],[0.335 0.511],'Color','y','String', 'x2 >=0');
    annotation('textarrow',[0.430 0.517],[0.592 0.638],'Color','y','String', 'x1 >=0');
    annotation('textarrow',[0.469 0.546],[0.445 0.6],'Color','r','String', 'Feasible region');
end
%Display the table in command window
disp(T)