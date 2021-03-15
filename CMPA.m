% PA 8  DIODE PARAMETER EXTRACTION
% JASON PRASAD 100196970 

% DEFINE INPUTS 
I_s=0.01e-12; % Saturation current 
I_b=0.1E-12 % bias current 
V_b=1.3; % bias voltage
G_p=0.1; % parallel resistor 

%creating a vector 
V_vec= linspace(-1.95,0.7,200);

I=I_s .*(exp((1.2/0.025).*V_vec)-1) + G_p.*V_vec - I_b.*(exp((-1.2/0.025).*(V_vec +V_b))-1);

vary = (rand(1,200)*.20);

Noise= I.*vary;
I_noise= I + Noise; 

figure
subplot(3,2,1)
plot(V_vec,I_noise)
title ('V vs I')
subplot(3,2,2)
semilogy(V_vec, abs(I_noise))
title('semilogy with polyfit')

%Polynomial fitting time 
fourth=polyfit(V_vec, I_noise,4) 
I_4=polyval(fourth, V_vec);

eight=polyfit(V_vec, I_noise,8)
I_8=polyval(eight,V_vec);
subplot(3,2,1);
hold on 
plot(V_vec,I_4)
plot(V_vec,I_8); 

subplot(3,2,2)
hold on 
semilogy(V_vec,abs(I_4))
semilogy(V_vec, abs(I_8))

% Non-linear fits
fo1 = fittype('A.*(exp(1.2*x/25e-3)-1) + 0.1.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff1 = fit(transpose(V_vec),transpose(I),fo1);
If1 = ff1(V_vec);
subplot(3, 2, 3)
plot(V_vec, If1);
title('Non-linear fit')
hold on
subplot(3, 2, 4)
semilogy(V_vec, abs(If1));
title('Semilogy Non-linear fit')
hold on

fo2 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff2 = fit(transpose(V_vec),transpose(I),fo2);
If2 = ff2(V_vec);
subplot(3, 2, 3)
plot(V_vec, If2);
hold on
subplot(3, 2, 4)
semilogy(V_vec, abs(If2));
hold on

fo3 = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');
ff3 = fit(transpose(V_vec),transpose(I),fo3);
If3 = ff3(V_vec);
subplot(3, 2, 3)
plot(V_vec, If3);
hold on
subplot(3, 2, 4)
semilogy(V_vec, abs(If3));
hold on

inputs = V_vec;
targets = I;
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);
view(net)
Inn = outputs;

figure (2)
plot(V_vec, Inn);
title('NN fit')

figure(3)
semilogy(V_vec, abs(Inn));
title('Semilogy NN fit')
