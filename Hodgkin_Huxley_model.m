%% Q1 Action Potential Properties

clc
clear

% creating the injected pulse of current

t = [0:0.01:30]; % time course

I = one_pulse(t,20,0,0.5); % one pulse of 20 nA injected for 0.5 ms
    % the function generates a one-pulse current in a specific time and length
    % see the full function below

% calculating the HH model parameters - see full function below

[v,n,m,h] = HH_model(t,I);

% plotting the action potential & the changes of n, m, h values

figure('Position', [150 150 1000 500])
subplot(1,2,1)
plot(t,v,t,I)
title('Action potential after current injection')
xlabel('Time (ms)')
ylabel('Voltage (mV)')
legend('Membrane Voltage', 'Injected Current')
subplot(1,2,2)
plot(t,n,t,m,t,h)
title('Probabilities of gates after current injection')
xlabel('Time (ms)')
ylabel('P (activation / inactivation)')
legend('n','m','h')

%% Q2 Firing rate during constant currents injection

clc
clear

% calculating membrane voltage during constant injection of a 30 nA current

t = [0:0.01:100]; %time
I_const = 30.*ones(1,length(t)); % constant current of 30 nA
v_of_const = HH_model(t,I_const); % membrane voltage by HH model
    % the results are displayed in a figure below

% calculating firing-rate for every current injected (IF curve)

current = [0:80]; % range of currents from 0 to 80 nA

% HH model for every current
for o = 1:length(current)
    I(1:length(t))=current(o);
    v = HH_model(t,I);
    v_peak = islocalmax(v); %finding firing peaks
    v_peaks_only = v(v_peak); %voltage in every peak

    % eliminating peaks that arn't action potentials
    for i=1:length(v_peaks_only)
        if v_peaks_only(i) < -40
            v_peaks_only(i) = 0;
        else
            v_peaks_only(i) = 1;
        end
    end

    sum_v_peaks = sum(v_peaks_only); % sum of all action potentials
    Hz(o) = sum_v_peaks/(100/1000); % firing rate in Hz
end

% plotting the action potential & IF curve
figure('Position', [150 150 1000 500])
subplot(1,2,1)
plot(t,v_of_const,t,I_const)
title('Membrane voltage during injection of 30 nA constatnt current')
xlabel('Time (ms)')
ylabel('Voltage (mV)')
legend('membrane potential','30 nA current')
subplot(1,2,2)
plot(current,Hz)
title('Frequency of firing in different currents')
xlabel('I (nA)')
ylabel('Firing rate (Hz)')


%% Q3 Refractory periods

%in this section, I will demonstrate the two types of refractory periods -
% Absolute & Relative

clc
clear

t = [0:0.01:30]; % time

% creating pulses for absolute and relative refractory periods
I_first = one_pulse(t,20,0,1); % first pulse injection
I_second_absolute_low = one_pulse(t,20,2,3); % second pulse of the same magnitude during absolute period
I_second_absolute_high = one_pulse(t,80,2,3); % second pulse of greater magnitude during absolute period
I_second_relative_low = one_pulse(t,20,8,9); % second pulse of the same magnitude during relative period
I_second_relative_high = one_pulse(t,80,8,9); % second pulse of greater magnitude during relative period

% combining the two pulses into vectors
I_absolute_low = I_first+I_second_absolute_low; 
I_absolute_high = I_first+I_second_absolute_high;
I_relative_low = I_first+I_second_relative_low;
I_relative_high = I_first+I_second_relative_high;

% calculating HH model for all pulses
v_absolute_low = HH_model(t,I_absolute_low);
v_absolute_high = HH_model(t,I_absolute_high);
v_relative_low = HH_model(t,I_relative_low);
v_relative_high = HH_model(t,I_relative_high);

% plotting them alltogether
figure('Position', [150 80 800 600])
subplot(2,2,1)
plot(t,v_absolute_low,t,I_absolute_low)
title('Absolute refractory period')
subtitle('Two pulses of 20nA, 1ms apart')
legend('membrane voltage','current')
xlabel('Time (ms)')
ylabel('Voltage (mV)')
ylim([-100 100])
subplot(2,2,3)
plot(t,v_absolute_high,t,I_absolute_high)
xlabel('Time (ms)')
ylabel('Voltage (mV)')
legend('membrane voltage','current')
subtitle('One pulse of 20nA and one of 80nA, 1ms apart')
subplot(2,2,2)
plot(t,v_relative_low,t,I_relative_low)
xlabel('Time (ms)')
ylabel('Voltage (mV)')
ylim([-100 100])
title('Relative refractory period')
subtitle('Two pulses of 20nA, 7ms apart')
legend('membrane voltage','current')
subplot(2,2,4)
plot(t,v_relative_high,t,I_relative_high)
xlabel('Time (ms)')
ylabel('Voltage (mV)')
subtitle('One pulse of 20nA and one of 80nA, 7ms apart')
legend('membrane voltage','current')




%% functions
% HH model function

function [v,n,m,h] = HH_model(t,I)

    % Constants
    g_k = 36; % permability for potassium
    g_na = 120; % permability for sodium
    g_l = 0.3; % permability of leak current
    e_k = -12; % nerst potential of potassium
    e_na = 115; % nerst potetial for sodium
    e_l = 10.6; % nerst potential for leak current
    c = 1; % capacity of membrane
    
    % Setting start point (steady-state before injection)
    v = 0;
    a_n = 0.01 * ( (10-v) / (exp((10-v)/10)-1) ); % alpha n equation
    b_n = 0.125*exp(-v/80); % beta n equation
    a_m = 0.1*( (25-v) / (exp((25-v)/10)-1) ); % alpha m equation
    b_m = 4*exp(-v/18); % beta m equation
    a_h = 0.07*exp(-v/20); % alpha h equation
    b_h = 1/(exp((30-v)/10)+1); % beta h equation
    tau_n = 1/(a_n+b_n); % tau n equation
    n_inf = a_n/(a_n+b_n); % n infinity equation
    tau_m = 1/(a_m+b_m); % tau m equation
    m_inf = a_m/(a_m+b_m); % m infinity equation
    tau_h = 1/(a_h+b_h); % tau h equation
    h_inf = a_h/(a_h+b_h); % h infinity equation
    
    % at start point, these equations apply
    n(1) = n_inf;
    m(1) = m_inf;
    h(1) = h_inf;

    % estimating the changes while injecting external current, using Euler
    % method of solving differetial equations

    for i = 1:(length(t)-1)
        a_n(i) = 0.01*((10-v(i))/(exp((10-v(i))/10)-1));
        b_n(i) = 0.125*exp(-v(i)/80);
        a_m(i) = 0.1*((25-v(i))/(exp((25-v(i))/10)-1));
        b_m(i) = 4*exp(-v(i)/18);
        a_h(i) = 0.07*exp(-v(i)/20);
        b_h(i) = 1/(exp((30-v(i))/10)+1);
        tau_n(i) = 1/(a_n(i)+b_n(i));
        n_inf(i) = a_n(i)/(a_n(i)+b_n(i));
        tau_m(i) = 1/(a_m(i)+b_m(i));
        m_inf(i) = a_m(i)/(a_m(i)+b_m(i));
        tau_h(i) = 1/(a_h(i)+b_h(i));
        h_inf(i) = a_h(i)/(a_h(i)+b_h(i));


        % calculating the different currents
        i_na = (m(i)^3) * g_na * h(i) * (v(i)-e_na); % current of sodium
        i_k = (n(i)^4) * g_k * (v(i)-e_k); % current of potassium
        i_l = g_l *(v(i)-e_l); % leak current
        i_tot = I(i) - i_k - i_na - i_l; % sum of all currents 
        
        % membrane voltage
        v(i+1) = v(i) + 0.01*i_tot/c; % the voltage change
        
        % n, m, h
        n(i+1) = n(i) + 0.01*((n_inf(i)-n(i))/tau_n(i)); 
        m(i+1) = m(i) + 0.01*((m_inf(i)-m(i))/tau_m(i)); 
        h(i+1) = h(i) + 0.01*((h_inf(i)-h(i))/tau_h(i));
    
    end

    v = v-70; % adjusting value for resting potential of the membrane

end


% one-pulse current function

    % t = time course 
    % A = current magnitude, 
    % pulse_start = start point of pulse injection
    % pulse_end = end point of pulse injection

function I = one_pulse(t,A,pulse_start,pulse_end)
    when = zeros(1,length(t)); 
    inter = t(1,length(t))/length(t); % intervals between time points
    
    % setting the start point

    if pulse_start == 0
        when(1) = 1;
    else
        when(1,round(pulse_start/inter)) = 1;
    end

    when(1,round(pulse_end/inter)+1) = 2; % setting end-point

    % creating the current accordingly
    I(1,1:find(when==1)) = 0;
    I(1,find(when==1):find(when==2)) = A;
    I(1,find(when==2)+1:length(t)) = 0;
end

