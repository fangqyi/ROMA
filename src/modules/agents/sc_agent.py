import torch
import torch.nn as nn
import torch.nn.functional as F


class SCAgent(nn.Module):  # FIXME: Does SCAgent need to extend nn.Module
    def __init__(self, input_shapes, output_shapes, args):
        super(SCAgent, self).__init__()
        self.args = args
        lstm_input_shape, dlstm_input_shape = input_shapes
        lstm_output_shape, dlstm_output_shape = output_shapes
        self.LSTM_agent = LSTMAgent(lstm_input_shape, lstm_output_shape, args)
        self.dLSTM_agent = DilatedLSTMAgent(dlstm_input_shape, dlstm_output_shape, args)
        self.g_lin_trans = nn.Linear(self.args.latent_state_dim, self.args.n_actions, bias=False)
        # bias set to off in linear projection to avoid constant zero goal

    def init_hidden(self, batch_size):
        lstm_init_state = self.LSTM_agent.init_hidden(batch_size)
        dlstm_init_state = self.dLSTM_agent.init_hidden(batch_size)
        return lstm_init_state, dlstm_init_state

    def forward(self, inputs, hidden_state, goals):
        lstm_inputs, dlstm_inputs = inputs
        lstm_hidden_state, dlstm_hidden_state = hidden_state
        lstm_outs, lstm_hidden_state = self.LSTM_agent(lstm_inputs, lstm_hidden_state)
        dlstm_outs, dlstm_hidden_state = self.dLSTM_agent(dlstm_inputs, dlstm_hidden_state)

        w_goals = self.g_lin_trans(goals)
        # linear projection in wt=φ(t∑i=t−c gi) eq. 4 in Feudal Net, goals = t∑i=t−c gi
        lstm_outs = F.softmax(torch.mul(w_goals, lstm_outs), dim=-1)
        # πt=SoftMax(Ut*wt) eq. 6 in Feudal Net

        return (lstm_outs, dlstm_outs), (lstm_hidden_state, dlstm_hidden_state)

    def parameters(self):
        return list(self.LSTM_agent.parameters()) + list(self.dLSTM_agent.parameters())

    def lstm_parameters(self):
        return self.LSTM_agent.parameters()

    def dlstm_parameters(self):
        return self.dLSTM_agent.parameters()


class DilatedLSTMAgent(nn.Module):
    def __init__(self, input_shape, output_shape, args):
        super(DilatedLSTMAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        self.fc1 = nn.Linear(input_shape, args.dilated_lstm_hidden_dim)
        self.dlstm = nn.LSTMCell(args.dilated_lstm_hidden_dim, args.dilated_lstm_hidden_dim)
        self.fc2 = nn.Linear(args.dilated_lstm_hidden_dim, output_shape)

    def init_hidden(self, batch_size):
        h0 = [(torch.zeros(batch_size * self.n_agents, self.args.dilated_lstm_hidden_dim),
               torch.zeros(batch_size * self.n_agents, self.args.dilated_lstm_hidden_dim)) for _ in range(self.args.horizon)]  # FIXEME: require_grad = is_training
        tick = 0
        return tick, h0

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        tick, hx = hidden_state
        hx[tick] = self.dlstm(x, hx[tick])
        outs = self.fc2(hx[tick][0])
        tick = (tick + 1) % self.args.horizon
        return outs, (tick, hx)

class LSTMAgent(nn.Module):
    def __init__(self, input_shape, output_shape, args):
        super(LSTMAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents

        self.fc1 = nn.Linear(input_shape, args.lstm_hidden_dim)
        # print("lstm fc1 input shape:{}".format(input_shape))
        # print("lstm fc1 output shape:{}".format(args.lstm_hidden_dim))
        self.lstm = nn.LSTMCell(args.lstm_hidden_dim, args.lstm_hidden_dim)
        self.fc2 = nn.Linear(args.lstm_hidden_dim, output_shape)

    def init_hidden(self, batch_size):
        # make hidden states on same device as model
        return (self.fc1.weight.new(batch_size * self.n_agents, self.args.lstm_hidden_dim).zero_(),
                self.fc1.weight.new(batch_size * self.n_agents, self.args.lstm_hidden_dim).zero_())

    def forward(self, inputs, hidden_state):
        # print("lstm forward fc1 shape:{}".format(inputs.shape))
        x = F.relu(self.fc1(inputs))
        h_in = (hidden_state[0].reshape(-1, self.args.lstm_hidden_dim),
                hidden_state[1].reshape(-1, self.args.lstm_hidden_dim))
        h = self.lstm(x, h_in)
        q = self.fc2(h[0])
        return q, h
