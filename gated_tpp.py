import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from sklearn.metrics import f1_score
import kernel_functions
sys.path.append('util')
sys.path.append('util')


class gated_tpp(nn.Module):

    def __init__(self, num_types, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_types = num_types
        self.encoder = Encoder(num_types, d_model)
        self.norm = nn.LayerNorm(d_model * 2, eps=1e-6)
        self.decoder = Decoder(num_types, d_model * 2, dropout)

    def forward(self, event_type, event_time):
        scores, embeddings, _ = self.encoder(event_type, event_time)

        hidden = torch.matmul(scores, embeddings)
        hidden = self.norm(hidden)
        return self.decoder(hidden)

    def calculate_loss(self, batch_arrival_times, sampled_arrival_times, batch_types, batch_probs):

        ## Shift the times because we are predicting for the next event.
        arrival_times = batch_arrival_times[:, 1:]
        sampled_times = sampled_arrival_times[:, :-1]

        ## l-1 loss
        loss = torch.abs(arrival_times - sampled_times)
        seq_length_mask = (batch_types[:, 1:] != 0) * 1
        batch_loss = loss * seq_length_mask
        time_loss = batch_loss.sum()

        non_event_mask_prob = torch.ones((batch_probs.size(0), batch_probs.size(1), 1)).to(batch_arrival_times.device)
        probs = torch.cat([non_event_mask_prob, batch_probs], dim=-1)
        one_hot_encodings = one_hot_embedding(batch_types[:, 1:], self.num_types + 1)
        cross_entropy_loss = -(one_hot_encodings * torch.log(probs[:, :-1, :])).sum(-1)
        cross_entropy_loss = cross_entropy_loss * seq_length_mask
        mark_loss = cross_entropy_loss.sum()

        return time_loss + mark_loss

    def train_epoch(self, dataloader, optimizer, params):

        epoch_loss = 0
        events = 0
        for batch in dataloader:
            optimizer.zero_grad()

            event_time, arrival_time, event_type, _ = map(lambda x: x.to(params.device), batch)
            predicted_times, probs = self(event_type, event_time)

            batch_loss = self.calculate_loss(arrival_time, predicted_times, event_type, probs)

            epoch_loss += batch_loss.item()
            events += ((event_type != 0).sum(-1) - 1).sum()
            events += ((event_type != 0).sum(-1) - 1).sum()
            batch_loss.backward()
            optimizer.step()
        return epoch_loss, events

    def validate_epoch(self, dataloader, device='cpu'):

        epoch_loss = 0
        events = 0
        with torch.no_grad():
            last_errors = []
            all_errors = []
            last_predicted_types = []
            last_actual_types = []
            accuracy = 0
            for batch in dataloader:

                event_time, arrival_time, event_type, _ = map(lambda x: x.to(device), batch)
                predicted_times, probs = self(event_type, event_time)
                batch_loss = self.calculate_loss(arrival_time, predicted_times, event_type, probs)

                epoch_loss += batch_loss
                events += ((event_type != 0).sum(-1) - 1).sum()

                last_event_index = (event_type != 0).sum(-1) - 2
                errors = predicted_times[:, :-1] - arrival_time[:, 1:]
                seq_index = 0

                predicted_events = torch.argmax(probs, dim=-1) + 1  ## Events go from 1 to N in the dataset
                type_prediction_hits = (predicted_events[:, :-1] == event_type[:, 1:]) * 1

                ## Clean Up TO DO
                actual_type = event_type[:, 1:]
                predicted_type = predicted_events[:, :-1]
                for idx in last_event_index:
                    last_errors.append(errors[seq_index][idx].unsqueeze(-1))
                    all_errors.append(errors[seq_index][:idx + 1])
                    last_predicted_types.append(predicted_type[seq_index][idx].item())
                    last_actual_types.append(actual_type[seq_index][idx].item())
                    accuracy += type_prediction_hits[seq_index][idx].item()

            last_errors = torch.cat(last_errors)
            last_RMSE = (last_errors ** 2).mean().sqrt()
            all_errors = torch.cat(all_errors)
            all_RMSE = (all_errors ** 2).mean().sqrt()
            last_event_accuracy = accuracy / len(dataloader.dataset.event_type)

            last_f1_score = f1_score(last_actual_types, last_predicted_types, average='micro')

            print(f'Micro F-1:{last_f1_score}')
        return epoch_loss, events, last_f1_score, last_RMSE, last_event_accuracy


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self,
                 num_types, d_model):
        super().__init__()

        self.d_model = d_model
        self.num_types = num_types

        self.embedding = BiasedPositionalEmbedding(d_model, max_len=4096)
        self.type_emb = nn.Embedding(num_types + 1, d_model, padding_idx=0)
        self.type_emb_prediction = nn.Embedding(num_types + 1, d_model, padding_idx=0)
        self.kernel = kernel_functions.sigmoid_gated_kernel(num_types, d_model)

    def forward(self, event_type, event_time):

        # Temporal Encoding
        temp_enc = self.embedding(event_type, event_time)
        # Type Encoding
        type_embedding = self.type_emb(event_type)
        # Calculate Pairwise Time and Type Encodings
        xd_bar, xd = get_pairwise_type_embeddings(type_embedding)
        combined_embeddings = torch.cat([xd_bar, xd], dim=-1)
        xt_bar, xt = get_pairwise_times(event_time)
        t_diff = torch.abs(xt_bar - xt)

        if self.num_types == 1:
            hidden_vector = temp_enc
        else:
            hidden_vector = torch.cat([temp_enc, type_embedding], dim=-1)

        # Future Masking
        subsequent_mask = get_subsequent_mask(event_type)
        scores = self.kernel(t_diff, combined_embeddings)
        scores = scores.masked_fill_(subsequent_mask == 0, value=0)

        return scores, hidden_vector, t_diff


class Decoder(nn.Module):


    def __init__(self,
                 num_types, d_model, dropout):
        super().__init__()

        self.d_model = d_model
        self.num_types = num_types
        self.predictor = generative_network(num_types, d_model, dropout)

    def forward(self, hidden):
        return self.predictor(hidden)


class generative_network(nn.Module):

    def __init__(self,num_types, d_model, dropout=0.1, layers=1, sample_size=50):
        super().__init__()

        self.d_model = d_model
        self.num_types = num_types
        self.samples = sample_size
        self.layers = layers


        self.mean = None
        self.std = None
        self.input_weights = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for i in range(layers)])
        self.noise_weights = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for i in range(layers)])

        self.event_time_calculator = nn.Linear(d_model, 1, bias=False)
        self.event_type_predictor = nn.Sequential(nn.Linear(d_model, num_types, bias=False))
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden):
        b_n, s_n, h_n = hidden.size()
        sample = self.samples

        mark_probs = F.softmax(self.event_type_predictor(hidden), -1)

        for i in range(self.layers):
            noise = torch.rand((b_n, s_n, sample, h_n), device=hidden.device)
            noise_sampled = self.noise_weights[i](noise)
            hidden = torch.relu(noise_sampled + self.input_weights[i](hidden)[:, :, None, :])

        mean = nn.functional.softplus(self.event_time_calculator(hidden)).squeeze(-1).mean(-1)
        std = nn.functional.softplus(self.event_time_calculator(hidden)).squeeze(-1).std(-1)

        return mean, mark_probs

def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=0)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    subsequent_mask = (subsequent_mask - 1) ** 2
    return subsequent_mask


class BiasedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        self.register_buffer('position', position)
        self.register_buffer('div_term', div_term)

        self.Wt = nn.Linear(1, d_model // 2, bias=False)

    def forward(self, x, interval):
        phi = self.Wt(interval.unsqueeze(-1))
        aa = len(x.size())
        if aa > 1:
            length = x.size(1)
        else:
            length = x.size(0)
        arc = (self.position[:length] * self.div_term).unsqueeze(0)
        pe_sin = torch.sin(arc + phi)
        pe_cos = torch.cos(arc + phi)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)

        return pe


def one_hot_embedding(labels, num_classes: int) -> torch.Tensor:
    """Embedding labels to one-hot form. Produces an easy-to-use mask to select components of the intensity.
    Args:
        labels: class labels, sized [N,].
        num_classes: number of classes.
    Returns:
        (tensor) encoded labels, sized [N, #classes].
    """
    device = labels.device
    y = torch.eye(num_classes).to(device)
    return y[labels]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_pairwise_times(event_time):
    xt_bar = event_time.unsqueeze(1). \
        expand(event_time.size(0), event_time.size(1), event_time.size(1))
    xt = xt_bar.transpose(1, 2)
    return xt_bar, xt


def get_pairwise_type_embeddings(embeddings):
    xd_bar = embeddings.unsqueeze(1).expand(embeddings.size(
        0), embeddings.size(1), embeddings.size(1), embeddings.size(-1))
    xd = xd_bar.transpose(1, 2)

    return xd_bar, xd
