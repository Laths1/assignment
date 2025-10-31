"""
This file contains the implementation of the Transformer FeedForward and it's integrated with Decoder
Classes have been added to this file for cluster training.
Also Lockdown is a transformer.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class WCST:
    
    def __init__(self, batch_size):
        self.colours = ['red','blue','green','yellow']
        self.shapes = ['circle','square','star','cross']
        self.quantities = ['1','2','3','4']
        self.categories = ['C1','C2','C3','C4']
        self.category_feature = np.random.choice([0])
        self.gen_deck()
        self.batch_size = batch_size

    def gen_deck(self):
        cards = []
        for colour in self.colours:
            for shape in self.shapes:
                for quantity in self.quantities:
                    cards = cards + [(colour, shape, quantity)]
        self.cards = np.array(cards)
        self.card_indices = np.arange(len(cards))

    def context_switch(self):
        self.category_feature = np.random.choice(np.delete([0,1,2],self.category_feature))

    def gen_batch(self):
        batch_size = self.batch_size
        while True:
            prev_feature = self.category_feature
            category_level = np.abs(self.category_feature - 2)+1
            card_partitions = [np.concatenate([np.arange(4**(category_level-1)) + feature_value*(4**(category_level-1))
                              + start for start in np.arange(0,64,int(4**(category_level)))])
                              for feature_value in range(4)]
            category_cards = np.vstack([np.random.choice(card_partition, batch_size, replace=True) for card_partition in card_partitions]).T
            category_cards = category_cards[np.arange(batch_size)[:,np.newaxis], [np.random.permutation(4) for _ in range(batch_size)]]
            category_cards_feature = (category_cards % (4**category_level)) // (4**(category_level-1))
            available_cards = np.delete(np.outer(np.ones((batch_size,1)),self.card_indices).reshape(-1),\
                                       (category_cards+np.arange(batch_size)[:,np.newaxis]*64).reshape(-1)).reshape(batch_size, 60)
            example_cards = available_cards[np.arange(batch_size),np.random.randint(0,60,(batch_size))]
            example_cards_feature = (example_cards % (4**category_level)) // (4**(category_level-1))
            example_labels = np.argmin(np.abs(category_cards_feature - example_cards_feature[:,np.newaxis]), axis=1)
            used_cards = np.hstack([category_cards,example_cards[:,np.newaxis]]).astype(int)
            available_cards = np.delete(np.outer(np.ones((batch_size,1)),self.card_indices).reshape(-1),\
                                       (used_cards+np.arange(batch_size)[:,np.newaxis]*64).reshape(-1)).reshape(batch_size, 59)
            question_cards = available_cards[np.arange(batch_size),np.random.randint(0,59,(batch_size))]
            question_cards_feature = (question_cards % (4**category_level)) // (4**(category_level-1))
            question_labels = np.argmin(np.abs(category_cards_feature - question_cards_feature[:,np.newaxis]),axis=1)
            yield np.hstack([category_cards,example_cards[:,np.newaxis],np.ones((batch_size,1))*68,\
                          example_labels[:,np.newaxis]+64,np.ones((batch_size,1))*69]),\
                   np.hstack([question_cards[:,np.newaxis],np.ones((batch_size,1))*68,question_labels[:,np.newaxis]+64])

    def visualise_batch(self,batch):
        trials = []
        batch = np.hstack(batch)
        for trial_idx in range(batch.shape[0]):
            trial = batch[trial_idx].astype(int)
            trial_cards = []
            for token_idx in trial:
                if token_idx < 64:
                    trial_cards = trial_cards + [self.cards[token_idx]]
                elif token_idx < 68:
                    trial_cards = trial_cards + [self.categories[token_idx-64]]
                elif token_idx == 68:
                    trial_cards = trial_cards + ['SEP']
                elif token_idx == 69:
                    trial_cards = trial_cards + ['EOS']
            trials = trials + [trial_cards]
            print(trial_cards)
        print("Feature for Classification: ", self.category_feature, "\n")
        return trials

    def get_card_features(self, card_index):
        if card_index < 0 or card_index >= 64:
            raise ValueError("Index must be between 0 and 63 for a card.")

        quantity_value = card_index % 4 
        shape_value = (card_index // 4) % 4
        colour_value = (card_index // 16) % 4
        return (colour_value, shape_value, quantity_value)

class Dataset_Loader:
    def __init__(self, training_batch, classification_batch, train_split, val_split, test_split, context_switch_interval):
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.classification_batch = classification_batch
        self.training_batch = training_batch
        self.context_switch_interval = context_switch_interval

    def _trial_key(self, input_batch, target_batch):
        """
        Create a unique key per trial including:
        - category cards
        - example card
        - example label
        - question card
        - question label (target)
        """
        keys = []
        for i in range(input_batch.shape[0]):
            cat_cards = tuple(input_batch[i, :4])
            example_card = input_batch[i, 4]
            example_label = input_batch[i, 5]
            question_card = input_batch[i, 6]
            question_label = target_batch[i, 0]
            keys.append((cat_cards,
                         example_card,
                         example_label,
                         question_card,
                         question_label))
        return keys

    def load_data(self):
        """
        The sum of train, val and test splits should equal training_batch.
        Ensures no duplicate trials across train/val/test.
        """
        wcst_env = WCST(self.classification_batch)
        train_data = []
        val_data = []
        test_data = []
        seen = set()  

        n_train = int(np.floor(self.training_batch * self.train_split))
        n_val = int(np.floor(self.training_batch * self.val_split))
        n_test = int(np.floor(self.training_batch * self.test_split))

        def fill_dataset(n_items, dataset_list, start_count=0):
            count = 0
            while count < n_items:
                if (start_count + count) % self.context_switch_interval == 0 and (start_count + count) != 0:
                    wcst_env.context_switch()

                input_batch, target_batch = next(wcst_env.gen_batch())
                keys = self._trial_key(input_batch, target_batch)

                new_indices = [i for i, k in enumerate(keys) if k not in seen]
                if len(new_indices) == 0:
                    continue  # skip this batch 

                for i in new_indices:
                    seen.add(keys[i])
                    dataset_list.append((input_batch[i:i+1], target_batch[i:i+1]))
                    count += 1
                    if count >= n_items:
                        break
            return count

        # fill each split ensuring uniqueness
        c1 = fill_dataset(n_train, train_data, start_count=0)
        c2 = fill_dataset(n_val, val_data, start_count=c1)
        c3 = fill_dataset(n_test, test_data, start_count=c1+c2)

        return train_data, val_data, test_data
class RuleDetector:
    def __init__(self, model, test_data, wcst_env, batch_size=32):
        self.model = model
        self.model.eval()
        self.test_loader = DataLoader(self._prepare_data(test_data), batch_size=batch_size)
        self.wcst_env = wcst_env
        self.num_heads = model.decoders[0].cross_attn.h
        self.num_layers = len(model.decoders)
        self.features = ['Colour', 'Shape', 'Quantity']

    def _prepare_data(self, data):
        inputs = np.vstack([d[0] for d in data])
        targets = np.vstack([d[1] for d in data])
        return TensorDataset(
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long)
        )

    def _get_match_positions(self, src_batch):
        match_positions_batch = []

        CAT_CARD_POSITIONS = torch.arange(4) 
        
        for i in range(src_batch.size(0)):
            src_i = src_batch[i].cpu().numpy()

            example_card_idx = src_i[4] 

            example_features = self.wcst_env.get_card_features(example_card_idx)

            match_indices = {'Colour': [], 'Shape': [], 'Quantity': []}

            for pos in CAT_CARD_POSITIONS:
                cat_card_idx = src_i[pos]
                cat_features = self.wcst_env.get_card_features(cat_card_idx)
                
                if cat_features[0] == example_features[0]: # Colour match
                    match_indices['Colour'].append(pos.item())
                if cat_features[1] == example_features[1]: # Shape match
                    match_indices['Shape'].append(pos.item())
                if cat_features[2] == example_features[2]: # Quantity match
                    match_indices['Quantity'].append(pos.item())
            
            match_positions_batch.append(match_indices)
        return match_positions_batch

    def analyze_attention(self):
        head_feature_scores = {
            (layer, head): {'Colour': 0.0, 'Shape': 0.0, 'Quantity': 0.0, 'Count': 0}
            for layer in range(self.num_layers)
            for head in range(self.num_heads)
        }
        
        with torch.no_grad():
            for src, tgt in self.test_loader:
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]

                _, attn_maps_all = self.model(src, tgt_input)

                batch_match_positions = self._get_match_positions(src)
                
                for layer_idx, attn_maps in enumerate(attn_maps_all):
                    
                    cross_w = attn_maps["cross"] 

                    cross_attention_final_query = cross_w[:, :, -1, :] 
                    
                    for batch_idx in range(src.size(0)):
                        attn_batch = cross_attention_final_query[batch_idx] # (H, T_src)
                        match_pos = batch_match_positions[batch_idx]
                        
                        for feature_key in self.features:
                            positions = match_pos[feature_key]
                            
                            if not positions:
                                continue 

                            attn_sum = attn_batch[:, positions].sum(dim=-1) 
                            
                            for head_idx in range(self.num_heads):
                                key = (layer_idx, head_idx)
                                head_feature_scores[key][feature_key] += attn_sum[head_idx].item()
                                head_feature_scores[key]['Count'] += 1

        average_scores = {}
        for key, data in head_feature_scores.items():
            if data['Count'] > 0:
                average_scores[key] = {
                    f: data[f] / data['Count']
                    for f in self.features
                }
            else:
                average_scores[key] = {f: 0.0 for f in self.features}
                
        return average_scores

    def visualize_specialization(self, average_scores):

        
        all_data = []
        for (layer, head), scores in average_scores.items():
            for feature, score in scores.items():
                all_data.append({
                    'Layer': f'L{layer}', 
                    'Head': head, 
                    'Feature': feature, 
                    'Attention_Score': score
                })
                
        df = pd.DataFrame(all_data)

        fig, axes = plt.subplots(self.num_layers, 1, figsize=(10, 3 * self.num_layers), sharex=True)
        
        if self.num_layers == 1:
            axes = [axes] 

        for layer_idx, ax in enumerate(axes):
            layer_df = df[df['Layer'] == f'L{layer_idx}']
            pivot_table = layer_df.pivot_table(
                index='Head', 
                columns='Feature', 
                values='Attention_Score'
            )
            
            sns.heatmap(
                pivot_table, 
                annot=True, 
                fmt=".3f", 
                cmap="YlGnBu", 
                cbar_kws={'label': 'Average Attention Score'},
                ax=ax
            )
            ax.set_title(f'Decoder Cross-Attention Specialization - Layer {layer_idx}')
            ax.set_ylabel('Head Index')
            
        plt.tight_layout()
        plt.savefig('all_attention.png')
        plt.show()

class SelfAttention(nn.Module):
    """
    Note:
        input vectors: X
        query vectors: Q = XW(q), where W(q) is the query weight matrix
        key vectors: K = XW(k), where W(k) is the key weight matrix
        value vectors: V = XW(v), where W(v) is the value weight matrix
        N(_) -> number of vectors
        D(_) -> dimension of vectors

        input_vectors -> Input embeddings shape(N(X) x D(X))
        key_matrix -> Key weight matrix shape(D(X) x D(Q))
        value_matrix -> Value weight matrix shape(D(X) x D(V))
        query_matrix -> Query weight matrix shape(D(X) x D(Q))
    """
    def __init__(self, embed_dim, qkv_dim):
        """
        Args:
            embed_dim: size of input embeddings (D(X))
            qkv_dim: size of Q, K, V vectors (D(Q)=D(K)=D(V))
        """
        super().__init__()
        # Learnable weight matrices
        self.W_q = nn.Linear(embed_dim, qkv_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, qkv_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, qkv_dim, bias=False)
        
    def scaled_dot_product(self,query, key, query_dim):
        scaling_factor = 1 / np.sqrt(query_dim)
        product = torch.einsum('bqd,bkd->bqk', query, key) #tensor product
        return scaling_factor * product

    def forward(self, x):
        """
        x: (batch_size, seq_len, embed_dim)
        """
        Q = self.W_q(x)  # shape(N(X) x D(Q))
        K = self.W_k(x)    # shape(N(X) x D(K))
        V = self.W_v(x)  # shape(N(X) x D(V))

        similarity = self.scaled_dot_product(Q, K, Q.shape[-1])  # shape(N(X) x N(X))
        attention_weights = torch.softmax(similarity, dim=-1)  # shape(N(X) x N(X)) softmax along columns
        output = torch.matmul(attention_weights, V)  # shape(N(X) x D(V))
        return output
        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.h = num_heads
        self.dh = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model, bias=True)
        self.Wk = nn.Linear(d_model, d_model, bias=True)
        self.Wv = nn.Linear(d_model, d_model, bias=True)
        self.Wo = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x):
        B, T, _ = x.size()
        return x.view(B, T, self.h, self.dh).transpose(1, 2)

    def _causal_mask(self, Tq, Tk, device):
        i = torch.arange(Tq, device=device).unsqueeze(1)
        j = torch.arange(Tk, device=device).unsqueeze(0)
        return j <= i

    def forward(self, query, key, value, attn_mask=None, additive_mask=None, causal=False):
        B, Tq, _ = query.size()
        _, Tk, _ = key.size()

        q = self._shape(self.Wq(query))
        k = self._shape(self.Wk(key))
        v = self._shape(self.Wv(value))

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / (self.dh ** 0.5)

        if causal:
            causal_keep = self._causal_mask(Tq, Tk, query.device)
            attn_logits = attn_logits.masked_fill(
                ~causal_keep.unsqueeze(0).unsqueeze(0),
                float('-inf')
            )

        if attn_mask is not None:
            attn_mask = attn_mask.bool().unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits.masked_fill(~attn_mask, float('-inf'))

        if additive_mask is not None:
            attn_logits = attn_logits + additive_mask

        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v)

        context = context.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        out = self.Wo(context)
        return out, attn_weights

###########
# Integrate Feed Forward
###########
class FeedForward(nn.Module):
    """
    Position-wise feed-forward network used inside Transformer blocks.
    Applies the same MLP to every time step independently: d_model -> d_ff -> d_model.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

        if activation.lower() == "gelu":
            self.act = nn.GELU()
        elif activation.lower() == "relu":
            self.act = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        embeddings = self.token_embedding(x) * torch.sqrt(
            torch.tensor(self.d_model, dtype=torch.float32, device=x.device)
        )
        return embeddings

class PositionalEncoder:
    """
    Sinusoidal Positional Encoder
    """
    @staticmethod
    def encode(input_seq, model_dim):
        """
        Args:
            input_seq -> The whole input sequence [batch, seq_len]
            model_dim -> The model dimension(hyperparameter)
            
        returns: [positional encodings] as tensor
        """
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        pe = np.zeros((batch_size, seq_len, model_dim))

        for pos in range(seq_len):
            for i in range(0, model_dim, 2):
                theta = pos / (10000 ** (2 * i / model_dim))
                pe[:, pos, i] = np.sin(theta)
                pe[:, pos, i+1] = np.cos(theta)

        return torch.tensor(pe, dtype=torch.float32)
  
class Encoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model) 
        
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model) 
        self.norm2 = nn.LayerNorm(d_model) 
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Feed forward
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x

class Decoder(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln_self = nn.LayerNorm(d_model) 
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

        self.ln_cross = nn.LayerNorm(d_model) 
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

        self.ln_ff = nn.LayerNorm(d_model) 
        self.ff = FeedForward(d_model=d_model, d_ff=4 * d_model, dropout=dropout, activation="gelu")

        self.drop = nn.Dropout(dropout)

    def _expand_key_mask(self, pad_mask: torch.Tensor, Tq: int, heads: int) -> torch.Tensor:
        
        B, Tk = pad_mask.size()
        keep = pad_mask.bool().unsqueeze(1).unsqueeze(2)        
        keep = keep.expand(B, heads, Tq, Tk)                    
        return keep

    def forward(self, x, enc_out, tgt_pad=None, src_pad=None):
        attn_maps = {}

        B, T_tgt, _ = x.size()
        self_keep = None
        if tgt_pad is not None:
            self_keep = self._expand_key_mask(tgt_pad, T_tgt, self.self_attn.h)

        x_norm = self.ln_self(x)
        self_out, self_w = self.self_attn(
            query=x_norm, key=x_norm, value=x_norm,
            attn_mask=self_keep,  
            causal=True
        )
        x = x + self.drop(self_out)
        attn_maps["self"] = self_w  

        B, T_src, _ = enc_out.size()
        cross_keep = None
        if src_pad is not None:
            cross_keep = self._expand_key_mask(src_pad, T_tgt, self.cross_attn.h)

        x_norm = self.ln_cross(x)
        cross_out, cross_w = self.cross_attn(
            query=x_norm, key=enc_out, value=enc_out,
            attn_mask=cross_keep,  
            causal=False
        )
        x = x + self.drop(cross_out)
        attn_maps["cross"] = cross_w  

        # Feed forward
        x_norm = self.ln_ff(x)
        ff_out = self.ff(x_norm)
        x = x + self.drop(ff_out)

        return x, attn_maps


class Transformer(nn.Module):
    def __init__(self,
                 num_encoder_layers,
                 num_decoder_layers,
                 d_model,
                 num_heads,
                 d_ff,
                 src_vocab_size,
                 tgt_vocab_size,
                 max_len=5000,
                 dropout=0.1):
        super().__init__()

        # Embeddings
        self.src_embedding = Embedding(src_vocab_size, d_model)
        self.tgt_embedding = Embedding(tgt_vocab_size, d_model)

        # Encoder/Decoder stacks
        self.encoder = Encoder(num_encoder_layers, d_model, num_heads, d_ff, dropout)
        self.decoders = nn.ModuleList([
            Decoder(d_model, num_heads, dropout)
            for _ in range(num_decoder_layers)
        ])

        # Output projection to target vocab
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        # Positional encoding
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: [batch, src_seq_len]
        tgt: [batch, tgt_seq_len]
        returns: logits over target vocab [batch, tgt_seq_len, tgt_vocab_size]
        """

        # ----- ENCODER -----
        src_embed = self.src_embedding(src)
        src_pe = PositionalEncoder.encode(src, src_embed.shape[-1]).to(src.device)
        src_embed = self.dropout(src_embed + src_pe)
        enc_out = self.encoder(src_embed, src_mask)

        # ----- DECODER -----
        tgt_embed = self.tgt_embedding(tgt)
        tgt_pe = PositionalEncoder.encode(tgt, tgt_embed.shape[-1]).to(tgt.device)
        tgt_embed = self.dropout(tgt_embed + tgt_pe)

        x = tgt_embed
        attn_maps_all = []

        for decoder in self.decoders:
            x, attn_maps = decoder(x, enc_out, tgt_pad=tgt_mask, src_pad=src_mask)
            attn_maps_all.append(attn_maps)

        logits = self.fc_out(x)
        return logits, attn_maps_all

class Trainer:
    def __init__(self, model, train_data, val_data, test_data, lr=0.001, batch_size=32):
        self.model = model
        self.train_data = self._prepare_data(train_data)
        self.val_data = self._prepare_data(val_data)
        self.test_data = self._prepare_data(test_data)
        self.batch_size = batch_size
        
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
    def _prepare_data(self, data):
        """Convert list of (input, target) pairs to proper tensors"""
        inputs = np.vstack([d[0] for d in data])
        targets = np.vstack([d[1] for d in data])
        return TensorDataset(
            torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long)
        )
    
    def create_masks(self, src, tgt):
        """Create masks for transformer"""
        src_mask = None
        tgt_mask = None
        return src_mask, tgt_mask
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        
        for src, tgt in tqdm(train_loader, desc="Training"):
            src, tgt = src.to(device), tgt.to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            self.optimizer.zero_grad()
            
            logits, _ = self.model(src, tgt_input)
            
            loss = self.criterion(logits.reshape(-1, logits.size(-1)), 
                                tgt_output.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits[:, -1, :], dim=-1)
            correct += (preds == tgt[:, -1]).sum().item()
            total += preds.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total if total > 0 else 0
        return avg_loss, accuracy
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        val_loader = DataLoader(self.val_data, batch_size=self.batch_size)
        
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                logits, _ = self.model(src, tgt_input)
                
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), 
                                    tgt_output.reshape(-1))
                
                total_loss += loss.item()
                
                preds = torch.argmax(logits[:, -1, :], dim=-1)
                correct += (preds == tgt[:, -1]).sum().item()
                total += preds.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total if total > 0 else 0
        return avg_loss, accuracy
    
    def test(self):
        self.model.eval()
        correct = 0
        total = 0
        
        test_loader = DataLoader(self.test_data, batch_size=self.batch_size)
        
        with torch.no_grad():
            for src, tgt in test_loader:
                src, tgt = src.to(device), tgt.to(device)
                
                tgt_input = tgt[:, :-1]
                logits, _ = self.model(src, tgt_input)
                
                preds = torch.argmax(logits[:, -1, :], dim=-1)
                correct += (preds == tgt[:, -1]).sum().item()
                total += preds.size(0)
        
        accuracy = correct / total if total > 0 else 0
        return accuracy
    
    def train(self, epochs=50, early_stopping_patience=10):
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("Starting training...")
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            epoch_time = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_transformer_wcst.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model for final testing
        self.model.load_state_dict(torch.load('best_transformer_wcst.pth'))
        
        # Final test
        test_acc = self.test()
        print(f"\nFinal Test Accuracy: {test_acc:.4f}")
        
        self.plot_training()
    
    def plot_training(self):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.val_accuracies, label='Val Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Validation Accuracy')
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.show()

if __name__ == "__main__":
    # Hyperparameters
    training_batch = 100000  # Total number of training examples
    classification_batch = 1000  # Batch size for data generation
    train_split = 0.7
    val_split = 0.15
    test_split = 0.15
    context_switch_interval = 1000000
    
    # Model parameters
    vocab_size = 70
    d_model = 128
    num_heads = 4
    num_encoder_layers = 6
    num_decoder_layers = 2
    d_ff = 1024
    dropout = 0.1
    
    # Training parameters
    batch_size = 64
    learning_rate = 0.001
    epochs = 20
    
    print("Loading dataset...")
    # Build Dataset
    data_loader = Dataset_Loader(
        training_batch=training_batch,
        classification_batch=classification_batch,
        train_split=train_split,
        val_split=val_split,
        test_split=test_split,
        context_switch_interval=context_switch_interval
    )
    train_data, val_data, test_data = data_loader.load_data()
    
    print(f"Dataset sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Build Model
    print("Building model...")
    model = Transformer(
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        dropout=dropout
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        lr=learning_rate,
        batch_size=batch_size
    )
    
    # Start training
    trainer.train(epochs=epochs)
    
    # Test on a few examples
    print("\nTesting on a few examples:")
    test_loader = DataLoader(trainer.test_data, batch_size=5, shuffle=True)
    src, tgt = next(iter(test_loader))
    src, tgt = src.to(device), tgt.to(device)
    
    with torch.no_grad():
        tgt_input = tgt[:, :-1]
        logits, _ = model(src, tgt_input)
        preds = torch.argmax(logits[:, -1, :], dim=-1)
        
        print("Predictions vs Targets:")
        for i in range(min(5, src.size(0))):
            print(f"  Example {i+1}: Predicted {preds[i].item()}, Target {tgt[i, -1].item()}")

    trainer.model.load_state_dict(torch.load('best_transformer_wcst.pth'))
    wcst_env = WCST(classification_batch) 

    test_wcst_env = WCST(classification_batch) 


    _, _, interpret_test_data = data_loader.load_data()

    detector = RuleDetector(
        model=model, 
        test_data=interpret_test_data, 
        wcst_env=test_wcst_env, 
        batch_size=32
    )

    print("Analyzing attention head specialization...")
    avg_scores = detector.analyze_attention()
    
    print("\nFeature Attention Scores per Head (Average Attention to Matching Card Positions):")
    for (layer, head), scores in avg_scores.items():
        # Print the score and identify the specialized feature
        specialized_feature = max(scores, key=scores.get)
        print(f"L{layer} H{head}: C:{scores['Colour']:.4f} | S:{scores['Shape']:.4f} | Q:{scores['Quantity']:.4f} -> Specialized: {specialized_feature}")
        
    print("\nGenerating heatmap visualization...")
    detector.visualize_specialization(avg_scores)
